"""The task-progress CVAE: the latent split, the length-normalized margin, and the shipped encoder.

Three properties carry the design and each is pinned directly rather than inferred from a training
curve:

* **the KL never sees ``z_phase``** -- that asymmetry is the entire reason the phase channel gets
  used, so it is asserted bit-for-bit;
* **a pair never crosses an episode boundary** -- rows are one per flat step, so a naive ``idx + 1``
  would silently pair one episode's last window with the next episode's first, and the hinge would
  be trained on a fabricated transition;
* **``phase()`` never touches the decoder** -- the encoder is the deployed artifact, so the
  inference path is tested with the decoder rigged to raise.

The ground-truth phase is held out of training by construction, and one test asserts the model has
no argument through which it could arrive.
"""

from __future__ import annotations

import inspect
import json
import math

import pytest
import torch

from fm_bc.pusht.chunks import OBS_DIM, embed_state
from fm_bc.pusht.constants import WORKSPACE_PX
from fm_bc.pusht.normalize import normalizer_for, stats_from_data
from fm_bc.pusht.prepared import prepare
from fm_bc.pusht.progress import (
    ProgressCVAE, load_progress_model, monotonicity_hinge, progress_rows, sample_pairs,
)
from fm_bc.pusht.progress_training import (
    ProgressConfig, evaluate_progress, phase_baselines, spearman, train_progress,
)

from test_pusht_data import LENGTHS, _toy  # noqa: E402 -- pytest puts tests/ on sys.path

TO = 2
_SMALL = {"hidden": (16, 16), "z_style": 2}


def _pos(ep):
    return normalizer_for("box", stats_from_data(ep.state[:, :2]), box=(0.0, WORKSPACE_PX))


def _model(ep, **kwargs):
    torch.manual_seed(0)
    opts = {**_SMALL, **kwargs}
    return ProgressCVAE(pos_norm=_pos(ep), obs_steps=TO, **opts)


def _batch(rows, n=None):
    n = n or rows["obs"].shape[0]
    return torch.arange(n)


def _trained(ep, steps: int = 3000):
    """Train on episodes 0-3, holding out episode 4. Shared by the behavioural and OOD tests."""
    prep = prepare(ep, n_grid=8, val_ratio=0.2, seed=0, max_train_episodes=4)
    cfg = ProgressConfig(obs_steps=TO, steps=steps, batch_size=64, lr=3e-3, eval_every=10**9,
                         log_every=10**9, hidden=(32, 32), z_style=2, kl_warmup=(50, 200), seed=0)
    return train_progress(ep, prep, cfg, verbose=False)[0], cfg


# -- rows and pairing --------------------------------------------------------------------


def test_rows_are_one_per_step_of_the_selected_episodes():
    ep = _toy()
    rows = progress_rows(ep, to=TO)
    assert rows["obs"].shape == (len(ep), TO, 5)
    assert rows["cond"].shape == (len(ep), 5)
    ids = torch.tensor([1, 3])
    subset = progress_rows(ep, to=TO, episode_ids=ids)
    assert subset["obs"].shape[0] == LENGTHS[1] + LENGTHS[3]


def test_condition_is_the_episode_initial_state_for_every_row():
    """Exhaustive, not sampled: one wrong row is a silently mislabelled training example."""
    ep = _toy()
    rows = progress_rows(ep, to=TO)
    want = ep.state[ep.episode_starts[ep.episode_index[rows["step"]]]]
    assert torch.allclose(rows["cond"], want, atol=0)


@pytest.mark.parametrize("gap", ["fixed", "loguniform"])
def test_a_partner_is_always_later_in_the_same_episode(gap: str):
    """The test that catches ``idx + 1``: rows are contiguous across episode boundaries."""
    ep = _toy()
    rows = progress_rows(ep, to=TO)
    idx = _batch(rows)
    gen = torch.Generator().manual_seed(0)
    partner, valid, _, delta = sample_pairs(rows, idx, gap=gap, stride=2, generator=gen)
    v = valid.nonzero().flatten()
    assert v.numel() > 0
    assert torch.equal(rows["step"][partner[v]], rows["step"][v] + delta[v].long())
    assert torch.equal(rows["episode"][partner[v]], rows["episode"][v])


@pytest.mark.parametrize("stride", [1, 3])
def test_rows_within_stride_of_an_episode_end_have_no_partner(stride: int):
    ep = _toy()
    rows = progress_rows(ep, to=TO)
    _, valid, _, _ = sample_pairs(rows, _batch(rows), gap="fixed", stride=stride)
    assert int((~valid).sum()) == ep.n_episodes * stride
    # ...and they are exactly the last `stride` steps of each episode.
    tail = rows["step"] >= (ep.episode_ends[rows["episode"]] - stride)
    assert torch.equal(~valid, tail)


@pytest.mark.parametrize("stride", [1, 3])
def test_pairs_never_reach_into_a_dropped_episode(stride: int):
    ep = _toy()
    ids = torch.tensor([1, 3])
    rows = progress_rows(ep, to=TO, episode_ids=ids)
    _, valid, _, _ = sample_pairs(rows, _batch(rows), gap="fixed", stride=stride)
    assert int(valid.sum()) == sum(LENGTHS[i] - stride for i in ids.tolist())


def test_the_margin_is_normalised_by_episode_length():
    """A fixed margin is either infeasible for the longest episode or slack for the shortest."""
    ep = _toy()
    rows = progress_rows(ep, to=TO)
    idx = _batch(rows)
    _, _, margin, delta = sample_pairs(rows, idx, gap="fixed", stride=1, kappa=0.5)
    want = 0.5 * delta / (rows["length"] - 1.0).clamp(min=1.0)
    assert torch.allclose(margin, want, atol=0)
    # The 11-step episode is asked for a strictly smaller per-step rise than the 4-step one.
    long_ep = margin[rows["episode"] == 2][0]
    short_ep = margin[rows["episode"] == 1][0]
    assert float(long_ep) < float(short_ep)


def test_loguniform_gaps_stay_inside_their_episode():
    ep = _toy()
    rows = progress_rows(ep, to=TO)
    gen = torch.Generator().manual_seed(3)
    for _ in range(20):
        _, _, _, delta = sample_pairs(rows, _batch(rows), gap="loguniform", generator=gen)
        assert bool((delta <= (rows["length"] - 1).clamp(min=1)).all())
        assert bool((delta >= 1).all())


# -- the latent split --------------------------------------------------------------------


def test_encode_and_decode_shapes_and_the_phase_range():
    ep = _toy()
    model = _model(ep)
    rows = progress_rows(ep, to=TO)
    u, z, mu, logvar = model.encode(rows["obs"], rows["cond"])
    assert u.shape == z.shape == (len(ep),)
    assert mu.shape == logvar.shape == (len(ep), 2)
    assert bool(((z > 0) & (z < 1)).all())
    pos_hat, v = model.decode(z, mu, rows["cond"])
    assert pos_hat.shape == (len(ep), TO, 4)
    assert v.shape == (len(ep), TO, 2)
    assert model.decode_state(z, mu, rows["cond"]).shape == (len(ep), TO, 5)


def test_decode_state_round_trips_through_the_embedding():
    """Pins the atan2 and the normalizer inversion together."""
    ep = _toy()
    model = _model(ep)
    rows = progress_rows(ep, to=TO)
    _, z, mu, _ = model.encode(rows["obs"], rows["cond"])
    state = model.decode_state(z, mu, rows["cond"])
    pos_hat, v = model.decode(z, mu, rows["cond"])
    emb = embed_state(state, model.pos)
    assert emb.shape[-1] == OBS_DIM
    assert torch.allclose(emb[..., :4], pos_hat, atol=1e-4)
    unit = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    assert torch.allclose(emb[..., 4:6], unit, atol=1e-5)


def test_the_kl_term_never_sees_the_phase_channel():
    """The central requirement: z_phase is the rate-free channel, so it carries no KL cost."""
    ep = _toy()
    model = _model(ep)
    rows = progress_rows(ep, to=TO)
    torch.manual_seed(1)
    _, before = model.loss(rows["obs"], rows["cond"])
    with torch.no_grad():
        model.phase_head.weight.add_(5.0)
        model.phase_head.bias.add_(2.0)
    torch.manual_seed(1)
    _, after = model.loss(rows["obs"], rows["cond"])
    assert after["kl_loss"] == before["kl_loss"]
    assert after["z_phase_mean"] != before["z_phase_mean"]  # it did change the phase


def test_the_heading_is_weighted_against_the_repo_px_per_radian_constant():
    """Unweighted, the cos/sin pair would take ~74 % of the reconstruction gradient."""
    ep = _toy()
    model = _model(ep)
    assert float(model.w_theta) == pytest.approx((100.0 / 256.000256) ** 2, rel=1e-4)
    obs = ep.state[:8].unsqueeze(1).repeat(1, TO, 1)
    x = embed_state(obs, model.pos)
    # A pure-heading perturbation of size d costs w_theta * d^2; a positional one costs d^2.
    d = 0.01
    pos_hat, v = x[..., :4].clone(), x[..., 4:6].clone()
    ang = model.recon_error(obs, pos_hat, v * (1 + d) + d)
    shifted = model.recon_error(obs, pos_hat + d, v)
    assert float(ang.mean()) < float(shifted.mean())


# -- the hinge ---------------------------------------------------------------------------


def test_the_hinge_is_zero_only_when_the_advance_clears_the_margin():
    valid = torch.ones(3, dtype=torch.bool)
    margin = torch.full((3,), 0.1)
    cur = torch.zeros(3)
    loss, adv = monotonicity_hinge(cur, torch.full((3,), 0.2), valid, margin)
    assert float(loss) == pytest.approx(0.0)
    assert torch.allclose(adv, torch.full((3,), 0.2))
    equal, _ = monotonicity_hinge(cur, cur.clone(), valid, margin)
    assert float(equal) == pytest.approx(1.0)  # the full relative deficit
    down, _ = monotonicity_hinge(cur, torch.full((3,), -0.1), valid, margin)
    assert float(down) > 1.0


def test_invalid_pairs_do_not_contribute_to_the_monotonicity_term():
    """Guards the ``clamp(min=0)`` sentinel: invalid rows point at row 0 and must be masked."""
    ep = _toy()
    model = _model(ep)
    rows = progress_rows(ep, to=TO)
    idx = _batch(rows)
    partner, valid, margin, _ = sample_pairs(rows, idx, gap="fixed", stride=1)
    obs_next = rows["obs"][partner]
    torch.manual_seed(2)
    base, _ = model.loss(rows["obs"], rows["cond"], obs_next, valid, margin)
    garbled = obs_next.clone()
    garbled[~valid] = 1e3
    torch.manual_seed(2)
    other, _ = model.loss(rows["obs"], rows["cond"], garbled, valid, margin)
    assert float(base.detach()) == pytest.approx(float(other.detach()), abs=1e-6)


def test_total_loss_is_its_weighted_terms():
    ep = _toy()
    model = _model(ep, beta=0.02, mono_weight=2.0, sat_weight=0.05)
    rows = progress_rows(ep, to=TO)
    partner, valid, margin, _ = sample_pairs(rows, _batch(rows), gap="fixed", stride=1)
    for scale in (0.0, 0.5, 1.0):
        torch.manual_seed(4)
        total, x = model.loss(rows["obs"], rows["cond"], rows["obs"][partner], valid, margin,
                              beta_scale=scale)
        want = (x["recon_loss"] + 0.02 * scale * x["kl_loss"]
                + 2.0 * x["mono_loss"] + 0.05 * x["sat_loss"])
        assert float(total.detach()) == pytest.approx(want, rel=1e-5)


def test_free_bits_floor_the_per_dimension_kl():
    ep = _toy()
    model = _model(ep, free_bits=0.25)
    with torch.no_grad():  # a standard-normal posterior: KL would otherwise be 0
        model.style_head.weight.zero_()
        model.style_head.bias.zero_()
    rows = progress_rows(ep, to=TO)
    _, x = model.loss(rows["obs"], rows["cond"])
    assert x["kl_loss"] == pytest.approx(2 * 0.25, rel=1e-5)  # z_style = 2


# -- the shipped encoder path ------------------------------------------------------------


def test_phase_never_touches_the_decoder():
    """The deployed artifact is the encoder, so the inference path must not need the decoder."""
    ep = _toy()
    model = _model(ep)
    rows = progress_rows(ep, to=TO)

    def explode(*args, **kwargs):
        raise AssertionError("phase() called the decoder")

    model.decode = explode
    z, ood = model.phase(rows["obs"], rows["cond"], return_ood=True)
    assert z.shape == ood.shape == (len(ep),)
    assert bool(((z > 0) & (z < 1)).all())


def test_dropping_the_condition_is_one_code_path():
    """`cond=None` and a per-row `cond_keep=False` must agree, or training and inference drift."""
    ep = _toy()
    model = _model(ep)
    rows = progress_rows(ep, to=TO)
    none = model.phase(rows["obs"], None)
    keep = torch.zeros(len(ep), dtype=torch.bool)
    _, dropped, _, _ = model.encode(rows["obs"], rows["cond"], cond_keep=keep)
    assert torch.allclose(none, dropped, atol=0)


def test_the_ood_score_flags_the_episode_the_encoder_never_saw():
    """The fixture hands us a genuine shift, so this tests the mechanism rather than a caricature.

    ``_toy``'s agent walks ``50 + 3*t`` in the **global** flat index, so the last episode sits
    entirely outside the coordinate range of the first four. Training on episodes 0-3 and scoring
    episode 4 is therefore real extrapolation -- exactly the situation the score exists for, and
    exactly where ``z_phase`` still returns a confident-looking number in ``(0, 1)``.

    The separation needs the KL to have actually shaped the posterior, so it strengthens with
    training: at 3000 steps every seed separates by 10x or more, while below ~2000 some do not.
    That is a property of the score, not a flaky test -- on real data, calibrate against the
    ``ood_p95`` written to ``metrics.json`` rather than assuming a fixed scale.
    """
    ep = _toy()
    model, _ = _trained(ep)
    rows = progress_rows(ep, to=TO)
    _, ood = model.phase(rows["obs"], rows["cond"], return_ood=True)
    held = rows["episode"] == 4
    assert float(ood[held].median()) > float(ood[~held].quantile(0.95))


def test_the_ood_score_ignores_the_phase_channel():
    ep = _toy()
    model = _model(ep)
    rows = progress_rows(ep, to=TO)
    _, before = model.phase(rows["obs"], rows["cond"], return_ood=True)
    with torch.no_grad():
        model.phase_head.weight.add_(3.0)
    _, after = model.phase(rows["obs"], rows["cond"], return_ood=True)
    assert torch.allclose(before, after, atol=0)


def test_save_load_round_trip_preserves_the_normalizer_and_outputs(tmp_path):
    ep = _toy()
    model = _model(ep)
    rows = progress_rows(ep, to=TO)
    want = model.phase(rows["obs"], rows["cond"])
    model.save(tmp_path / "final.pt", info={"hello": "world"})
    again, ckpt = load_progress_model(tmp_path / "final.pt")
    assert ckpt["kind"] == "progress_cvae"
    assert ckpt["info"] == {"hello": "world"}
    assert torch.allclose(again.pos.scale, model.pos.scale, atol=0)
    assert torch.allclose(again.phase(rows["obs"], rows["cond"]), want, atol=0)


def test_constructor_rejects_the_five_dimensional_state_normalizer():
    ep = _toy()
    full = normalizer_for("box", stats_from_data(ep.state), box=(0.0, WORKSPACE_PX))
    with pytest.raises(ValueError, match="agent_pos"):
        ProgressCVAE(pos_norm=full, obs_steps=TO, **_SMALL)


# -- rank correlation and training -------------------------------------------------------


def test_spearman_sign_convention_and_ties():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    # Ties take the average rank: [1,1,2,2] ranks as [0.5, 0.5, 2.5, 2.5], giving 4/(sqrt(5)*2).
    assert spearman(a, torch.tensor([1.0, 1.0, 2.0, 2.0])) == pytest.approx(0.8944272, rel=1e-6)
    assert math.isnan(spearman(a, torch.ones(4)))  # a constant column has no rank order


def test_null_baselines_are_finite():
    ep = _toy()
    prep = prepare(ep, n_grid=8, val_ratio=0.2, seed=0, max_train_episodes=2)
    out = phase_baselines(ep, prep, prep.unused_ids, obs_steps=TO)
    assert set(out) >= {"rho_geom", "rho_linear"}
    assert all(abs(v) <= 1.0 for v in out.values())


def test_training_smoke_writes_a_complete_run_directory(tmp_path):
    ep = _toy()
    prep = prepare(ep, n_grid=8, val_ratio=0.2, seed=0, max_train_episodes=2)
    cfg = ProgressConfig(obs_steps=TO, steps=60, batch_size=32, eval_every=30, log_every=20,
                         **_SMALL)
    model, final = train_progress(ep, prep, cfg, out_dir=tmp_path, verbose=False)
    for name in ("config.json", "log.csv", "final.pt", "metrics.json"):
        assert (tmp_path / name).exists(), name
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    for key in ("spearman_per_episode_mean", "spearman_pooled", "calibration_gap",
                "spearman_no_cond", "phase_ablation", "ood_p95", "active_units",
                "rho_geom", "rho_linear", "train_config", "val_split"):
        assert key in metrics, key
    again, _ = load_progress_model(tmp_path / "final.pt")
    assert isinstance(again, ProgressCVAE)


def test_training_recovers_monotone_progress_on_the_episodes_it_saw():
    """The behavioural test: does the objective produce a monotone progress code at all?

    Scored on the **training** episodes, deliberately. ``_toy``'s agent ramp is global over the
    flat buffer, so a held-out episode lies outside the coordinate range the encoder ever saw and
    the fixture simply cannot test generalization -- that shift is what
    ``test_the_ood_score_flags_the_episode_the_encoder_never_saw`` pins instead. What *is* testable
    here is that the hinge plus reconstruction recover the ordering, which they do exactly
    (per-episode rho = 1.0 on every training episode), so the bar is tight rather than loose.

    Per-episode rather than pooled, because the global ramp means a pooled correlation would pass
    even if the model had learnt nothing about within-episode progress.
    """
    ep = _toy()
    model, cfg = _trained(ep)
    rows = progress_rows(ep, to=TO, episode_ids=torch.tensor([0, 1, 2, 3]))
    out = evaluate_progress(model, rows, cfg, full=False)
    assert out["spearman_per_episode_mean"] > 0.95
    assert out["violation_local"] == 0.0
    # And the decoder genuinely used the phase channel -- without this the rho means nothing.
    assert out["phase_ablation"] > 5.0


def test_the_model_has_no_way_to_receive_the_ground_truth_phase():
    """A contract test, in the spirit of ``test_pusht_imports_without_theseus``."""
    for method in (ProgressCVAE.loss, ProgressCVAE.encode, ProgressCVAE.phase):
        assert "phase" not in inspect.signature(method).parameters, method.__name__
    rows = progress_rows(_toy(), to=TO)
    assert "phase" not in rows and "phase_holdout" in rows
