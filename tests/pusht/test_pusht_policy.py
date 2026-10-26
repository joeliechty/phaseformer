"""Push-T policy arms: rows, the agent score, the regularizer schedule and training."""

from __future__ import annotations

import math

import pytest
import torch

from fm_bc.pusht.chunks import tee_points
from fm_bc.pusht.constants import N_TEE_POINTS, TEE_CENTROIDS, TEE_POINTS, TEE_VERTICES
from fm_bc.pusht.policy import (
    ARCHS, PushTBigTransformer, PushTLargeMLP, PushTLargeTransformer, PushTMediumDeepTransformer,
    PushTMediumDeepMLP, PushTMediumMLP, PushTMediumTransformer, PushTMLP, PushTTransformer,
    agent_score,
    load_pusht_policy,
    make_pusht_policy,
)
from fm_bc.pusht.policy_training import (
    ARCH_SUFFIX, PolicyConfig, evaluate_policy_offline, lr_factor, method_name, policy_rows,
    score_weight_at, train_pusht_policy, validation_loss,
)
from fm_bc.pusht.prepared import prepare

from test_pusht_data import LENGTHS, _toy  # noqa: E402 -- pytest puts tests/ on sys.path


def _prep(ep):
    return prepare(ep, n_grid=8, val_ratio=0.2, seed=0, max_train_episodes=3)


# A transformer small enough to train in a test; PushTTransformer's own defaults are 0.91M
# parameters and PushTBigTransformer's are 67M.
_TF_NET = {"hidden_dim": 32, "num_layers": 2, "num_heads": 2, "phase_dim": 16}


def _tf(pos, **kw):
    return PushTTransformer(pos, **{**_TF_NET, **kw})


def _tfbig(pos, **kw):
    return PushTBigTransformer(pos, **{**_TF_NET, **kw})


def test_rows_target_the_phase_velocity_and_integrate_back_to_the_next_position():
    ep = _toy()
    prep = _prep(ep)
    rows = policy_rows(ep, prep, torch.arange(len(LENGTHS)), normalizer="box")
    pos = prep.normalizer("agent_pos", "box")
    assert torch.equal(rows["step"], torch.arange(len(ep)))
    assert torch.equal(rows["obs"][:, -1], ep.state)
    assert torch.equal(rows["phase"], ep.phase)
    # drho is exactly the phase increment between consecutive steps of an episode.
    same = ep.episode_index[1:] == ep.episode_index[:-1]
    assert torch.allclose(rows["dphase"][:-1][same], (ep.phase[1:] - ep.phase[:-1])[same])
    # One Euler step x + u * drho lands on the next demonstrated position.
    nxt = pos.invert(pos.apply(ep.state[:-1, :2]) + rows["velocity"][:-1] * rows["dphase"][:-1, None])
    assert torch.allclose(nxt[same], ep.state[1:, :2][same], atol=1e-3)
    # The last step of an episode holds the previous phase velocity (edge-held agent_delta).
    last = ep.episode_ends - 1
    assert torch.allclose(rows["velocity"][last], rows["velocity"][last - 1])


def test_rows_respect_the_episode_selection():
    ep = _toy()
    prep = _prep(ep)
    rows = policy_rows(ep, prep, prep.train_ids)
    assert set(rows["episode"].tolist()) == set(prep.train_ids.tolist())
    assert rows["obs"].shape[0] == int(ep.lengths[prep.train_ids].sum())


def test_agent_score_matches_the_shared_kernel_estimator():
    divergence = pytest.importorskip("fm_bc.divergence")  # scipy; absent on the laptop
    g = torch.Generator().manual_seed(0)
    q = torch.rand(40, 2, generator=g, dtype=torch.float64)
    s = torch.rand(40, 25, 2, generator=g, dtype=torch.float64)
    pad = lambda x: torch.cat([x, torch.zeros_like(x[..., :1])], -1)  # noqa: E731
    want = divergence.score_kernel_at(pad(q), pad(s), 0.1, metric="xy")
    assert torch.allclose(agent_score(q, s, 0.1, chunk=7), want, atol=1e-10)


def test_agent_score_points_toward_the_samples():
    q = torch.zeros(1, 2)
    s = torch.tensor([[[1.0, 0.0], [1.2, 0.1], [0.9, -0.1]]])
    assert agent_score(q, s, 1.0)[0, 0] > 0


def test_score_weight_follows_the_skate_cosine_schedule():
    cfg = PolicyConfig(score_weight=0.05, score_anneal_frac=0.5, device="cpu")
    assert math.isclose(score_weight_at(cfg, 0, 100), 0.05)
    assert math.isclose(score_weight_at(cfg, 25, 100), 0.025, rel_tol=1e-9)
    assert score_weight_at(cfg, 50, 100) == pytest.approx(0.0, abs=1e-12)
    assert score_weight_at(cfg, 90, 100) == pytest.approx(0.0, abs=1e-12)


def test_anneal_frac_zero_holds_the_score_weight_constant():
    cfg = PolicyConfig(score_weight=2.5, score_anneal_frac=0.0, device="cpu")
    assert [score_weight_at(cfg, t, 100) for t in (0, 50, 99, 100)] == [2.5] * 4
    assert method_name(cfg) == "score_w2.5_const"
    assert method_name(PolicyConfig(score_weight=2.5, score_anneal_frac=0.0,
                                    use_phase=False, device="cpu")) == "mlp_score_w2.5_const"


def test_lr_warms_up_then_follows_the_diffusers_cosine():
    cfg = PolicyConfig(lr_warmup_steps=10, device="cpu")
    assert lr_factor(cfg, 0, 110) == 0.0
    assert lr_factor(cfg, 5, 110) == pytest.approx(0.5)
    assert lr_factor(cfg, 10, 110) == pytest.approx(1.0)
    assert lr_factor(cfg, 60, 110) == pytest.approx(0.5)
    assert lr_factor(cfg, 110, 110) == pytest.approx(0.0, abs=1e-12)


def test_method_names_match_the_skate_runs():
    assert method_name(PolicyConfig(use_phase=False, device="cpu")) == "mlp"
    assert method_name(PolicyConfig(device="cpu")) == "fm"
    assert method_name(PolicyConfig(score_weight=0.05, device="cpu")) == "score_w0.05_cos0.5"
    assert method_name(PolicyConfig(score_weight=2.5, score_anneal_frac=1.0, bandwidth_px=64,
                                    device="cpu")) == "score_w2.5_cos1_h64"
    assert method_name(PolicyConfig(use_phase=False, score_weight=2.5, device="cpu")) == (
        "mlp_score_w2.5_cos0.5")


def test_the_phase_input_is_ignored_without_use_phase():
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    torch.manual_seed(0)
    m = PushTMLP(pos, hidden=(8,), use_phase=False)
    obs = ep.state[:4, None]
    assert torch.equal(m(obs, torch.zeros(4)), m(obs, torch.ones(4)))
    m2 = PushTMLP(pos, hidden=(8,), use_phase=True)
    assert not torch.equal(m2(obs, torch.zeros(4)), m2(obs, torch.ones(4)))


@pytest.mark.parametrize("score_weight,use_phase,arch", [
    (0.0, True, "mlp"), (0.05, True, "mlp"), (0.05, False, "mlp"),
    (0.0, True, "transformer"), (0.05, True, "transformer"),
    (0.05, True, "transformer_big"),
])
def test_training_selects_the_best_validation_checkpoint_and_saves_periodically(
        tmp_path, score_weight, use_phase, arch):
    ep = _toy()
    prep = _prep(ep)
    cfg = PolicyConfig(score_weight=score_weight, use_phase=use_phase, arch=arch,
                       net_kwargs={} if arch == "mlp" else _TF_NET,
                       num_epochs=60, batch_size=16, lr=3e-3,
                       lr_warmup_steps=5, hidden=(64, 64), checkpoint_every=20, device="cpu",
                       bandwidth_px=16.0, normalizer="box")
    rows = policy_rows(ep, prep, prep.train_ids, normalizer="box")
    torch.manual_seed(0)
    untrained_u = evaluate_policy_offline(
        make_pusht_policy(arch, **cfg.model_kwargs(prep.normalizer("agent_pos", "box"))),
        rows)["velocity_rmse"]
    model, final = train_pusht_policy(ep, prep, cfg, tmp_path, verbose=False)
    assert final["best"]["train"]["velocity_rmse"] < untrained_u
    names = {p.name for p in tmp_path.iterdir()}
    assert {"best.pt", "last.pt", "epoch_0020.pt", "epoch_0040.pt", "epoch_0060.pt",
            "log.csv", "metrics.json", "config.json"} <= names

    # best.pt is the checkpoint at the logged minimum of the EMA validation loss.
    import csv
    log = list(csv.DictReader(open(tmp_path / "log.csv")))
    best_row = min(log, key=lambda r: float(r["val_loss_ema"]))
    assert int(best_row["epoch"]) == final["best"]["epoch"]
    loaded, ckpt = load_pusht_policy(tmp_path / "best.pt")
    val = policy_rows(ep, prep, prep.val_ids, normalizer="box")
    assert validation_loss(loaded, val) == pytest.approx(float(best_row["val_loss_ema"]), rel=1e-5)
    assert ckpt["epoch"] == final["best"]["epoch"]
    assert loaded.next_position(rows["obs"], rows["phase"], rows["dphase"]).shape == (
        rows["obs"].shape[0], 2)


def test_rows_in_minmax_units_still_integrate_to_the_next_position():
    ep = _toy()
    prep = _prep(ep)
    rows = policy_rows(ep, prep, torch.arange(len(LENGTHS)), normalizer="minmax")
    pos = prep.normalizer("agent_pos", "minmax")
    same = ep.episode_index[1:] == ep.episode_index[:-1]
    nxt = pos.invert(pos.apply(ep.state[:-1, :2]) + rows["velocity"][:-1] * rows["dphase"][:-1, None])
    assert torch.allclose(nxt[same], ep.state[1:, :2][same], atol=1e-3)


def test_the_input_is_object_pose_then_agent_position_then_phase():
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    m = PushTMLP(pos, hidden=(8,))
    s = ep.state[:3]
    x = m.embed(s[:, None], torch.tensor([0.0, 0.5, 1.0]))
    th = s[:, 4]
    want = torch.cat([pos.apply(s[:, 2:4]), th.cos()[:, None], th.sin()[:, None],
                      pos.apply(s[:, :2]), torch.tensor([[-1.0], [0.0], [1.0]])], dim=-1)
    assert torch.allclose(x, want)


# -- the transformer backbone -------------------------------------------------------------------
#
# PushTTransformer is an alternative backbone for the same arms: same call signature, same target,
# same checkpoint contract. These pin the properties the arm comparison depends on.


def test_the_transformer_has_the_mlps_call_signature_and_shapes():
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    obs, phase = ep.state[:4, None], torch.rand(4)
    assert _tf(pos)(obs, phase).shape == (4, 2)
    assert _tf(pos, horizon=5)(obs, phase).shape == (4, 5, 2)
    # Two observed steps give 2 * To context tokens, and the output is still one velocity.
    window = ep.state[:4].unsqueeze(1).expand(4, 2, 5).contiguous()
    m = _tf(pos, obs_steps=2)
    assert m.tokens(window).shape == (4, 4, m.hidden_dim)
    assert m(window, phase).shape == (4, 2)


def test_the_transformer_phase_input_is_ignored_without_use_phase():
    """The property every arm comparison rests on -- the mirror of the MLP's test above."""
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    obs = ep.state[:4, None]
    torch.manual_seed(0)
    off, on = _tf(pos, use_phase=False), _tf(pos, use_phase=True)
    # The output head is zero-initialized, so perturb the weights before comparing.
    for net in (off, on):
        for p in net.parameters():
            p.data.add_(torch.randn_like(p) * 0.05)
    assert torch.equal(off(obs, torch.zeros(4)), off(obs, torch.ones(4)))
    assert not torch.equal(on(obs, torch.zeros(4)), on(obs, torch.ones(4)))


def test_the_transformer_is_deterministic():
    """No noise input and no sampling, in train() as well as eval(): dropout defaults to 0, so the
    training loss is as deterministic as the MLP arms'."""
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    obs, phase = ep.state[:4, None], torch.rand(4)
    m = _tf(pos)
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.05)
    assert m.dropout == 0.0
    m.train()
    assert torch.equal(m(obs, phase), m(obs, phase))
    m.eval()
    assert torch.equal(m(obs, phase), m(obs, phase))


def test_the_transformer_blocks_start_as_the_identity():
    """adaLN-Zero: the modulations are zero-initialized, so every residual branch is gated off and
    an untrained network predicts exactly zero."""
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    obs, phase = ep.state[:4, None], torch.rand(4)
    m = _tf(pos)
    assert torch.equal(m(obs, phase), torch.zeros(4, 2))
    net, context = m.net, m.tokens(obs)
    h = queries = net.tokens(4, None)
    for block in net.blocks:
        h = block(h, net.embed_phase(phase, 4), context)
    assert torch.equal(h, queries)


def test_the_transformer_round_trips_through_a_checkpoint(tmp_path):
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    rows = policy_rows(ep, _prep(ep), torch.arange(len(LENGTHS)), normalizer="box")
    m = _tf(pos)
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.05)
    m.eval()
    m.save(tmp_path / "best.pt", epoch=7)
    loaded, ckpt = load_pusht_policy(tmp_path / "best.pt")
    assert isinstance(loaded, PushTTransformer) and ckpt["epoch"] == 7
    assert torch.equal(loaded(rows["obs"], rows["phase"]), m(rows["obs"], rows["phase"]))
    assert loaded.next_position(rows["obs"], rows["phase"], rows["dphase"]).shape == (
        rows["obs"].shape[0], 2)
    # config() must be complete: the training loop builds the EMA copy this way.
    assert PushTTransformer(**m.config()).config() == m.config()


def test_the_loader_dispatches_on_kind(tmp_path):
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    PushTMLP(pos, hidden=(8,)).save(tmp_path / "mlp.pt")
    _tf(pos).save(tmp_path / "tf.pt")
    assert isinstance(load_pusht_policy(tmp_path / "mlp.pt")[0], PushTMLP)
    assert isinstance(load_pusht_policy(tmp_path / "tf.pt")[0], PushTTransformer)
    torch.save({"kind": "nonsense", "config": {}, "state_dict": {}}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="not a Push-T policy checkpoint"):
        load_pusht_policy(tmp_path / "bad.pt")


def test_the_arch_suffix_leaves_the_mlp_run_names_untouched():
    """Existing run directories and figure labels are keyed on these names."""
    for kw in ({}, {"use_phase": False}, {"score_weight": 0.05},
               {"score_weight": 0.05, "use_phase": False}, {"score_anneal_frac": 0.0}):
        cfg = PolicyConfig(device="cpu", **kw)
        assert method_name(cfg) == method_name(PolicyConfig(device="cpu", arch="mlp", **kw))
        assert not method_name(cfg).endswith("_tf")
    assert method_name(PolicyConfig(arch="transformer", device="cpu")) == "fm_tf"
    assert method_name(PolicyConfig(arch="transformer", use_phase=False, device="cpu")) == "mlp_tf"
    assert method_name(PolicyConfig(arch="transformer", score_weight=0.05, device="cpu")) == (
        "score_w0.05_cos0.5_tf")
    with pytest.raises(ValueError, match="arch must be one of"):
        PolicyConfig(arch="cnn", device="cpu")


def test_model_kwargs_only_pass_the_selected_backbones_options():
    assert sorted(ARCHS) == ["mlp", "mlp_large", "mlp_medium", "mlp_medium_deep", "transformer",
                             "transformer_big", "transformer_large", "transformer_medium",
                             "transformer_medium_deep"]
    mlp = PolicyConfig(device="cpu").model_kwargs("norm")
    assert mlp["hidden"] and mlp["activation"] and "hidden_dim" not in mlp
    tf = PolicyConfig(arch="transformer", net_kwargs={"hidden_dim": 8},
                      device="cpu").model_kwargs("norm")
    assert tf["hidden_dim"] == 8 and "hidden" not in tf and "activation" not in tf
    # An un-overridden backbone passes no size kwargs at all, so the class's own defaults stand.
    # This is what keeps --arch transformer_big from being shrunk back to the small backbone.
    big = PolicyConfig(arch="transformer_big", device="cpu").model_kwargs("norm")
    assert "hidden_dim" not in big and "hidden" not in big


# -- the scaled published backbone: medium (wide), medium deep, large --------------------------
#
# PushTTransformer widened, deepened or both, nothing else: scale-only tests.

# (class, arch, run suffix, (hidden_dim, num_layers, num_heads, phase_dim), parameters)
_SCALED = [
    (PushTMediumTransformer, "transformer_medium", "_tfmed", (180, 2, 4, 180), 1_791_002),
    (PushTMediumDeepTransformer, "transformer_medium_deep", "_tfdeep", (128, 4, 4, 128),
     1_730_818),
    (PushTLargeTransformer, "transformer_large", "_tflarge", (180, 4, 4, 180), 3_416_762),
]


@pytest.mark.parametrize("cls, arch, suffix, shape, n_params", _SCALED)
def test_a_scaled_backbone_is_the_published_one_at_another_size(cls, arch, suffix, shape,
                                                                 n_params):
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    small, big = PushTTransformer(pos), cls(pos)
    assert (n(small), n(big)) == (907_522, n_params)
    assert (big.hidden_dim, big.num_layers, big.num_heads, big.phase_dim) == shape
    # Overriding the defaults back to the published ones *is* the published backbone: the design
    # did not change, only the size.
    as_small = cls(pos, hidden_dim=128, num_layers=2, phase_dim=128)
    assert {k: v.shape for k, v in as_small.state_dict().items()} == {
        k: v.shape for k, v in small.state_dict().items()}


@pytest.mark.parametrize("cls, arch, suffix, shape, n_params", _SCALED)
def test_a_scaled_backbone_saves_loads_and_gets_its_own_run_names(tmp_path, cls, arch, suffix,
                                                                  shape, n_params):
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    m = cls(pos, **_TF_NET)
    m.save(tmp_path / "m.pt")
    loaded, _ = load_pusht_policy(tmp_path / "m.pt")
    assert type(loaded) is cls and loaded.config() == m.config()
    assert "hidden_dim" not in PolicyConfig(arch=arch, device="cpu").model_kwargs("norm")
    names = set()
    for kw in ({}, {"use_phase": False}, {"score_weight": 2.5, "score_anneal_frac": 0.0}):
        small = method_name(PolicyConfig(arch="transformer", device="cpu", **kw))
        name = method_name(PolicyConfig(arch=arch, device="cpu", **kw))
        assert name == small.removesuffix("_tf") + suffix and not name.endswith("_tf")
        names.add(name)
    assert "score_w2.5_const" + suffix in names
    # No two backbones' suffixes can be confused by an endswith match (plot_pusht's lookup).
    others = [s for s in ARCH_SUFFIX.values() if s and s != suffix]
    assert not any(n.endswith(o) for n in names for o in others)


# -- the 67M backbone ----------------------------------------------------------------------------
#
# PushTBigTransformer is PushTTransformer scaled up around cross-attention: the T tokenized as
# points on it (12 context tokens instead of 2) and no self-attention. These pin both the new
# behaviour and, more importantly, that the published 0.9M backbone did not move.


def test_the_big_backbone_leaves_the_published_one_untouched():
    """The whole reason this is a separate class. If this fails, existing checkpoints and every
    published `_tf` number are in question."""
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    keys = set(PushTTransformer(pos).state_dict())
    assert "net.blocks.0.attn.qkv.weight" in keys       # self-attention still built
    assert not any("point_" in k for k in keys)         # and no tee tokenizer leaked in
    assert sum(p.numel() for p in PushTTransformer(pos).parameters()) == 907_522
    assert sum(p.numel() for p in PushTTransformer(pos, use_phase=False).parameters()) == 874_498
    assert set(PushTTransformer(pos).config()) == {
        "pos_norm", "obs_steps", "horizon", "use_phase", "hidden_dim", "num_layers", "num_heads",
        "mlp_ratio", "dropout", "phase_dim"}


def test_the_big_backbone_defaults_to_the_configuration_we_mean_to_train():
    """Following tests/skate/test_baselines.py's ACT check: a constructor kwarg that fails to reach
    the network shows up only as a suspiciously small model."""
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    m = PushTBigTransformer(pos)
    assert (m.hidden_dim, m.num_layers, m.num_heads, m.phase_dim) == (768, 8, 12, 256)
    n = sum(p.numel() for p in m.parameters())
    assert 60e6 < n < 70e6, n
    assert n == 66_877_698


def test_the_tee_points_ride_the_block_rigidly():
    """`world = block_xy + R(theta) @ local`. Rigidity is what catches a transposed rotation,
    which a shape check cannot see."""
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    local = torch.tensor(TEE_POINTS)
    # theta = 0: the points are the body-frame polygon translated to the block's position.
    state = torch.tensor([[10.0, 20.0, 100.0, 200.0, 0.0]])
    got = pos.invert(tee_points(state, pos))
    assert torch.allclose(got[0], local + torch.tensor([100.0, 200.0]), atol=1e-4)
    # Any theta: every pairwise distance is preserved, because the map is a rigid motion.
    state = torch.tensor([[10.0, 20.0, 100.0, 200.0, 0.9], [0.0, 0.0, 300.0, 50.0, -2.3]])
    got = pos.invert(tee_points(state, pos))
    want = torch.cdist(local[None], local[None])
    assert torch.allclose(torch.cdist(got, got), want.expand(2, -1, -1), atol=1e-3)
    # The two centroids really are the centroids of the two rectangles the T is built from.
    assert TEE_POINTS == TEE_VERTICES + TEE_CENTROIDS and len(TEE_POINTS) == N_TEE_POINTS
    crossbar = torch.tensor(TEE_VERTICES[:2] + TEE_VERTICES[2:3] + TEE_VERTICES[7:8])
    assert torch.allclose(crossbar.mean(0), torch.tensor(TEE_CENTROIDS[0]))


def test_the_tee_tokens_add_no_information():
    """They are a re-encoding of the block pose, not extra observation: a fixed function of
    (block_x, block_y, theta) alone, so the arm comparison is still about the backbone."""
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    m = _tfbig(pos)
    same_block = torch.tensor([[[1.0, 2.0, 100.0, 200.0, 0.4]], [[90.0, 7.0, 100.0, 200.0, 0.4]]])
    pts = m.point_proj(tee_points(same_block, m.pos, m.tee_local))
    assert torch.equal(pts[0], pts[1])            # differing only in the agent changes nothing
    moved = torch.tensor([[[1.0, 2.0, 101.0, 200.0, 0.4]]])
    assert not torch.equal(tee_points(moved, m.pos), tee_points(same_block[:1], m.pos))


def test_the_big_backbone_tokenizes_the_block_as_points_and_drops_self_attention():
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    obs, phase = ep.state[:4, None], torch.rand(4)
    m = _tfbig(pos)
    assert m.n_context_tokens == 12 and m.tokens(obs).shape == (4, 12, m.hidden_dim)
    assert m(obs, phase).shape == (4, 2)
    window = ep.state[:4].unsqueeze(1).expand(4, 2, 5).contiguous()
    m2 = _tfbig(pos, obs_steps=2)
    assert m2.n_context_tokens == 24 and m2.tokens(window).shape == (4, 24, m2.hidden_dim)
    # No self-attention anywhere, but cross-attention and the phase path are intact.
    keys = set(m.state_dict())
    assert not any(".attn." in k or ".mod1." in k for k in keys)
    assert any("cross_attn" in k for k in keys)
    assert not any("tee_local" in k for k in keys)  # a constant of the env, not of the checkpoint
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.05)
    assert not torch.equal(m(obs, torch.zeros(4)), m(obs, torch.ones(4)))       # reads rho
    assert not torch.equal(m(obs, phase), m(obs.flip(0), phase))                # reads obs


def test_self_attention_has_no_gradient_at_one_query_token():
    """Why the big backbone drops it. A softmax over a single key is constant however the query is
    projected, so q and k are dead weight at horizon=1 -- but not at horizon>1."""
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    obs, phase = ep.state[:4, None], torch.rand(4)

    def qkv_grads(horizon):
        torch.manual_seed(0)
        m = _tf(pos, horizon=horizon)
        for p in m.parameters():           # adaLN-Zero gates everything off until perturbed
            p.data.add_(torch.randn_like(p) * 0.05)
        m(obs, phase).pow(2).sum().backward()
        d = m.hidden_dim
        g = m.net.blocks[0].attn.qkv.weight.grad
        return g[:d].abs().max(), g[d:2 * d].abs().max(), g[2 * d:].abs().max(), m

    q, k, v, m = qkv_grads(1)
    assert q == 0.0 and k == 0.0, (q, k)
    assert v > 0 and m.net.blocks[0].cross_attn.q.weight.grad.abs().max() > 0
    q3, k3, _, _ = qkv_grads(3)
    assert q3 > 0 and k3 > 0


def test_the_big_backbone_starts_as_the_identity_and_round_trips(tmp_path):
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    rows = policy_rows(ep, _prep(ep), torch.arange(len(LENGTHS)), normalizer="box")
    obs, phase = ep.state[:4, None], torch.rand(4)
    # adaLN-Zero still holds with the self-attention branch absent.
    assert torch.equal(_tfbig(pos)(obs, phase), torch.zeros(4, 2))
    m = _tfbig(pos)
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.05)
    m.eval()
    m.save(tmp_path / "best.pt", epoch=7)
    loaded, ckpt = load_pusht_policy(tmp_path / "best.pt")
    assert isinstance(loaded, PushTBigTransformer) and ckpt["epoch"] == 7
    assert torch.equal(loaded(rows["obs"], rows["phase"]), m(rows["obs"], rows["phase"]))
    assert PushTBigTransformer(**m.config()).config() == m.config()
    # Dispatch is on `kind`, so the two backbones never load as each other.
    _tf(pos).save(tmp_path / "small.pt")
    assert type(load_pusht_policy(tmp_path / "small.pt")[0]) is PushTTransformer


def test_the_big_backbone_gets_its_own_run_names():
    """`_tfbig` runs must never land in a published `_tf` directory."""
    for kw in ({}, {"use_phase": False}, {"score_weight": 2.5, "score_anneal_frac": 0.0}):
        small = method_name(PolicyConfig(arch="transformer", device="cpu", **kw))
        big = method_name(PolicyConfig(arch="transformer_big", device="cpu", **kw))
        assert big == small + "big" and big.endswith("_tfbig") and small != big
    assert method_name(PolicyConfig(arch="transformer_big", device="cpu")) == "fm_tfbig"
    assert method_name(PolicyConfig(arch="transformer_big", use_phase=False,
                                    device="cpu")) == "mlp_tfbig"
    assert method_name(PolicyConfig(arch="transformer_big", score_weight=2.5,
                                    score_anneal_frac=0.0, device="cpu")) == (
        "score_w2.5_const_tfbig")


# -- the resized MLPs: matched to the medium and large transformers ----------------------------


@pytest.mark.parametrize("cls, arch, suffix, hidden, n_params, tf_twin", [
    (PushTMediumMLP, "mlp_medium", "_mlpmed", (768,) * 4, 1_779_458, 1_791_002),
    (PushTMediumDeepMLP, "mlp_medium_deep", "_mlpdeep", (512,) * 8, 1_843_714, 1_730_818),
    (PushTLargeMLP, "mlp_large", "_mlplarge", (768,) * 7, 3_551_234, 3_416_762),
])
def test_a_resized_mlp_matches_its_transformer_and_keeps_its_own_size(cls, arch, suffix, hidden,
                                                                       n_params, tf_twin):
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    m = cls(pos)
    assert m.hidden == hidden and n(m) == n_params and abs(n_params / tf_twin - 1) < 0.07
    # Overriding the defaults back to the published ones *is* the published MLP.
    assert [v.shape for v in cls(pos, hidden=(512,) * 4).state_dict().values()] == [
        v.shape for v in PushTMLP(pos).state_dict().values()]
    # The config passes no width, so PolicyConfig's (512,) * 4 default cannot shrink it back.
    kwargs = PolicyConfig(arch=arch, device="cpu").model_kwargs(pos)
    assert "hidden" not in kwargs and cls(**kwargs).hidden == hidden
    for kw in ({}, {"use_phase": False}, {"score_weight": 2.5, "score_anneal_frac": 0.0}):
        published = method_name(PolicyConfig(arch="mlp", device="cpu", **kw))
        name = method_name(PolicyConfig(arch=arch, device="cpu", **kw))
        assert name == published + suffix
        assert not any(name.endswith(s) for s in ARCH_SUFFIX.values() if s and s != suffix)


def test_a_resized_mlp_saves_and_loads_as_itself(tmp_path):
    ep = _toy()
    pos = _prep(ep).normalizer("agent_pos", "box")
    m = PushTMediumMLP(pos, hidden=(8, 8))
    m.save(tmp_path / "m.pt")
    loaded, _ = load_pusht_policy(tmp_path / "m.pt")
    assert type(loaded) is PushTMediumMLP and loaded.config() == m.config()
    PushTMLP(pos, hidden=(8, 8)).save(tmp_path / "p.pt")
    assert type(load_pusht_policy(tmp_path / "p.pt")[0]) is PushTMLP
