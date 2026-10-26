"""The Push-T chunk baselines: shapes, normalization and the checkpoint round trip."""

from __future__ import annotations

import pytest
import torch

from fm_bc.pusht.baselines import (
    PUSHT_BASELINES, cosine_betas, load_pusht_baseline, make_pusht_baseline,
)

NORM = {"scale": [256.0, 256.0], "offset": [256.0, 256.0]}
# ACT at its real defaults is 40M parameters; this keeps a unit test to a few thousand.
_SMALL_ACT = {"hidden_dim": 32, "dim_feedforward": 64, "nheads": 2, "enc_layers": 1,
              "dec_layers": 1, "vae_enc_layers": 1, "latent_dim": 8}


def policy(kind: str, **kw):
    # A 5-step DDPM chain keeps the sampler's cost in a unit test; the schedule is tested apart.
    kw.setdefault("num_train_timesteps", 5) if kind == "dp" else None
    if kind == "act":
        kw = {**_SMALL_ACT, **kw}
    model = make_pusht_baseline(kind, pos_norm=NORM, act_norm=NORM, obs_steps=2, horizon=8,
                                action_steps=4, **kw)
    return model.eval()  # MIP and ACT have dropout: sampling is deterministic only in eval mode


@pytest.mark.parametrize("kind", sorted(PUSHT_BASELINES))
def test_loss_and_sample_shapes(kind):
    torch.manual_seed(0)
    model = policy(kind)
    obs, act = torch.rand(3, 2, 5) * 400, torch.rand(3, 8, 2) * 512
    loss, _ = model.loss(obs, act)
    assert loss.isfinite() and loss.ndim == 0
    chunk = model.sample(obs, torch.Generator().manual_seed(0))
    assert chunk.shape == (3, 8, 2)


@pytest.mark.parametrize("kind", sorted(PUSHT_BASELINES))
def test_sampling_is_reproducible_from_the_generator(kind):
    torch.manual_seed(0)
    model = policy(kind)
    obs = torch.rand(2, 2, 5) * 400
    a = model.sample(obs, torch.Generator().manual_seed(7))
    b = model.sample(obs, torch.Generator().manual_seed(7))
    assert torch.allclose(a, b)


@pytest.mark.parametrize("kind", sorted(PUSHT_BASELINES))
def test_checkpoint_round_trip(kind, tmp_path):
    torch.manual_seed(0)
    model = policy(kind)
    model.save(tmp_path / "m.pt", info={"method": kind})
    loaded, extra = load_pusht_baseline(tmp_path / "m.pt")
    obs = torch.rand(2, 2, 5) * 400
    assert extra["info"]["method"] == kind
    assert torch.allclose(loaded.sample(obs, torch.Generator().manual_seed(1)),
                          model.sample(obs, torch.Generator().manual_seed(1)))


def test_cosine_betas_matches_the_skate_schedule():
    pytest.importorskip("theseus")
    from fm_bc.skate.baselines import cosine_betas as skate_betas  # noqa: PLC0415
    assert torch.allclose(cosine_betas(20), skate_betas(20))


def test_action_normalization_is_the_action_box():
    model = policy("mip")
    corners = torch.tensor([[0.0, 0.0], [512.0, 512.0]])
    assert torch.allclose(model.act.apply(corners), torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))


def test_act_is_deterministic_whatever_the_generator():
    """ACT fixes the latent to the prior mean at inference, so the chunk is a function of the
    observation alone. This is the property that makes it the deterministic contrast to DP and
    MIP, and dropout in train() mode is what would silently break it."""
    torch.manual_seed(0)
    model = policy("act")
    obs = torch.rand(3, 2, 5) * 400
    a = model.sample(obs, torch.Generator().manual_seed(1))
    b = model.sample(obs, torch.Generator().manual_seed(999))
    assert torch.allclose(a, b) and torch.allclose(a, model.sample(obs, None))


def test_act_validation_loss_is_controlled_by_the_generator():
    """``validation_loss`` passes a fixed generator so that ``best.pt`` selection is a
    deterministic function of the weights. That only holds if the generator reaches the CVAE's
    reparameterization draw -- ``randn_like`` would compile, run, and quietly randomize it."""
    torch.manual_seed(0)
    model = policy("act")
    obs, act = torch.rand(3, 2, 5) * 400, torch.rand(3, 8, 2) * 512
    normalized = model.act.apply(act)

    def loss(seed):
        return model.loss_normalized(obs, normalized,
                                     generator=torch.Generator().manual_seed(seed))[0]

    assert torch.allclose(loss(7), loss(7))
    assert not torch.allclose(loss(7), loss(8))


def test_act_kl_weight_scales_only_the_kl_term():
    """Guards the CVAE actually being wired into the loss: at ``kl_weight = 0`` the objective is
    the bare L1, and the gap at ``kl_weight = 1`` is exactly the reported KL."""
    torch.manual_seed(0)
    obs, act = torch.rand(3, 2, 5) * 400, torch.rand(3, 8, 2) * 512
    free, penalized = policy("act", kl_weight=0.0), policy("act", kl_weight=1.0)
    penalized.load_state_dict(free.state_dict())
    normalized = free.act.apply(act)
    l0, e0 = free.loss_normalized(obs, normalized, generator=torch.Generator().manual_seed(3))
    l1, _ = penalized.loss_normalized(obs, normalized,
                                      generator=torch.Generator().manual_seed(3))
    assert e0["act_kl"] > 0 and abs(l0.item() - e0["act_l1"]) < 1e-5
    assert abs((l1 - l0).item() - e0["act_kl"]) < 1e-5
