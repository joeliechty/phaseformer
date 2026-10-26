"""The action-chunk baselines: chunk windows, the receding-horizon rollout, the four objectives
and samplers, and a training smoke run per method."""

import json

import numpy as np
import pytest
import torch

from fm_bc.skate.baseline_training import BaselineConfig, ema_decay, lr_factor, train_baseline
from fm_bc.skate.baselines import (
    BASELINES, ACTPolicy, ChunkPolicy, DiffusionPolicy, MeanFlowPolicy, MIPPolicy, cosine_betas,
    load_policy,
)
from fm_bc.skate.chunks import make_windows, normalize_actions, unnormalize_actions, window_indices
from fm_bc.skate.evaluation import rollout_chunk_policy
from fm_bc.skate.training import BASELINE_METHODS, TrainConfig, method_name, parse_method, \
    prepare_training_set
from skate_planner.planner import SkateLimits
from test_policy import _toy_dataset

D = torch.double
# Networks small enough for CPU tests.
_SMALL_UNET = {"down_dims": [8, 16], "n_groups": 4, "diffusion_step_embed_dim": 8}
_SMALL = {
    "dp": {**_SMALL_UNET, "num_train_timesteps": 10},
    "meanflow": _SMALL_UNET,
    "mip": {"emb_dim": 16, "n_layers": 2, "timestep_emb_dim": 8},
    # ACT at its real defaults is 40M parameters; this is a few thousand.
    "act": {"hidden_dim": 32, "dim_feedforward": 64, "nheads": 2, "enc_layers": 1,
            "dec_layers": 1, "vae_enc_layers": 1, "latent_dim": 8},
}


def test_window_padding():
    ds = _toy_dataset(n=4, n_steps=10)
    w = make_windows(ds, 2, 16, "cpu")
    assert w["obs"].shape == (40, 2, 3) and w["act"].shape == (40, 16, 2)
    obs, act = w["obs"].double().numpy(), w["act"].double().numpy()
    np.testing.assert_allclose(obs[0], ds.poses[0, [0, 0]], atol=1e-6)  # start repeated
    np.testing.assert_allclose(obs[3], ds.poses[0, [2, 3]], atol=1e-6)
    # k = 3: controls 2 .. 17, the last 8 edge-padded with control 9.
    np.testing.assert_allclose(act[3, :8], ds.controls[0, 2:10], atol=1e-6)
    np.testing.assert_allclose(act[3, 8:], np.repeat(ds.controls[0, 9:10], 8, 0), atol=1e-6)
    np.testing.assert_allclose(act[10, 0], ds.controls[1, 0], atol=1e-6)  # next demo, padded
    np.testing.assert_allclose(act[10, 1], ds.controls[1, 0], atol=1e-6)
    o, a = window_indices(0, 10, 2, 16)
    assert o.tolist() == [0, 0] and a[:3].tolist() == [0, 0, 1]


def test_action_normalization_round_trip():
    lim = SkateLimits()
    corners = torch.tensor([[0.0, -2.0], [2.0, 2.0]], dtype=D)
    np.testing.assert_allclose(normalize_actions(corners, lim), [[-1, -1], [1, 1]])
    u = torch.randn(5, 16, 2, dtype=D)
    torch.testing.assert_close(unnormalize_actions(normalize_actions(u, lim), lim), u)


class _ReplayChunks(ChunkPolicy):
    """Samples demo ``i``'s own chunk at step ``k``, counting its calls to know ``k``."""

    def __init__(self, controls, **kw):
        super().__init__(**kw)
        self.controls, self.calls = controls, 0

    def sample(self, obs, generator=None):
        k = self.calls * self.action_steps
        self.calls += 1
        _, act = window_indices(k, self.controls.shape[1], self.obs_steps, self.horizon)
        return self.controls[:, act]


@pytest.mark.parametrize("to,tp,ta", [(2, 16, 8), (2, 16, 3), (1, 4, 4)])
def test_chunk_replay_reproduces_demos(to, tp, ta):
    """Executing the demos' own windows retraces the demos exactly: the executed slice lines
    up with the observation window."""
    ds = _toy_dataset(n=8, n_steps=10)
    controls = torch.as_tensor(ds.controls)
    policy = _ReplayChunks(controls, obs_steps=to, horizon=tp, action_steps=ta)
    poses, executed, _ = rollout_chunk_policy(policy, torch.as_tensor(ds.poses[:, 0]), 10, ds.dt)
    torch.testing.assert_close(executed, controls)
    np.testing.assert_allclose(poses.numpy(), ds.poses, atol=1e-9)


def test_ddpm_schedule_and_oracle_sampler():
    betas = cosine_betas(100)
    assert (betas > 0).all() and (betas <= 0.999).all()
    ac100 = torch.cumprod(1 - betas, dim=0)
    assert (ac100[1:] < ac100[:-1]).all() and ac100[0] > 0.99 and ac100[-1] < 0.01
    policy = DiffusionPolicy(**_SMALL["dp"]).double()
    ac = policy.alphas_cumprod
    # With the exact epsilon of a point mass at c, the chain lands exactly on c.
    c = torch.linspace(-0.8, 0.8, 32, dtype=D).reshape(1, 16, 2)
    policy.eps = lambda x, t, obs: ((x - ac[t].sqrt()[:, None, None] * c)
                                    / (1 - ac[t]).sqrt()[:, None, None])
    obs = torch.zeros(3, 2, 3, dtype=D)
    x = policy.sample_normalized(obs, torch.Generator().manual_seed(0))
    torch.testing.assert_close(x, c.expand(3, -1, -1))


def test_meanflow_jvp_matches_finite_difference():
    torch.manual_seed(0)
    policy = MeanFlowPolicy(**_SMALL["meanflow"]).double()
    z = torch.randn(4, 16, 2, dtype=D)
    v = torch.randn_like(z)
    t, r = torch.tensor([0.9, 0.5, 0.7, 0.3], dtype=D), torch.tensor([0.1, 0.5, 0.2, 0.0], dtype=D)
    cond = torch.randn(4, 8, dtype=D)
    f = lambda z_, r_, t_: policy.u(z_, r_, t_, cond)
    _, dudt = torch.func.jvp(f, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
    # time_scale 1000 puts sinusoid frequencies up to ~1000 on t: the central difference's own
    # truncation error is ~1e-5 here, hence the tolerance.
    h = 1e-6
    fd = (f(z + h * v, r, t + h) - f(z - h * v, r, t - h)) / (2 * h)
    torch.testing.assert_close(dudt, fd, rtol=1e-4, atol=1e-4)


def test_meanflow_time_sampling():
    policy = MeanFlowPolicy(**_SMALL["meanflow"], ratio_r_neq_t=0.0)
    r, t = policy.sample_times(1000, "cpu", torch.float32)
    assert torch.equal(r, t)  # plain flow matching: the target is v
    policy.ratio_r_neq_t = 0.25
    r, t = policy.sample_times(20000, "cpu", torch.float32)
    assert (r <= t).all() and 0.2 < (r < t).float().mean() < 0.3
    assert ((t > 0) & (t < 1)).all()


def test_act_is_deterministic():
    """ACT sets the latent to the prior mean at inference, so the chunk depends on the
    observation alone. This is why it is here: the chunking baseline that cannot split modes."""
    policy = ACTPolicy(**_SMALL["act"]).eval()
    obs = torch.randn(5, 2, 3)
    a = policy.sample(obs, torch.Generator().manual_seed(0))
    b = policy.sample(obs, torch.Generator().manual_seed(1))
    assert a.shape == (5, 16, 2) and torch.equal(a, b)
    loss, parts = policy.loss(obs, torch.rand(5, 16, 2))
    assert loss.ndim == 0 and set(parts) == {"act_l1", "act_kl"}


def test_act_attends_over_the_sequence_not_the_batch():
    """``nn.MultiheadAttention`` is sequence-first by default; fed ``(B, S, H)`` without
    ``batch_first=True`` it attends over the batch, which trains and passes every shape check.
    A row's output must not depend on who else is in the batch -- including at ``B == horizon``,
    where the mistake would go unnoticed."""
    torch.manual_seed(0)
    policy = ACTPolicy(**_SMALL["act"]).eval()
    obs = torch.randn(16, 2, 3)  # B == horizon == 16: the case a shape check cannot catch
    alone, batched = policy.sample(obs[:1]), policy.sample(obs)
    torch.testing.assert_close(alone[0], batched[0])
    torch.testing.assert_close(policy.sample(obs[:5])[0], batched[0])


def test_act_kl_weight_scales_only_the_kl_term():
    """The CVAE has to be wired into the objective: at ``kl_weight = 0`` the loss is the bare L1,
    and raising it to 1 adds exactly the reported KL. Catches a posterior that is built, reported
    and then never penalized -- the failure that lets z carry the chunk at training time."""
    torch.manual_seed(0)
    free, penalized = ACTPolicy(**_SMALL["act"]), ACTPolicy(**_SMALL["act"], kl_weight=1.0)
    free.kl_weight = 0.0
    penalized.load_state_dict(free.state_dict())
    obs, controls = torch.randn(4, 2, 3), torch.rand(4, 16, 2)
    torch.manual_seed(1)
    l0, parts = free.loss(obs, controls)
    torch.manual_seed(1)
    l1, _ = penalized.loss(obs, controls)
    assert parts["act_kl"] > 0
    assert abs(l0.item() - parts["act_l1"]) < 1e-5
    assert abs((l1 - l0).item() - parts["act_kl"]) < 1e-5


def test_act_matches_the_reference_parameter_count():
    """At LeRobot's defaults ACT is ~40M parameters -- most of it the two 4-layer encoders, the
    decoder being a single layer on purpose (see fm_bc.nets.act). Pinned because a constructor
    kwarg that fails to reach the network shows up only as a suspiciously small model."""
    n = sum(p.numel() for p in ACTPolicy().parameters())
    assert 39e6 < n < 41e6


def test_mip_is_deterministic():
    policy = MIPPolicy(**_SMALL["mip"]).eval()
    obs = torch.randn(5, 2, 3)
    a = policy.sample(obs, torch.Generator().manual_seed(0))
    b = policy.sample(obs, torch.Generator().manual_seed(1))
    assert a.shape == (5, 16, 2) and torch.equal(a, b)
    loss, parts = policy.loss(obs, torch.rand(5, 16, 2))
    assert loss.ndim == 0 and set(parts) == {"mip_loss0", "mip_loss1"}


def test_schedules():
    dp = BaselineConfig(method="dp", steps=1000)
    assert lr_factor(dp, 0) == 0 and lr_factor(dp, 250) == 0.5 and lr_factor(dp, 500) == 1
    assert lr_factor(dp, 1000) == pytest.approx(0.0)
    assert ema_decay(dp, 1) == 0 and 0 < ema_decay(dp, 10) < ema_decay(dp, 1000) <= 0.9999
    mip = BaselineConfig(method="mip", steps=1000)
    assert mip.batch_size == 1024 and mip.grad_clip == 10
    assert lr_factor(mip, 0) == 1 and lr_factor(mip, 1000) == 1 and ema_decay(mip, 1) == 0.995
    assert BaselineConfig(method="dp", batch_size=64).batch_size == 64  # overrides win


def test_baseline_method_names():
    assert BASELINE_METHODS == tuple(BASELINES)
    for name in BASELINE_METHODS:
        p = parse_method(name)
        assert p["baseline"] == name and not p["use_phase"] and p["cdm_weight"] == 0
        assert method_name(**p) == name
    assert parse_method("fm")["baseline"] is None and parse_method("cdm_w0.05")["baseline"] is None


@pytest.mark.parametrize("method", BASELINE_METHODS)
def test_baseline_training_smoke(tmp_path, method):
    ds = prepare_training_set(_toy_dataset(), TrainConfig(k=4, device="cpu"))
    test = prepare_training_set(_toy_dataset(seed=1), TrainConfig(k=4, device="cpu"))
    cfg = BaselineConfig(method=method, steps=60, batch_size=64, warmup_steps=5,
                         model=_SMALL[method], k=4, eval_every=30, log_every=10, device="cpu")
    _, final = train_baseline(ds, test, cfg, tmp_path, verbose=False)
    log = np.genfromtxt(tmp_path / "log.csv", delimiter=",", names=True)
    assert np.isfinite(log["total_loss"]).all() and np.isfinite(log["mmd_mean"][-1])
    assert "model_div_per_phase" not in final and np.isfinite(final["div_curve_mae"])
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["train_config"]["method"] == method
    model, ckpt = load_policy(tmp_path / "final.pt")
    assert isinstance(model, BASELINES[method]) and ckpt["kind"] == method
    arrays = np.load(tmp_path / "rollouts.npz")
    assert arrays["poses"].shape == test.poses.shape


class _StepChunks(ChunkPolicy):
    """Queried at every step (temporal ensembling): at call ``k`` samples demo ``i``'s own
    window at step ``k``, or, with ``stamp``, a chunk filled with the query step ``k``."""

    def __init__(self, controls, stamp=False, **kw):
        super().__init__(**kw)
        self.controls, self.stamp, self.calls = controls, stamp, 0

    def sample(self, obs, generator=None):
        k, self.calls = self.calls, self.calls + 1
        if self.stamp:
            return self.controls.new_full((obs.shape[0], self.horizon, 2), float(k))
        _, act = window_indices(k, self.controls.shape[1], self.obs_steps, self.horizon)
        return self.controls[:, act]


@pytest.mark.parametrize("to,tp", [(2, 16), (1, 4)])
def test_temporal_ensemble_of_consistent_chunks_reproduces_demos(to, tp):
    """Chunks that agree with each other average to themselves, whatever the weights."""
    ds = _toy_dataset(n=8, n_steps=10)
    controls = torch.as_tensor(ds.controls)
    policy = _StepChunks(controls, obs_steps=to, horizon=tp, action_steps=1)
    poses, executed, _ = rollout_chunk_policy(policy, torch.as_tensor(ds.poses[:, 0]), 10, ds.dt,
                                              temporal_ensemble=0.3)
    assert policy.calls == 10
    torch.testing.assert_close(executed, controls)
    np.testing.assert_allclose(poses.numpy(), ds.poses, atol=1e-9)


@pytest.mark.parametrize("m", [0.01, 0.7])
def test_temporal_ensemble_weights_oldest_first(m):
    """Step t averages the future parts chunk[To-1:] of every query q <= t still covering t,
    weighted exp(-m i) with i = 0 the oldest query -- ACT's temporal_agg."""
    to, tp, n = 2, 6, 12
    policy = _StepChunks(torch.zeros(3, n, 2, dtype=D), stamp=True, obs_steps=to, horizon=tp,
                         action_steps=1)
    _, _, cmd = rollout_chunk_policy(policy, torch.zeros(3, 3, dtype=D), n, 0.1,
                                     temporal_ensemble=m)
    future = tp - to + 1
    for t in range(n):
        q = np.arange(max(0, t - future + 1), t + 1)  # oldest first
        w = np.exp(-m * np.arange(len(q)))
        np.testing.assert_allclose(cmd[:, t].numpy(), (w @ q) / w.sum(), rtol=1e-12)


def test_temporal_ensemble_method_names():
    for name in ("fm_chunk_te0.01", "dp_te0.01", "mip_te0.5"):
        p = parse_method(name)
        assert p["temporal_ensemble"] == float(name.rsplit("_te", 1)[1])
        assert method_name(**p) == name
    assert parse_method("dp_te0.01")["baseline"] == "dp"
    assert parse_method("fm_chunk_te0.01")["horizon"] == 16
    assert parse_method("dp")["temporal_ensemble"] is None
