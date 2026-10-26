"""The policy's divergence, the closed-loop integrator, the MMD metric, the divergence
regularizers and a training smoke run."""

import json
import math

import numpy as np
import pytest
import torch

from fm_bc.skate.baselines import load_policy
from fm_bc.skate.chunks import window_indices
from fm_bc.skate.dataset import TrajectoryDataset
from fm_bc.skate.evaluation import collision_breakdown, dense_clearance, mmd, rollout_policy
from fm_bc.skate.policy import ARCHS, FlowMLP, FlowTransformer, make_flow_policy, model_divergence
from fm_bc.skate.scenarios import get_scenario
from fm_bc.se2 import se2_exp_retract
from fm_bc.skate.training import (
    ARCH_SUFFIX, TrainConfig, cdm_regularizer, cdm_residuals, cdm_weight_at,
    chunk_score_residuals, make_rows, method_name, open_loop_poses, parse_method,
    prepare_training_set, run_dir, stratified_subset, train,
)
from skate_planner.planner import SkateLimits
from skate_planner.skate_model import body_twist, rollout

D = torch.double
# Projection onto the admissible body directions (forward, yaw) of a skate.
_ADMISSIBLE = torch.diag(torch.tensor([1.0, 0.0, 1.0], dtype=D))


def _poses(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g, dtype=D) * torch.tensor([10.0, 8.0], dtype=D)
    th = (torch.rand(n, 1, generator=g, dtype=D) * 2 - 1) * 3.0
    return torch.cat([xy, th], dim=1)


def _jacobian(model, poses, phase):
    """Full body-frame Jacobian ``d xi_i / d e_j`` at ``e = 0``, ``(B, 3, 3)``."""
    eps = torch.zeros_like(poses, requires_grad=True)
    twist = model.twist(se2_exp_retract(poses, eps), phase)
    rows = [torch.autograd.grad(twist[:, i].sum(), eps, retain_graph=True, allow_unused=True)[0]
            for i in range(3)]
    return torch.stack([torch.zeros_like(poses) if r is None else r for r in rows], dim=1)


def _quadratic_form_var(a):
    """``Var(w^T A w)`` for ``w ~ N(0, I)``: ``2 |sym(A)|_F^2``, per batch row."""
    sym = 0.5 * (a + a.transpose(1, 2))
    return 2 * (sym**2).sum(dim=(1, 2))


class _Analytic(FlowMLP):
    """``v = a (x cos th + y sin th)``, ``omega = b th``: a body-frame x-perturbation moves the
    position along the heading, so ``dv/de_x = a``; ``domega/de_theta = b``. Divergence
    ``a + b`` everywhere."""

    def __init__(self, a, b):
        super().__init__(hidden=(1,))
        self.a, self.b = a, b

    def forward(self, poses, phase):
        x, y, th = poses.unbind(-1)
        return torch.stack([self.a * (x * torch.cos(th) + y * torch.sin(th)), self.b * th], -1)


class _Replay(FlowMLP):
    """Returns control ``k`` of a fixed sequence at phase ``k / N``."""

    def __init__(self, controls):
        super().__init__(hidden=(1,))
        self.controls = controls

    def forward(self, poses, phase):
        k = int(round(float(phase[0]) * self.controls.shape[1]))
        return self.controls[:, k]


class _ReplayChunks(FlowMLP):
    """Returns the edge-padded chunk ``u_k .. u_{k+horizon-1}`` at phase ``k / N``, and fails
    if it is queried between re-plans."""

    def __init__(self, controls, horizon, action_steps):
        super().__init__(hidden=(1,), horizon=horizon, action_steps=action_steps)
        self.controls = controls

    def forward(self, poses, phase):
        n = self.controls.shape[1]
        k = int(round(float(phase[0]) * n))
        assert k % self.action_steps == 0
        return self.controls[:, window_indices(k, n, 1, self.horizon)[1]]


def test_divergence_matches_analytic_field():
    poses, phase = _poses(), torch.rand(64, dtype=D)
    _, div = model_divergence(_Analytic(0.7, -0.3).double(), poses, phase, create_graph=False)
    assert torch.allclose(div, torch.full_like(div, 0.4), atol=1e-10)


def test_exact_divergence_matches_finite_differences():
    torch.manual_seed(0)
    model = FlowMLP(hidden=(32, 32)).double()
    poses, phase = _poses(), torch.rand(64, dtype=D)
    _, div = model_divergence(model, poses, phase, create_graph=False)
    h, fd = 1e-5, torch.zeros(64, dtype=D)
    for i in range(3):
        e = torch.zeros(64, 3, dtype=D)
        e[:, i] = h
        plus = model.twist(se2_exp_retract(poses, e), phase)[:, i]
        minus = model.twist(se2_exp_retract(poses, -e), phase)[:, i]
        fd += (plus - minus) / (2 * h)
    assert torch.allclose(div, fd, atol=1e-7)


def test_hutchinson_is_unbiased():
    torch.manual_seed(1)
    model = FlowMLP(hidden=(32, 32)).double()
    poses, phase = _poses(16), torch.rand(16, dtype=D)
    _, exact = model_divergence(model, poses, phase, create_graph=False)
    n = 4000
    _, hutch = model_divergence(model, poses, phase, "hutchinson", n_probes=n, create_graph=False)
    # Probes are N(0, P), P = diag(1, 0, 1), so z^T J z = w^T (P J P) w; allow 5 standard errors.
    jac = _jacobian(model, poses, phase)
    tol = 5 * torch.sqrt(_quadratic_form_var(_ADMISSIBLE @ jac @ _ADMISSIBLE) / n)
    assert ((hutch - exact).abs() <= tol).all()


def test_hutchinson_probes_are_admissible():
    """Probing only the forward and yaw body directions leaves the trace alone (the lateral
    twist row is identically zero) and drops the lateral cross terms' variance."""
    torch.manual_seed(5)
    model = FlowMLP(hidden=(32, 32)).double()
    poses, phase = _poses(8), torch.rand(8, dtype=D)
    n = 20000
    # One probe per row: n independent single-probe estimates of each pose, in one call.
    _, est = model_divergence(model, poses.repeat(n, 1), phase.repeat(n), "hutchinson",
                              create_graph=False)
    jac = _jacobian(model, poses, phase)
    admissible = _quadratic_form_var(_ADMISSIBLE @ jac @ _ADMISSIBLE)
    assert torch.allclose(est.reshape(n, 8).var(dim=0), admissible, rtol=0.15)
    assert (admissible < _quadratic_form_var(jac)).all()


def test_divergence_is_differentiable_in_parameters():
    torch.manual_seed(2)
    model = FlowMLP(hidden=(16, 16)).double()
    _, div = model_divergence(model, _poses(8), torch.rand(8, dtype=D), create_graph=True)
    div.pow(2).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_cdm_residuals_and_regularizers():
    pred = torch.tensor([[1.0, 0.0, 0.5], [2.0, 0.0, -1.0]])
    target = torch.tensor([[1.5, 0.0, 0.0], [1.0, 0.0, -1.0]])
    # Large lateral scores (7, -3) that must never enter: both twists' lateral slots are zero.
    score = torch.tensor([[2.0, 7.0, -1.0], [0.5, -3.0, 4.0]])
    div_v, div_u = torch.tensor([0.3, -0.2]), torch.tensor([0.1, 0.4])
    a, b = cdm_residuals(pred, target, div_v, div_u, score)
    assert torch.allclose(a, torch.tensor([-0.2, 0.6]))
    assert torch.allclose(b, torch.tensor([0.5 * 2.0 + 0.5, -1.0 * 0.5]))
    both, div, sc = (cdm_regularizer(a, b, t).item() for t in ("both", "div", "score"))
    assert (both, div, sc) == pytest.approx((0.7, 0.4, 1.0))
    assert both <= div + sc
    # The score term alone needs no model divergence.
    a0, b0 = cdm_residuals(pred, target, None, div_u, score)
    assert a0 is None and cdm_regularizer(a0, b0, "score").item() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cdm_regularizer(a, b, "lateral")


def test_cdm_weight_schedule():
    const = TrainConfig(cdm_weight=0.05, steps=1000)
    assert [cdm_weight_at(const, s) for s in (1, 500, 1000)] == [0.05] * 3
    cos = TrainConfig(cdm_weight=0.05, cdm_schedule="cosine", cdm_anneal_frac=0.4, steps=1000)
    w = np.array([cdm_weight_at(cos, s) for s in range(1001)])
    assert w[0] == pytest.approx(0.05) and w[200] == pytest.approx(0.025)
    assert (w[400:] == 0).all() and (np.diff(w) <= 0).all()


def test_train_config_rejects_unknown_regularizers():
    for bad in ({"cdm_terms": "lateral"}, {"cdm_schedule": "linear"}, {"cdm_anneal_frac": 0.0}):
        with pytest.raises(ValueError):
            TrainConfig(**bad)


def test_replay_rollout_reproduces_skate_rollout():
    g = torch.Generator().manual_seed(3)
    starts = _poses(20, seed=3)
    controls = torch.stack([torch.rand(20, 12, generator=g, dtype=D) * 2,
                            (torch.rand(20, 12, generator=g, dtype=D) * 2 - 1) * 2], dim=-1)
    ours, ctrl, cmd = rollout_policy(_Replay(controls).double(), starts, 12, dt=0.15)
    ref = rollout(starts, controls, 0.15)
    assert torch.allclose(ctrl, controls) and torch.equal(cmd, ctrl)
    assert torch.allclose(ours[..., :2], ref[..., :2], atol=1e-10)
    assert torch.allclose(torch.cos(ours[..., 2] - ref[..., 2]), torch.ones((), dtype=D), atol=1e-12)


def test_rollout_saturates_to_planner_limits():
    limits = _toy_dataset().limits
    assert limits == SkateLimits()  # no limits recorded -> the planner's defaults
    starts = _poses(4, seed=5)
    wild = torch.tensor([[5.0, -3.0], [-1.0, 0.5], [1.0, 2.5], [1.5, -0.5]], dtype=D)
    wild = wild[:, None].expand(4, 6, 2)
    poses, executed, commanded = rollout_policy(_Replay(wild).double(), starts, 6, 0.15, limits)
    expected = torch.tensor([[2.0, -2.0], [0.0, 0.5], [1.0, 2.0], [1.5, -0.5]], dtype=D)
    assert torch.equal(commanded, wild)
    assert torch.equal(executed, expected[:, None].expand(4, 6, 2))
    # The path driven is the Theseus rollout of the *saturated* controls.
    ref = rollout(starts, executed, 0.15)
    assert torch.allclose(poses[..., :2], ref[..., :2], atol=1e-10)


def test_dense_clearance_matches_substep_rollout():
    world = get_scenario("dual_path").world
    starts = torch.tensor([[5.0, 3.6, math.pi / 2], [2.0, 6.0, 0.3]], dtype=D)
    controls = torch.tensor([[1.8, 0.0], [1.0, -0.5]], dtype=D)[:, None].expand(2, 10, 2)
    poses = rollout(starts, controls, 0.15)
    ours = dense_clearance(poses, controls, 0.15, world.analytic_sdf, substeps=8)
    dense = rollout(starts, controls, 0.15, substeps=8).numpy()
    np.testing.assert_allclose(ours, world.analytic_sdf(dense[..., :2]).min(axis=1), atol=1e-9)
    assert ours[0] < 0  # straight on out of the lane drives through the block


def test_collision_breakdown_separates_box_from_wall():
    world = get_scenario("dual_path").world
    # Straight on out of the lane into the block; straight up the left edge into the top wall.
    starts = torch.tensor([[5.0, 3.6, math.pi / 2], [1.0, 6.0, math.pi / 2]], dtype=D)
    controls = torch.tensor([1.8, 0.0], dtype=D).expand(2, 10, 2)
    poses = rollout(starts, controls, 0.15)
    obstacle, boundary = collision_breakdown(poses, controls, 0.15, world, 0.3)
    assert obstacle.tolist() == [True, False]
    assert boundary.tolist() == [False, True]


def test_mmd_zero_for_identical_and_grows_with_shift():
    a = _poses(200, seed=4)
    assert abs(mmd(a, a)) < 1e-12
    shifted = [mmd(a, a + torch.tensor([s, 0.0, 0.0], dtype=D)) for s in (0.1, 0.3, 1.0)]
    assert shifted[0] < shifted[1] < shifted[2]


def _toy_dataset(n=64, n_steps=10, seed=0):
    """Straight-ish skate paths from the dual_path start distribution (no planner needed)."""
    spec = get_scenario("dual_path")
    rng = np.random.default_rng(seed)
    starts = spec.start.sample(n, rng)
    controls = np.stack([1.0 + 0.2 * rng.standard_normal((n, n_steps)),
                         0.3 * rng.standard_normal((n, n_steps))], axis=-1)
    poses = rollout(torch.as_tensor(starts), torch.as_tensor(controls), 0.15).numpy()
    return TrajectoryDataset(
        scenario="dual_path", poses=poses, controls=controls,
        twists=body_twist(torch.as_tensor(controls)).numpy(), mode=np.arange(n) % 2,
        starts=starts, goals=np.tile(spec.goals[0].mean, (n, 1)), dt=0.15,
    )


def test_stratified_subset_recomputes_targets():
    ds = prepare_training_set(_toy_dataset(), TrainConfig(k=4, device="cpu"))
    assert ds.divergence is not None and ds.div_config["k"] == 4
    sub = stratified_subset(ds, 32, np.random.default_rng(0))
    assert sub.divergence is None and np.bincount(sub.mode).tolist() == [16, 16]
    sub = prepare_training_set(ds, TrainConfig(n_train=32, k=4, device="cpu"))
    assert len(sub) == 32 and sub.divergence.shape == (32, 11)


def test_state_only_mlp_ignores_phase(tmp_path):
    torch.manual_seed(4)
    model = FlowMLP(hidden=(16, 16), use_phase=False).double()
    assert model.net[0].in_features == 4
    poses = _poses(8)
    assert torch.equal(model(poses, torch.zeros(8, dtype=D)), model(poses, torch.ones(8, dtype=D)))
    model.save(tmp_path / "m.pt")
    loaded, _ = FlowMLP.load(tmp_path / "m.pt")
    assert loaded.use_phase is False
    # Checkpoints from before the flag existed are phase-conditioned.
    ckpt = torch.load(tmp_path / "m.pt")
    ckpt["config"].pop("use_phase")
    assert FlowMLP(**ckpt["config"]).use_phase is True


def test_method_names():
    assert method_name(0.0) == "fm" and method_name(0.05) == "cdm_w0.05"
    assert method_name(0.0, use_phase=False) == "mlp"
    assert method_name(0.05, use_phase=False) == "mlp_cdm_w0.05"
    assert method_name(0.05, cdm_terms="div", cdm_schedule="cosine") == "div_w0.05_cos0.5"
    assert (method_name(0.05, cdm_terms="score", cdm_schedule="cosine", cdm_anneal_frac=0.3)
            == "score_w0.05_cos0.3")
    for name in ("fm", "mlp", "cdm_w0.05", "mlp_cdm_w0.05", "cdm_w0.05_cos0.5",
                 "div_w0.05_cos0.5", "score_w1e-05_cos0.3", "mlp_div_w0.05"):
        assert method_name(**parse_method(name)) == name
    assert parse_method("score_w0.05_cos0.5")["cdm_terms"] == "score"
    with pytest.raises(ValueError):
        parse_method("lateral_w0.05")


@pytest.mark.parametrize("cdm_weight,use_phase,terms,schedule", [
    (0.0, True, "both", "constant"), (0.05, True, "both", "constant"),
    (0.0, False, "both", "constant"), (0.05, True, "both", "cosine"),
    (0.05, True, "div", "cosine"), (0.05, True, "score", "cosine"),
])
def test_training_smoke(tmp_path, cdm_weight, use_phase, terms, schedule):
    ds = prepare_training_set(_toy_dataset(), TrainConfig(k=4, device="cpu"))
    test = prepare_training_set(_toy_dataset(seed=1), TrainConfig(k=4, device="cpu"))
    cfg = TrainConfig(cdm_weight=cdm_weight, cdm_terms=terms, cdm_schedule=schedule,
                      use_phase=use_phase, steps=200, batch_size=128, hidden=(64, 64), k=4,
                      eval_every=100, log_every=50, device="cpu")
    _, final = train(ds, test, cfg, tmp_path, verbose=False)
    log = np.genfromtxt(tmp_path / "log.csv", delimiter=",", names=True)
    assert log["velocity_loss"][-1] < log["velocity_loss"][0]
    for key in ("cdm_loss", "div_l1", "score_l1"):  # logged for every arm
        assert np.isfinite(log[key]).all()
    np.testing.assert_allclose(log["cdm_weight"], [cdm_weight_at(cfg, s) for s in log["step"]])
    if schedule == "cosine":  # annealed to exactly 0 by half of the 200 steps
        assert log["cdm_weight"][0] == pytest.approx(cdm_weight, rel=1e-3)
        assert (log["cdm_weight"][log["step"] >= 100] == 0).all()
    assert all(math.isfinite(v) for v in final["model_div_per_phase"])
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["train_config"]["cdm_weight"] == cdm_weight
    assert saved["train_config"]["cdm_terms"] == terms
    assert saved["train_config"]["cdm_schedule"] == schedule
    assert saved["train_config"]["use_phase"] == use_phase
    model, _ = FlowMLP.load(tmp_path / "final.pt")
    assert model.hidden == (64, 64) and model.use_phase == use_phase


@pytest.mark.parametrize("horizon,action_steps", [(16, 8), (4, 4), (5, 3)])
def test_chunk_replay_reproduces_skate_rollout(horizon, action_steps):
    g = torch.Generator().manual_seed(6)
    starts = _poses(10, seed=6)
    controls = torch.stack([torch.rand(10, 20, generator=g, dtype=D) * 2,
                            (torch.rand(10, 20, generator=g, dtype=D) * 2 - 1) * 2], dim=-1)
    policy = _ReplayChunks(controls, horizon, action_steps).double()
    ours, ctrl, _ = rollout_policy(policy, starts, 20, dt=0.15)
    assert torch.allclose(ctrl, controls)
    assert torch.allclose(ours[..., :2], rollout(starts, controls, 0.15)[..., :2], atol=1e-10)


def test_chunk_mlp_shapes_twist_and_checkpoint(tmp_path):
    torch.manual_seed(7)
    model = FlowMLP(hidden=(16, 16), horizon=16, action_steps=8).double()
    poses, phase = _poses(8), torch.rand(8, dtype=D)
    chunk = model(poses, phase)
    assert chunk.shape == (8, 16, 2)
    assert torch.equal(model.twist(poses, phase), body_twist(chunk[:, 0]))
    _, div = model_divergence(model, poses, phase, create_graph=False)
    assert div.shape == (8,) and torch.isfinite(div).all()
    model.save(tmp_path / "m.pt")
    loaded, _ = FlowMLP.load(tmp_path / "m.pt")
    assert (loaded.horizon, loaded.action_steps) == (16, 8)
    assert torch.equal(loaded.double()(poses, phase), chunk)
    # Checkpoints from before chunking existed predict one control.
    ckpt = torch.load(tmp_path / "m.pt")
    for key in ("horizon", "action_steps"):
        ckpt["config"].pop(key)
    assert FlowMLP(**ckpt["config"])(poses.float(), phase.float()).shape == (8, 2)
    with pytest.raises(ValueError):
        FlowMLP(horizon=4, action_steps=5)


def test_chunk_rows_are_edge_padded():
    ds = prepare_training_set(_toy_dataset(n=4, n_steps=10), TrainConfig(k=2, device="cpu"))
    rows = make_rows(ds, "cpu", horizon=4)
    chunk = rows["chunk"].reshape(4, 10, 4, 2)
    controls = torch.as_tensor(ds.controls, dtype=torch.float32)
    assert torch.equal(chunk[:, 3], controls[:, 3:7])
    assert torch.equal(chunk[:, 8], controls[:, [8, 9, 9, 9]])
    assert torch.equal(chunk[:, :, 0], rows["controls"].reshape(4, 10, 2))


def test_chunk_method_name_and_config():
    assert method_name(0.0, horizon=16) == "fm_chunk"
    assert parse_method("fm_chunk")["horizon"] == 16 and parse_method("fm")["horizon"] == 1
    assert method_name(**parse_method("fm_chunk")) == "fm_chunk"
    assert method_name(0.05, cdm_terms="score", cdm_schedule="cosine", horizon=16) \
        == "chunk_score_w0.05_cos0.5"
    p = parse_method("chunk_score_w0.05_cos0.5")
    assert p["horizon"] == 16 and p["use_phase"] and p["cdm_terms"] == "score"
    assert method_name(**p) == "chunk_score_w0.05_cos0.5"
    TrainConfig(horizon=16, action_steps=8, cdm_weight=0.05, cdm_terms="score")
    for bad in ({"cdm_weight": 0.05}, {"cdm_weight": 0.05, "cdm_terms": "div"},
                {"use_phase": False}, {"action_steps": 17}):
        with pytest.raises(ValueError):
            TrainConfig(**{"horizon": 16, "action_steps": 8, **bad})


def test_chunk_training_smoke(tmp_path):
    ds = prepare_training_set(_toy_dataset(), TrainConfig(k=4, device="cpu"))
    test = prepare_training_set(_toy_dataset(seed=1), TrainConfig(k=4, device="cpu"))
    cfg = TrainConfig(horizon=6, action_steps=3, steps=200, batch_size=128, hidden=(64, 64),
                      k=4, eval_every=100, log_every=50, device="cpu")
    _, final = train(ds, test, cfg, tmp_path, verbose=False)
    log = np.genfromtxt(tmp_path / "log.csv", delimiter=",", names=True)
    assert log["velocity_loss"][-1] < log["velocity_loss"][0]
    for key in ("cdm_loss", "div_l1", "score_l1"):
        assert np.isfinite(log[key]).all()
    assert all(math.isfinite(v) for v in final["model_div_per_phase"])
    model, _ = FlowMLP.load(tmp_path / "final.pt")
    assert (model.horizon, model.action_steps) == (6, 3)


def test_open_loop_demo_chunk_scores_like_the_stored_score():
    """Integrating a demo's own controls lands on its poses, so the chunk score at ``x̂^(j)``
    is the stored score at step ``k + j``; a perfect chunk has zero residual."""
    ds = prepare_training_set(_toy_dataset(n=32, n_steps=10), TrainConfig(k=4, device="cpu"))
    poses = torch.as_tensor(ds.poses, dtype=torch.float32)
    rows = make_rows(ds, "cpu", horizon=4)
    k = 2
    start, target = poses[:, k], rows["chunk"].reshape(32, 10, 4, 2)[:, k]
    states = open_loop_poses(start, target, ds.dt)
    assert torch.allclose(states, poses[:, k:k + 4], atol=1e-4)
    step = torch.full((32,), k)
    demo = poses.transpose(0, 1)
    bw = ds.div_config["bandwidth"]
    assert torch.equal(chunk_score_residuals(target, target, start, step, demo, ds.dt, bw),
                       torch.zeros(32, 4))
    # A chunk off only in v by 1 m/s scores -score_x at each state: +1 on the target's side.
    pred = target - torch.tensor([1.0, 0.0])
    b = chunk_score_residuals(pred, target, start, step, demo, ds.dt, bw)
    states = open_loop_poses(start, pred, ds.dt)
    from fm_bc.divergence import score_kernel_at
    for j in range(4):
        s = score_kernel_at(states[:, j], demo[k + j], bw)
        assert torch.allclose(b[:, j], s[:, 0], atol=1e-5)
    s0 = torch.as_tensor(ds.score[:, k], dtype=torch.float32)
    assert torch.allclose(b[:, 0], s0[:, 0], rtol=1e-2, atol=1e-2)  # float32 vs float64


def test_chunk_score_training_smoke(tmp_path):
    ds = prepare_training_set(_toy_dataset(), TrainConfig(k=4, device="cpu"))
    test = prepare_training_set(_toy_dataset(seed=1), TrainConfig(k=4, device="cpu"))
    cfg = TrainConfig(horizon=6, action_steps=3, cdm_weight=0.05, cdm_terms="score",
                      cdm_schedule="cosine", steps=200, batch_size=128, hidden=(64, 64), k=4,
                      eval_every=100, log_every=50, device="cpu")
    _, final = train(ds, test, cfg, tmp_path, verbose=False)
    log = np.genfromtxt(tmp_path / "log.csv", delimiter=",", names=True)
    assert log["velocity_loss"][-1] < log["velocity_loss"][0]
    assert np.isfinite(log["score_chunk_l1"]).all() and (log["score_chunk_l1"] > 0).all()
    assert np.isfinite(final["mmd_mean"])


class _StampChunks(FlowMLP):
    """A chunked policy whose chunk at phase ``k / N`` is filled with ``k``."""

    def __init__(self, n, horizon):
        super().__init__(hidden=(1,), horizon=horizon, action_steps=1)
        self.n = n

    def forward(self, poses, phase):
        k = round(float(phase[0]) * self.n)
        return poses.new_full((poses.shape[0], self.horizon, 2), float(k))


def test_temporal_ensemble_rollout_of_chunked_mlp():
    n, horizon, m = 20, 5, 0.4
    _, _, cmd = rollout_policy(_StampChunks(n, horizon).double(), _poses(4), n, 0.1,
                               temporal_ensemble=m)
    for t in range(n):
        q = np.arange(max(0, t - horizon + 1), t + 1)
        w = np.exp(-m * np.arange(len(q)))
        np.testing.assert_allclose(cmd[:, t].numpy(), (w @ q) / w.sum(), rtol=1e-12)
    # Consistent chunks: the demo's own windows reproduce it at every step.
    g = torch.Generator().manual_seed(9)
    controls = torch.stack([torch.rand(6, n, generator=g, dtype=D) * 2,
                            torch.rand(6, n, generator=g, dtype=D) - 0.5], dim=-1)
    replay = _ReplayChunks(controls, 16, 1).double()
    ours, ctrl, _ = rollout_policy(replay, _poses(6), n, 0.15, temporal_ensemble=0.01)
    torch.testing.assert_close(ctrl, controls)
    with pytest.raises(ValueError):
        rollout_policy(FlowMLP(hidden=(4,)).double(), _poses(2), 3, 0.1, temporal_ensemble=0.01)


# -- the transformer backbone -------------------------------------------------------------------
#
# FlowTransformer is an alternative backbone for the same arms: same call signature, same target,
# and it must stay usable by model_divergence, which differentiates through it twice.

_TF_NET = {"hidden_dim": 32, "num_layers": 2, "num_heads": 2, "phase_dim": 16}


def _tf(**kw):
    # double throughout, like the rest of this file (_poses returns torch.double).
    return FlowTransformer(**{**_TF_NET, **kw}).double()


def _jittered(model, scale=0.05, seed=0):
    """The output head is zero-initialized, so an untrained model predicts exactly zero."""
    torch.manual_seed(seed)
    for p in model.parameters():
        p.data.add_(torch.randn_like(p) * scale)
    return model


def test_the_transformer_has_the_flow_mlps_call_signature_and_shapes():
    poses, phase = _poses(7), torch.rand(7, dtype=D)
    m = _tf()
    assert m(poses, phase).shape == (7, 2)
    assert m.twist(poses, phase).shape == (7, 3)
    assert torch.equal(m.twist(poses, phase)[:, 1], torch.zeros(7, dtype=D))  # skate constraint
    # Leading batch dimensions are preserved, as FlowMLP's are.
    lead = torch.randn(3, 5, 3, dtype=D)
    assert m(lead, torch.rand(3, 5, dtype=D)).shape == (3, 5, 2)
    chunked = _tf(horizon=4, action_steps=2)
    assert chunked(poses, phase).shape == (7, 4, 2)
    assert chunked.twist(poses, phase).shape == (7, 3)  # the first control's
    with pytest.raises(ValueError):
        _tf(horizon=2, action_steps=3)


def test_the_transformer_phase_input_is_ignored_without_use_phase():
    poses = _poses(6)
    on, off = _jittered(_tf(use_phase=True)), _jittered(_tf(use_phase=False), seed=1)
    assert torch.equal(off(poses, torch.zeros(6, dtype=D)), off(poses, torch.ones(6, dtype=D)))
    assert not torch.equal(on(poses, torch.zeros(6, dtype=D)), on(poses, torch.ones(6, dtype=D)))


def test_the_transformer_is_deterministic_and_starts_as_the_identity():
    poses, phase = _poses(6), torch.rand(6, dtype=D)
    fresh = _tf()
    assert fresh.dropout == 0.0
    # adaLN-Zero gates every branch off and the output head is zeroed, so this is exactly 0.
    assert torch.equal(fresh(poses, phase), torch.zeros(6, 2, dtype=D))
    m = _jittered(_tf())
    m.train()
    assert torch.equal(m(poses, phase), m(poses, phase))
    m.eval()
    assert torch.equal(m(poses, phase), m(poses, phase))


@pytest.mark.parametrize("estimator", ["exact", "hutchinson"])
def test_model_divergence_differentiates_the_transformer_twice(estimator):
    """The CDM term backpropagates through the input Jacobian, so the backbone must support double
    backward -- which is why its attention is written out rather than using SDPA."""
    m = _jittered(_tf())
    poses, phase = _poses(5), torch.rand(5, dtype=D)
    twist, div = model_divergence(m, poses, phase, estimator, create_graph=True)
    assert twist.shape == (5, 3) and div.shape == (5,)
    assert torch.isfinite(div).all()
    loss = twist[:, [0, 2]].pow(2).mean() + 0.05 * div.abs().mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_the_transformer_round_trips_through_a_checkpoint(tmp_path):
    # float32 here, unlike the rest of this file: config() carries no dtype, so load() rebuilds a
    # float32 model and a double checkpoint would come back rounded rather than bitwise equal.
    m = _jittered(FlowTransformer(**_TF_NET)).eval()
    poses, phase = _poses(5).float(), torch.rand(5)
    m.save(tmp_path / "final.pt", info={"note": "smoke"})
    loaded, ckpt = FlowTransformer.load(tmp_path / "final.pt")
    assert torch.equal(loaded.eval()(poses, phase), m(poses, phase))
    assert ckpt["kind"] == "flow_transformer" and ckpt["info"]["note"] == "smoke"
    assert FlowTransformer(**m.config()).config() == m.config()
    # load_policy dispatches: FlowMLP has no `kind`, the transformer does.
    FlowMLP(hidden=(4,)).save(tmp_path / "mlp.pt")
    assert isinstance(load_policy(tmp_path / "mlp.pt")[0], FlowMLP)
    assert isinstance(load_policy(tmp_path / "final.pt")[0], FlowTransformer)


def test_make_flow_policy_passes_only_the_selected_backbones_options():
    assert sorted(ARCHS) == ["mlp", "transformer"]
    assert isinstance(make_flow_policy("mlp", hidden=(8,)), FlowMLP)
    assert isinstance(make_flow_policy("transformer", **_TF_NET), FlowTransformer)
    with pytest.raises(ValueError, match="arch must be one of"):
        make_flow_policy("cnn")
    with pytest.raises(ValueError, match="unexpected kwargs"):
        make_flow_policy("mlp", hidden_dim=64)
    with pytest.raises(ValueError, match="arch must be one of"):
        TrainConfig(arch="cnn")


def test_the_arch_suffix_leaves_every_existing_run_name_untouched():
    """Run directories and figure labels are keyed on these names, and parse_method is their
    inverse -- both halves have to stay in step."""
    legacy = ["fm", "mlp", "fm_chunk", "dp", "mip", "meanflow", "act", "cdm_w0.05",
              "mlp_cdm_w0.05", "score_w0.05_cos0.5", "div_w0.1", "chunk_score_w0.05_cos0.5",
              "dp_te0.01", "fm_chunk_te0.01", "mlp_score_w2.5_cos1"]
    for name in legacy:
        parsed = parse_method(name)
        assert parsed["arch"] == "mlp"
        assert method_name(**parsed) == name

    # The suffix goes inside _te<m>, so the two nest and still round-trip.
    for name in ["fm_tf", "mlp_tf", "score_w0.05_cos0.5_tf", "cdm_w0.05_tf", "fm_chunk_tf",
                 "fm_chunk_tf_te0.01", "chunk_score_w0.05_cos0.5_tf"]:
        parsed = parse_method(name)
        assert parsed["arch"] == "transformer"
        assert method_name(**parsed) == name
    assert parse_method("fm_chunk_tf_te0.01")["horizon"] == 16
    assert ARCH_SUFFIX["mlp"] == ""

    assert run_dir("runs", "dual_path", 64, TrainConfig()).as_posix().endswith("n64/fm/seed0")
    assert run_dir("runs", "dual_path", 64,
                   TrainConfig(arch="transformer")).as_posix().endswith("n64/fm_tf/seed0")
