"""Roll a policy out from the start distribution and compare it with held-out demonstrations.

Every metric is distributional: a flow-matching policy is deterministic given its start, so
multimodality has to come from how it splits the *start distribution* between modes. The
rollout starts are the held-out set's own sampled starts, so the two sets of trajectories are
draws from the same start distribution and can be compared phase by phase.

The generative baselines (:class:`~fm_bc.skate.baselines.ChunkPolicy`) are stochastic instead: they
sample a chunk of controls from the last poses and execute part of it before re-planning, so
they can also split between modes *within* one start. Their sampling noise is drawn from a
seeded generator, so their rollouts are reproducible too.

Any chunk policy can instead be rolled out with **temporal ensembling** (ACT, Zhao et al. 2023):
a chunk is predicted at *every* control step, and the control executed at step ``t`` is the
weighted mean of every prediction made so far for ``t``, ``sum_i w_i u_i / sum_i w_i`` with
``w_i = exp(-m i)`` and ``i = 0`` the oldest -- ACT's ``temporal_agg``. ``m`` is the
``temporal_ensemble`` argument (ACT uses 0.01).
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from skate_planner.planner import SkateLimits
from skate_planner.skate_model import body_twist

from ..divergence import divergence_knn
from ..se2 import se2_exp_retract
from .baselines import ChunkPolicy
from .dataset import TrajectoryDataset
from .policy import FlowMLP, model_divergence

__all__ = ["rollout_policy", "rollout_chunk_policy", "temporal_ensemble_weights",
           "dense_clearance", "collision_breakdown", "mmd", "evaluate_policy"]


def temporal_ensemble_weights(n: int, m: float) -> torch.Tensor:
    """ACT's ``exp(-m i)`` for ``n`` predictions of one step, oldest first, normalized."""
    w = torch.exp(-m * torch.arange(n, dtype=torch.float64))
    return w / w.sum()


class _Ensemble:
    """Every chunk predicted so far, and their temporally weighted mean at a step."""

    def __init__(self, m: float):
        self.m, self.queries = m, []  # (query step k, future controls (B, L, 2) for k .. k+L-1)

    def add(self, k: int, future: torch.Tensor) -> None:
        self.queries.append((k, future))
        # A query can no longer cover this or any later step once its chunk has run out.
        self.queries = [(q, f) for q, f in self.queries if k - q < f.shape[1]]

    def at(self, t: int) -> torch.Tensor:
        preds = [f[:, t - q] for q, f in self.queries if 0 <= t - q < f.shape[1]]
        w = temporal_ensemble_weights(len(preds), self.m).to(preds[0])
        return torch.einsum("i,ibc->bc", w, torch.stack(preds))


@torch.no_grad()
def rollout_policy(
    model: FlowMLP, starts: torch.Tensor, n_steps: int, dt: float,
    limits: SkateLimits | None = None, temporal_ensemble: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-loop ``T <- T Exp(dt [v, 0, omega])`` with the phase advancing ``k / n_steps``.

    With ``limits`` the commanded controls are saturated to the robot's control box before
    they are executed -- an actuator cannot exceed them, so the rollout is what a real skate
    robot would drive. Controls are held for ``dt``, exactly the zero-order hold the
    demonstrations were rolled out with, so a policy that reproduces a demo's controls
    reproduces its poses exactly.

    A chunked ``FlowMLP`` (``horizon > 1``) is queried only every ``action_steps`` steps, at
    that step's pose and phase, and executes its chunk's first ``action_steps`` controls. With
    ``temporal_ensemble = m`` it is queried at every step instead and executes the temporal
    ensemble of all its chunks (see the module docstring); the commanded control is that mean.

    Returns poses ``(M, n_steps + 1, 3)``, the executed controls ``(M, n_steps, 2)`` and the
    commanded (pre-saturation) controls.
    """
    chunked, ta = model.horizon > 1, model.action_steps
    if temporal_ensemble is not None and not chunked:
        raise ValueError("temporal ensembling needs a chunked policy (horizon > 1)")
    ensemble = _Ensemble(temporal_ensemble) if temporal_ensemble is not None else None
    pose = starts
    poses, executed, commanded = [pose], [], []
    for k in range(n_steps):
        phase = pose.new_full((pose.shape[0],), k / n_steps)
        if ensemble is not None:
            ensemble.add(k, model(pose, phase))
            u = ensemble.at(k)
        elif not chunked:
            u = model(pose, phase)
        else:
            if k % ta == 0:
                chunk = model(pose, phase)
            u = chunk[:, k % ta]
        u_exec = limits.clamp(u) if limits is not None else u
        pose = se2_exp_retract(pose, dt * body_twist(u_exec))
        poses.append(pose)
        executed.append(u_exec)
        commanded.append(u)
    return (torch.stack(poses, dim=1), torch.stack(executed, dim=1),
            torch.stack(commanded, dim=1))


@torch.no_grad()
def rollout_chunk_policy(
    policy: ChunkPolicy, starts: torch.Tensor, n_steps: int, dt: float,
    limits: SkateLimits | None = None, generator: torch.Generator | None = None,
    temporal_ensemble: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Receding-horizon rollout of a chunk policy: every ``Ta`` steps it samples a chunk from
    the last ``To`` poses (the start repeated before there are ``To``) and executes its
    ``chunk[To-1 : To-1+Ta]`` -- the window convention of :mod:`fm_bc.skate.chunks`. Saturation, the
    zero-order hold and the returns are those of :func:`rollout_policy`.

    With ``temporal_ensemble = m`` a chunk is sampled at every step and the executed control is
    the temporal ensemble of the future parts ``chunk[To-1:]`` of all chunks so far."""
    to, ta = policy.obs_steps, policy.action_steps
    ensemble = _Ensemble(temporal_ensemble) if temporal_ensemble is not None else None
    pose = starts
    poses, executed, commanded = [pose], [], []
    for k in range(n_steps):
        replan = ensemble is not None or k % ta == 0
        if replan:
            obs = torch.stack([poses[max(k - to + 1 + j, 0)] for j in range(to)], dim=1)
            future = policy.sample(obs, generator)[:, to - 1:]
        if ensemble is not None:
            ensemble.add(k, future)
            u = ensemble.at(k)
        else:
            if replan:
                chunk = future[:, :ta]
            u = chunk[:, k % ta]
        u_exec = limits.clamp(u) if limits is not None else u
        pose = se2_exp_retract(pose, dt * body_twist(u_exec))
        poses.append(pose)
        executed.append(u_exec)
        commanded.append(u)
    return (torch.stack(poses, dim=1), torch.stack(executed, dim=1),
            torch.stack(commanded, dim=1))


def dense_clearance(
    poses: torch.Tensor, controls: torch.Tensor, dt: float, sdf, substeps: int = 8
) -> np.ndarray:
    """Minimum signed distance along each ``(M, N+1, 3)`` trajectory, ``(M,)``.

    Each control interval is an exact arc, densified by ``substeps`` -- the same check
    ``skate_model.evaluate`` applies to the demonstrations, so a fast skate cannot cut a corner
    between knots unseen.
    """
    s = torch.linspace(0.0, 1.0, substeps + 1, dtype=poses.dtype, device=poses.device)
    step = dt * body_twist(controls)  # (M, N, 3)
    dense = se2_exp_retract(poses[:, :-1, None, :], s[None, None, :, None] * step[:, :, None, :])
    xy = dense[..., :2].reshape(poses.shape[0], -1, 2).cpu().numpy()
    return sdf(xy).min(axis=1)


def collision_breakdown(
    poses: torch.Tensor, controls: torch.Tensor, dt: float, world, radius: float,
    substeps: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Which rollouts come within ``radius`` of an obstacle, and which of the world boundary,
    ``(M,)`` bools each (a rollout can do both; their union is the collision test).

    The demos' goals sit close to the top wall, so a rollout that overshoots its goal by a few
    centimetres touches the boundary. Splitting the two keeps the obstacle rate -- the
    mode-averaging failure -- readable on its own.
    """
    obstacle = dense_clearance(poses, controls, dt, world._obstacle_sdf, substeps) < radius
    boundary = dense_clearance(poses, controls, dt, world._boundary_sdf, substeps) < radius
    return obstacle, boundary


def _embed(poses: torch.Tensor, rot_scale: float) -> torch.Tensor:
    th = poses[..., 2]
    return torch.stack(
        [poses[..., 0], poses[..., 1], rot_scale * torch.cos(th), rot_scale * torch.sin(th)], dim=-1
    )


def mmd(a: torch.Tensor, b: torch.Tensor, bandwidth: float = 0.3, rot_scale: float = 1.0) -> float:
    """Squared MMD (biased V-statistic) between pose sets ``(M, 3)`` and ``(M', 3)``.

    Gaussian kernel on ``[x, y, l cos theta, l sin theta]``: a Euclidean embedding, so the kernel
    is positive definite by construction (a Gaussian of the SE(2) log distance need not be). Its
    chord distance ``2 l sin(dtheta/2)`` matches ``l dtheta`` for small heading differences. The
    bandwidth is fixed, not the median heuristic, so values are comparable across runs.
    """
    ea, eb = _embed(a, rot_scale), _embed(b, rot_scale)
    gamma = 1.0 / (2.0 * bandwidth**2)

    def k(p, q):
        return torch.exp(-gamma * torch.cdist(p, q).pow(2)).mean()

    return float(k(ea, ea) + k(eb, eb) - 2.0 * k(ea, eb))


def evaluate_policy(
    model: FlowMLP | ChunkPolicy,
    ds_test: TrajectoryDataset,
    bandwidth: float = 0.3,
    rot_scale: float = 1.0,
    k: int = 16,
    goal_sigmas: float = 3.0,
    full: bool = True,
    clamp: bool = True,
    device: str = "cpu",
    sample_seed: int = 0,
    temporal_ensemble: float | None = None,
) -> tuple[dict, dict]:
    """Metrics ``dict`` of scalars and per-phase lists, and the rollout arrays.

    ``clamp`` saturates commands to the planner's control limits recorded with the demos
    (:attr:`TrajectoryDataset.limits`) before executing them; collisions and every other
    metric are then measured on the path actually driven. ``full=False`` skips the divergence
    diagnostics (for cheap evaluations during training).

    A :class:`ChunkPolicy` is rolled out with :func:`rollout_chunk_policy`, its noise seeded by
    ``sample_seed``. It has no deterministic velocity field, so it reports no
    ``model_div_per_phase``; the rollouts' kNN divergence (``div_curve_mae``) is still measured.

    ``temporal_ensemble = m`` rolls a chunk policy out with ACT's temporal ensembling instead of
    executing ``action_steps`` of each chunk (see the module docstring).
    """
    spec = ds_test.spec
    was_training = model.training
    model.eval()
    dtype = next(model.parameters()).dtype
    to = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

    n_steps = ds_test.controls.shape[1]
    test = to(ds_test.poses)
    limits = ds_test.limits
    chunked = isinstance(model, ChunkPolicy)
    if chunked:
        gen = torch.Generator(device=device).manual_seed(sample_seed)
        poses, controls, commanded = rollout_chunk_policy(
            model, test[:, 0], n_steps, ds_test.dt, limits if clamp else None, gen,
            temporal_ensemble)
    else:
        poses, controls, commanded = rollout_policy(model, test[:, 0], n_steps, ds_test.dt,
                                                    limits if clamp else None, temporal_ensemble)

    mmd_phase = [mmd(poses[:, t], test[:, t], bandwidth, rot_scale) for t in range(n_steps + 1)]

    radius = float(ds_test.config.get("radius", 0.3))
    obstacle_hit, boundary_hit = collision_breakdown(poses, controls, ds_test.dt, spec.world,
                                                     radius)
    collided = obstacle_hit | boundary_hit

    final = poses[:, -1, :2].cpu().numpy()
    reach = np.zeros(len(final), dtype=bool)
    for g in spec.goals:
        tol = spec.reach_radius if spec.reach_radius is not None else goal_sigmas * g.sigma
        reach |= np.linalg.norm(final - [g.x, g.y], axis=-1) <= tol
    success = reach & ~collided

    poses_np = poses.cpu().numpy()
    labels = spec.label_mode(poses_np)
    frac = [float(np.mean(labels == m)) for m in range(spec.n_modes)]
    demo_frac = [float(np.mean(ds_test.mode == m)) for m in range(spec.n_modes)]
    undecided = float(np.mean(labels < 0))
    # Total variation between the mode histograms, with "undecided" as its own bin.
    mode_tv = 0.5 * (sum(abs(f - d) for f, d in zip(frac, demo_frac)) + undecided)

    c = controls.cpu().numpy()
    cmd = commanded.cpu().numpy()
    tol = float(ds_test.config.get("limit_tol", 0.05))
    # Commands outside the control box by more than the demos' own tolerance: with clamping
    # these steps leaned on the saturation; without it the robot drove beyond its limits.
    excess = np.maximum.reduce([
        cmd[..., 0] - limits.v_max, limits.v_min - cmd[..., 0],
        np.abs(cmd[..., 1]) - limits.omega_max,
    ])
    saturated = excess > tol

    metrics = {
        "mmd_mean": float(np.mean(mmd_phase)),
        "mmd_final": mmd_phase[-1],
        "mmd_per_phase": mmd_phase,
        "collision_rate": float(collided.mean()),
        "obstacle_collision_rate": float(obstacle_hit.mean()),
        "boundary_collision_rate": float(boundary_hit.mean()),
        "reach_rate": float(reach.mean()),
        "success_rate": float(success.mean()),
        "mode_fractions": frac,
        "demo_mode_fractions": demo_frac,
        "undecided_rate": undecided,
        "mode_tv": float(mode_tv),
        "saturation_rate": float(saturated.any(axis=1).mean()),
        "saturated_step_fraction": float(saturated.mean()),
        "clamped": bool(clamp),
        "temporal_ensemble": temporal_ensemble,
        "limits": asdict(limits),
    }
    arrays = {"poses": poses_np, "controls": c, "commanded": cmd, "labels": labels,
              "collided": collided}

    if full:
        p64 = poses.double()
        roll_div = divergence_knn(p64, ds_test.dt, k=k).mean(dim=0).cpu().numpy()
        if ds_test.divergence is not None:
            demo_div = ds_test.divergence.mean(axis=0)
        else:
            demo_div = divergence_knn(test.double(), ds_test.dt, k=k).mean(dim=0).cpu().numpy()
        metrics.update({
            "rollout_div_per_phase": roll_div.tolist(),
            "demo_div_per_phase": demo_div.tolist(),
            "div_curve_mae": float(np.mean(np.abs(roll_div - demo_div))),
        })
        if not chunked:
            phase = to(np.repeat(np.arange(n_steps) / n_steps, poses.shape[0]))
            flat = poses[:, :-1].transpose(0, 1).reshape(-1, 3)
            _, div_v = model_divergence(model, flat, phase, create_graph=False)
            model_div = div_v.detach().reshape(n_steps, -1).mean(dim=1).cpu().numpy()
            metrics["model_div_per_phase"] = model_div.tolist()

    model.train(was_training)
    return metrics, arrays
