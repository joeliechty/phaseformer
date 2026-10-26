"""Generate demonstration distributions with the Theseus skate planner, keeping only valid ones.

Each batch row is an independent planning problem -- its own start, goal and seed via point --
so one batched Levenberg-Marquardt solve yields a whole chunk of demonstrations. What is kept
is the **rolled-out** trajectory (controls re-integrated from the start), not the optimizer's
pose variables: only the rollout satisfies the skate model exactly, and it is what a policy
trained on ``(pose, twist)`` pairs will be asked to reproduce.

A trajectory is rejected if the rollout

* touches an obstacle (dense rollout vs. the analytic SDF) or misses the goal,
* exceeds the control limits (they are soft hinge costs in the optimizer), or
* lands in a different homotopy class than the one it was seeded for.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from skate_planner.initializers import interpolate
from skate_planner.planner import SkateLimits, SkatePlanner
from skate_planner.skate_model import body_twist, rollout

from ..divergence import divergence_kde, divergence_knn, score_kernel
from .scenarios import ScenarioSpec, get_scenario

__all__ = ["GenerationConfig", "TrajectoryDataset", "generate", "plan_problems", "add_div_and_score"]


@dataclass
class GenerationConfig:
    horizon: int = 50
    dt: float = 0.15
    cell_size: float = 0.1
    chunk: int = 512
    via_jitter: float = 0.3
    limit_tol: float = 0.05
    goal_tol: float = 0.25
    balance: bool = True
    max_rounds: int = 40
    seed: int = 0
    # Two switches back to the planner before commit 2eaf39d, which the shipped single_path
    # demos were made with (scripts/skate/reproduce_datasets.py): the distance-transform grid
    # SDF (ObstacleField.grid_sdf, method "edt") and seeds whose last knot snaps to the goal
    # heading. The defaults are today's planner.
    sdf: str = "exact"
    seed_goal_heading: bool = False


@dataclass
class TrajectoryDataset:
    """Accepted demonstrations for one scenario, all on the shared phase grid ``k / N``."""

    scenario: str
    poses: np.ndarray  # (M, N+1, 3) [x, y, theta], rolled out
    controls: np.ndarray  # (M, N, 2) [v, omega]
    twists: np.ndarray  # (M, N, 3) body twists [v, 0, omega]
    mode: np.ndarray  # (M,)
    starts: np.ndarray  # (M, 3) sampled start
    goals: np.ndarray  # (M, 3) sampled goal
    dt: float
    config: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    divergence: np.ndarray | None = None  # (M, N+1) SE(2), 1/s
    divergence_xy: np.ndarray | None = None  # (M, N+1) positions only, 1/s
    score: np.ndarray | None = None  # (M, N+1, 3) body-frame [1/m, 1/m, 1/rad]
    div_config: dict = field(default_factory=dict)

    @property
    def phase(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.poses.shape[1])

    @property
    def spec(self) -> ScenarioSpec:
        return get_scenario(self.scenario)

    @property
    def limits(self) -> SkateLimits:
        """The planner's control box these demos were generated under: the robot's physical
        limits, which a policy's commands are saturated to when it is executed."""
        return SkateLimits(**self.config.get("limits", {}))

    def __len__(self) -> int:
        return self.poses.shape[0]

    def save(self, path: str | Path) -> None:
        arrays = {
            k: v for k, v in asdict(self).items()
            if isinstance(v, np.ndarray)
        }
        meta = {
            "scenario": self.scenario, "dt": self.dt, "config": self.config,
            "stats": self.stats, "div_config": self.div_config,
        }
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryDataset":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            arrays = {k: f[k] for k in f.files if k != "meta"}
        return cls(**meta, **arrays)


def _limits_ok(controls: np.ndarray, limits: SkateLimits, tol: float) -> np.ndarray:
    v, w = controls[..., 0], controls[..., 1]
    ok = (v >= limits.v_min - tol) & (v <= limits.v_max + tol) & (np.abs(w) <= limits.omega_max + tol)
    return ok.all(axis=1)


def _solve_batch(planner: SkatePlanner, sdf, spec: ScenarioSpec, cfg: GenerationConfig,
                 starts: np.ndarray, goals: np.ndarray, vias: np.ndarray, modes: np.ndarray,
                 goal_heading: bool, device: str):
    """One batched solve: ``(rolled poses, controls, feasible, within limits, right mode)``."""
    to = lambda a: torch.as_tensor(a, dtype=torch.double, device=device)
    start_t, goal_t = to(starts), to(goals)
    # A position-only goal lets the seed arrive along its own tangent (unless
    # seed_goal_heading asks for the old behaviour); a pose goal snaps it to the heading.
    seeds, seed_ctrls = interpolate(start_t, goal_t, cfg.horizon, cfg.dt, via=to(vias),
                                    goal_heading=goal_heading)
    sol = planner.solve(
        sdf, start_t, goal_t, seeds, seed_ctrls, clearance_fn=spec.world.analytic_sdf
    )
    rolled = rollout(start_t, sol.controls, cfg.dt).cpu().numpy()
    controls = sol.controls.cpu().numpy()

    feasible = sol.diagnostics.feasible(planner.radius, goal_tol=cfg.goal_tol)
    limits = _limits_ok(controls, planner.limits, cfg.limit_tol)
    right_mode = spec.label_mode(rolled) == modes
    return rolled, controls, feasible, limits, right_mode


def plan_problems(
    spec: ScenarioSpec,
    starts: np.ndarray,
    modes: np.ndarray,
    config: GenerationConfig | None = None,
    device: str = "cpu",
) -> tuple[TrajectoryDataset, np.ndarray]:
    """Plan given ``starts (n, 3)`` to the sides ``modes (n,)``, each to its mode's goal mean
    through its unjittered via chain, in chunks of ``config.chunk``.

    Returns every problem as a :class:`TrajectoryDataset`, in the order given, and the ``(n,)``
    mask of the ones :func:`generate` would accept. What to do with the rest is the caller's.
    """
    cfg = config or GenerationConfig()
    planner = SkatePlanner.for_world(
        spec.world, cell_size=cfg.cell_size, horizon=cfg.horizon, dt=cfg.dt,
        goal_mode=spec.goal_mode, device=device,
    )
    sdf = spec.world.grid_sdf(cfg.cell_size, method=cfg.sdf)
    goal_heading = cfg.seed_goal_heading or spec.goal_mode == "pose"
    chains = np.stack([np.asarray(v, dtype=float).reshape(-1, 2) for v in spec.vias])
    goal_means = np.stack([spec.goals[g].mean for g in spec.goal_of_mode])
    modes = np.asarray(modes, dtype=int)
    out = {"poses": [], "controls": [], "ok": []}
    for lo in range(0, len(starts), cfg.chunk):
        sl = slice(lo, lo + cfg.chunk)
        rolled, controls, feasible, limits, right_mode = _solve_batch(
            planner, sdf, spec, cfg, starts[sl], goal_means[modes[sl]], chains[modes[sl]],
            modes[sl], goal_heading, device)
        out["poses"].append(rolled)
        out["controls"].append(controls)
        out["ok"].append(feasible & limits & right_mode)
    controls = np.concatenate(out["controls"])
    ds = TrajectoryDataset(
        scenario=spec.name,
        poses=np.concatenate(out["poses"]),
        controls=controls,
        twists=body_twist(torch.as_tensor(controls)).numpy(),
        mode=modes.copy(),
        starts=np.asarray(starts, dtype=float).copy(),
        goals=goal_means[modes],
        dt=cfg.dt,
        config={**asdict(cfg), "device": device, "radius": planner.radius,
                "safety_margin": planner.safety_margin, "limits": asdict(planner.limits)},
    )
    return ds, np.concatenate(out["ok"])


def generate(
    spec: ScenarioSpec,
    n_target: int = 1024,
    config: GenerationConfig | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> TrajectoryDataset:
    """Plan until ``n_target`` valid demonstrations are collected (split evenly across modes
    when ``config.balance``).

    A ``spec.deterministic`` scenario plans each mode once, from its fixed start, goal and via
    chain, and repeats that one trajectory to the mode's count: every demonstration of a mode
    is then bit-identical.
    """
    cfg = config or GenerationConfig()
    rng = np.random.default_rng(cfg.seed)
    via_jitter = cfg.via_jitter if spec.via_jitter is None else spec.via_jitter
    planner = SkatePlanner.for_world(
        spec.world, cell_size=cfg.cell_size, horizon=cfg.horizon, dt=cfg.dt,
        goal_mode=spec.goal_mode, device=device,
    )
    sdf = spec.world.grid_sdf(cfg.cell_size, method=cfg.sdf)
    per_mode = (
        [math.ceil(n_target / spec.n_modes)] * spec.n_modes if cfg.balance
        else [math.ceil(n_target * w) for w in spec.mode_weights]
    )

    kept: list[list[dict]] = [[] for _ in range(spec.n_modes)]
    tried = np.zeros(spec.n_modes, dtype=int)
    valid = np.zeros(spec.n_modes, dtype=int)  # before trimming to the per-mode target
    reasons = {"collision_or_goal": 0, "limits": 0, "wrong_mode": 0}
    t0 = time.time()

    def count(m: int) -> int:
        return sum(len(b["poses"]) for b in kept[m])

    # A pose goal fixes the arrival heading, so the seed's last knot should take it.
    goal_heading = cfg.seed_goal_heading or spec.goal_mode == "pose"
    for rnd in range(cfg.max_rounds):
        if spec.deterministic and rnd > 0:
            break  # one solve per mode; a second would be the same problem again
        deficit = np.array([max(0, per_mode[m] - count(m)) for m in range(spec.n_modes)])
        if not deficit.any():
            break
        # Spend the chunk on the modes still short, in proportion to how short they are.
        modes = (np.arange(spec.n_modes) if spec.deterministic
                 else rng.choice(spec.n_modes, size=cfg.chunk, p=deficit / deficit.sum()))
        starts, goals, vias, modes = spec.sample(len(modes), rng, modes, via_jitter)
        rolled, controls, feasible, limits, right_mode = _solve_batch(
            planner, sdf, spec, cfg, starts, goals, vias, modes, goal_heading, device)
        ok = feasible & limits & right_mode
        reasons["collision_or_goal"] += int((~feasible).sum())
        reasons["limits"] += int((feasible & ~limits).sum())
        reasons["wrong_mode"] += int((feasible & limits & ~right_mode).sum())

        for m in range(spec.n_modes):
            sel = np.flatnonzero((modes == m) & ok)[: max(0, per_mode[m] - count(m))]
            if spec.deterministic:
                sel = np.repeat(sel, per_mode[m])
            tried[m] += int((modes == m).sum())
            valid[m] += int(((modes == m) & ok).sum())
            kept[m].append({
                "poses": rolled[sel], "controls": controls[sel],
                "starts": starts[sel], "goals": goals[sel],
            })
        if verbose:
            have = ", ".join(f"{spec.mode_names[m]} {count(m)}/{per_mode[m]}"
                             for m in range(spec.n_modes))
            print(f"  [{spec.name}] round {rnd}: accepted {int(ok.sum())}/{len(modes)} -> {have}"
                  f"  ({time.time() - t0:.1f}s)")
    if any(count(m) < per_mode[m] for m in range(spec.n_modes)):
        print(f"  [{spec.name}] WARNING: stopped short of target.")

    def cat(key: str) -> np.ndarray:
        return np.concatenate([b[key] for m in range(spec.n_modes) for b in kept[m]])

    mode = np.concatenate([
        np.full(sum(len(b["poses"]) for b in kept[m]), m) for m in range(spec.n_modes)
    ])
    controls = cat("controls")
    stats = {
        "n": int(len(mode)),
        "per_mode": [count(m) for m in range(spec.n_modes)],
        "tried_per_mode": tried.tolist(),
        "acceptance_per_mode": [float(valid[m] / max(1, tried[m])) for m in range(spec.n_modes)],
        "rejections": reasons,
        "seconds": time.time() - t0,
    }
    if spec.deterministic:
        stats["unique_trajectories"] = int(sum(count(m) > 0 for m in range(spec.n_modes)))
    return TrajectoryDataset(
        scenario=spec.name,
        poses=cat("poses"),
        controls=controls,
        twists=body_twist(torch.as_tensor(controls)).numpy(),
        mode=mode,
        starts=cat("starts"),
        goals=cat("goals"),
        dt=cfg.dt,
        config={**asdict(cfg), "device": device, "radius": planner.radius,
                "safety_margin": planner.safety_margin, "limits": asdict(planner.limits)},
        stats=stats,
    )


def add_div_and_score(
    ds: TrajectoryDataset,
    method: str = "knn",
    k: int = 16,
    bandwidth: float = 0.2,
    div_bandwidth: float | None = None,
    rot_scale: float = 1.0,
    window_length: int = 9,
    polyorder: int = 2,
    leave_one_out: bool = True,
    device: str = "cpu",
) -> TrajectoryDataset:
    """Fill ``divergence`` (SE(2)), ``divergence_xy`` and ``score`` in place and return ``ds``.

    The SE(2) analogue of robomimic's ``add_div_and_score_to_training_data``. No phase
    binning is needed: every demonstration already shares the fixed-horizon phase grid.

    ``method`` picks the divergence estimator: ``"knn"`` (k-th neighbour radius, the default)
    or ``"kde"`` (Gaussian KDE with ``div_bandwidth``, defaulting to the score's
    ``bandwidth``). ``k`` is only used by ``"knn"``. kNN with ``k=16`` is the default: on flows
    of known divergence its error is almost pure zero-mean noise (RMS bias 0.02 1/s), which
    averages out in a regression target, while a fixed-bandwidth KDE is systematically
    attenuated wherever the distribution is narrower than its kernel (RMS bias 0.16 1/s at
    ``h=0.2``). kNN attenuates too once the spread drops below the neighbour spacing, but much
    later: at scenario A's 0.025 m corridor it recovers ~60% of a known compression vs. ~3% for
    KDE ``h=0.2``. ``scripts/skate/paper_divergence_figures.py`` reproduces all of this.
    """
    if method not in ("kde", "knn"):
        raise ValueError(f"method must be 'kde' or 'knn', got {method!r}")
    div_bandwidth = bandwidth if div_bandwidth is None else div_bandwidth
    poses = torch.as_tensor(ds.poses, dtype=torch.double, device=device)
    common = dict(window_length=window_length, polyorder=polyorder, rot_scale=rot_scale)

    def divergence(metric: str) -> np.ndarray:
        if method == "kde":
            d = divergence_kde(poses, ds.dt, bandwidth=div_bandwidth, metric=metric,
                               leave_one_out=leave_one_out, **common)
        else:
            d = divergence_knn(poses, ds.dt, k=k, metric=metric, **common)
        return d.cpu().numpy()

    ds.divergence = divergence("se2")
    ds.divergence_xy = divergence("xy")
    ds.score = score_kernel(poses, bandwidth, metric="se2", rot_scale=rot_scale).cpu().numpy()
    ds.div_config = {"method": method, "k": k, "bandwidth": bandwidth,
                     "div_bandwidth": div_bandwidth, "rot_scale": rot_scale,
                     "window_length": window_length, "polyorder": polyorder,
                     "leave_one_out": leave_one_out}
    return ds
