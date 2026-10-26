"""The skate motion planner: a Theseus ``Objective`` plus the plumbing to drive it.

The trajectory optimization is a nonlinear least-squares problem over ``N+1`` SE(2) poses and
``N`` controls, solved with Levenberg-Marquardt. Theseus batches the whole thing over dim 0,
which is what makes multi-start planning cheap: every restart is one row of the batch and they
all solve in a single call.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import theseus as th
import torch

from .costs import SkateDynamics, XYGoal
from .initializers import interpolate, multistart_seeds
from .skate_model import (
    TrajectoryDiagnostics,
    evaluate,
    se2_tensor,
    xytheta,
)
from .world import GridSDF, ObstacleField

__all__ = ["SkateLimits", "SkateWeights", "SkateSolution", "SkatePlanner"]


@dataclass
class SkateLimits:
    """Control box. ``v_min = 0`` forbids reversing, which is the usual skate assumption."""

    v_max: float = 2.0
    v_min: float = 0.0
    omega_max: float = 2.0

    def clamp(self, controls: torch.Tensor) -> torch.Tensor:
        """Saturate ``(..., 2)`` controls ``[v, omega]`` to the box, as the actuators would.

        The planner only penalizes leaving the box (soft hinge costs); anything that *executes*
        controls on the robot -- e.g. a learned policy's rollout -- should pass them through
        this first.
        """
        v = controls[..., 0].clamp(self.v_min, self.v_max)
        omega = controls[..., 1].clamp(-self.omega_max, self.omega_max)
        return torch.stack([v, omega], dim=-1)


@dataclass
class SkateWeights:
    """Relative importance of each term, expressed on the **squared** error.

    Theseus cost weights are whitening factors -- ``ScaleCostWeight(s)`` contributes
    ``s^2 ||e||^2 / 2`` -- so the planner passes ``sqrt(w)`` internally. Stating the weights this
    way means doubling a number here doubles that term's share of the cost, which is what most
    people expect a weight to do.
    """

    # 2000 chosen by sweep: at 400 the collision term wins trades against the dynamics residual
    # and the rolled-out path drifts ~0.17 m off the optimized poses; above ~50000 the problem
    # becomes stiff enough that LM stalls after a few iterations.
    dynamics: float = 2000.0
    collision: float = 60.0
    limits: float = 60.0
    smoothness: float = 2.0
    effort: float = 0.05
    boundary: float = 10000.0
    slip: float = 200.0  # penalty mode only


@dataclass
class SkateSolution:
    """Result of a solve. All trajectory tensors carry the batch dimension."""

    poses: torch.Tensor  # (B, N+1, 3) as [x, y, theta]
    controls: torch.Tensor  # (B, N, 2)  as [v, omega]
    diagnostics: TrajectoryDiagnostics
    info: th.NonlinearOptimizerInfo
    best_index: int = 0
    feasible: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def best_poses(self) -> torch.Tensor:
        return self.poses[self.best_index]

    @property
    def best_controls(self) -> torch.Tensor:
        return self.controls[self.best_index]

    def report(self) -> str:
        n_feasible = int(self.feasible.sum())
        head = (
            f"{n_feasible}/{len(self.feasible)} restarts feasible; "
            f"best = #{self.best_index}\n"
        )
        return head + "  " + self.diagnostics.summary(self.best_index)


class SkatePlanner:
    """Builds the factor graph once, then solves it for any world / start / goal.

    Parameters
    ----------
    grid_shape:
        ``(rows, cols)`` of the SDF that will be supplied at solve time. Fixed at construction
        because the SDF is an aux variable whose shape is baked into the objective.
    twist_mode:
        ``"structural"`` (default) uses 2-dof controls ``[v, omega]`` and the custom
        :class:`~skate_planner.costs.SkateDynamics`, so side-slip is impossible by construction.
        ``"penalty"`` uses Theseus's own ``GPMotionModel`` + ``Nonholonomic`` over 3-dof body
        twists, where slip is merely penalized. The second exists mainly as a cross-check that
        the custom cost agrees with the library's machinery.
    goal_mode:
        ``"pose"`` constrains the final heading; ``"position"`` leaves it free.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        horizon: int = 40,
        dt: float = 0.15,
        radius: float = 0.3,
        safety_margin: float = 0.15,
        limits: Optional[SkateLimits] = None,
        weights: Optional[SkateWeights] = None,
        twist_mode: Literal["structural", "penalty"] = "structural",
        goal_mode: Literal["pose", "position"] = "pose",
        dtype: torch.dtype = torch.double,
        device: str | torch.device = "cpu",
        max_iterations: int = 60,
        step_size: float = 0.6,
    ):
        if twist_mode not in ("structural", "penalty"):
            raise ValueError(f"twist_mode must be 'structural' or 'penalty', got {twist_mode!r}")
        if goal_mode not in ("pose", "position"):
            raise ValueError(f"goal_mode must be 'pose' or 'position', got {goal_mode!r}")

        self.grid_shape = grid_shape
        self.horizon = horizon
        self.dt = dt
        self.radius = radius
        self.safety_margin = safety_margin
        self.limits = limits or SkateLimits()
        self.weights = weights or SkateWeights()
        self.twist_mode = twist_mode
        self.goal_mode = goal_mode
        self.dtype = dtype
        self.device = torch.device(device)

        self.objective = self._build_objective()
        self.objective.to(device=self.device)
        optimizer = th.LevenbergMarquardt(
            self.objective,
            linear_solver_cls=th.CholeskyDenseSolver,
            max_iterations=max_iterations,
            step_size=step_size,
        )
        self.layer = th.TheseusLayer(optimizer)
        self.layer.to(device=self.device)

    # -- construction -----------------------------------------------------------------------

    def _scale(self, weight: float) -> th.ScaleCostWeight:
        return th.ScaleCostWeight(
            torch.tensor(math.sqrt(weight), dtype=self.dtype, device=self.device)
        )

    def _var(self, tensor: torch.Tensor, name: str) -> th.Variable:
        return th.Variable(tensor.to(dtype=self.dtype, device=self.device), name=name)

    def _build_objective(self) -> th.Objective:
        n = self.horizon
        rows, cols = self.grid_shape
        objective = th.Objective(dtype=self.dtype)

        # Aux variables. Every one of these is a key in the solve-time input dict.
        start_var = th.SE2(name="start", dtype=self.dtype)
        goal_var: th.Manifold = (
            th.SE2(name="goal", dtype=self.dtype)
            if self.goal_mode == "pose"
            else th.Point2(name="goal", dtype=self.dtype)
        )
        sdf_origin = th.Point2(name="sdf_origin", dtype=self.dtype)
        sdf_data = self._var(torch.zeros(1, rows, cols), "sdf_data")
        cell_size = self._var(torch.ones(1, 1), "cell_size")
        cost_eps = self._var(
            torch.tensor([[self.radius + self.safety_margin]]), "cost_eps"
        )
        dt_var = self._var(torch.tensor([[self.dt]]), "dt")

        poses = [th.SE2(name=f"pose_{i}", dtype=self.dtype) for i in range(n + 1)]

        w_boundary = self._scale(self.weights.boundary)
        objective.add(th.Difference(poses[0], start_var, w_boundary, name="start_pose"))
        goal_cost = (
            th.Difference(poses[-1], goal_var, w_boundary, name="goal_pose")
            if self.goal_mode == "pose"
            else XYGoal(poses[-1], goal_var, w_boundary, name="goal_pose")
        )
        objective.add(goal_cost)

        # Obstacle avoidance is identical in both modes: Collision2D takes the SE2 pose
        # directly and chains the jacobian of SE2.xy() internally.
        w_collision = self._scale(self.weights.collision)
        for i in range(1, n + 1):
            objective.add(
                th.eb.Collision2D(
                    poses[i], sdf_origin, sdf_data, cell_size, cost_eps,
                    w_collision, name=f"collision_{i}",
                )
            )

        # Names the caller is expected to supply at solve time. Deliberately *not* the whole
        # aux-var set: cost weights register auto-named scale variables of their own
        # ("Variable__48"), which are constants and would make a missing-key check meaningless.
        self._required_inputs = {
            "start", "goal", "sdf_origin", "sdf_data", "cell_size", "cost_eps", "dt",
            *(f"pose_{i}" for i in range(n + 1)),
        }

        if self.twist_mode == "structural":
            self._add_structural_dynamics(objective, poses, dt_var)
            self._required_inputs |= {f"ctrl_{i}" for i in range(n)}
        else:
            self._add_penalty_dynamics(objective, poses, dt_var)
            self._required_inputs |= {f"vel_{i}" for i in range(n + 1)}
        return objective

    def _add_structural_dynamics(self, objective, poses, dt_var) -> None:
        """2-dof controls with the skate constraint baked into the residual."""
        n = self.horizon
        controls = [th.Vector(2, name=f"ctrl_{i}", dtype=self.dtype) for i in range(n)]

        zero = th.Vector(
            tensor=torch.zeros(1, 2, dtype=self.dtype, device=self.device), name="zero_ctrl"
        )
        lower = self._var(
            torch.tensor([[self.limits.v_min, -self.limits.omega_max]]), "ctrl_lower"
        )
        upper = self._var(
            torch.tensor([[self.limits.v_max, self.limits.omega_max]]), "ctrl_upper"
        )
        # HingeCost requires down/up/threshold all shaped (1, dof). Passing typed Variables
        # rather than floats also avoids its internal torch.ones(1, dof) float32 fallback.
        margin = self._var(torch.zeros(1, 2), "ctrl_margin")

        w_dyn = self._scale(self.weights.dynamics)
        w_lim = self._scale(self.weights.limits)
        w_eff = self._scale(self.weights.effort)
        w_smooth = self._scale(self.weights.smoothness)

        for i in range(n):
            objective.add(
                SkateDynamics(poses[i], poses[i + 1], controls[i], dt_var, w_dyn,
                              name=f"dynamics_{i}")
            )
            objective.add(
                th.eb.HingeCost(controls[i], lower, upper, margin, w_lim, name=f"limits_{i}")
            )
            objective.add(th.Difference(controls[i], zero, w_eff, name=f"effort_{i}"))
            if i + 1 < n:
                objective.add(
                    th.Between(controls[i], controls[i + 1], zero, w_smooth,
                               name=f"smooth_{i}")
                )

    def _add_penalty_dynamics(self, objective, poses, dt_var) -> None:
        """3-dof body twists with slip merely penalized -- the library-native formulation."""
        n = self.horizon
        vels = [th.Vector(3, name=f"vel_{i}", dtype=self.dtype) for i in range(n + 1)]

        # DoubleIntegrator's error is [local(p_i, p_{i+1}) - dt*vel_i ; vel_{i+1} - vel_i], so
        # its first three rows are exactly SkateDynamics' residual with a free lateral slot.
        # A diagonal weight lets the dynamics half and the smoothness half be weighted
        # separately, matching what structural mode does with two distinct cost functions.
        #
        # GPMotionModel is the more idiomatic choice here, but GPCostWeight whitens with the
        # Cholesky of a block matrix that couples the position and velocity halves, so the
        # dynamics rows do not end up weighted at `dynamics` and the two modes stop being
        # comparable -- which is the whole reason this mode exists.
        diagonal = torch.tensor(
            [[math.sqrt(self.weights.dynamics)] * 3
             + [math.sqrt(self.weights.smoothness)] * 3],
            dtype=self.dtype, device=self.device,
        )
        dynamics_weight = th.DiagonalCostWeight(th.Variable(diagonal, name="gp_diagonal"))

        inf = torch.inf
        lower = self._var(
            torch.tensor([[self.limits.v_min, -inf, -self.limits.omega_max]]), "vel_lower"
        )
        upper = self._var(
            torch.tensor([[self.limits.v_max, inf, self.limits.omega_max]]), "vel_upper"
        )
        margin = self._var(torch.zeros(1, 3), "vel_margin")
        w_lim = self._scale(self.weights.limits)
        w_slip = self._scale(self.weights.slip)

        for i in range(n):
            objective.add(
                th.eb.DoubleIntegrator(poses[i], vels[i], poses[i + 1], vels[i + 1],
                                       dt_var, dynamics_weight, name=f"gp_{i}")
            )
        for i in range(n + 1):
            objective.add(
                th.eb.Nonholonomic(poses[i], vels[i], w_slip, name=f"nonholonomic_{i}")
            )
            objective.add(
                th.eb.HingeCost(vels[i], lower, upper, margin, w_lim, name=f"limits_{i}")
            )

    # -- solving ----------------------------------------------------------------------------

    def _validate_inputs(self, inputs: dict[str, torch.Tensor]) -> None:
        """Reject unknown keys.

        Theseus only emits ``warnings.warn`` for a name it does not recognize, so a typo in an
        input key silently does nothing at all and the solve quietly optimizes from stale
        values. Failing loudly here is worth the few lines.
        """
        known = set(self.objective.optim_vars) | set(self.objective.aux_vars)
        unknown = sorted(set(inputs) - known)
        if unknown:
            raise KeyError(
                f"Unknown input keys {unknown}. Theseus would have ignored these with only a "
                f"warning. Known names include: {sorted(known)[:8]} ..."
            )
        missing = sorted(self._required_inputs - set(inputs))
        if missing:
            raise KeyError(
                f"Missing required inputs {missing[:8]}"
                f"{' ...' if len(missing) > 8 else ''}. Theseus would have silently optimized "
                "from whatever values these variables held previously."
            )

    def _batched(self, tensor: torch.Tensor, batch: int) -> torch.Tensor:
        tensor = tensor.to(dtype=self.dtype, device=self.device)
        if tensor.shape[0] == batch:
            return tensor
        if tensor.shape[0] != 1:
            raise ValueError(f"Cannot broadcast batch {tensor.shape[0]} to {batch}.")
        return tensor.expand(batch, *tensor.shape[1:]).contiguous()

    def solve(
        self,
        sdf: GridSDF,
        start: torch.Tensor,
        goal: torch.Tensor,
        init_poses: torch.Tensor,
        init_controls: torch.Tensor,
        clearance_fn=None,
        verbose: bool = False,
        damping: float = 1e-3,
    ) -> SkateSolution:
        """Run the optimizer. ``start``/``goal`` are ``(B, 3)``, seeds are ``(B, N+1, 3)`` /
        ``(B, N, 2)``."""
        batch = init_poses.shape[0]
        start = self._batched(start.reshape(-1, 3), batch)
        goal = self._batched(goal.reshape(-1, 3), batch)

        to = lambda t: torch.as_tensor(t, dtype=self.dtype, device=self.device)
        inputs: dict[str, torch.Tensor] = {
            "start": se2_tensor(start),
            "goal": se2_tensor(goal) if self.goal_mode == "pose" else goal[:, :2].contiguous(),
            "sdf_origin": self._batched(to(sdf.origin).reshape(1, 2), batch),
            "sdf_data": self._batched(to(sdf.data)[None], batch),
            "cell_size": self._batched(to([[sdf.cell_size]]), batch),
            "cost_eps": self._batched(to([[self.radius + self.safety_margin]]), batch),
            "dt": self._batched(to([[self.dt]]), batch),
        }
        for i in range(self.horizon + 1):
            inputs[f"pose_{i}"] = se2_tensor(self._batched(init_poses[:, i], batch))

        if self.twist_mode == "structural":
            for i in range(self.horizon):
                inputs[f"ctrl_{i}"] = self._batched(init_controls[:, i], batch)
        else:
            # 3-dof twists: seed the lateral slot at zero, the value it should keep.
            zeros = torch.zeros(batch, 1, dtype=self.dtype, device=self.device)
            padded = torch.cat(
                [init_controls[:, :, :1], zeros[:, None].expand(batch, self.horizon, 1),
                 init_controls[:, :, 1:]], dim=-1
            )
            for i in range(self.horizon + 1):
                k = min(i, self.horizon - 1)
                inputs[f"vel_{i}"] = self._batched(padded[:, k], batch)

        self._validate_inputs(inputs)
        solution, info = self.layer.forward(
            inputs,
            optimizer_kwargs={
                "track_best_solution": True,
                "verbose": verbose,
                "damping": damping,
                "adaptive_damping": True,
            },
        )
        # Theseus stores best_solution as detached CPU clones (see _maybe_init_best_solution),
        # so it has to be moved back before it meets any device-resident tensor. Note also that
        # it is detached: anyone wanting to backprop through the planner must use `solution`
        # from forward() instead, and accept the last iterate rather than the best one.
        values = info.best_solution if info.best_solution is not None else solution
        get = lambda key: values[key].to(dtype=self.dtype, device=self.device)

        poses = torch.stack(
            [xytheta(get(f"pose_{i}")) for i in range(self.horizon + 1)], dim=1
        )
        if self.twist_mode == "structural":
            controls = torch.stack([get(f"ctrl_{i}") for i in range(self.horizon)], dim=1)
        else:
            twists = torch.stack([get(f"vel_{i}") for i in range(self.horizon)], dim=1)
            controls = twists[..., [0, 2]].contiguous()

        diagnostics = evaluate(
            poses, controls, start, goal, self.dt,
            clearance_fn=clearance_fn if clearance_fn is not None else _no_obstacles,
        )
        feasible = diagnostics.feasible(self.radius)
        best = _pick_best(diagnostics, feasible)
        return SkateSolution(poses, controls, diagnostics, info, best, feasible)

    # -- convenience ------------------------------------------------------------------------

    def plan(
        self,
        world: ObstacleField,
        start: torch.Tensor | np.ndarray,
        goal: torch.Tensor | np.ndarray,
        cell_size: float = 0.1,
        n_starts: int = 1,
        rng: np.random.Generator | None = None,
        verbose: bool = False,
    ) -> SkateSolution:
        """Rasterize the world, build seeds, solve, and score -- the one-call entry point."""
        sdf = world.grid_sdf(cell_size)
        if tuple(sdf.data.shape) != tuple(self.grid_shape):
            raise ValueError(
                f"World rasterizes to {sdf.data.shape} but this planner was built for "
                f"{self.grid_shape}. Rebuild it with grid_shape={sdf.data.shape}, or use "
                f"SkatePlanner.for_world(...)."
            )
        start_t = torch.as_tensor(start, dtype=self.dtype, device=self.device).reshape(-1, 3)
        goal_t = torch.as_tensor(goal, dtype=self.dtype, device=self.device).reshape(-1, 3)

        if n_starts == 1:
            seeds, seed_ctrls = interpolate(start_t, goal_t, self.horizon, self.dt)
        else:
            seeds, seed_ctrls = multistart_seeds(
                start_t, goal_t, self.horizon, self.dt, n_starts, world=world, rng=rng
            )
        return self.solve(
            sdf, start_t, goal_t, seeds, seed_ctrls,
            clearance_fn=world.analytic_sdf, verbose=verbose,
        )

    @classmethod
    def for_world(
        cls, world: ObstacleField, cell_size: float = 0.1, **kwargs
    ) -> "SkatePlanner":
        """Build a planner sized to ``world``'s rasterization at ``cell_size``."""
        rows, cols = world.grid_sdf(cell_size).data.shape
        return cls(grid_shape=(rows, cols), **kwargs)


def _no_obstacles(xy: np.ndarray) -> np.ndarray:
    return np.full(xy.shape[:-1], np.inf)


def _pick_best(diagnostics: TrajectoryDiagnostics, feasible: np.ndarray) -> int:
    """Prefer feasible restarts; among them, the shortest path. Otherwise least-bad clearance."""
    if feasible.any():
        scores = np.where(feasible, diagnostics.path_length, np.inf)
        return int(np.argmin(scores))
    return int(np.argmax(diagnostics.min_clearance))
