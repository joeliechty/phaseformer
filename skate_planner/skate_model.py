"""First-order skate kinematics on SE(2).

The robot has state ``q = (x, y, theta)`` and input ``u = (v, omega)``::

    xdot = v cos(theta)      ydot = v sin(theta)      thetadot = omega

with the no-side-slip ("skate") constraint ``xdot sin(theta) - ydot cos(theta) = 0``.

Everything here is built on one idea: expressed as a **body-frame twist**, that model is

.. code-block:: text

    xi = [v, 0, omega]
              ^
              the literal zero IS the skate constraint

Theseus's SE(2) tangent space is ordered ``[u_x, u_y, theta]`` (verified empirically), so the
lateral slot of the twist is index 1. Because it is a hard-coded zero rather than a decision
variable, side-slip is impossible by construction -- there is no penalty weight to tune and no
residual slip to apologize for.

Integration is *exact*, not an Euler step: holding ``(v, omega)`` constant over ``dt`` traces a
circular arc, and ``pose.retract(dt * xi) == pose @ Exp(dt * xi)`` is precisely that arc.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
import theseus as th
import torch

__all__ = [
    "LATERAL_INDEX",
    "body_twist",
    "twist_selector",
    "integrate",
    "rollout",
    "se2_tensor",
    "xytheta",
    "wrap_angle",
    "lateral_slip",
    "TrajectoryDiagnostics",
    "evaluate",
]

#: Index of the lateral component within an SE(2) body twist ``[u_x, u_y, theta]``.
LATERAL_INDEX = 1


def twist_selector(dtype: torch.dtype = torch.double, device=None) -> torch.Tensor:
    """The constant ``(3, 2)`` matrix ``S`` with ``S @ [v, omega] == [v, 0, omega]``.

    Also the (negated, scaled) control jacobian of the dynamics residual, which is why it is
    factored out here rather than written inline.
    """
    return torch.tensor(
        [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=dtype, device=device
    )


def body_twist(controls: torch.Tensor) -> torch.Tensor:
    """Map controls ``[v, omega]`` of shape ``(..., 2)`` to twists ``[v, 0, omega]``.

    This is the single place the skate constraint is expressed.
    """
    v = controls[..., 0]
    omega = controls[..., 1]
    zero = torch.zeros_like(v)
    return torch.stack([v, zero, omega], dim=-1)


# -- SE(2) tensor conversions ---------------------------------------------------------------
#
# Theseus stores an SE2 as (batch, 4) = [x, y, cos(theta), sin(theta)]. Only the `x_y_theta=`
# constructor kwarg accepts 3 columns, so anything fed through a TheseusLayer input dict must be
# converted first. These two helpers are the chokepoint for that.


def se2_tensor(x_y_theta: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` ``[x, y, theta]`` -> ``(..., 4)`` ``[x, y, cos, sin]``."""
    x, y, theta = x_y_theta[..., 0], x_y_theta[..., 1], x_y_theta[..., 2]
    return torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1)


def xytheta(se2: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` ``[x, y, cos, sin]`` -> ``(..., 3)`` ``[x, y, theta]``."""
    return torch.stack(
        [se2[..., 0], se2[..., 1], torch.atan2(se2[..., 3], se2[..., 2])], dim=-1
    )


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles into ``(-pi, pi]``."""
    return torch.atan2(torch.sin(theta), torch.cos(theta))


# -- kinematics -----------------------------------------------------------------------------


def integrate(pose: th.SE2, controls: torch.Tensor, dt: float | torch.Tensor) -> th.SE2:
    """One exact skate step: ``pose @ Exp(dt * [v, 0, omega])``."""
    delta = _as_scalar(dt, controls) * body_twist(controls)
    return pose.retract(delta)


def rollout(
    start: torch.Tensor,
    controls: torch.Tensor,
    dt: float | torch.Tensor,
    substeps: int = 1,
) -> torch.Tensor:
    """Forward-integrate controls from a start pose.

    Parameters
    ----------
    start:
        ``(B, 3)`` start poses as ``[x, y, theta]``.
    controls:
        ``(B, N, 2)`` control sequence.
    dt:
        Timestep.
    substeps:
        Intermediate samples per control interval. Because exact arc integration composes
        exactly -- ``Exp(dt * xi) == Exp((dt/m) * xi)^m`` -- raising this adds points *along*
        each arc without moving the knots at all. Use it to check clearance between knots,
        where a fast-moving skate can otherwise cut a corner through a thin obstacle unseen.

    Returns
    -------
    ``(B, N * substeps + 1, 3)`` poses as ``[x, y, theta]``, dynamically consistent by
    construction.
    """
    if substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    sub_dt = _as_scalar(dt, controls) / substeps
    pose = th.SE2(x_y_theta=start)
    poses = [pose.tensor]
    for k in range(controls.shape[1]):
        for _ in range(substeps):
            pose = integrate(pose, controls[:, k], sub_dt)
            poses.append(pose.tensor)
    return xytheta(torch.stack(poses, dim=1))


def lateral_slip(poses: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
    """Lateral body velocity implied by consecutive poses.

    ``poses`` is ``(B, N + 1, 3)``; returns ``(B, N)``. A trajectory that truly obeys the skate
    model has this identically zero. Applied to the *optimizer's* poses it measures how far the
    soft dynamics residual let the solution drift off-model.
    """
    b, n1, _ = poses.shape
    flat = poses.reshape(-1, 3)
    a = th.SE2(x_y_theta=flat.reshape(b, n1, 3)[:, :-1].reshape(-1, 3))
    c = th.SE2(x_y_theta=flat.reshape(b, n1, 3)[:, 1:].reshape(-1, 3))
    twist = th.local(a, c) / _as_scalar(dt, flat)
    return twist[:, LATERAL_INDEX].reshape(b, n1 - 1)


def _as_scalar(dt: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if isinstance(dt, torch.Tensor):
        return dt.to(dtype=like.dtype, device=like.device).reshape(-1, 1)[0]
    return torch.as_tensor(dt, dtype=like.dtype, device=like.device)


# -- honesty check --------------------------------------------------------------------------


@dataclass
class TrajectoryDiagnostics:
    """Per-batch-element quality metrics. Every field is a ``(B,)`` array.

    Soft dynamics residuals mean the optimizer's pose sequence is only *approximately*
    reachable. A trajectory can look clean in a plot and still not be executable, so these
    numbers -- especially ``rollout_drift`` -- are the ones that decide whether a plan is real.
    """

    goal_error_xy: np.ndarray
    goal_error_theta: np.ndarray
    optimizer_goal_error_xy: np.ndarray
    rollout_drift: np.ndarray
    max_lateral_slip: np.ndarray
    min_clearance: np.ndarray
    max_speed: np.ndarray
    max_yaw_rate: np.ndarray
    path_length: np.ndarray

    def feasible(self, radius: float, goal_tol: float = 0.25) -> np.ndarray:
        """Boolean ``(B,)``: the *executed* path is collision-free and reaches the goal.

        Both conditions are checked on the rolled-out trajectory rather than the optimizer's
        pose variables, because only the rollout is something the robot can actually drive.
        """
        return (self.min_clearance >= radius) & (self.goal_error_xy <= goal_tol)

    def as_dict(self) -> dict[str, np.ndarray]:
        return asdict(self)

    def compact(self, index: int = 0) -> str:
        """Short one-liner for figure subtitles, where width is scarce."""
        return (
            f"goal {self.goal_error_xy[index]:.3f} m  ·  "
            f"drift {self.rollout_drift[index]:.3f} m  ·  "
            f"clearance {self.min_clearance[index]:.3f} m  ·  "
            f"path {self.path_length[index]:.2f} m"
        )

    def summary(self, index: int = 0) -> str:
        return (
            f"goal xy err {self.goal_error_xy[index]:.4f} m "
            f"(optimizer thought {self.optimizer_goal_error_xy[index]:.4f}) | "
            f"goal th err {self.goal_error_theta[index]:.4f} rad | "
            f"drift {self.rollout_drift[index]:.4f} m | "
            f"slip {self.max_lateral_slip[index]:.2e} m/s | "
            f"clearance {self.min_clearance[index]:.4f} m | "
            f"path {self.path_length[index]:.3f} m"
        )


def evaluate(
    poses: torch.Tensor,
    controls: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    dt: float,
    clearance_fn: Callable[[np.ndarray], np.ndarray],
    substeps: int = 8,
) -> TrajectoryDiagnostics:
    """Score optimized trajectories, re-integrating the controls to check executability.

    ``poses`` ``(B, N+1, 3)``, ``controls`` ``(B, N, 2)``, ``start``/``goal`` ``(B, 3)``.
    ``clearance_fn`` maps ``(..., 2)`` positions to signed distances (use
    :meth:`~skate_planner.world.ObstacleField.analytic_sdf`, so clearance is reported against
    exact geometry rather than the grid the solver optimized on).

    Two deliberate choices here, both about not flattering the result:

    * clearance is measured on the **rolled-out** path, densified by ``substeps``, so it
      reflects where the robot would actually go rather than where the optimizer's knots sit;
    * it uses the **analytic** SDF, so grid interpolation error in the solver's field cannot
      hide a collision.
    """
    rolled = rollout(start, controls, dt)
    dense = rollout(start, controls, dt, substeps=substeps)

    drift = torch.linalg.norm(rolled[..., :2] - poses[..., :2], dim=-1).max(dim=1).values
    # Goal error is measured where the robot actually ends up, not where the optimizer's last
    # pose variable sits. The two differ by exactly the collocation slack, and reporting the
    # latter would flatter every result by that margin.
    goal_xy = torch.linalg.norm(rolled[:, -1, :2] - goal[:, :2], dim=-1)
    goal_th = wrap_angle(rolled[:, -1, 2] - goal[:, 2]).abs()
    optimizer_goal_xy = torch.linalg.norm(poses[:, -1, :2] - goal[:, :2], dim=-1)

    dense_np = dense.detach().cpu().numpy()
    clearance = clearance_fn(dense_np[..., :2]).min(axis=1)

    steps = torch.linalg.norm(rolled[:, 1:, :2] - rolled[:, :-1, :2], dim=-1).sum(dim=1)

    to_np = lambda t: t.detach().cpu().numpy()
    return TrajectoryDiagnostics(
        goal_error_xy=to_np(goal_xy),
        goal_error_theta=to_np(goal_th),
        optimizer_goal_error_xy=to_np(optimizer_goal_xy),
        rollout_drift=to_np(drift),
        max_lateral_slip=to_np(lateral_slip(poses, dt).abs().max(dim=1).values),
        min_clearance=np.asarray(clearance),
        max_speed=to_np(controls[..., 0].abs().max(dim=1).values),
        max_yaw_rate=to_np(controls[..., 1].abs().max(dim=1).values),
        path_length=to_np(steps),
    )
