"""Trajectory seeds for the optimizer.

Trajectory optimization is local: the seed decides which homotopy class the solution lands in.
A straight line through an obstacle has no gradient telling it which way to go around, so the
multi-start seeds here deliberately commit to going *left* or *right* of the straight path, and
let the batched solver sort out which commitment was better.
"""

from __future__ import annotations

import numpy as np
import theseus as th
import torch

from .skate_model import LATERAL_INDEX, se2_tensor, wrap_angle
from .world import ObstacleField

__all__ = ["controls_from_poses", "interpolate", "multistart_seeds"]


def controls_from_poses(poses: torch.Tensor, dt: float) -> torch.Tensor:
    """Project a pose sequence onto skate controls.

    ``poses`` is ``(B, N+1, 3)``; returns ``(B, N, 2)`` as ``[v, omega]``.

    The relative body twist ``Log(T_k^-1 T_{k+1}) / dt`` is ``[v, v_lat, omega]``; keeping the
    first and last components and discarding the lateral one is the natural projection onto the
    skate model. For a seed that is already skate-feasible the discarded part is zero.
    """
    b, n1, _ = poses.shape
    a = th.SE2(x_y_theta=poses[:, :-1].reshape(-1, 3))
    c = th.SE2(x_y_theta=poses[:, 1:].reshape(-1, 3))
    twist = (th.local(a, c) / dt).reshape(b, n1 - 1, 3)
    keep = [i for i in range(3) if i != LATERAL_INDEX]
    return twist[..., keep].contiguous()


def _headings_from_path(xy: torch.Tensor, start_th: torch.Tensor, goal_th: torch.Tensor) -> torch.Tensor:
    """Heading at each knot: tangent to the path, with the given endpoint headings kept.

    Pointing the body along the direction of travel matters more here than it would for a
    holonomic robot -- a skate can only move along its heading, so a seed whose heading fights
    its own path direction starts out badly infeasible.
    """
    deltas = xy[:, 1:] - xy[:, :-1]
    tangent = torch.atan2(deltas[..., 1], deltas[..., 0])
    # Interior knots average the incoming and outgoing tangents (via unit vectors, so the
    # average is well behaved across the +/-pi branch cut).
    inc, out = tangent[:, :-1], tangent[:, 1:]
    avg = torch.atan2(
        torch.sin(inc) + torch.sin(out), torch.cos(inc) + torch.cos(out)
    )
    return torch.cat([start_th[:, None], avg, goal_th[:, None]], dim=1)


def _quadratic_bspline(
    start: torch.Tensor, vias: torch.Tensor, goal: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """Clamped quadratic B-spline on the control polygon ``start, vias..., goal``.

    ``start``/``goal`` are ``(B, 2)``, ``vias`` ``(B, K, 2)``, ``s`` ``(T,)`` in ``[0, 1]``;
    returns ``(B, T, 2)``. The curve leaves ``start`` heading for the first via, arrives at
    ``goal`` from the last, and in between is tangent to every polygon edge at its midpoint --
    ``K`` quadratic Bezier pieces joined with matching tangents. With one via it is exactly the
    quadratic Bezier ``(start, via, goal)``.
    """
    k = vias.shape[1]
    ctrl = torch.cat([start[:, None], vias, goal[:, None]], dim=1)  # (B, K+2, 2)
    mid = 0.5 * (ctrl[:, 1:-2] + ctrl[:, 2:-1])  # midpoints of the edges between vias
    first = torch.cat([ctrl[:, :1], mid], dim=1)  # each piece's start, (B, K, 2)
    last = torch.cat([mid, ctrl[:, -1:]], dim=1)  # and end
    u = s * k
    j = torch.clamp(u.floor().long(), max=k - 1)  # piece index per sample
    t = (u - j)[None, :, None]
    return (1 - t) ** 2 * first[:, j] + 2 * (1 - t) * t * vias[:, j] + t**2 * last[:, j]


def interpolate(
    start: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    dt: float,
    via: torch.Tensor | None = None,
    goal_heading: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seed poses and controls, optionally bending through via points.

    ``start``/``goal`` are ``(B, 3)``; ``via`` is ``(B, 2)``, a chain of waypoints
    ``(B, K, 2)``, or ``None``. Returns ``(poses (B, N+1, 3), controls (B, N, 2))``.

    ``goal_heading=False`` is for position-only goals: the last knot keeps the path's own
    arrival tangent instead of snapping to ``goal[:, 2]``. Forcing an arbitrary heading onto
    it spins the seed's final control (up to ~pi / dt when the path arrives from the other
    side) and can trap the solve.
    """
    b = start.shape[0]
    device, dtype = start.device, start.dtype
    s = torch.linspace(0.0, 1.0, horizon + 1, device=device, dtype=dtype)[None, :, None]

    if via is None:
        xy = start[:, None, :2] + s * (goal[:, None, :2] - start[:, None, :2])
    else:
        # Smooth through the via points rather than cornering at them, so the optimizer does
        # not spend iterations rounding off kinks.
        xy = _quadratic_bspline(start[:, :2], via if via.dim() == 3 else via[:, None], goal[:, :2],
                                s[0, :, 0])

    theta = _headings_from_path(xy, start[:, 2], goal[:, 2])
    if not goal_heading:
        arrive = xy[:, -1] - xy[:, -2]
        theta = torch.cat([theta[:, :-1], torch.atan2(arrive[:, 1], arrive[:, 0])[:, None]], dim=1)
    poses = torch.cat([xy, theta[..., None]], dim=-1)
    poses[:, 0] = start
    if goal_heading:
        poses[:, -1] = goal
    return poses, controls_from_poses(poses, dt)


def multistart_seeds(
    start: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    dt: float,
    n_starts: int,
    world: ObstacleField | None = None,
    rng: np.random.Generator | None = None,
    max_offset: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``n_starts`` distinct seeds, returned batched as ``(B, N+1, 3)`` / ``(B, N, 2)``.

    Seed 0 is the straight line. The rest fan out perpendicular to it, alternating sides, with
    any remainder drawn from random free space in ``world``.
    """
    start = start.reshape(1, 3)
    goal = goal.reshape(1, 3)
    device, dtype = start.device, start.dtype

    direction = goal[0, :2] - start[0, :2]
    normal = torch.stack([-direction[1], direction[0]])
    normal = normal / (torch.linalg.norm(normal) + 1e-12)
    midpoint = 0.5 * (start[0, :2] + goal[0, :2])

    vias: list[torch.Tensor | None] = [None]
    n_offsets = n_starts - 1
    if world is not None:
        n_offsets = max(0, n_starts - 1 - max(0, (n_starts - 1) // 3))

    for k in range(n_offsets):
        # Alternate sides, growing the offset: +d1, -d1, +d2, -d2, ...
        step = k // 2 + 1
        sign = 1.0 if k % 2 == 0 else -1.0
        magnitude = max_offset * step / (n_offsets // 2 + 1)
        vias.append((midpoint + sign * magnitude * normal)[None, :])

    n_random = n_starts - len(vias)
    if n_random > 0:
        if world is None:
            raise ValueError("Random seeds requested but no world was supplied.")
        rng = rng or np.random.default_rng()
        samples = world.sample_free(n_random, clearance=0.0, rng=rng)
        for point in samples:
            vias.append(torch.as_tensor(point, dtype=dtype, device=device)[None, :])

    poses, controls = [], []
    for via in vias[:n_starts]:
        p, c = interpolate(start, goal, horizon, dt, via=via)
        poses.append(p)
        controls.append(c)
    return torch.cat(poses, dim=0), torch.cat(controls, dim=0)
