"""Resample variable-length episodes onto a shared phase grid.

The divergence and score estimators in :mod:`fm_bc.divergence` take neighbours *at the same
phase*, so they need a rectangular ``(M, N+1, D)`` array. The skate task gets that for free --
every demo is planned to the same fixed horizon -- but Push-T episodes range from ~30 to 246
steps, so the grid has to be built.

**Uniform resampling, geodesic in SE(2).** Each episode's own normalized time ``k / (L_e - 1)``
is resampled onto ``linspace(0, 1, N+1)``. Positions interpolate linearly; the block pose
interpolates along the SE(2) geodesic with :func:`~fm_bc.se2.se2_log` and
:func:`~fm_bc.se2.se2_exp_retract`, so an interpolated pose is genuinely on the screw motion
between its two bracketing poses. (The reference SE(3) implementation attempted the same thing
with a log/exp pair that does not round-trip -- mean position error 3.02 -- so its interpolated
poses were frequently not even between the poses they bracketed. The pair used here round-trips
to float64 precision, and ``tests/test_se2.py`` checks it against Theseus.) Wrapping is handled
by construction: ``se2_log`` wraps the heading difference into ``(-pi, pi]``, so no ``unwrap``
pass is needed and the grid's ``block_theta`` comes out in ``(-pi, pi]`` rather than the source's
``[0, 2pi)``.

**Why not scatter-and-gap-fill.** The alternative -- allocate ``N+1`` bins, scatter each real
step into its nearest bin, then interpolate across the gaps -- preserves original samples
exactly and records which entries were fabricated. But the mask it produces is then ignored by
the kNN and KDE estimators, so a 60-step episode contributes ~75% fabricated points to every
neighbour query while a 246-step episode contributes none. That artifact scales with how short
the episode is, which correlates with task difficulty, which is the very signal the divergence
target is meant to carry. Uniform resampling fabricates *more* points but distributes the
artifact evenly across episodes, has exact endpoints and monotone phase by construction, and
needs no mask. The cost is that interpolation is a low-pass, so the spread at each phase is
slightly smoothed -- uniformly, and below the kNN estimator's own attenuation floor.

**Per-episode ``dt``.** After gridding, one phase step is ``dt_e = (L_e - 1) * DT / N`` seconds,
which differs across episodes by up to 4x. :func:`~fm_bc.divergence.divergence_knn` takes a
single scalar ``dt`` and is exactly linear in ``1/dt``, so passing a global value would scale a
half-length episode's divergence by exactly 0.5. Divide by :attr:`PushTGrid.grid_dt` instead::

    div = divergence_knn(grid.block_poses(), dt=1.0, ...) / grid.grid_dt[:, None]  # 1/s
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..se2 import se2_exp_retract, se2_log
from .constants import DT
from .data import PushTEpisodes

__all__ = ["PushTGrid", "resample_to_phase_grid"]


@dataclass
class PushTGrid:
    """Every episode resampled onto the shared phase grid ``linspace(0, 1, N+1)``."""

    grid_phase: torch.Tensor  # (N+1,)
    grid_state: torch.Tensor  # (E, N+1, 5) block_theta in (-pi, pi]
    grid_action: torch.Tensor  # (E, N+1, 2)
    grid_agent_delta: torch.Tensor  # (E, N+1, 2)
    grid_dt: torch.Tensor  # (E,) seconds per phase step -- see the module docstring
    lengths: torch.Tensor  # (E,) original episode lengths

    @property
    def n_grid(self) -> int:
        return int(self.grid_phase.shape[0])

    def agent_positions(self) -> torch.Tensor:
        """``(E, N+1, 2)`` -- feed to ``divergence_knn(metric="xy")`` (D = 2)."""
        return self.grid_state[..., :2]

    def block_poses(self) -> torch.Tensor:
        """``(E, N+1, 3)`` SE(2) -- feed to ``divergence_knn(metric="se2")`` (D = 3)."""
        return self.grid_state[..., 2:5]


def resample_to_phase_grid(
    ep: PushTEpisodes,
    n_grid: int = 128,
    device: str | torch.device = "cpu",
    dt: float = DT,
) -> PushTGrid:
    """Resample every episode of ``ep`` onto ``n_grid`` shared phase points.

    ``n_grid`` is ``N + 1``: the number of phase points, not the number of intervals. The
    default sits near the mean episode length (~124.5), so a typical episode is resampled
    roughly 1:1, the 246-step tail is downsampled ~2:1 and short episodes are upsampled.
    """
    if n_grid < 2:
        raise ValueError(f"n_grid must be at least 2, got {n_grid}")
    ep = ep.to(device)
    starts, lengths = ep.episode_starts, ep.lengths
    if bool((lengths < 2).any()):
        raise ValueError("every episode needs at least 2 steps to define a phase grid")

    target = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=torch.float64)
    last = (lengths - 1).to(torch.float64)[:, None]  # (E, 1)

    # Source phase is uniform in k, so the bracketing pair is available in closed form.
    pos = target[None, :] * last  # (E, N+1) fractional source index
    lo = pos.floor().clamp(max=last)
    alpha = (pos - lo).clamp(0.0, 1.0)  # (E, N+1)
    lo_i = lo.to(torch.int64)
    hi_i = (lo_i + 1).clamp(max=(lengths - 1)[:, None])
    flat_lo = starts[:, None] + lo_i  # (E, N+1) indices into the flat buffer
    flat_hi = starts[:, None] + hi_i

    def lerp(x: torch.Tensor) -> torch.Tensor:
        a, b = x[flat_lo], x[flat_hi]
        return (a + alpha[..., None].to(a) * (b - a)).to(torch.float32)

    # Block pose travels the SE(2) geodesic rather than three independent lines. float64
    # throughout: se2.py's small-angle Taylor cutoffs are chosen against float64 resolution.
    pose = ep.state[:, 2:5].to(torch.float64)
    pose_lo, pose_hi = pose[flat_lo], pose[flat_hi]
    grid_pose = se2_exp_retract(pose_lo, alpha[..., None] * se2_log(pose_lo, pose_hi))

    grid_state = torch.cat([lerp(ep.state[:, :2]), grid_pose.to(torch.float32)], dim=-1)
    return PushTGrid(
        grid_phase=target.to(torch.float32),
        grid_state=grid_state,
        grid_action=lerp(ep.action),
        grid_agent_delta=lerp(ep.agent_delta),
        grid_dt=((lengths - 1).to(torch.float32) * dt / (n_grid - 1)),
        lengths=lengths.clone(),
    )
