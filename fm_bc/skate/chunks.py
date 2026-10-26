"""Observation / action-chunk windows for the generative baselines (Diffusion Policy, MIP,
MeanFlow), in Diffusion Policy's convention.

At control step ``k`` a policy sees the last ``To`` poses ``T_{k-To+1} .. T_k`` and predicts the
``Tp`` controls ``u_{k-To+1} .. u_{k-To+Tp}``: the chunk starts at the *oldest* observed step, so
its first ``To - 1`` entries are the past and the ``Ta`` executed ones are
``chunk[To-1 : To-1+Ta] = u_k .. u_{k+Ta-1}``. Windows that reach past either end of a demo are
edge-padded (the first pose / control repeated before the start, the last control after the
end), exactly as Diffusion Policy's ``SequenceSampler`` pads.

Actions are normalized to ``[-1, 1]`` by the robot's control box (:class:`SkateLimits`), not by
data min/max: the scale is then the same for every subset and seed, and a sample clipped to
``[-1, 1]`` is exactly a command the rollout's saturation would leave untouched.
"""

from __future__ import annotations

import numpy as np
import torch

from skate_planner.planner import SkateLimits

from ..windows import window_indices
from .dataset import TrajectoryDataset
from .scenarios import BOUNDS

__all__ = ["OBS_DIM", "ACT_DIM", "embed_poses", "window_indices", "make_windows",
           "normalize_actions", "unnormalize_actions"]

OBS_DIM = 4  # [(x - cx)/hx, (y - cy)/hy, cos theta, sin theta]
ACT_DIM = 2  # [v, omega]


def embed_poses(poses: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` poses -> ``(..., 4)``, the same embedding as ``FlowMLP.embed`` minus phase."""
    x0, y0, x1, y1 = BOUNDS
    center = poses.new_tensor([(x0 + x1) / 2, (y0 + y1) / 2])
    scale = poses.new_tensor([(x1 - x0) / 2, (y1 - y0) / 2])
    th = poses[..., 2:3]
    return torch.cat([(poses[..., :2] - center) / scale, torch.cos(th), torch.sin(th)], dim=-1)


def _box(limits: SkateLimits, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (like.new_tensor([limits.v_min, -limits.omega_max]),
            like.new_tensor([limits.v_max, limits.omega_max]))


def normalize_actions(controls: torch.Tensor, limits: SkateLimits) -> torch.Tensor:
    lo, hi = _box(limits, controls)
    return 2 * (controls - lo) / (hi - lo) - 1


def unnormalize_actions(actions: torch.Tensor, limits: SkateLimits) -> torch.Tensor:
    lo, hi = _box(limits, actions)
    return (actions + 1) / 2 * (hi - lo) + lo


def make_windows(ds: TrajectoryDataset, to: int, tp: int, device: str
                 ) -> dict[str, torch.Tensor]:
    """Every ``(demo, k)`` window, ``k = 0 .. N-1``: ``obs`` poses ``(M N, To, 3)`` and
    ``act`` physical controls ``(M N, Tp, 2)``."""
    m, n = ds.controls.shape[:2]
    obs_idx, act_idx = window_indices(np.arange(n), n, to, tp)  # (N, To), (N, Tp)
    obs = ds.poses[:, obs_idx]  # (M, N, To, 3)
    act = ds.controls[:, act_idx]  # (M, N, Tp, 2)
    to_t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    return {"obs": to_t(obs.reshape(m * n, to, 3)), "act": to_t(act.reshape(m * n, tp, 2))}
