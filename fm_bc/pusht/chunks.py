"""Observation / action-chunk windows for Push-T, in Diffusion Policy's convention.

The same interface as :mod:`fm_bc.skate.chunks` -- the last ``To`` observations in, a chunk of
``Tp`` actions out, ``Ta`` of them executed before re-planning -- but over a flat,
episode-concatenated buffer, so every window is clipped to its own episode
(:func:`~fm_bc.windows.episode_window_indices`).

Two things differ from the skate task and are worth stating plainly:

* **Actions are absolute target positions in pixels**, not body twists. The skate task could
  normalize actions by the robot's control box because its actions *were* the velocity field;
  here the analogous box is the environment's action space, ``[0, 512]^2``, which is the
  ``box`` mode of :mod:`fm_bc.pusht.normalize`.
* **The pusher has no orientation.** It is a circle, so the agent contributes two translational
  dimensions and SE(2) enters only through the block's pose.

The default observation is 6-dimensional: normalized agent ``xy``, normalized block ``xy``, and
``cos``/``sin`` of the block heading. The trigonometric pair mirrors
:func:`fm_bc.skate.chunks.embed_poses` and exists because the heading wraps -- a network should not
have to learn that ``0`` and ``2*pi`` are the same place.
"""

from __future__ import annotations

import torch

from ..windows import episode_window_indices
from .constants import TEE_POINTS
from .data import PushTEpisodes
from .normalize import Normalizer

__all__ = ["OBS_DIM", "ACT_DIM", "embed_state", "make_windows", "tee_points"]

OBS_DIM = 6  # [agent_x_n, agent_y_n, block_x_n, block_y_n, cos theta, sin theta]
ACT_DIM = 2  # [target_x, target_y]


def embed_state(state: torch.Tensor, pos: Normalizer) -> torch.Tensor:
    """``(..., 5)`` raw state -> ``(..., 6)`` network input.

    ``pos`` normalizes a 2-vector of pixel coordinates; agent and block share it because they
    live in the same workspace box.
    """
    theta = state[..., 4:5]
    return torch.cat(
        [pos.apply(state[..., :2]), pos.apply(state[..., 2:4]),
         torch.cos(theta), torch.sin(theta)],
        dim=-1,
    )


def tee_points(state: torch.Tensor, pos: Normalizer,
               points: torch.Tensor | None = None) -> torch.Tensor:
    """``(..., 5)`` raw state -> ``(..., P, 2)`` normalized positions of points on the T block.

    ``world = block_xy + R(theta) @ local`` for each of ``points`` (default
    :data:`~fm_bc.pusht.constants.TEE_POINTS`, ``P = 10``), given in the body-origin frame the
    dataset's block pose is in.

    **This adds no information to** :func:`embed_state`: every point is a fixed function of the
    same ``(block_x, block_y, theta)``. It is a re-encoding -- the block's pose expressed as
    positions of parts of it rather than as a centre and a heading -- for a network that reads
    its observation as a set of tokens and can then attend to one part of the block at a time.

    ``pos`` is the same 2-D position normalizer :func:`embed_state` puts the agent and block
    through, so the outputs are in comparable units. The rotation happens in pixels and the
    whole ``(..., P, 2)`` is normalized afterwards, which is what keeps this exact under an
    anisotropic (per-dimension ``minmax``) normalizer. Note the stem tip sits up to 120 px from
    the body origin, so a normalized point can reach ~±1.6 where the agent's own range is ±1;
    nothing clips it, and block ``xy`` already leaves ``[-1, 1]`` on occasion.
    """
    if points is None:
        points = torch.as_tensor(TEE_POINTS, dtype=state.dtype, device=state.device)
    theta = state[..., 4:5]
    cos, sin = torch.cos(theta), torch.sin(theta)  # (..., 1)
    lx, ly = points[:, 0], points[:, 1]  # (P,)
    x = state[..., 2:3] + cos * lx - sin * ly  # (..., P)
    y = state[..., 3:4] + sin * lx + cos * ly
    return pos.apply(torch.stack([x, y], dim=-1))


def make_windows(
    ep: PushTEpisodes,
    to: int = 2,
    tp: int = 16,
    device: str | torch.device = "cpu",
    episode_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Every ``(episode, step)`` window, in physical units.

    Returns ``obs`` ``(T', To, 5)`` raw states and ``act`` ``(T', Tp, 2)`` raw absolute actions,
    plus ``agent_delta`` ``(T', Tp, 2)`` for the velocity-style action head and the flat
    ``obs_idx`` / ``act_idx`` used to gather them (so a lazy ``Dataset`` can reuse the same
    indices instead of materializing). Normalization is left to the policy, exactly as
    :func:`fm_bc.skate.chunks.make_windows` returns physical controls.

    ``episode_ids`` restricts the windows to those episodes -- this is how a train/val split or
    an ``--n-train`` sweep is applied.
    """
    ep = ep.to(device)
    obs_idx, act_idx = episode_window_indices(
        ep.episode_starts, ep.episode_ends, ep.episode_index, to, tp
    )
    if episode_ids is not None:
        keep = torch.isin(ep.episode_index, episode_ids.to(device))
        obs_idx, act_idx = obs_idx[keep], act_idx[keep]
    return {
        "obs": ep.state[obs_idx],
        "act": ep.action[act_idx],
        "agent_delta": ep.agent_delta[act_idx],
        "obs_idx": obs_idx,
        "act_idx": act_idx,
    }
