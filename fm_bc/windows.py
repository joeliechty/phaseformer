"""Observation / action-chunk window indices, in Diffusion Policy's convention.

Two flavours, one convention:

* :func:`window_indices` -- the fixed-horizon case. Every demo has the same length, so one set
  of index arrays serves every row. This is what the skate task uses (see
  :mod:`fm_bc.skate.chunks`).
* :func:`episode_window_indices` -- the variable-length case. Demos are concatenated into one
  flat buffer with per-episode bounds, so each window has to be clipped to *its own* episode.
  This is what Push-T uses (see :mod:`fm_bc.pusht.chunks`).

In both, at control step ``k`` a policy sees the last ``To`` observations and predicts ``Tp``
actions starting at the *oldest* observed step, so the ``Ta`` executed entries are
``chunk[To-1 : To-1+Ta]``. Windows that reach past either end are edge-padded -- the first
sample repeated before the start, the last repeated after the end -- exactly as Diffusion
Policy's ``SequenceSampler`` pads.

This module is deliberately free of Theseus and of any task-specific import, so it can be used
from either task's pipeline.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["window_indices", "episode_window_indices"]


def window_indices(k: np.ndarray | int, n_steps: int, to: int, tp: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Pose indices ``(..., To)`` into ``poses[:, :N+1]`` and control indices ``(..., Tp)`` into
    ``controls[:, :N]`` of the window at step ``k``, edge-padded.

    The asymmetry -- observations clipped to ``n_steps``, actions to ``n_steps - 1`` -- is the
    skate task's ``N+1`` poses against ``N`` controls.
    """
    k = np.asarray(k)[..., None]
    obs = np.clip(k - to + 1 + np.arange(to), 0, n_steps)
    act = np.clip(k - to + 1 + np.arange(tp), 0, n_steps - 1)
    return obs, act


def episode_window_indices(
    starts: torch.Tensor,
    ends: torch.Tensor,
    episode_index: torch.Tensor,
    to: int,
    tp: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Windows for every step of a flat, episode-concatenated buffer.

    Parameters
    ----------
    starts, ends:
        ``(E,)`` inclusive start and **exclusive** end of each episode in the flat buffer.
    episode_index:
        ``(T,)`` the episode each flat step belongs to.
    to, tp:
        Observation steps and prediction horizon.

    Returns
    -------
    ``obs (T, To)`` and ``act (T, Tp)`` index tensors into the flat buffers, one window per
    step, each clipped to the bounds of *its own* episode. No window ever reads a step
    belonging to a different episode: before an episode's start its first sample repeats, after
    its last step its last sample repeats.

    Unlike :func:`window_indices` there is no ``N+1``/``N`` asymmetry -- Push-T stores a state
    and an action at every step -- so both clamp to ``end - 1``.
    """
    if to < 1 or tp < 1:
        raise ValueError(f"need to >= 1 and tp >= 1, got to={to}, tp={tp}")
    device = starts.device
    k = torch.arange(len(episode_index), device=device)[:, None]
    lo = starts[episode_index][:, None]
    hi = (ends[episode_index] - 1)[:, None]
    base = k - to + 1
    obs = (base + torch.arange(to, device=device)).clamp_(min=0)
    act = (base + torch.arange(tp, device=device)).clamp_(min=0)
    return torch.clamp(obs, lo, hi), torch.clamp(act, lo, hi)
