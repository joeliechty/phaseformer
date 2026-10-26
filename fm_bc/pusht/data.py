"""The canonical Push-T demonstration container: variable-length episodes in one flat buffer.

Unlike the skate task's :class:`~fm_bc.skate.dataset.TrajectoryDataset`, whose demos all share a
fixed horizon, Push-T episodes vary in length (206 episodes, 25 650 steps, mean ~124.5, max 246).
They are stored **concatenated along time with per-episode bounds**, mirroring the source zarr,
rather than padded into a rectangle:

* padding to the longest episode would waste half the buffer (mean/max is ~2x), and
* more importantly, a padded array forces a validity mask that every consumer has to remember
  to honour. Concatenated-plus-bounds makes it impossible to read a fabricated sample by
  accident.

The rectangular, phase-gridded view that the divergence and score estimators need is a separate
artifact -- see :mod:`fm_bc.pusht.grid`.

Arrays are torch tensors; ``save``/``load`` go through ``.npz`` with a JSON ``meta`` string, the
same idiom as :meth:`fm_bc.skate.dataset.TrajectoryDataset.save`, so the artifact stays portable and
``allow_pickle=False``-safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

__all__ = ["PushTEpisodes"]

# Arrays that are stored but fully determined by `episode_ends`. `load` recomputes and checks
# them, so the redundancy is self-checking rather than a second source of truth.
_DERIVED = ("episode_starts", "episode_index", "step_index", "phase")


@dataclass
class PushTEpisodes:
    """Push-T demonstrations, ``T`` steps across ``E`` episodes.

    ``state`` is ``[agent_x, agent_y, block_x, block_y, block_theta]`` in pixels and radians. The
    agent (the pusher) is a circle and has **no** orientation; SE(2) enters only through the
    block's pose, ``state[:, 2:5]``.
    """

    state: torch.Tensor  # (T, 5)
    action: torch.Tensor  # (T, 2) absolute agent target, pixels
    episode_ends: torch.Tensor  # (E,) exclusive one-past-last, as in the zarr
    vel_action: torch.Tensor  # (T, 2) action[t] - agent_pos[t], commanded displacement
    agent_delta: torch.Tensor  # (T, 2) agent_pos[t+1] - agent_pos[t], achieved; edge-held
    episode_starts: torch.Tensor  # (E,)
    episode_index: torch.Tensor  # (T,)
    step_index: torch.Tensor  # (T,)
    phase: torch.Tensor  # (T,) step_index / (L_e - 1), in [0, 1]
    keypoint: torch.Tensor | None = None  # (T, 9, 2)
    n_contacts: torch.Tensor | None = None  # (T, 1)
    meta: dict = field(default_factory=dict)

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        state: torch.Tensor,
        action: torch.Tensor,
        episode_ends: torch.Tensor,
        keypoint: torch.Tensor | None = None,
        n_contacts: torch.Tensor | None = None,
        meta: dict | None = None,
    ) -> "PushTEpisodes":
        """Build from the three required arrays, deriving everything else."""
        state = torch.as_tensor(state, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        episode_ends = torch.as_tensor(episode_ends, dtype=torch.int64)
        starts, index, step, phase = derive_indices(episode_ends, len(state))
        agent_pos = state[:, :2]
        return cls(
            state=state,
            action=action,
            episode_ends=episode_ends,
            vel_action=(action - agent_pos).to(torch.float32),
            agent_delta=_agent_delta(agent_pos, starts, episode_ends),
            episode_starts=starts,
            episode_index=index,
            step_index=step,
            phase=phase,
            keypoint=None if keypoint is None else torch.as_tensor(keypoint, dtype=torch.float32),
            n_contacts=(None if n_contacts is None
                        else torch.as_tensor(n_contacts, dtype=torch.float32)),
            meta=dict(meta or {}),
        )

    # -- views --------------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of steps, not episodes -- use :attr:`n_episodes` for that."""
        return int(self.state.shape[0])

    @property
    def n_episodes(self) -> int:
        return int(self.episode_ends.shape[0])

    @property
    def lengths(self) -> torch.Tensor:
        """``(E,)`` episode lengths."""
        return self.episode_ends - self.episode_starts

    def episode_slice(self, e: int) -> slice:
        """The flat-buffer slice of episode ``e``."""
        return slice(int(self.episode_starts[e]), int(self.episode_ends[e]))

    def to(self, device: str | torch.device) -> "PushTEpisodes":
        """Move every tensor to ``device`` (a new container; tensors are not copied on CPU)."""
        moved = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in self.__dict__.items()
        }
        return PushTEpisodes(**moved)

    # -- io -----------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        arrays = {
            k: v.cpu().numpy() for k, v in self.__dict__.items()
            if isinstance(v, torch.Tensor)
        }
        np.savez_compressed(path, meta=json.dumps(self.meta), **arrays)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "PushTEpisodes":
        """Load and re-derive the index fields, raising if the stored ones disagree."""
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            arrays = {k: torch.as_tensor(f[k]) for k in f.files if k != "meta"}
        ds = cls(meta=meta, **arrays).to(device)
        ds.check()
        return ds

    # -- invariants ---------------------------------------------------------------------

    def check(self) -> None:
        """Re-derive the index fields from ``episode_ends`` and assert the stored ones match.

        Cheap, and it turns any future off-by-one in the converter into a loud failure at load
        time rather than a silently misaligned training set.
        """
        n = len(self)
        ends = self.episode_ends
        if ends.ndim != 1 or ends.numel() == 0:
            raise ValueError(f"episode_ends must be a non-empty 1-D tensor, got {tuple(ends.shape)}")
        if not bool((ends[1:] > ends[:-1]).all()):
            raise ValueError("episode_ends must be strictly increasing")
        if int(ends[-1]) != n:
            raise ValueError(f"episode_ends[-1] is {int(ends[-1])}, expected {n} steps")
        starts, index, step, phase = derive_indices(ends, n)
        for name, want in zip(_DERIVED, (starts, index, step, phase)):
            got = getattr(self, name)
            if got.shape != want.shape or not torch.allclose(got.to(want.dtype), want, atol=1e-6):
                raise ValueError(f"stored {name!r} disagrees with episode_ends")
        for name in ("state", "action", "vel_action", "agent_delta"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")


def derive_indices(
    episode_ends: torch.Tensor, n_steps: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(episode_starts, episode_index, step_index, phase)`` implied by ``episode_ends``.

    ``phase`` is normalized time, ``step_index / (L_e - 1)`` in ``[0, 1]`` -- the same definition
    the reference SE(3) pipeline uses, and 0 for a degenerate single-step episode.
    """
    device = episode_ends.device
    starts = torch.cat([episode_ends.new_zeros(1), episode_ends[:-1]])
    lengths = episode_ends - starts
    index = torch.repeat_interleave(
        torch.arange(len(episode_ends), device=device), lengths
    )
    step = torch.arange(n_steps, device=device) - starts[index]
    denom = (lengths[index] - 1).clamp(min=1).to(torch.float32)
    phase = torch.where(lengths[index] > 1, step / denom, torch.zeros_like(denom))
    return starts, index, step, phase.to(torch.float32)


def _agent_delta(
    agent_pos: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    """``agent_pos[t+1] - agent_pos[t]``, with each episode's final step **edge-held**.

    The last step of an episode has no ``t+1``. Holding the previous delta rather than writing a
    zero matters: a zero reads as a full stop, which a divergence estimator would take at face
    value. Single-step episodes get a genuine zero because there is nothing else to hold.
    """
    delta = torch.zeros_like(agent_pos)
    delta[:-1] = agent_pos[1:] - agent_pos[:-1]
    last = ends - 1
    prev = (last - 1).clamp(min=0)
    # Where the episode has >= 2 steps, hold; otherwise leave the zero.
    multi = (ends - starts) > 1
    delta[last[multi]] = delta[prev[multi]]
    delta[last[~multi]] = 0.0
    return delta
