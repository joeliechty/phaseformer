"""The derived Push-T artifact: split labels, normalization stats and the phase grid.

Everything here is a deterministic function of ``pusht_episodes.npz`` plus a handful of flags
(``--n-grid``, ``--seed``, ``--val-ratio``, ``--max-train-episodes``), so this file is cheap to
regenerate and is never the source of truth. It exists so that a training run reads one
self-describing artifact instead of re-deriving splits and statistics -- and, more importantly,
so the statistics are pinned to a recorded split rather than recomputed per run.

Normalization stats are computed over **training episodes only**. All three modes
(``box``/``minmax``/``gaussian``) are written for every key, so switching is a config change
rather than a re-run; ``meta["normalizer"]["mode"]`` records which one is active.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .constants import WORKSPACE_PX
from .data import PushTEpisodes
from .grid import PushTGrid, resample_to_phase_grid
from .normalize import MODES, NO_BOX_KEYS, Normalizer, normalizer_for, stats_from_data
from .splits import TEST, TRAIN, UNUSED, VAL, episode_split, ratio_split

__all__ = ["PushTPrepared", "NORM_KEYS", "prepare"]

# Key -> how to pull it out of a PushTEpisodes. `agent_pos` is separate from `state` because a
# policy that observes only the pusher wants its own two-dimensional normalizer.
NORM_KEYS = ("state", "action", "agent_pos", "keypoint", "vel_action", "agent_delta")


def _field(ep: PushTEpisodes, key: str) -> torch.Tensor | None:
    if key == "agent_pos":
        return ep.state[:, :2]
    return getattr(ep, key)


@dataclass
class PushTPrepared:
    """Splits, normalization stats and the phase grid for one configuration."""

    split: torch.Tensor  # (E,) int8: TRAIN / VAL / UNUSED / TEST
    grid: PushTGrid
    norm: dict[str, dict[str, Normalizer]]  # key -> mode -> Normalizer
    meta: dict = field(default_factory=dict)

    @property
    def train_ids(self) -> torch.Tensor:
        return torch.nonzero(self.split == TRAIN, as_tuple=False).flatten()

    @property
    def val_ids(self) -> torch.Tensor:
        return torch.nonzero(self.split == VAL, as_tuple=False).flatten()

    @property
    def unused_ids(self) -> torch.Tensor:
        """The episodes Diffusion Policy's convention leaves out of both sets.

        Never trained on, so they are held out in the sense that matters, and there are 112 of
        them against the split's 4 validation episodes -- enough to put a standard error on a
        per-episode metric.
        """
        return torch.nonzero(self.split == UNUSED, as_tuple=False).flatten()

    @property
    def test_ids(self) -> torch.Tensor:
        """The test episodes of a ratio split (empty under DP's convention)."""
        return torch.nonzero(self.split == TEST, as_tuple=False).flatten()

    @property
    def held_out_ids(self) -> torch.Tensor:
        """The episodes to report final scores on: the test set if the split has one, else the
        episodes DP's convention leaves unused, else the validation set."""
        for ids in (self.test_ids, self.unused_ids, self.val_ids):
            if ids.numel():
                return ids
        return self.val_ids

    def normalizer(self, key: str, mode: str | None = None) -> Normalizer:
        """The normalizer for ``key``; ``mode`` defaults to the artifact's active mode."""
        return self.norm[key][mode or self.meta["normalizer"]["mode"]]

    # -- io -----------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "split": self.split.cpu().numpy(),
            **{k: v.cpu().numpy() for k, v in self.grid.__dict__.items()},
        }
        for key, modes in self.norm.items():
            for mode, n in modes.items():
                arrays[f"norm_{key}_{mode}_scale"] = n.scale.cpu().numpy()
                arrays[f"norm_{key}_{mode}_offset"] = n.offset.cpu().numpy()
        np.savez_compressed(path, meta=json.dumps(self.meta), **arrays)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "PushTPrepared":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            arrays = {k: torch.as_tensor(f[k]).to(device) for k in f.files if k != "meta"}
        grid = PushTGrid(**{k: arrays[k] for k in PushTGrid.__dataclass_fields__})
        norm: dict[str, dict[str, Normalizer]] = {}
        for name in arrays:
            if not name.startswith("norm_") or not name.endswith("_scale"):
                continue
            key, mode = name[len("norm_"):-len("_scale")].rsplit("_", 1)
            norm.setdefault(key, {})[mode] = Normalizer(
                scale=arrays[name], offset=arrays[f"norm_{key}_{mode}_offset"]
            )
        return cls(split=arrays["split"], grid=grid, norm=norm, meta=meta)


def prepare(
    ep: PushTEpisodes,
    n_grid: int = 128,
    val_ratio: float = 0.02,
    seed: int = 42,
    max_train_episodes: int | None = 90,
    mode: str = "box",
    device: str | torch.device = "cpu",
    split_mode: str = "dp",
    fractions: tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> PushTPrepared:
    """Draw the split, fit the normalizers on training episodes, and build the phase grid.

    ``split_mode="dp"`` is Diffusion Policy's convention (``val_ratio``, ``max_train_episodes``);
    ``"ratio"`` splits every episode into train / val / test by ``fractions``.
    """
    if mode not in MODES:
        raise ValueError(f"normalizer mode must be one of {MODES}, got {mode!r}")
    if split_mode == "dp":
        split = torch.as_tensor(episode_split(ep.n_episodes, val_ratio, seed, max_train_episodes))
        split_meta = {"mode": "dp", "val_ratio": val_ratio, "seed": seed,
                      "max_train_episodes": max_train_episodes}
    elif split_mode == "ratio":
        split = torch.as_tensor(ratio_split(ep.n_episodes, tuple(fractions), seed))
        split_meta = {"mode": "ratio", "fractions": list(fractions), "seed": seed}
    else:
        raise ValueError(f"split_mode must be 'dp' or 'ratio', got {split_mode!r}")
    # Steps belonging to training episodes -- stats must never see validation data.
    train_steps = torch.isin(
        ep.episode_index, torch.nonzero(split == TRAIN, as_tuple=False).flatten()
    )

    norm: dict[str, dict[str, Normalizer]] = {}
    for key in NORM_KEYS:
        value = _field(ep, key)
        if value is None:  # e.g. keypoint dropped at conversion time
            continue
        stats = stats_from_data(value[train_steps])
        box = None if key in NO_BOX_KEYS else (0.0, WORKSPACE_PX)
        norm[key] = {m: normalizer_for(m, stats, box) for m in MODES}

    grid = resample_to_phase_grid(ep, n_grid=n_grid, device=device)
    meta = {
        "source_meta": ep.meta,
        "n_grid": n_grid,
        "n_episodes": ep.n_episodes,
        "split": {**split_meta,
                  "n_train": int((split == TRAIN).sum()), "n_val": int((split == VAL).sum()),
                  "n_test": int((split == TEST).sum()),
                  "n_unused": int((split == UNUSED).sum())},
        "normalizer": {"mode": mode, "modes": list(MODES),
                       "fitted_on": "training episodes only",
                       "no_box_keys": list(NO_BOX_KEYS),
                       "convention": "normalized = (raw - offset) / scale"},
        "grid": {"theta_convention": "(-pi, pi]",
                 "resample": "uniform in phase; block pose along the SE(2) geodesic",
                 "grid_dt": "seconds per phase step, PER EPISODE -- divide divergence by it"},
    }
    return PushTPrepared(split=split.to(device), grid=grid, norm=norm, meta=meta)
