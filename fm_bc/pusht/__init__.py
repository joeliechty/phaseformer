"""Push-T manipulation (Chi et al. 2023): demonstration loading, processing and windowing.

Push-T is the repo's second SE(2) task. None of the Lie-group machinery changes -- what changes
is that the demonstrations are *human* rather than planner output, that episodes vary in length,
and that actions are absolute pixel targets rather than body twists.

Two artifacts, because there are two consumers:

``data/pusht/pusht_episodes.npz`` (:class:`PushTEpisodes`)
    The canonical conversion of the source zarr: variable-length episodes concatenated into one
    flat buffer with per-episode bounds. Feeds the action-chunk baselines through
    :func:`fm_bc.pusht.chunks.make_windows`.

``data/pusht/pusht_prepared.npz`` (:class:`PushTPrepared`)
    Derived and regenerable: split labels, normalization statistics fitted on training episodes,
    and the rectangular phase grid that :mod:`fm_bc.divergence` needs.

A third consumer reads both: :mod:`fm_bc.pusht.progress`, a task-progress CVAE whose latent splits
into a monotone scalar phase and a Gaussian style code. It is trained *without* ever seeing
:attr:`PushTEpisodes.phase` -- that field is the held-out yardstick -- and only its encoder ships.

Typical use::

    from fm_bc.pusht import PushTEpisodes, PushTPrepared

    ep = PushTEpisodes.load("data/pusht/pusht_episodes.npz")
    prep = PushTPrepared.load("data/pusht/pusht_prepared.npz")
    poses = prep.grid.block_poses()                      # (E, N+1, 3) SE(2)

Submodules import lazily, so ``fm_bc.pusht`` needs torch but not Theseus.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "ACTION_BOX": ".constants",
    "DT": ".constants",
    "FPS": ".constants",
    "GOAL_POSE": ".constants",
    "ROT_SCALE_PX_PER_RAD": ".constants",
    "STATE_KEYS": ".constants",
    "SUCCESS_COVERAGE": ".constants",
    "WORKSPACE_PX": ".constants",
    "PushTEpisodes": ".data",
    "from_zarr": ".convert",
    "PushTGrid": ".grid",
    "resample_to_phase_grid": ".grid",
    "Normalizer": ".normalize",
    "PushTPrepared": ".prepared",
    "prepare": ".prepared",
    "ProgressCVAE": ".progress",
    "progress_rows": ".progress",
    "load_progress_model": ".progress",
    "ProgressConfig": ".progress_training",
    "train_progress": ".progress_training",
    "episode_split": ".splits",
    "subset_episodes": ".splits",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_EXPORTS[name], __name__)
    globals()[name] = value = getattr(module, name)
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
