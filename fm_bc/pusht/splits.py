"""Episode-level train / validation splits, reproducing Diffusion Policy's convention.

:func:`val_mask` and :func:`downsample` are ports of ``get_val_mask`` and ``downsample_mask``
from ``diffusion_policy/common/sampler.py``, kept semantically identical so that a run here is
comparable to the published Push-T numbers. Two details matter and are easy to get wrong:

* each function creates its **own** ``default_rng(seed)`` -- they do not share a stream, so
  drawing the validation set does not perturb the training subsample, and
* ``downsample`` draws indices *into the array of training indices*, not into the episode range.

The task config (``diffusion_policy/config/task/pusht_lowdim.yaml``) uses ``seed=42``,
``val_ratio=0.02`` and ``max_train_episodes=90``, which on Push-T's 206 episodes gives 4
validation, 90 training and 112 held out unused. So the headline "Push-T, 200 demos" results
actually train on 90 episodes.

Unlike the skate task there are no homotopy modes here, so
``fm_bc.skate.training.stratified_subset`` does not apply; :func:`subset_episodes` is its
single-mode degenerate case.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TRAIN", "VAL", "UNUSED", "TEST", "val_mask", "downsample", "episode_split",
           "ratio_split", "subset_episodes"]

TRAIN, VAL, UNUSED, TEST = 0, 1, 2, 3


def val_mask(n_episodes: int, val_ratio: float = 0.02, seed: int = 42) -> np.ndarray:
    """``(E,) bool``, True for validation episodes. Port of DP's ``get_val_mask``."""
    mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    mask[np.random.default_rng(seed=seed).choice(n_episodes, size=n_val, replace=False)] = True
    return mask


def downsample(mask: np.ndarray, max_n: int | None, seed: int = 42) -> np.ndarray:
    """Keep at most ``max_n`` of the True entries. Port of DP's ``downsample_mask``."""
    if max_n is None or mask.sum() <= max_n:
        return mask.copy()
    idxs = np.nonzero(mask)[0]
    keep = idxs[np.random.default_rng(seed=seed).choice(len(idxs), size=int(max_n),
                                                        replace=False)]
    out = np.zeros_like(mask)
    out[keep] = True
    return out


def episode_split(
    n_episodes: int,
    val_ratio: float = 0.02,
    seed: int = 42,
    max_train_episodes: int | None = 90,
) -> np.ndarray:
    """``(E,) int8`` of :data:`TRAIN` / :data:`VAL` / :data:`UNUSED`.

    Three labels rather than DP's two booleans: the episodes dropped by ``max_train_episodes``
    are *unused*, which is not the same thing as *validation*. Keeping them distinct lets an
    ``--n-train`` sweep reclaim them without ever touching the held-out set.
    """
    is_val = val_mask(n_episodes, val_ratio, seed)
    is_train = downsample(~is_val, max_train_episodes, seed)
    split = np.full(n_episodes, UNUSED, dtype=np.int8)
    split[is_train] = TRAIN
    split[is_val] = VAL
    return split


def ratio_split(
    n_episodes: int,
    fractions: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
) -> np.ndarray:
    """``(E,) int8`` of :data:`TRAIN` / :data:`VAL` / :data:`TEST` by fractions of all episodes.

    Not DP's convention: every episode is used, and a proper test set is carved out because there
    is no simulator to test against. One ``default_rng(seed)`` permutation; validation and test
    sizes are rounded and training takes the remainder (206 episodes -> 144 / 41 / 21).
    """
    if len(fractions) != 3 or min(fractions) < 0 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must be three non-negative numbers summing to 1, got {fractions}")
    n_val, n_test = round(n_episodes * fractions[1]), round(n_episodes * fractions[2])
    if n_val + n_test >= n_episodes:
        raise ValueError(f"{n_episodes} episodes leave no training set at {fractions}")
    perm = np.random.default_rng(seed=seed).permutation(n_episodes)
    split = np.full(n_episodes, TRAIN, dtype=np.int8)
    split[perm[:n_val]] = VAL
    split[perm[n_val:n_val + n_test]] = TEST
    return split


def subset_episodes(train_ids: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` training episode ids drawn uniformly without replacement, sorted.

    Push-T has no modes, so this is what ``fm_bc.skate.training.stratified_subset`` degenerates to
    when every demonstration carries the same label.
    """
    if n >= len(train_ids):
        return np.sort(train_ids)
    return np.sort(rng.choice(train_ids, size=int(n), replace=False))
