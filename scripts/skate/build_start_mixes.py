"""Assemble the start-noise mixtures: p% of the demos with noisy starts, the rest the two fixed
trajectories.

    python scripts/skate/build_start_mixes.py

For each ``p`` in ``fm_bc.skate.scenarios.MIX_PERCENTS``, ``dual_path_mix{p:02d}.npz`` takes
``round(p% of 1024)`` demos from ``dual_path_noisy_start`` (C's start noise, no via jitter, the
fixed goal pose) and fills the rest with ``dual_path_fixed`` (one start pose, the left and right
trajectory repeated), half of each per mode. The first demos of each mode are taken, so a larger
mixture contains every noisy demo of a smaller one. Built for the training set and the held-out
set alike (``data/skate_deterministic`` and its ``test/``), with the divergence and score
recomputed on the mixture.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate import TrajectoryDataset, add_div_and_score  # noqa: E402
from fm_bc.skate.scenarios import MIX_PERCENTS  # noqa: E402

N = 1024
ARRAYS = ("poses", "controls", "twists", "mode", "starts", "goals")


def take(ds: TrajectoryDataset, per_mode: int) -> dict[str, np.ndarray]:
    idx = np.concatenate([np.flatnonzero(ds.mode == m)[:per_mode] for m in (0, 1)])
    return {k: getattr(ds, k)[idx] for k in ARRAYS}


def noisy_count(pct: int) -> int:
    """Noisy-start demos in the ``pct`` mixture: even, so both modes get the same number."""
    return 2 * round(pct / 100 * N / 2)


def mix(fixed: TrajectoryDataset, noisy: TrajectoryDataset, pct: int) -> TrajectoryDataset:
    n_noisy = noisy_count(pct)
    a, b = take(noisy, n_noisy // 2), take(fixed, (N - n_noisy) // 2)
    arrays = {k: np.concatenate([a[k], b[k]]) for k in ARRAYS}
    order = np.argsort(arrays["mode"], kind="stable")  # modes contiguous, as generated sets are
    return replace(
        noisy, scenario=f"dual_path_mix{pct:02d}", **{k: v[order] for k, v in arrays.items()},
        stats={"n": N, "noisy_start_demos": int(n_noisy), "fixed_demos": int(N - n_noisy),
               "noisy_fraction": n_noisy / N},
        divergence=None, divergence_xy=None, score=None, div_config={},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/skate_deterministic"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for d in (args.data, args.data / "test"):
        fixed = TrajectoryDataset.load(d / "dual_path_fixed.npz")
        noisy = TrajectoryDataset.load(d / "dual_path_noisy_start.npz")
        for pct in MIX_PERCENTS:
            ds = add_div_and_score(mix(fixed, noisy, pct), device=args.device)
            ds.save(d / f"{ds.scenario}.npz")
            print(f"{d / ds.scenario}.npz: {ds.stats['noisy_start_demos']} noisy-start + "
                  f"{ds.stats['fixed_demos']} fixed demos")


if __name__ == "__main__":
    main()
