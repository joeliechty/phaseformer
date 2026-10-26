"""Estimate SE(2) divergence and score for generated datasets and store them in the .npz.

    python scripts/skate/compute_divergence.py --data data/ --div-method knn --k 16 --bandwidth 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate import EXTRA_SCENARIOS, SCENARIOS, TrajectoryDataset, add_div_and_score  # noqa: E402


def _describe(ds: TrajectoryDataset) -> None:
    quarters = np.array_split(np.arange(ds.poses.shape[1]), 4)
    for m, name in enumerate(ds.spec.mode_names):
        sel = ds.mode == m
        means = [f"{ds.divergence[sel][:, q].mean():+.3f}" for q in quarters]
        means_xy = [f"{ds.divergence_xy[sel][:, q].mean():+.3f}" for q in quarters]
        print(f"   {name:>11}: mean div per phase quarter  se2 {means}  xy {means_xy}")
    score_norm = np.linalg.norm(ds.score, axis=-1)
    print(f"   |score| median {np.median(score_norm):.3f}, 95% {np.percentile(score_norm, 95):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--scenario", default="all", choices=("all", *SCENARIOS, *EXTRA_SCENARIOS))
    parser.add_argument("--div-method", default="knn", choices=("knn", "kde"),
                        help="divergence estimator")
    parser.add_argument("--k", type=int, default=16, help="k-th nearest neighbour (knn only)")
    parser.add_argument("--bandwidth", type=float, default=0.2, help="score kernel bandwidth")
    parser.add_argument("--div-bandwidth", type=float, default=None,
                        help="KDE divergence bandwidth (default: --bandwidth)")
    parser.add_argument("--rot-scale", type=float, default=1.0,
                        help="metres per radian in the SE(2) metric")
    parser.add_argument("--window", type=int, default=9, help="Savitzky-Golay window length")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    for name in names:
        path = args.data / f"{name}.npz"
        ds = TrajectoryDataset.load(path)
        add_div_and_score(ds, method=args.div_method, k=args.k, bandwidth=args.bandwidth,
                          div_bandwidth=args.div_bandwidth, rot_scale=args.rot_scale,
                          window_length=args.window, device=args.device)
        ds.save(path)
        print(f"== {name}: {len(ds)} trajectories, {args.div_method} divergence/score "
              f"written to {path}")
        _describe(ds)


if __name__ == "__main__":
    main()
