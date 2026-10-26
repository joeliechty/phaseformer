"""Compare kNN and KDE divergence estimators on the generated datasets.

    python scripts/skate/compare_divergence.py --data data/ --ks 4 16 --bandwidths 0.1 0.2

For each estimator, per mode:
  band   std across trajectories, averaged over phase (the width of the plotted band)
  |mu|   magnitude of the mean curve, averaged over phase (the signal)
  corr   correlation of the mean curve with the first kNN estimator's
A band that shrinks together with |mu| is attenuation, not a cleaner signal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate import SCENARIOS, TrajectoryDataset  # noqa: E402
from fm_bc.divergence import divergence_kde, divergence_knn  # noqa: E402
from fm_bc.skate.plots import plot_divergence_estimators  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("output/skate/divergence"))
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--bandwidths", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--rot-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name in SCENARIOS:
        path = args.data / f"{name}.npz"
        if not path.exists():
            print(f"skip {name}: {path} missing")
            continue
        ds = TrajectoryDataset.load(path)
        poses = torch.as_tensor(ds.poses, device=args.device)
        est: dict[str, np.ndarray] = {}
        for k in args.ks:
            est[f"kNN k={k}"] = divergence_knn(
                poses, ds.dt, k=k, rot_scale=args.rot_scale).cpu().numpy()
        for h in args.bandwidths:
            est[f"KDE h={h}"] = divergence_kde(
                poses, ds.dt, bandwidth=h, rot_scale=args.rot_scale).cpu().numpy()

        modes = range(ds.spec.n_modes)
        curves = {key: np.concatenate([d[ds.mode == m].mean(0) for m in modes])
                  for key, d in est.items()}
        ref = next(iter(curves.values()))
        print(f"== {name}")
        for key, d in est.items():
            band = np.mean([d[ds.mode == m].std(0).mean() for m in modes])
            mu = curves[key]
            print(f"   {key:>9}: band {band:.3f}  |mu| {np.abs(mu).mean():.3f}  "
                  f"|mu|/band {np.abs(mu).mean() / band:.2f}  "
                  f"corr {np.corrcoef(mu, ref)[0, 1]:+.2f}")

        save = args.out / f"{name}_estimators.png"
        plot_divergence_estimators(ds, est, save=save)
        plt.close("all")
        print(f"   wrote {save}")


if __name__ == "__main__":
    main()
