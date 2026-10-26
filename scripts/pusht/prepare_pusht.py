"""Draw the train/val split, fit normalizers and build the shared phase grid.

    python scripts/pusht/prepare_pusht.py --n-grid 128 --normalizer box
    python scripts/pusht/prepare_pusht.py --split ratio --fractions 0.9 0.1 0.0 --seed 42 \
        --n-grid 128 --normalizer minmax --out data/pusht/pusht_prepared_9010.npz   # the paper's
    python scripts/pusht/prepare_pusht.py --split ratio --fractions 0.7 0.2 0.1 \
        --out data/pusht/pusht_prepared_702010.npz

Pure torch, so it reruns cheaply for any grid size, seed or normalizer. The defaults reproduce
Diffusion Policy's Push-T split (seed 42, val_ratio 0.02, 90 training episodes); ``--split ratio``
splits every episode into train / val / test instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.pusht.data import PushTEpisodes  # noqa: E402
from fm_bc.pusht.normalize import MODES  # noqa: E402
from fm_bc.pusht.prepared import prepare  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/pusht/pusht_episodes.npz"))
    parser.add_argument("--out", type=Path, default=Path("data/pusht/pusht_prepared.npz"))
    parser.add_argument("--n-grid", type=int, default=128,
                        help="phase points N+1; the default sits near the mean episode length")
    parser.add_argument("--split", default="dp", choices=["dp", "ratio"])
    parser.add_argument("--fractions", type=float, nargs=3, default=[0.7, 0.2, 0.1],
                        help="train val test, with --split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-episodes", type=int, default=90,
                        help="Diffusion Policy's default; -1 to keep every training episode")
    parser.add_argument("--normalizer", default="box", choices=MODES,
                        help="the active mode; all of them are written either way")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"would prepare {args.data} -> {args.out} (n_grid={args.n_grid})")
        return

    ep = PushTEpisodes.load(args.data, device=args.device)
    prep = prepare(
        ep,
        n_grid=args.n_grid,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train_episodes=None if args.max_train_episodes < 0 else args.max_train_episodes,
        mode=args.normalizer,
        device=args.device,
        split_mode=args.split,
        fractions=tuple(args.fractions),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prep.save(args.out)

    dt = prep.grid.grid_dt
    print(f"saved -> {args.out}")
    print("   " + json.dumps(prep.meta["split"]))
    print(f"   grid {tuple(prep.grid.grid_state.shape)}  normalizer {args.normalizer}")
    # A constant grid_dt would mean the per-episode time scaling had been lost.
    print(f"   grid_dt {float(dt.min()):.4f}..{float(dt.max()):.4f} s per phase step "
          f"({float(dt.max() / dt.min()):.2f}x spread)")


if __name__ == "__main__":
    main()
