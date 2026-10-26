"""Convert the Push-T replay buffer (zarr) into the canonical episode dataset.

    python scripts/pusht/convert_pusht.py --zarr data/pusht/pusht_cchi_v7_replay.zarr

Run scripts/pusht/download_pusht.sh first. This is the only step that needs zarr
(``pip install -e '.[pusht]'``); everything downstream reads the .npz it writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.pusht.convert import from_zarr, sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zarr", type=Path,
                        default=Path("data/pusht/pusht_cchi_v7_replay.zarr"))
    parser.add_argument("--out", type=Path, default=Path("data/pusht/pusht_episodes.npz"))
    parser.add_argument("--zip", type=Path, default=None,
                        help="the downloaded pusht.zip; its sha256 is recorded in the metadata")
    parser.add_argument("--no-keypoints", action="store_true",
                        help="drop the 9 block keypoints (needed for the 20-dim DP obs)")
    parser.add_argument("--no-contacts", action="store_true")
    # Not "--n": `conda run` would swallow it as an abbreviation of its own --name.
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="convert only the first N episodes, for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"would convert {args.zarr} -> {args.out}")
        return

    digest = sha256(args.zip) if args.zip is not None else None
    ds = from_zarr(
        args.zarr,
        keypoints=not args.no_keypoints,
        n_contacts=not args.no_contacts,
        max_episodes=args.max_episodes,
        zip_sha256=digest,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.save(args.out)

    size_mb = args.out.stat().st_size / 1e6
    print(f"saved {ds.n_episodes} episodes / {len(ds)} steps -> {args.out}  ({size_mb:.1f} MB)")
    print("   " + json.dumps(ds.meta["length_stats"]))
    if ds.keypoint is not None:
        print(f"   keypoints {tuple(ds.keypoint.shape)}")


if __name__ == "__main__":
    main()
