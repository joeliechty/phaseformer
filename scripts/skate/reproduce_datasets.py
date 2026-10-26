"""Rebuild the shipped skate nav datasets bit for bit, and check them against their checksums.

    python scripts/skate/reproduce_datasets.py                 # -> data/reproduced/, then checked
    python scripts/skate/reproduce_datasets.py --check data/   # only check an existing copy

The six files in ``data/`` (training, seed 0) and ``data/test/`` (held out, seed 1), one per
scenario, were made by two generations of the planner, and ``generate_datasets.py`` today
reproduces only one of them:

* ``single_path`` was planned before commit ``2eaf39d``: against the distance-transform grid SDF,
  from seeds whose last knot snaps to the goal heading. Today's planner accepts ~98 % of the
  problems that one accepted 34 % of, so every array differs.
* ``dual_goal`` and ``dual_path`` were planned by ``2eaf39d``, which is still today's planner.

All six divergence / score targets were then computed with the SE(2) log's
``(theta/2) cot(theta/2)`` in its form before commit ``8c0f517``, which moves their last bits
(by up to ~3e-8).

This script switches exactly those three things back -- ``GenerationConfig(sdf="edt",
seed_goal_heading=True)`` for single_path, :func:`fm_bc.se2.legacy_half_cot` for the targets --
and otherwise runs the pipeline as it was run: 1024 demos per scenario on CUDA, then
``add_div_and_score`` with ``compute_divergence.py``'s defaults. Every array of every file is then
compared with ``nav_datasets.sha256.json`` (the SHA-256 of the shipped files); a mismatch is
named and the exit status is 1. On the RTX 4090 with torch 2.1.2+cu121 all 54 arrays match. On
other hardware the planner's solves may differ in their last bits, and the check will say so.

It never overwrites a file without ``--overwrite``: ``data/`` holds the only copy of the shipped
demonstrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.se2 import legacy_half_cot  # noqa: E402
from fm_bc.skate import (  # noqa: E402
    SCENARIOS, GenerationConfig, add_div_and_score, generate, get_scenario,
)

MANIFEST = Path(__file__).with_name("nav_datasets.sha256.json")
N_TRAJ = 1024
# Subdirectory -> seed: the training demos, and the independent held-out set.
SPLITS = {"": 0, "test": 1}
# The pre-2eaf39d planner, for the scenario it made; the others use today's defaults.
LEGACY = {"single_path": {"sdf": "edt", "seed_goal_heading": True}}


def digests(path: Path) -> dict[str, str]:
    """SHA-256 of every array in an ``.npz`` (dtype, shape and bytes). The metadata is left out:
    it records wall-clock time, and a rebuilt file's config also records the legacy switches."""
    out = {}
    with np.load(path, allow_pickle=False) as f:
        for name in sorted(f.files):
            if name == "meta":
                continue
            a = np.ascontiguousarray(f[name])
            h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
            h.update(a.tobytes())
            out[name] = h.hexdigest()
    return out


def check(root: Path) -> bool:
    """Compare every file the manifest names, under ``root``; one line per file."""
    manifest = {k: v for k, v in json.loads(MANIFEST.read_text()).items() if not k.startswith("_")}
    ok = True
    for rel, want in manifest.items():
        path = root / rel
        if not path.exists():
            print(f"  MISSING  {path}")
            ok = False
            continue
        got = digests(path)
        bad = sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))
        print(f"  {'DIFFERS' if bad else 'ok':8s} {path}" + (f"  ({', '.join(bad)})" if bad else ""))
        ok &= not bad
    print("all arrays match the shipped datasets" if ok else "MISMATCH against the shipped datasets")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/reproduced"))
    parser.add_argument("--check", type=Path, default=None, metavar="DIR",
                        help="only check DIR (e.g. data/) against the checksums")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.check is not None:
        sys.exit(0 if check(args.check) else 1)

    targets = [args.out / sub / f"{name}.npz" for sub in SPLITS for name in SCENARIOS]
    existing = [t for t in targets if t.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{len(existing)} of the files exist already, e.g. {existing[0]}; "
                         "pass --overwrite to replace them")

    for sub, seed in SPLITS.items():
        (args.out / sub).mkdir(parents=True, exist_ok=True)
        for name in SCENARIOS:
            legacy = LEGACY.get(name, {})
            print(f"== {sub or 'train'}/{name}, seed {seed}"
                  + (" (planner before 2eaf39d)" if legacy else ""))
            ds = generate(get_scenario(name), n_target=N_TRAJ,
                          config=GenerationConfig(seed=seed, **legacy), device=args.device)
            with legacy_half_cot():
                add_div_and_score(ds)
            ds.save(args.out / sub / f"{name}.npz")

    print(f"checking {args.out} against {MANIFEST.name}")
    sys.exit(0 if check(args.out) else 1)


if __name__ == "__main__":
    main()
