"""Audit a Push-T artifact: print its shape and check every invariant it should satisfy.

    python scripts/pusht/inspect_pusht.py --data data/pusht/pusht_episodes.npz
    python scripts/pusht/inspect_pusht.py --data data/pusht/pusht_prepared.npz

Read-only, and **exits non-zero if any check fails**, so it can gate a pipeline run. It detects
which of the two artifacts it was given from the keys present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.se2 import wrap_angle  # noqa: E402
from fm_bc.pusht.constants import DT, WORKSPACE_PX  # noqa: E402
from fm_bc.pusht.data import PushTEpisodes  # noqa: E402
from fm_bc.pusht.prepared import PushTPrepared  # noqa: E402


class Checks:
    """Collects pass/fail lines so the whole audit runs before the exit status is decided."""

    def __init__(self) -> None:
        self.failed = 0

    def __call__(self, ok: bool, label: str, detail: str = "") -> None:
        self.failed += not ok
        print(f"   [{'ok' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")


def inspect_episodes(path: Path, check: Checks) -> None:
    ep = PushTEpisodes.load(path)  # .load() already re-derives and asserts the index fields
    print(f"== {path}  (canonical episodes)")
    print(f"   {ep.n_episodes} episodes / {len(ep)} steps")
    print("   " + json.dumps(ep.meta.get("length_stats", {})))

    lengths = ep.lengths
    check(int(ep.episode_ends[-1]) == len(ep), "episode_ends[-1] == n_steps")
    check(bool((lengths > 0).all()), "every episode is non-empty")
    check(bool((ep.phase >= 0).all() and (ep.phase <= 1).all()), "phase within [0, 1]")

    for e in range(ep.n_episodes):
        sl = ep.episode_slice(e)
        if float(ep.phase[sl][0]) != 0.0 or float(ep.phase[sl][-1]) != 1.0:
            check(False, "phase runs 0 -> 1 in every episode", f"episode {e}")
            break
    else:
        check(True, "phase runs 0 -> 1 in every episode")

    lo, hi = ep.state.min(0).values, ep.state.max(0).values
    print(f"   state min {np.round(lo.numpy(), 2).tolist()}")
    print(f"   state max {np.round(hi.numpy(), 2).tolist()}")
    out = int(((ep.action < 0) | (ep.action > WORKSPACE_PX)).sum())
    check(out == 0, f"actions inside [0, {WORKSPACE_PX:.0f}]", f"{out} outside" if out else "")
    theta = ep.state[:, 4]
    check(bool(((theta >= 0) & (theta < 2 * np.pi + 1e-4)).all()),
          "block_theta in the released [0, 2pi) convention")

    # agent_delta must be the forward difference, with each episode's last step edge-held.
    err = 0.0
    for e in range(ep.n_episodes):
        s, t = int(ep.episode_starts[e]), int(ep.episode_ends[e])
        want = ep.state[s + 1:t, :2] - ep.state[s:t - 1, :2]
        err = max(err, float((ep.agent_delta[s:t - 1] - want).abs().max()))
        err = max(err, float((ep.agent_delta[t - 1] - ep.agent_delta[t - 2]).abs().max()))
    check(err < 1e-4, "agent_delta is the forward difference, last step edge-held", f"{err:.2e}")


def inspect_prepared(path: Path, check: Checks) -> None:
    prep = PushTPrepared.load(path)
    grid, split = prep.grid, prep.split
    print(f"== {path}  (prepared: splits, normalizers, phase grid)")
    print("   " + json.dumps(prep.meta["split"]))
    print(f"   grid {tuple(grid.grid_state.shape)}  "
          f"normalizer {prep.meta['normalizer']['mode']}")

    train, val = prep.train_ids, prep.val_ids
    check(len(np.intersect1d(train.numpy(), val.numpy())) == 0, "train and val are disjoint")
    check(len(train) == prep.meta["split"]["n_train"], f"{len(train)} training episodes")
    check(len(val) == prep.meta["split"]["n_val"], f"{len(val)} validation episodes")
    test = prep.test_ids
    check(len(np.intersect1d(test.numpy(), np.concatenate([train.numpy(), val.numpy()]))) == 0,
          "test is disjoint from train and val")
    check(len(test) == prep.meta["split"].get("n_test", 0), f"{len(test)} test episodes")
    check(int(split.shape[0]) == int(grid.grid_state.shape[0]), "one split label per episode")

    phase = grid.grid_phase
    check(bool((phase.diff() > 0).all()), "grid phase is strictly increasing")
    check(abs(float(phase[0])) < 1e-6 and abs(float(phase[-1]) - 1) < 1e-6,
          "grid phase spans [0, 1]")

    # The per-episode dt is the fix for a global-dt divergence bug; a constant one means it was
    # lost, and every short episode's divergence would be silently mis-scaled.
    dt = grid.grid_dt
    spread = float(dt.max() / dt.min())
    check(spread > 1.0, "grid_dt varies across episodes",
          f"{float(dt.min()):.4f}..{float(dt.max()):.4f} s ({spread:.2f}x)")
    want = (grid.lengths - 1).float() * DT / (grid.n_grid - 1)
    check(float((dt - want).abs().max()) < 1e-6, "grid_dt == (L_e - 1) * DT / N")

    theta = grid.grid_state[..., 4]
    check(bool((theta.abs() <= np.pi + 1e-4).all()), "gridded block_theta in (-pi, pi]")
    # Compare *wrapped* steps: a raw difference of ~2pi is just the (-pi, pi] seam and says
    # nothing. What would indicate a bad interpolation is a wrapped step near pi -- the geodesic
    # having taken the long way round between two bracketing poses.
    step = float(wrap_angle(theta.diff(dim=1)).abs().max())
    check(step < np.pi / 2, "block heading takes the short arc between poses",
          f"max wrapped |dtheta| {step:.3f} rad")

    worst = 0.0
    for key, modes in sorted(prep.norm.items()):
        for mode, n in modes.items():
            x = torch.randn(64, n.scale.shape[-1]) * 100 + 200
            worst = max(worst, float((n.invert(n.apply(x)) - x).abs().max()))
    check(worst < 1e-3, "every normalizer round-trips", f"max err {worst:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/pusht/pusht_episodes.npz"))
    args = parser.parse_args()

    if not args.data.exists():
        sys.exit(f"no such artifact: {args.data}")
    with np.load(args.data, allow_pickle=False) as f:
        keys = set(f.files)

    check = Checks()
    if "grid_state" in keys:
        inspect_prepared(args.data, check)
    elif "episode_ends" in keys:
        inspect_episodes(args.data, check)
    else:
        sys.exit(f"{args.data} is not a Push-T artifact (keys: {sorted(keys)[:8]})")

    print(f"\n{'FAILED: ' + str(check.failed) + ' check(s)' if check.failed else 'all checks passed'}")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
