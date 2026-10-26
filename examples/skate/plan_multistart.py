"""Batched multi-start planning: every restart is one row of a single Theseus solve.

    python examples/skate/plan_multistart.py --scenario slalom --starts 12

Trajectory optimization is local, so the seed decides which way around an obstacle the solution
goes. Theseus batches over dim 0, so N restarts cost one solve rather than N.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from skate_planner import SkatePlanner, plot_multistart, scenario  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "output" / "skate" / "planner"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="slalom", choices=["slalom", "gap", "clutter"])
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=45)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--cell-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cpu",
        help="cpu is faster below ~32 restarts; cuda pulls ahead above that",
    )
    args = parser.parse_args()

    world, start, goal = scenario(args.scenario)
    planner = SkatePlanner.for_world(
        world, cell_size=args.cell_size, horizon=args.horizon, dt=args.dt,
        radius=args.radius, device=args.device,
    )

    elapsed = time.perf_counter()
    solution = planner.plan(
        world, start, goal, cell_size=args.cell_size, n_starts=args.starts,
        rng=np.random.default_rng(args.seed),
    )
    elapsed = time.perf_counter() - elapsed

    print(f"{args.starts} restarts solved in one batched call in {elapsed:.2f}s "
          f"on {args.device}")
    print(solution.report())

    diagnostics = solution.diagnostics
    print(f"\n{'#':>3s} {'feas':>5s} {'clearance':>10s} {'goal err':>9s} {'drift':>8s} "
          f"{'path':>8s}")
    for i in range(args.starts):
        marker = " *" if i == solution.best_index else "  "
        print(
            f"{i:>3d}{marker[1]} {str(bool(solution.feasible[i])):>5s} "
            f"{diagnostics.min_clearance[i]:10.3f} {diagnostics.goal_error_xy[i]:9.4f} "
            f"{diagnostics.rollout_drift[i]:8.4f} {diagnostics.path_length[i]:8.3f}"
        )
    print("\n(clearance and goal error are measured on the rolled-out path, not the "
          "optimizer's pose variables)")

    OUT.mkdir(parents=True, exist_ok=True)
    figure = OUT / f"{args.scenario}_multistart.png"
    plot_multistart(
        world, solution, radius=planner.radius, safety_margin=planner.safety_margin,
        save=figure,
    )
    print(f"wrote {figure}")


if __name__ == "__main__":
    main()
