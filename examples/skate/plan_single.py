"""Plan one skate trajectory through an obstacle field, then plot and animate it.

    python examples/skate/plan_single.py --scenario slalom
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from skate_planner import SkatePlanner, animate, plot_solution, scenario  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "output" / "skate" / "planner"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="slalom", choices=["slalom", "gap", "clutter"])
    parser.add_argument("--horizon", type=int, default=45)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--cell-size", type=float, default=0.1)
    parser.add_argument(
        "--device", default="cpu",
        help="cpu is faster below ~32 restarts; cuda pulls ahead above that",
    )
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()

    world, start, goal = scenario(args.scenario)
    planner = SkatePlanner.for_world(
        world, cell_size=args.cell_size, horizon=args.horizon, dt=args.dt,
        radius=args.radius, device=args.device,
    )
    print(
        f"scenario={args.scenario} device={args.device} "
        f"grid={planner.grid_shape} cost functions={planner.objective.size_cost_functions()}"
    )

    solution = planner.plan(world, start, goal, cell_size=args.cell_size, n_starts=1)
    print(solution.report())
    if not solution.feasible[solution.best_index]:
        print(
            "  NOTE: the single straight-line seed did not find a feasible plan. This is the "
            "expected failure mode for a local optimizer -- try examples/skate/plan_multistart.py."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    figure = OUT / f"{args.scenario}_single.png"
    plot_solution(
        world, solution, radius=planner.radius, safety_margin=planner.safety_margin,
        save=figure, title=f"Skate plan - {args.scenario} (single start)",
    )
    print(f"wrote {figure}")

    if not args.no_animation:
        gif = OUT / f"{args.scenario}_single.gif"
        animate(
            world, solution, radius=planner.radius, dt=planner.dt,
            safety_margin=planner.safety_margin, save=gif,
        )
        print(f"wrote {gif}")


if __name__ == "__main__":
    main()
