"""Generate collision-free demonstration distributions for the Fig. 6 scenarios.

    python scripts/skate/generate_datasets.py --scenario all --n-traj 1024 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate import EXTRA_SCENARIOS, SCENARIOS, GenerationConfig, generate, get_scenario  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="all", choices=("all", *SCENARIOS, *EXTRA_SCENARIOS))
    # Not "--n": `conda run` would swallow it as an abbreviation of its own --name.
    parser.add_argument("--n-traj", type=int, default=1024,
                        help="accepted trajectories per scenario")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--chunk", type=int, default=512, help="problems per batched solve")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-balance", action="store_true",
                        help="keep the natural mode split instead of equal counts per mode")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    for name in names:
        cfg = GenerationConfig(horizon=args.horizon, dt=args.dt, chunk=args.chunk,
                               seed=args.seed, balance=not args.no_balance)
        print(f"== {name}")
        ds = generate(get_scenario(name), n_target=args.n_traj, config=cfg, device=args.device)
        path = args.out / f"{name}.npz"
        ds.save(path)
        print(f"   saved {len(ds)} trajectories -> {path}")
        print("   " + json.dumps(ds.stats))


if __name__ == "__main__":
    main()
