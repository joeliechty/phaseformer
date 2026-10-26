"""Build the start-spread sets: the start spread varied with everything else fixed, sides random
or sorted, and the start-noise mixtures' noisy demos on their own.

    python scripts/skate/build_start_spread.py

Every set is C's world driven to the fixed goal pose (5.0, 7.0, heading up) with no via jitter,
1024 demos, exactly 512 per side (``fm_bc.skate.scenarios``, ``START_SPREAD_SCENARIOS``):

* ``dual_path_spread{02,04,08,12}`` -- start (5 + σε_x, 0.9 + σε_y, π/2 + 0.05 (σ / 8 cm) ε_θ)
  for σ = 2, 4, 8, 12 cm, the side from one fixed random assignment (independent of the start);
* ``dual_path_sorted{02,08}`` -- the same starts, the side sorted: left exactly when the start
  is left of centre (ε_x < 0);
* ``dual_path_mixonly{05,10,15}`` -- the 52 / 102 / 154 noisy-start demos of
  ``dual_path_mix05/10/15`` alone (``build_start_mixes.py``), without the fixed copies.

The ε are common random numbers: one draw of standard normals per split (seed 0 for the training
sets, 1 for the held-out ``test/`` sets), shared by every σ and both side rules, in antithetic
pairs -- 512 draws, each used as ε and -ε. The start sample is then exactly symmetric about the
centre line, so the sorted rule gives exactly 512 per side; i.i.d. draws would put the split at
the sample median instead (-0.12 for seed 0's draw, a 3σ fluctuation that also shifts every
start 0.8 cm left on average at σ = 8 cm). A start whose robot disc (radius 0.3 m) touches a
wall, or a problem the planner fails in *any* set, has its pair replaced by the next spare draw
in *every* set; this repeats until every set is complete. A (σ, ε, side) problem is planned once and shared, so where the
random and sorted rules agree the two sets hold the same demo. Each file's ``stats`` records the
redraws, the corr(start x, side) and the per-side counts. The divergence and score targets are
then computed as for every other set (``add_div_and_score``'s defaults).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_start_mixes import noisy_count, take  # noqa: E402
from fm_bc.skate import GenerationConfig, TrajectoryDataset, add_div_and_score  # noqa: E402
from fm_bc.skate.dataset import plan_problems  # noqa: E402
from fm_bc.skate.scenarios import MIX_PERCENTS, SORTED_CM, SPREAD_CM, get_scenario  # noqa: E402
from skate_planner.skate_model import body_twist  # noqa: E402

N = 1024
SPARES = 512
SPLITS = {"": 0, "test": 1}  # subdirectory -> seed of its draws, as generate_datasets.py
# (name, sigma in cm, side rule), one per planned set.
SETS = [(f"dual_path_spread{c:02d}", c, "random") for c in SPREAD_CM] + \
       [(f"dual_path_sorted{c:02d}", c, "sorted") for c in SORTED_CM]


def starts_of(eps: np.ndarray, cm: int) -> np.ndarray:
    s = cm / 100
    return np.column_stack([5.0 + s * eps[:, 0], 0.9 + s * eps[:, 1],
                            np.pi / 2 + 0.05 * (s / 0.08) * eps[:, 2]])


def sorted_sides(eps_x: np.ndarray) -> np.ndarray:
    """Left (0) left of centre, right (1) right of it: exactly half each, the draws being
    antithetic."""
    return (eps_x > 0).astype(int)


def build_split(seed: int, device: str) -> dict[str, TrajectoryDataset]:
    rng = np.random.default_rng(seed)
    half = N // 2
    pool = rng.standard_normal((half + SPARES, 3))
    random_sides = rng.permutation(np.repeat([0, 1], half))
    spec = get_scenario(SETS[0][0])  # world, goal pose and vias: the same for every set
    radius = 0.3
    rows = np.arange(half)  # the pool row of each antithetic pair: slots i and i + N/2
    sign = np.repeat([1.0, -1.0], half)
    spare, redraws, start_hits, rejected = half, 0, 0, 0
    # (cm, pool row, sign, side) -> (poses, controls, ok)
    cache: dict[tuple[int, int, float, int], tuple] = {}
    cfg = GenerationConfig(seed=seed)
    config = None  # the planner's, as recorded by the first solve
    def key_of(cm, i, side):
        return cm, int(rows[i % half]), float(sign[i]), int(side)

    while True:
        eps = sign[:, None] * np.concatenate([pool[rows], pool[rows]])
        want = {name: (starts_of(eps, cm), random_sides if rule == "random"
                       else sorted_sides(eps[:, 0])) for name, cm, rule in SETS}
        # Starts that put the robot disc in a wall are redrawn without planning.
        bad = np.zeros(N, dtype=bool)
        for name, cm, _ in SETS:
            bad |= spec.world.analytic_sdf(want[name][0][:, :2]) < radius
        start_hits += int(bad.sum())
        todo = {}
        for name, cm, _ in SETS:
            starts, sides = want[name]
            for i in np.flatnonzero(~bad):
                key = key_of(cm, i, sides[i])
                if key not in cache:
                    todo[key] = (starts[i], sides[i])
        if todo:
            keys = list(todo)
            ds, ok = plan_problems(spec, np.stack([todo[k][0] for k in keys]),
                                   np.array([todo[k][1] for k in keys]), cfg, device)
            config = config or ds.config
            for j, key in enumerate(keys):
                cache[key] = (ds.poses[j], ds.controls[j], bool(ok[j]))
            rejected += int((~ok).sum())
            print(f"  seed {seed}: planned {len(keys)}, {int((~ok).sum())} rejected")
        for name, cm, _ in SETS:
            sides = want[name][1]
            for i in np.flatnonzero(~bad):
                bad[i] |= not cache[key_of(cm, i, sides[i])][2]
        if not bad.any():
            break
        pairs = np.unique(np.flatnonzero(bad) % half)
        for j in pairs:
            if spare == len(pool):
                raise RuntimeError("ran out of spare draws")
            rows[j], spare = spare, spare + 1
            redraws += 1
        print(f"  seed {seed}: redrawing {len(pairs)} antithetic pairs")

    out = {}
    for name, cm, rule in SETS:
        starts, sides = want[name]
        planned = [cache[key_of(cm, i, sides[i])] for i in range(N)]
        order = np.argsort(sides, kind="stable")  # modes contiguous, as generated sets are
        controls = np.stack([p[1] for p in planned])[order]
        out[name] = TrajectoryDataset(
            scenario=name, poses=np.stack([p[0] for p in planned])[order], controls=controls,
            twists=body_twist(torch.as_tensor(controls)).numpy(), mode=sides[order].copy(),
            starts=starts[order], goals=np.tile(spec.goals[0].mean, (N, 1)), dt=cfg.dt,
            config=config,
            stats={"n": N, "per_mode": np.bincount(sides, minlength=2).tolist(),
                   "sigma_cm": cm, "sides": rule, "crn_seed": seed, "redraws": redraws,
                   "start_collisions": start_hits, "planner_rejections": rejected,
                   "corr_start_x_side": float(np.corrcoef(starts[:, 0], sides)[0, 1]),
                   # Pearson of a median split of a Gaussian tops out at sqrt(2/pi) = 0.80;
                   # this is the one that reads 1 for sorted sides.
                   "corr_start_sign_side": float(np.corrcoef(starts[:, 0] > 5.0, sides)[0, 1])},
        )
    return out


def mix_only(noisy: TrajectoryDataset, pct: int) -> TrajectoryDataset:
    n = noisy_count(pct)
    arrays = take(noisy, n // 2)
    return replace(
        noisy, scenario=f"dual_path_mixonly{pct:02d}", **arrays,
        stats={"n": n, "per_mode": [n // 2, n // 2], "noisy_start_demos": n,
               "from": f"dual_path_mix{pct:02d}"},
        divergence=None, divergence_xy=None, score=None, div_config={},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/skate_deterministic"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split", choices=("train", "test"), default=None,
                        help="build one split only (two processes can then run side by side)")
    args = parser.parse_args()
    for sub, seed in SPLITS.items():
        if args.split and args.split != (sub or "train"):
            continue
        d = args.data / sub
        names = [s[0] for s in SETS]
        if args.overwrite or not all((d / f"{n}.npz").exists() for n in names):
            for ds in build_split(seed, args.device).values():
                ds = add_div_and_score(ds, device=args.device)
                ds.save(d / f"{ds.scenario}.npz")
                st = ds.stats
                print(f"{d / ds.scenario}.npz: per side {st['per_mode']}, corr(start x, side) "
                      f"{st['corr_start_x_side']:+.3f}, corr(sign, side) "
                      f"{st['corr_start_sign_side']:+.3f}, redraws {st['redraws']} "
                      f"(start collisions {st['start_collisions']}, planner rejections "
                      f"{st['planner_rejections']})")
        noisy = TrajectoryDataset.load(d / "dual_path_noisy_start.npz")
        for pct in MIX_PERCENTS:
            path = d / f"dual_path_mixonly{pct:02d}.npz"
            if path.exists() and not args.overwrite:
                continue
            ds = add_div_and_score(mix_only(noisy, pct), device=args.device)
            ds.save(path)
            print(f"{path}: {len(ds)} noisy-start demos, per side {ds.stats['per_mode']}")


if __name__ == "__main__":
    main()
