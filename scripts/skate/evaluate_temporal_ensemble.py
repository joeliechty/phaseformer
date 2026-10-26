"""Re-evaluate trained chunk policies with temporal ensembling (ACT's ``temporal_agg``).

    python scripts/skate/evaluate_temporal_ensemble.py --methods fm_chunk dp mip --scenarios all \
        --n-train 1024 --seeds 0 1 2 3 4 --m 0.01 --device cuda

Nothing is retrained. For every finished ``{runs}/{scenario}/n{n}/{method}/seed{s}/`` the saved
policy (``final.pt``; the EMA weights for the generative baselines) is rolled out again with a
chunk predicted at *every* control step, and the control executed at step ``t`` is
``sum_i w_i u_i / sum_i w_i`` over all predictions made so far for ``t``, ``w_i = exp(-m i)``,
oldest first (``fm_bc.skate.evaluation``). The evaluation is otherwise the original run's: the same
held-out starts, clamping, MMD bandwidth, kNN rank and sampling seed.

Writes a new run next to the original, ``{method}_te{m}/seed{s}/``, with ``metrics.json`` and
``rollouts.npz`` from the new rollouts and, from the original run, ``config.json``, ``log.csv``
(the training curve is the same) and ``final.pt`` as a symlink -- so ``evaluate_policies.py``
aggregates it like any other arm. Skips runs that already have a ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate.baselines import load_policy  # noqa: E402
from fm_bc.skate.dataset import TrajectoryDataset  # noqa: E402
from fm_bc.skate.evaluation import evaluate_policy  # noqa: E402
from fm_bc.skate.scenarios import SCENARIOS  # noqa: E402
from fm_bc.skate.training import method_name, parse_method  # noqa: E402

# What a run's metrics.json carries besides the evaluation itself.
_INFO_KEYS = ("train_config", "scenario", "n_train", "n_test", "n_params", "div_config",
              "train_seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--methods", nargs="+", default=["fm_chunk", "dp", "mip"])
    parser.add_argument("--scenarios", nargs="+", default=["all"])
    parser.add_argument("--n-train", type=int, nargs="+", default=[1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--m", type=float, default=0.01,
                        help="ACT's exponential weight: w_i = exp(-m i), oldest prediction first")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--test", type=Path, default=Path("data/test"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    scenarios = SCENARIOS if args.scenarios == ["all"] else args.scenarios
    sys.stdout.reconfigure(line_buffering=True)

    t0, done = time.time(), 0
    for scenario in scenarios:
        test = TrajectoryDataset.load(args.test / f"{scenario}.npz")
        for n in args.n_train:
            for method in args.methods:
                target = method_name(**{**parse_method(method), "temporal_ensemble": args.m})
                for seed in args.seeds:
                    src = args.runs / scenario / f"n{n}" / method / f"seed{seed}"
                    out = args.runs / scenario / f"n{n}" / target / f"seed{seed}"
                    if not (src / "metrics.json").exists():
                        print(f"skip {src} (not trained)")
                        continue
                    if (out / "metrics.json").exists() and not args.overwrite:
                        print(f"skip {out} (done)")
                        continue
                    orig = json.loads((src / "metrics.json").read_text())
                    cfg = orig["train_config"]
                    model, _ = load_policy(src / "final.pt", map_location=args.device)
                    model = model.to(args.device)
                    t1 = time.time()
                    metrics, arrays = evaluate_policy(
                        model, test, bandwidth=cfg["mmd_bandwidth"], k=cfg["k"], full=True,
                        clamp=cfg["clamp_controls"], device=args.device, sample_seed=seed,
                        temporal_ensemble=args.m)
                    metrics.update({key: orig[key] for key in _INFO_KEYS if key in orig})
                    metrics["eval_seconds"] = time.time() - t1
                    out.mkdir(parents=True, exist_ok=True)
                    for name in ("config.json", "log.csv"):
                        shutil.copy2(src / name, out / name)
                    link = out / "final.pt"
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(os.path.relpath(src / "final.pt", out))
                    np.savez_compressed(out / "rollouts.npz", **arrays)
                    # Last: metrics.json marks a finished run.
                    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
                    done += 1
                    print(f"{out}: mmd {metrics['mmd_mean']:.4f} (was {orig['mmd_mean']:.4f})  "
                          f"obstacle {metrics['obstacle_collision_rate']:.3f} "
                          f"(was {orig.get('obstacle_collision_rate', float('nan')):.3f})  "
                          f"success {metrics['success_rate']:.3f} (was {orig['success_rate']:.3f})"
                          f"  ({metrics['eval_seconds']:.0f}s)")
    print(f"\n{done} runs re-evaluated in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
