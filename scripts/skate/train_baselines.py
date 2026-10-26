"""Train the action-chunk baselines -- Diffusion Policy (``dp``), MIP (``mip``), MeanFlow
(``meanflow``) and ACT (``act``) -- over scenarios x training-set sizes x seeds.

    python scripts/skate/train_baselines.py --methods dp mip act --scenarios all \
        --n-train 1024 --seeds 0 1 2 3 4 --device cuda

Writes ``{out}/{scenario}/n{n_train}/{method}/seed{s}/`` with the same files as
``train_policy.py``, so ``evaluate_policies.py`` aggregates them next to FM, MLP and the CDM
arms. Each method keeps its paper's optimization recipe (``fm_bc.skate.baseline_training``); the
flags below override it for every method given. Runs in one process, skips any run that
already has a ``metrics.json`` and shares each ``(n_train, seed)`` training subset across the
methods, like ``run_experiments.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate.baseline_training import (  # noqa: E402
    BaselineConfig, baseline_run_dir, train_baseline,
)
from fm_bc.skate.scenarios import SCENARIOS  # noqa: E402
from fm_bc.skate.training import BASELINE_METHODS, prepare_training_set  # noqa: E402
from train_policy import load_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--methods", nargs="+", default=list(BASELINE_METHODS),
                        choices=BASELINE_METHODS)
    parser.add_argument("--scenarios", nargs="+", default=["all"])
    parser.add_argument("--n-train", type=int, nargs="+", default=[1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--data", type=Path, default=Path("data"), help="training .npz dir")
    parser.add_argument("--test", type=Path, default=Path("data/test"), help="held-out .npz dir")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=None, help="default: per method")
    parser.add_argument("--batch-size", type=int, default=None, help="default: per method")
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--k", type=int, default=16, help="kNN rank of the divergence")
    parser.add_argument("--mmd-bandwidth", type=float, default=0.3)
    parser.add_argument("--no-clamp", action="store_true",
                        help="execute raw commands instead of saturating them to SkateLimits")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    scenarios = SCENARIOS if args.scenarios == ["all"] else args.scenarios
    sys.stdout.reconfigure(line_buffering=True)

    t0, done = time.time(), 0
    for scenario in scenarios:
        ds, test = load_split(args, scenario)
        for n_train in args.n_train:
            for seed in args.seeds:
                ds_train = None
                for method in args.methods:
                    cfg = BaselineConfig(
                        method=method, steps=args.steps, lr=args.lr, batch_size=args.batch_size,
                        n_train=n_train, k=args.k, eval_every=args.eval_every,
                        log_every=args.log_every, mmd_bandwidth=args.mmd_bandwidth,
                        clamp_controls=not args.no_clamp, seed=seed, device=args.device,
                    )
                    out = baseline_run_dir(args.out, scenario, min(n_train, len(ds)), cfg)
                    if (out / "metrics.json").exists() and not args.overwrite:
                        print(f"skip {out} (done)")
                        continue
                    if ds_train is None:
                        ds_train = prepare_training_set(ds, cfg)
                    print(f"\n=== {out}  [{time.time() - t0:.0f}s elapsed]")
                    _, final = train_baseline(ds_train, test, cfg, out)
                    done += 1
                    print(f"    mmd {final['mmd_mean']:.4f}  collision "
                          f"{final['collision_rate']:.3f}  success {final['success_rate']:.3f}  "
                          f"modes {[round(f, 3) for f in final['mode_fractions']]}  "
                          f"({final['train_seconds']:.0f}s)")
    print(f"\n{done} runs trained in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
