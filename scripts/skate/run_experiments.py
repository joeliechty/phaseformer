"""Train the policy grid: scenarios x training-set sizes x regularizer arms x seeds.

    python scripts/skate/run_experiments.py --scenarios all --cdm-weights 0 0.05 \
        --n-train 1024 128 --seeds 0 1 2 3 4 --device cuda

``--no-phase --cdm-weights 0`` trains the state-only MLP baseline on the same subsets
(``runs/.../mlp/``), and ``--cdm-weights 0 --horizon 16 --action-steps 8`` FM with action
chunking (``runs/.../fm_chunk/``); add ``--cdm-weights 0.05 --cdm-terms score --cdm-schedule cosine``
to that for the score regularizer along the chunk (``runs/.../chunk_score_w0.05_cos0.5/``). The CDM ablation -- each regularizer term alone and both together, with
the weight annealed to 0 over the first half of training:

    python scripts/skate/run_experiments.py --cdm-weights 0.05 --cdm-terms both div score \
        --cdm-schedule cosine --cdm-anneal-frac 0.5 --n-train 1024 --seeds 0 1 2 3 4

Runs in one process (one conda start-up) and skips any run that already has a
``metrics.json``, so an interrupted grid resumes where it stopped. For each ``(n_train,
seed)`` the training subset -- and its divergence / score -- is shared by every arm, so
the arms differ only in the loss.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate.scenarios import SCENARIOS  # noqa: E402
from fm_bc.skate.training import CDM_TERMS, prepare_training_set, run_dir, train  # noqa: E402
from train_policy import add_train_args, config_from_args, load_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenarios", nargs="+", default=["all"])
    parser.add_argument("--cdm-weights", type=float, nargs="+", default=[0.0, 0.05])
    parser.add_argument("--cdm-terms", nargs="+", default=["both"], choices=CDM_TERMS,
                        help="regularizers trained at every nonzero weight")
    parser.add_argument("--n-train", type=int, nargs="+", default=[1024, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--overwrite", action="store_true")
    add_train_args(parser)
    args = parser.parse_args()
    scenarios = SCENARIOS if args.scenarios == ["all"] else args.scenarios
    # Weight 0 is FM whatever the terms, so it is trained once.
    arms = [(w, t) for w in args.cdm_weights for t in (args.cdm_terms if w > 0 else ["both"])]
    # The grid is usually run with stdout redirected to a log; flush per line so it can be tailed.
    sys.stdout.reconfigure(line_buffering=True)

    t0, done = time.time(), 0
    for scenario in scenarios:
        ds, test = load_split(args, scenario)
        for n_train in args.n_train:
            for seed in args.seeds:
                ds_train = None
                for w, terms in arms:
                    cfg = config_from_args(args, cdm_weight=w, cdm_terms=terms, n_train=n_train,
                                           seed=seed)
                    n = min(n_train, len(ds))
                    out = run_dir(args.out, scenario, n, cfg)
                    if (out / "metrics.json").exists() and not args.overwrite:
                        print(f"skip {out} (done)")
                        continue
                    if ds_train is None:
                        ds_train = prepare_training_set(ds, cfg)
                    print(f"\n=== {out}  [{time.time() - t0:.0f}s elapsed]")
                    _, final = train(ds_train, test, cfg, out)
                    done += 1
                    print(f"    mmd {final['mmd_mean']:.4f}  collision "
                          f"{final['collision_rate']:.3f}  success {final['success_rate']:.3f}  "
                          f"modes {[round(f, 3) for f in final['mode_fractions']]}")
    print(f"\n{done} runs trained in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
