"""Train the supervised phase estimators used as the Push-T rollout clock.

    python scripts/pusht/train_pusht_phase.py --seeds 0 1 2 3 4

One three-layer MLP per seed, fit by MSE on every demonstration's ``rho = k / (L - 1)``
(:mod:`fm_bc.pusht.phase_mlp`). The seeds match the policy seeds so a rollout can pair each policy
with the estimator trained on the same seed. Writes ``runs/pusht/phase/seed{s}/model.pt`` and
``metrics.json`` (``runs/pusht/phase_start/`` with ``--use-start``, the estimator the paper's
rollouts use). Overfitting is fine and expected: this is an instrument for injecting task
progress, not a generalization result.

``--arch transformer`` trains :class:`fm_bc.pusht.phase_mlp.PhaseClockTransformer` instead, the
robot and the object as separate tokens, e.g. the size twin of the 724 x 3 MLP clock:

    python scripts/pusht/train_pusht_phase.py --arch transformer --use-start \
        --hidden-dim 148 --num-layers 4 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --out runs/pusht/phase_start_xf_large
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.pusht.data import PushTEpisodes  # noqa: E402
from fm_bc.pusht.phase_mlp import train_phase_mlp  # noqa: E402
from fm_bc.pusht.prepared import PushTPrepared  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episodes", type=Path, default=Path("data/pusht/pusht_episodes.npz"))
    p.add_argument("--prepared", type=Path, default=Path("data/pusht/pusht_prepared_9010.npz"))
    p.add_argument("--out", type=Path, default=None,
                   help="default: runs/pusht/phase_start with --use-start, else runs/pusht/phase")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--arch", default="mlp", choices=["mlp", "transformer"],
                   help="mlp: PhaseMLP; transformer: PhaseClockTransformer, the robot and the "
                        "object as separate tokens")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512], help="--arch mlp only")
    tf = p.add_argument_group("--arch transformer", "default: the class's own (104 / 2 / 4)")
    tf.add_argument("--hidden-dim", type=int, default=None)
    tf.add_argument("--num-layers", type=int, default=None)
    tf.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--use-start", action="store_true",
                   help="also feed the episode's initial block pose (xy, cos, sin)")
    p.add_argument("--num-epochs", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="linear learning-rate warmup (0: none, the MLP clocks' recipe)")
    p.add_argument("--cosine", action="store_true",
                   help="anneal the learning rate to 0 over the run (default: constant)")
    p.add_argument("--train-split-only", action="store_true",
                   help="fit on the training episodes only (default: every episode)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.out is None:
        args.out = Path("runs/pusht/phase_start" if args.use_start else "runs/pusht/phase")

    # Only the flags actually passed, so an un-overridden size is the class's own default.
    net_kwargs = {k: v for k, v in {"hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
                                    "num_heads": args.num_heads}.items() if v is not None}
    ep = PushTEpisodes.load(args.episodes, device=args.device)
    prep = PushTPrepared.load(args.prepared, device=args.device)
    for seed in args.seeds:
        out = args.out / f"seed{seed}"
        if (out / "metrics.json").exists() and not args.overwrite:
            print(f"skip {out} (done)")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"seed {seed}:")
        model, metrics = train_phase_mlp(
            ep, prep, episode_ids=prep.train_ids if args.train_split_only else None,
            hidden=args.hidden, use_start=args.use_start, num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr, seed=seed, device=args.device, verbose=True,
            arch=args.arch, net_kwargs=net_kwargs, warmup_steps=args.warmup_steps,
            cosine=args.cosine)
        model.save(out / "model.pt", metrics=metrics)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  rmse {metrics['rmse']:.4f}  (train {metrics.get('rmse_train', float('nan')):.4f}, "
              f"test {metrics.get('rmse_test', float('nan')):.4f})")


if __name__ == "__main__":
    main()
