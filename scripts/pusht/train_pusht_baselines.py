"""Train the Push-T chunk baselines (Diffusion Policy, MIP, ACT) on the MLP arms' pipeline.

    python scripts/pusht/train_pusht_baselines.py --prepared data/pusht/pusht_prepared_9010.npz \
        --out runs --methods dp mip --seeds 0 1 2
    python scripts/pusht/train_pusht_baselines.py --methods act --lr 1e-5 --no-ema --seeds 0 1 2

Same split, epochs, optimizer, EMA and best-validation-loss selection as
``scripts/pusht/train_pusht_policy.py``; writes ``{out}/pusht/policy/{dp,mip,act}/seed{s}/``.

ACT is trained on its own recipe (``--lr 1e-5 --no-ema``, LeRobot's defaults). Its published
batch size is 8 rather than 256; this script keeps 256, because an epoch-based loop at batch 8
costs 32x the optimizer steps for the same data. The deviation is recorded in the run's
``config.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.pusht.baseline_training import (  # noqa: E402
    BaselineConfig, baseline_run_dir, train_pusht_baseline,
)
from fm_bc.pusht.data import PushTEpisodes  # noqa: E402
from fm_bc.pusht.prepared import PushTPrepared  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episodes", type=Path, default=Path("data/pusht/pusht_episodes.npz"))
    p.add_argument("--prepared", type=Path, default=Path("data/pusht/pusht_prepared_9010.npz"))
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--methods", nargs="+", default=["dp", "mip", "act"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--obs-steps", type=int, default=2)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--action-steps", type=int, default=8)
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="ACT's own recipe is --lr 1e-5 --no-ema")
    p.add_argument("--no-ema", action="store_true",
                   help="train without an EMA and select on the raw model's validation loss "
                        "(ACT is trained without one)")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ep = PushTEpisodes.load(args.episodes, device=args.device)
    prep = PushTPrepared.load(args.prepared, device=args.device)
    for kind in args.methods:
        for seed in args.seeds:
            cfg = BaselineConfig(kind=kind, obs_steps=args.obs_steps, horizon=args.horizon,
                                 action_steps=args.action_steps, num_epochs=args.num_epochs,
                                 batch_size=args.batch_size, lr=args.lr,
                                 use_ema=not args.no_ema,
                                 checkpoint_every=args.checkpoint_every, seed=seed,
                                 device=args.device)
            out = baseline_run_dir(args.out, cfg)
            if (out / "metrics.json").exists() and not args.overwrite:
                print(f"skip {out} (done)")
                continue
            print(f"{kind} seed {seed} -> {out}", flush=True)
            _, m = train_pusht_baseline(ep, prep, cfg, out)
            print(f"done in {m['train_seconds']:.0f}s: best epoch {m['best']['epoch']} "
                  f"val {m['best']['val_loss_ema']:.4f}", flush=True)


if __name__ == "__main__":
    main()
