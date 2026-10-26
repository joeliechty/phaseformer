"""Train the Push-T task-progress CVAE, or just print the null baselines it has to beat.

    python scripts/pusht/train_progress.py --baselines-only
    python scripts/pusht/train_progress.py --seeds 0 1 2 --out runs

Run ``--baselines-only`` first. It costs no training and reports two label-free nulls -- a
hand-coded ``-d_SE2(block, goal)`` and a linear probe on the same window -- which the model has to
clearly beat before any of its numbers mean anything.

Progress is scored against ``ep.phase``, which the model never sees during training. The headline
held-out set is the 112 episodes Diffusion Policy's split convention leaves UNUSED, not its 4
validation episodes; those are reported separately under ``val_split``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.pusht.data import PushTEpisodes  # noqa: E402
from fm_bc.pusht.normalize import MODES  # noqa: E402
from fm_bc.pusht.prepared import PushTPrepared  # noqa: E402
from fm_bc.pusht.progress_training import (  # noqa: E402
    ProgressConfig, phase_baselines, progress_run_dir, train_progress,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episodes", type=Path, default=Path("data/pusht/pusht_episodes.npz"))
    p.add_argument("--prepared", type=Path, default=Path("data/pusht/pusht_prepared.npz"))
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--baselines-only", action="store_true",
                   help="print the null baselines and exit; run this first")
    p.add_argument("--overwrite", action="store_true", help="redo runs that have metrics.json")
    p.add_argument("--dry-run", action="store_true")
    # data / objective
    p.add_argument("--obs-steps", type=int, default=2)
    p.add_argument("--normalizer", default="box", choices=MODES)
    p.add_argument("--gap", default="loguniform", choices=("loguniform", "fixed"))
    p.add_argument("--stride", type=int, default=1, help="only used with --gap fixed")
    p.add_argument("--kappa", type=float, default=0.5,
                   help="margin fraction; kappa=1 is exact phase supervision, never run it")
    p.add_argument("--kappa-hi", type=float, default=None,
                   help="upper band on the advance rate; turn on if calibration_gap is large")
    p.add_argument("--cond-dropout", type=float, default=0.1)
    # model
    p.add_argument("--z-style", type=int, default=8)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--activation", default="silu")
    p.add_argument("--beta", type=float, default=5e-3)
    p.add_argument("--free-bits", type=float, default=0.03)
    p.add_argument("--mono-weight", type=float, default=1.0)
    p.add_argument("--style-dropout", type=float, default=0.0)
    # optimization
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.dry_run:
        print(f"would train on {args.episodes} / {args.prepared} -> {args.out}, "
              f"seeds {args.seeds}")
        return

    ep = PushTEpisodes.load(args.episodes, device=args.device)
    prep = PushTPrepared.load(args.prepared, device=args.device)
    eval_ids = prep.held_out_ids
    print(f"{ep.n_episodes} episodes: {prep.train_ids.numel()} train, "
          f"{prep.val_ids.numel()} val, {eval_ids.numel()} held out for scoring")

    nulls = phase_baselines(ep, prep, eval_ids, args.obs_steps, args.device)
    print("nulls (per-episode spearman vs held-out phase):")
    print(f"   rho_geom   {nulls['rho_geom']:+.3f}   (-d_SE2(block, goal), hand-coded)")
    print(f"   rho_linear {nulls['rho_linear']:+.3f}   (least-squares probe on the same window)")
    print(f"   -> the CVAE must clearly beat {max(nulls['rho_geom'], nulls['rho_linear']):+.3f}")
    if args.baselines_only:
        return

    longest = int(ep.lengths.max())
    print(f"smallest per-step rise asked of the longest episode ({longest} steps): "
          f"{args.kappa / max(longest - 1, 1):.5f}")

    for seed in args.seeds:
        cfg = ProgressConfig(
            obs_steps=args.obs_steps, normalizer=args.normalizer, gap=args.gap,
            stride=args.stride, kappa=args.kappa, kappa_hi=args.kappa_hi,
            cond_dropout=args.cond_dropout, z_style=args.z_style, hidden=tuple(args.hidden),
            activation=args.activation, beta=args.beta, free_bits=args.free_bits,
            mono_weight=args.mono_weight, style_dropout=args.style_dropout, steps=args.steps,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps, grad_clip=args.grad_clip,
            log_every=args.log_every, eval_every=args.eval_every, seed=seed,
            device=args.device,
        )
        run = progress_run_dir(args.out, cfg)
        if (run / "metrics.json").exists() and not args.overwrite:
            print(f"skip {run} (metrics.json exists)")
            continue
        print(f"\n== {run}")
        _, final = train_progress(ep, prep, cfg, out_dir=run, eval_ids=eval_ids)
        _report(final, nulls)


def _report(final: dict, nulls: dict) -> None:
    """Print the gate first, then the headline numbers -- the order they must be read in."""
    ablation = final["phase_ablation"]
    gate = "OK" if ablation >= 5.0 else "FAILED -- the decoder ignored z_phase"
    print(f"   gate: phase_ablation {ablation:+.1f}%  [{gate}]")
    if ablation < 5.0:
        print("   nothing below is meaningful: the hinge alone does not ground z_phase.")
    rho = final["spearman_per_episode_mean"]
    best_null = max(nulls["rho_geom"], nulls["rho_linear"])
    print(f"   spearman/episode {rho:+.3f}  (nulls {best_null:+.3f}, "
          f"margin {rho - best_null:+.3f})")
    print(f"   pooled {final['spearman_pooled']:+.3f}  "
          f"calibration_gap {final['calibration_gap']:+.3f}")
    print(f"   no-cond {final['spearman_no_cond']:+.3f}  "
          f"terciles {final['spearman_t1']:+.2f}/{final['spearman_t2']:+.2f}/"
          f"{final['spearman_t3']:+.2f}")
    print(f"   violations local {final['violation_local']:.3f} "
          f"long {final['violation_long']:.3f}  roughness {final['roughness']:.3f}")
    print(f"   active_units {final['active_units']}  ood_p95 {final['ood_p95']:.2f}  "
          f"rmse_agent {final['rmse_agent_px']:.1f}px")
    print("   " + json.dumps({k: round(final[k], 3) for k in
                              ("c_ablation", "style_ablation", "kl_total")}))


if __name__ == "__main__":
    main()
