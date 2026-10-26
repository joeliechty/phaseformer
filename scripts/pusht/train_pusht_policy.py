"""Train the Push-T policy arms -- plain, +phase, +phase+score -- on the agent's phase
velocity dx/drho, with Diffusion Policy's Push-T training pipeline, and score them offline.

    python scripts/pusht/train_pusht_policy.py --methods mlp fm score --seeds 0 1 2 3 4 \
        --device cuda
    python scripts/pusht/train_pusht_policy.py --methods score --seeds 0 1 2 \
        --score-weights 0.5 2.5 10 --score-anneal-fracs 0.5 1 --bandwidths-px 16 32 64
    python scripts/pusht/train_pusht_policy.py --arch transformer --methods mlp fm score \
        --seeds 0 1 2
    python scripts/pusht/train_pusht_policy.py --arch transformer_big --methods score \
        --score-weights 2.5 --score-anneal-fracs 0 --seeds 0 1 2 3 4

``--arch`` picks the backbone: ``mlp`` concatenates ``2 rho - 1`` to its input, ``transformer``
injects the same phase by adaptive layer norm. ``transformer_medium`` is that backbone widened
to twice the parameters (1.8M) and otherwise unchanged, ``transformer_medium_deep`` the same
budget spent on depth (four blocks, 1.7M) and ``transformer_large`` both at once (3.4M).
``mlp_medium`` (768 x 4, 1.8M), ``mlp_medium_deep`` (512 x 8, 1.8M) and ``mlp_large`` (768 x 7,
3.6M) resize the MLP to match the medium and large transformers. ``transformer_big`` is the same idea at
~67M parameters -- the T tokenized as points on it so cross-attention has twelve keys rather than
two, and no self-attention, which is dead weight at one query token. Each backbone appends its own
suffix to the run name (``_mlpmed``, ``_mlpdeep``, ``_mlplarge``, ``_tf``, ``_tfmed``, ``_tfdeep``,
``_tflarge``, ``_tfbig``), so no two ever share a run directory and existing runs are
untouched. The transformer size flags below default to the selected backbone's own
values, so ``--arch transformer_big`` needs nothing else.

``mlp`` is state-only, ``fm`` adds the oracle phase ``t/(L-1)``, ``mlp_score`` is the state-only
network with the score regularizer (no phase input), ``score`` adds the agent-score regularizer with lambda annealed 0.05 -> 0 over the first half (``score_w0.05_cos0.5``). Inputs
are ``[o, x, rho]``. See :mod:`fm_bc.pusht.policy_training` for the pipeline. Writes
``{out}/pusht/policy/{method}/seed{s}/`` -- ``best.pt`` (lowest EMA validation loss),
``epoch_XXXX.pt`` every ``--checkpoint-every`` epochs, ``last.pt``, ``log.csv``, ``metrics.json``
-- and skips runs that already have ``metrics.json`` unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.pusht.data import PushTEpisodes  # noqa: E402
from fm_bc.pusht.policy import ARCHS  # noqa: E402
from fm_bc.pusht.policy_training import (  # noqa: E402
    PolicyConfig, is_mlp, method_name, policy_run_dir, train_pusht_policy,
)
from fm_bc.pusht.prepared import PushTPrepared  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episodes", type=Path, default=Path("data/pusht/pusht_episodes.npz"))
    p.add_argument("--prepared", type=Path, default=Path("data/pusht/pusht_prepared_9010.npz"),
                   help="split artifact: the 90/10 split, whose runs <out>/pusht/policy holds; "
                        "pusht_prepared_702010.npz is 70/20/10, pusht_prepared.npz DP's 90/4")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--methods", nargs="+", default=["mlp", "fm", "score"],
                   choices=["mlp", "fm", "score", "mlp_score"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--score-weights", type=float, nargs="+", default=[0.05],
                   help="several values sweep the score arm (with the two lists below)")
    p.add_argument("--score-anneal-fracs", type=float, nargs="+", default=[0.5])
    p.add_argument("--bandwidths-px", type=float, nargs="+", default=[32.0])
    p.add_argument("--obs-steps", type=int, default=1,
                   help="1: inputs [o, x, rho] at the current step; >1 prepends past [o, x]")
    p.add_argument("--normalizer", default="minmax", choices=["box", "minmax", "gaussian"])
    p.add_argument("--arch", default="mlp", choices=sorted(ARCHS),
                   help="backbone; 'mlp_medium' / 'mlp_medium_deep' / 'mlp_large' append "
                        "_mlpmed / _mlpdeep / _mlplarge to the run name, 'transformer' _tf, 'transformer_medium' _tfmed, "
                        "'transformer_medium_deep' _tfdeep, 'transformer_large' _tflarge, "
                        "'transformer_big' _tfbig")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512, 512],
                   help="--arch mlp only (mlp_medium, mlp_medium_deep and mlp_large are 768 x 4, "
                        "512 x 8 and 768 x 7)")
    p.add_argument("--activation", default="silu", help="--arch mlp only")
    tf = p.add_argument_group(
        "--arch transformer / transformer_medium / transformer_medium_deep / transformer_large / "
        "transformer_big",
        "Each defaults to its backbone's own value (transformer 128/2/4/128, transformer_medium "
        "180/2/4/180, transformer_medium_deep 128/4/4/128, transformer_large 180/4/4/180, "
        "transformer_big 768/8/12/256), so leaving these alone is what selects the "
        "published size. Passing one "
        "overrides it for a sweep.",
    )
    tf.add_argument("--hidden-dim", type=int, default=None)
    tf.add_argument("--num-layers", type=int, default=None)
    tf.add_argument("--num-heads", type=int, default=None)
    tf.add_argument("--phase-dim", type=int, default=None)
    tf.add_argument("--dropout", type=float, default=None,
                    help="0 keeps the arm deterministic in training too, as the MLP arms are")
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--lr-warmup-steps", type=int, default=500)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ep = PushTEpisodes.load(args.episodes, device=args.device)
    prep = PushTPrepared.load(args.prepared, device=args.device)
    # Only the selected backbone's kwargs reach the config, so an MLP run records no stray keys --
    # and only the ones actually passed, so an un-overridden run takes the backbone class's own
    # defaults. Spelling them out here instead would silently shrink --arch transformer_big back
    # to the small backbone's 128 x 2.
    _tf_flags = {
        "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "num_heads": args.num_heads, "phase_dim": args.phase_dim, "dropout": args.dropout,
    }
    net_kwargs = {} if is_mlp(args.arch) else {
        k: v for k, v in _tf_flags.items() if v is not None
    }
    arms = []  # (use_phase, score_weight, anneal_frac, bandwidth)
    for method in args.methods:
        if method in ("score", "mlp_score"):
            arms += [(method == "score", w, f, h) for w in args.score_weights
                     for f in args.score_anneal_fracs for h in args.bandwidths_px]
        else:
            arms.append((method != "mlp", 0.0, 0.5, 32.0))
    for use_phase, weight, frac, bandwidth in arms:
        for seed in args.seeds:
            cfg = PolicyConfig(
                use_phase=use_phase, score_weight=weight, score_anneal_frac=frac,
                bandwidth_px=bandwidth, obs_steps=args.obs_steps, normalizer=args.normalizer,
                arch=args.arch, net_kwargs=net_kwargs,
                hidden=tuple(args.hidden), activation=args.activation,
                num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay, lr_warmup_steps=args.lr_warmup_steps,
                use_ema=not args.no_ema, val_every=args.val_every,
                checkpoint_every=args.checkpoint_every, seed=seed, device=args.device,
            )
            out = policy_run_dir(args.out, cfg)
            if (out / "metrics.json").exists() and not args.overwrite:
                print(f"skip {out} (done)")
                continue
            print(f"{method_name(cfg)} seed {seed} -> {out}", flush=True)
            _, final = train_pusht_policy(ep, prep, cfg, out, verbose=False)
            b, last = final["best"], final["last"]
            print(f"done in {final['train_seconds']:.0f}s: best epoch {b['epoch']} val loss "
                  f"{b['val_loss_ema']:.3f}  val u rmse {b['val_split']['velocity_rmse']:.3f} "
                  f"(last {last['val_split']['velocity_rmse']:.3f}; "
                  f"u=0 {b['val_split']['zero_velocity_rmse']:.3f})", flush=True)

if __name__ == "__main__":
    main()
