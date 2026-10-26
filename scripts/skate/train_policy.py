"""Train one SE(2) flow-matching policy -- FM, FM plus a divergence regularizer with
``--cdm-weight > 0``, or the state-only MLP with ``--no-phase`` -- and evaluate it against the
held-out demonstrations.

    python scripts/skate/train_policy.py --scenario dual_path --cdm-weight 0.05 --device cuda
    python scripts/skate/train_policy.py --scenario dual_path --cdm-weight 0.05 --cdm-terms div \
        --cdm-schedule cosine --device cuda

``--cdm-terms`` picks the regularizer (``both`` is CDM ``E|a + b|``, ``div`` is ``E|a|``,
``score`` is ``E|b|``; see ``fm_bc.skate.training``); ``--cdm-schedule cosine`` anneals its weight
to 0 over the first ``--cdm-anneal-frac`` of training. ``--horizon 16 --action-steps 8`` is FM with
action chunking (MLP+phase+chunk, ``fm_chunk``): it predicts 16 controls from the current pose
and phase and executes 8 before re-planning, like Diffusion Policy and MIP; with
``--cdm-weight 0.05 --cdm-terms score`` the score regularizer is applied to every control of the
chunk at its open-loop state and phase (``chunk_score_w<λ>``, MLP+phase+AC+score).

Writes ``{out}/{scenario}/n{n_train}/{method}/seed{s}/`` -- ``method`` is ``fm``, ``mlp``,
``fm_chunk``, ``cdm_w<λ>``, ``div_w<λ>_cos<frac>``, ... -- with ``config.json``, ``log.csv``, ``final.pt``,
``metrics.json`` and ``rollouts.npz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate.dataset import TrajectoryDataset  # noqa: E402
from fm_bc.skate.scenarios import EXTRA_SCENARIOS, SCENARIOS  # noqa: E402
from fm_bc.skate.training import (  # noqa: E402
    CDM_SCHEDULES, CDM_TERMS, TrainConfig, prepare_training_set, run_dir, train,
)


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=Path("data"), help="training .npz dir")
    parser.add_argument("--test", type=Path, default=Path("data/test"), help="held-out .npz dir")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arch", default="mlp", choices=["mlp", "transformer"],
                        help="backbone; 'transformer' appends _tf to the run name")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256],
                        help="--arch mlp only")
    parser.add_argument("--activation", default="silu", choices=["silu", "relu", "tanh"],
                        help="--arch mlp only")
    tf = parser.add_argument_group("--arch transformer")
    tf.add_argument("--hidden-dim", type=int, default=128)
    tf.add_argument("--num-layers", type=int, default=2)
    tf.add_argument("--num-heads", type=int, default=4)
    tf.add_argument("--phase-dim", type=int, default=128)
    tf.add_argument("--dropout", type=float, default=0.0,
                    help="0 keeps the arm deterministic in training too, as the MLP arms are")
    parser.add_argument("--no-phase", action="store_true",
                        help="state-only network: drop the phase / flow-time input")
    parser.add_argument("--horizon", type=int, default=1,
                        help="controls per predicted chunk; 16 (with --action-steps 8) is the "
                             "MLP+phase+chunk arm")
    parser.add_argument("--action-steps", type=int, default=1,
                        help="controls of each chunk executed before re-planning")
    parser.add_argument("--cdm-schedule", default="constant", choices=CDM_SCHEDULES,
                        help="cosine: anneal the regularizer weight to 0 over --cdm-anneal-frac "
                             "of the steps, then train pure FM")
    parser.add_argument("--cdm-anneal-frac", type=float, default=0.5)
    parser.add_argument("--div-estimator", default="exact", choices=["exact", "hutchinson"])
    parser.add_argument("--n-probes", type=int, default=1)
    parser.add_argument("--k", type=int, default=16, help="kNN rank of the divergence target")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--mmd-bandwidth", type=float, default=0.3)
    parser.add_argument("--no-clamp", action="store_true",
                        help="execute raw commands instead of saturating them to the planner's "
                             "SkateLimits (not a physical robot)")
    parser.add_argument("--device", default="cuda")


def config_from_args(args, **overrides) -> TrainConfig:
    # Only the selected backbone's kwargs reach the config, so an MLP run records no stray keys.
    net_kwargs = {} if args.arch == "mlp" else {
        "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "num_heads": args.num_heads, "phase_dim": args.phase_dim, "dropout": args.dropout,
    }
    return TrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, hidden=tuple(args.hidden),
        arch=args.arch, net_kwargs=net_kwargs,
        activation=args.activation, use_phase=not args.no_phase,
        horizon=args.horizon, action_steps=args.action_steps,
        cdm_schedule=args.cdm_schedule, cdm_anneal_frac=args.cdm_anneal_frac,
        div_estimator=args.div_estimator, n_probes=args.n_probes,
        k=args.k, eval_every=args.eval_every, log_every=args.log_every,
        mmd_bandwidth=args.mmd_bandwidth, clamp_controls=not args.no_clamp, device=args.device,
        **overrides,
    )


def load_split(args, scenario: str) -> tuple[TrajectoryDataset, TrajectoryDataset]:
    return (TrajectoryDataset.load(args.data / f"{scenario}.npz"),
            TrajectoryDataset.load(args.test / f"{scenario}.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", required=True, choices=(*SCENARIOS, *EXTRA_SCENARIOS))
    parser.add_argument("--cdm-weight", type=float, default=0.0, help="0 = FM baseline")
    parser.add_argument("--cdm-terms", default="both", choices=CDM_TERMS,
                        help="regularizer: both = CDM E|a + b|, div = E|a|, score = E|b|")
    parser.add_argument("--n-train", type=int, default=None, help="default: every demo")
    parser.add_argument("--seed", type=int, default=0)
    add_train_args(parser)
    args = parser.parse_args()

    ds, test = load_split(args, args.scenario)
    cfg = config_from_args(args, cdm_weight=args.cdm_weight, cdm_terms=args.cdm_terms,
                           n_train=args.n_train, seed=args.seed)
    ds_train = prepare_training_set(ds, cfg)
    out = run_dir(args.out, args.scenario, len(ds_train), cfg)
    print(f"{args.scenario}: {len(ds_train)} demos, {out.parent.name}, "
          f"phase input {cfg.use_phase} -> {out}")
    _, final = train(ds_train, test, cfg, out)
    print(f"done in {final['train_seconds']:.0f}s: mmd {final['mmd_mean']:.4f}  "
          f"collision {final['collision_rate']:.3f}  success {final['success_rate']:.3f}  "
          f"modes {final['mode_fractions']}  div mae {final['div_curve_mae']:.3f}")


if __name__ == "__main__":
    main()
