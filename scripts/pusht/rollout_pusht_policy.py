"""Roll the trained Push-T policies out closed loop in gym-pusht and write their scores.

    python scripts/pusht/rollout_pusht_policy.py --methods mlp fm score_w2.5_cos0.5 --workers 12

For every ``{--runs}/{method}/seed*/best.pt`` this runs Diffusion Policy's test protocol
(50 envs, seeds 100000..100049, at most 300 steps, score = max reward, success = coverage > 0.95)
with the phase from :class:`fm_bc.pusht.sim.NominalClock` -- the median training-episode length
of the prepared split, or ``--phase-length`` steps to reach rho = 1 -- or, with
``--clock phase_mlp``, from the monotone state-based estimator of the same seed -- and the inverse-PD command (see :mod:`fm_bc.pusht.sim`). Writes
``rollout_{clock}_{command}.json`` and ``.npz`` next to the checkpoint; skips runs that have the
json unless ``--overwrite``. ``--checkpoint last.pt`` (the final, overfit model rather than the
lowest-validation-loss one) appends ``_last`` to the tag, so both evaluations coexist.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fm_bc.pusht.data import PushTEpisodes  # noqa: E402
from fm_bc.pusht.policy import load_pusht_policy  # noqa: E402
from fm_bc.pusht.prepared import PushTPrepared  # noqa: E402
from fm_bc.pusht.phase_mlp import PHASE_EPS, MonotonePhase, load_phase_mlp  # noqa: E402
from fm_bc.pusht.sim import COMMANDS, MAX_STEPS, TEST_SEEDS, NominalClock, rollout_policy  # noqa: E402


def _one(job: tuple) -> str:
    (ckpt, median_length, phase_length, command, n_envs, max_steps, tag, phase_ckpt,
     eps, smooth, max_rate, step, ratchet, device) = job
    torch.set_num_threads(1)
    model, _ = load_pusht_policy(ckpt, map_location=device)
    if phase_ckpt is None:
        clock = NominalClock(median_length, phase_length)
        described = {"clock": "nominal", "median_length": median_length,
                     "phase_length": clock.phase_length, "dphase": clock.dphase}
    else:
        estimator, _ = load_phase_mlp(phase_ckpt, map_location=device)
        clock = MonotonePhase(estimator, eps=eps, smooth=smooth, max_rate=max_rate, step=step,
                              ratchet=ratchet, device=device)
        described = {**clock.describe, "phase_model": str(phase_ckpt)}
    r = rollout_policy(model, clock, seeds=TEST_SEEDS[:n_envs], max_steps=max_steps,
                       command=command, device=device)
    out = Path(ckpt).parent
    metrics = {
        **described,
        "command": command, "checkpoint": Path(ckpt).name, "n_envs": n_envs,
        "max_steps": max_steps, "seeds": [int(s) for s in r["seeds"]],
        "mean_score": float(r["score"].mean()), "success_rate": float(r["success"].mean()),
        "mean_final_coverage": float(r["final_coverage"].mean()),
        "mean_steps": float(r["steps"].mean()),
        "score": r["score"].tolist(), "success": r["success"].tolist(),
    }
    np.savez_compressed(out / f"rollout_{tag}.npz", states=r["states"], actions=r["actions"],
                        score=r["score"], success=r["success"], steps=r["steps"],
                        seeds=r["seeds"])
    (out / f"rollout_{tag}.json").write_text(json.dumps(metrics, indent=2))
    return f"{out}: score {metrics['mean_score']:.3f}  success {metrics['success_rate']:.2f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs/pusht/policy"),
                   help="the paper's 90/10 runs by default, with their --prepared and --phase-runs; "
                        "the 70/20/10 runs are runs/pusht/policy_702010 with "
                        "data/pusht/pusht_prepared_702010.npz (see fm_bc/pusht/STATUS.md)")
    p.add_argument("--episodes", type=Path, default=Path("data/pusht/pusht_episodes.npz"))
    p.add_argument("--prepared", type=Path, default=Path("data/pusht/pusht_prepared_9010.npz"))
    p.add_argument("--methods", nargs="+",
                   default=["mlp", "mlp_score_w2.5_cos0.5", "fm", "score_w2.5_cos0.5"])
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--clock", default="nominal", choices=["nominal", "phase_mlp"],
                   help="nominal: the open-loop median-length clock (the baseline). phase_mlp: "
                        "the monotone state-based estimate of scripts/pusht/train_pusht_phase.py, "
                        "paired seed for seed with the policy")
    p.add_argument("--phase-runs", type=Path, default=Path("runs/pusht/phase_start"),
                   help="phase estimators for --clock phase_mlp (default: the start-pose ones "
                        "the paper's rollouts use)")
    p.add_argument("--phase-eps", type=float, default=PHASE_EPS)
    p.add_argument("--phase-smooth", type=float, default=1.0,
                   help="EMA coefficient on the estimate before the monotone ratchet (1 = raw)")
    p.add_argument("--phase-max-rate", type=float, default=None,
                   help="cap on one step's phase advance")
    p.add_argument("--phase-fixed-step", action="store_true",
                   help="decouple: the estimate conditions the policy, the step stays the median "
                        "demo's 1 / (L_med - 1)")
    p.add_argument("--phase-no-ratchet", action="store_true",
                   help="drop the monotone bound: the policy is conditioned on the raw estimate, "
                        "which may stand still or go back (needs --phase-fixed-step)")
    p.add_argument("--tag-suffix", default="", help="appended to the output file name")
    p.add_argument("--phase-length", type=float, default=None,
                   help="steps for rho to reach 1 (default: the median length); the step size "
                        "stays the median's")
    p.add_argument("--command", default="inverse_pd", choices=COMMANDS)
    p.add_argument("--n-envs", type=int, default=len(TEST_SEEDS))
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda",
                   help="where the policy and phase estimator run; the simulator is CPU either "
                        "way, so the small MLP arms gain nothing here and the big ones gain a lot")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.phase_no_ratchet and not args.phase_fixed_step:
        p.error("--phase-no-ratchet needs --phase-fixed-step: the estimate's own increment can be "
                "zero or negative")

    ep = PushTEpisodes.load(args.episodes)
    prep = PushTPrepared.load(args.prepared)
    median_length = NominalClock.from_episodes(ep.lengths[prep.train_ids]).median_length
    if args.clock == "phase_mlp":
        tag = f"phase_mlp{args.tag_suffix}_{args.command}"
    elif args.phase_length is None:
        tag = f"nominal_{args.command}"
    else:
        tag = f"nominal_L{args.phase_length:g}_{args.command}"
    if Path(args.checkpoint).stem != "best":  # keep best.pt's rollouts; e.g. last.pt -> *_last
        tag = f"{tag}_{Path(args.checkpoint).stem}"
    print(f"nominal clock: median training-episode length {median_length:g} "
          f"({len(prep.train_ids)} episodes) -> drho {1 / (median_length - 1):.5f}")
    jobs = []
    for method in args.methods:
        for ckpt in sorted((args.runs / method).glob(f"seed*/{args.checkpoint}")):
            if (ckpt.parent / f"rollout_{tag}.json").exists() and not args.overwrite:
                print(f"skip {ckpt.parent} (done)")
                continue
            phase_ckpt = None
            if args.clock == "phase_mlp":
                phase_ckpt = args.phase_runs / ckpt.parent.name / "model.pt"
                if not phase_ckpt.exists():
                    raise SystemExit(f"no phase estimator for {ckpt.parent.name}: {phase_ckpt} "
                                     "(run scripts/pusht/train_pusht_phase.py)")
                phase_ckpt = str(phase_ckpt)
            jobs.append((str(ckpt), median_length, args.phase_length, args.command, args.n_envs,
                         args.max_steps, tag, phase_ckpt, args.phase_eps, args.phase_smooth,
                         args.phase_max_rate,
                         1.0 / (median_length - 1.0) if args.phase_fixed_step else None,
                         not args.phase_no_ratchet, args.device))
    # CUDA does not survive fork, so the workers have to start clean. The parent never touches
    # the GPU, but a forked child inherits its whole address space and initializing CUDA there
    # deadlocks or errors; spawn gives each worker its own context.
    ctx = get_context("spawn") if str(args.device).startswith("cuda") else None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for line in pool.map(_one, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
