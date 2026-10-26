"""Roll the Push-T chunk baselines (DP, MIP, ACT) out closed loop, on the MLP arms' protocol.

    python scripts/pusht/rollout_pusht_baselines.py --runs runs/pusht/policy --workers 6

Diffusion Policy's test protocol (50 envs, seeds 100000..100049, at most 300 steps, score = max
reward, success = coverage > 0.95). A chunk baseline predicts the environment's action directly,
so there is no phase clock and no inverse PD (:func:`fm_bc.pusht.sim.rollout_chunk_policy`).
Both loops of the skate comparison: ``--loop open`` executes ``action_steps`` of each chunk (the
policy as trained, ``rollout_chunk.*``) and ``--loop closed`` re-plans every step and temporally
ensembles ACT-style (``rollout_chunk_te<m>.*``). Writes next to the checkpoint;
``--checkpoint last.pt`` (the final, overfit model) appends ``_last`` to the tag.

Note that the ``act`` arm is reported **open loop**: LeRobot's ACT defaults to
``temporal_ensemble_coeff = None``, i.e. execute the chunk as trained, and that is the
configuration the community benchmarks. ``--loop closed`` still works on it -- it is the ACT
paper's own ``temporal_agg`` -- but it is not the default here.
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

from fm_bc.pusht.baselines import load_pusht_baseline  # noqa: E402
from fm_bc.pusht.sim import MAX_STEPS, TEST_SEEDS, rollout_chunk_policy  # noqa: E402

TAG = "chunk"


def _one(job: tuple) -> str:
    ckpt, n_envs, max_steps, device, ensemble, tag = job
    torch.set_num_threads(2)
    model, _ = load_pusht_baseline(ckpt, map_location=device)
    r = rollout_chunk_policy(model, seeds=TEST_SEEDS[:n_envs], max_steps=max_steps, device=device,
                             temporal_ensemble=ensemble)
    out = Path(ckpt).parent
    metrics = {
        "clock": "none (chunk policy)", "command": "action",
        "loop": "closed (temporal ensembling)" if ensemble else "open (chunk as trained)",
        "temporal_ensemble": ensemble, "checkpoint": Path(ckpt).name,
        "n_envs": n_envs, "max_steps": max_steps, "obs_steps": model.obs_steps,
        "horizon": model.horizon, "action_steps": model.action_steps,
        "seeds": [int(s) for s in r["seeds"]],
        "mean_score": float(r["score"].mean()), "success_rate": float(r["success"].mean()),
        "mean_final_coverage": float(r["final_coverage"].mean()),
        "mean_steps": float(r["steps"].mean()),
        "score": r["score"].tolist(), "success": r["success"].tolist(),
    }
    np.savez_compressed(out / f"rollout_{tag}.npz", states=r["states"], actions=r["actions"],
                        score=r["score"], success=r["success"], steps=r["steps"], seeds=r["seeds"])
    (out / f"rollout_{tag}.json").write_text(json.dumps(metrics, indent=2))
    return f"{out}: score {metrics['mean_score']:.3f}  success {metrics['success_rate']:.2f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs/pusht/policy"))
    p.add_argument("--methods", nargs="+", default=["dp", "mip", "act"])
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--n-envs", type=int, default=len(TEST_SEEDS))
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--loop", default="open", choices=["open", "closed"],
                   help="open: execute action_steps of each chunk (as trained). closed: re-plan "
                        "every step and temporally ensemble, ACT-style")
    p.add_argument("--m", type=float, default=0.01, help="temporal-ensembling weight (closed loop)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ensemble = args.m if args.loop == "closed" else None
    tag = TAG if args.loop == "open" else f"{TAG}_te{args.m:g}"
    if Path(args.checkpoint).stem != "best":  # keep best.pt's rollouts; e.g. last.pt -> *_last
        tag = f"{tag}_{Path(args.checkpoint).stem}"
    jobs = []
    for method in args.methods:
        for ckpt in sorted((args.runs / method).glob(f"seed*/{args.checkpoint}")):
            if (ckpt.parent / f"rollout_{tag}.json").exists() and not args.overwrite:
                print(f"skip {ckpt.parent} (done)")
                continue
            jobs.append((str(ckpt), args.n_envs, args.max_steps, args.device, ensemble, tag))
    # CUDA does not survive fork: a forked worker inherits the parent's address space and
    # initializing CUDA there deadlocks or errors. Spawn gives each worker its own context.
    ctx = get_context("spawn") if str(args.device).startswith("cuda") else None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for line in pool.map(_one, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
