"""MMD of the trained Push-T policies' rollouts to the held-out test demonstrations.

    python scripts/pusht/evaluate_pusht_mmd.py --workers 14

For every ``{--runs}/{method}/seed*/best.pt``: roll out from the start state of each test
demonstration for that demonstration's length with its own clock
(:func:`fm_bc.pusht.sim.rollout_from_states`), then compare the rollouts with the demonstrations at
matched phase (:mod:`fm_bc.pusht.evaluation`). Writes ``mmd_demo_clock.json`` and ``.npz`` next to
the checkpoint, and ``{--runs}/mmd_reference.json`` with two references on the same test
set: the validation demonstrations (the finite-sample floor of real demonstrations) and a pusher
that never moves (the start states held for the whole episode).

It needs a test split, so it defaults to the 70/20/10 runs in ``runs/pusht/policy_702010``; the
90/10 split (``runs/pusht/policy``) has none -- its rollouts are the test.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fm_bc.pusht.data import PushTEpisodes  # noqa: E402
from fm_bc.pusht.evaluation import MMD_BANDWIDTH_PX, per_phase_mmd  # noqa: E402
from fm_bc.pusht.policy import load_pusht_policy  # noqa: E402
from fm_bc.pusht.prepared import PushTPrepared  # noqa: E402
from fm_bc.pusht.sim import rollout_from_states  # noqa: E402

OUT = "mmd_demo_clock"


def _demos(ep: PushTEpisodes, ids) -> list[torch.Tensor]:
    return [ep.state[ep.episode_slice(int(e))].to(torch.float64) for e in ids]


def _one(job: tuple) -> str:
    ckpt, episodes, prepared = job
    torch.set_num_threads(1)
    ep, prep = PushTEpisodes.load(episodes), PushTPrepared.load(prepared)
    demos = _demos(ep, prep.test_ids)
    model, _ = load_pusht_policy(ckpt)
    trajs = rollout_from_states(model, [d[0].numpy() for d in demos], [len(d) for d in demos])
    r = per_phase_mmd([torch.as_tensor(t) for t in trajs], demos)
    out = Path(ckpt).parent
    lengths = np.array([len(t) for t in trajs])
    padded = np.zeros((len(trajs), lengths.max(), 5))
    for i, t in enumerate(trajs):
        padded[i, :len(t)], padded[i, len(t):] = t, t[-1]
    np.savez_compressed(out / f"{OUT}.npz", states=padded, lengths=lengths,
                        episodes=prep.test_ids.numpy())
    (out / f"{OUT}.json").write_text(json.dumps(
        {**r, "checkpoint": Path(ckpt).name, "n_episodes": len(trajs), "clock": "demo"}, indent=2))
    return f"{out}: mmd_mean {r['mmd_mean']:.4f}  mmd_final {r['mmd_final']:.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs/pusht/policy_702010"))
    p.add_argument("--episodes", type=Path, default=Path("data/pusht/pusht_episodes.npz"))
    p.add_argument("--prepared", type=Path, default=Path("data/pusht/pusht_prepared_702010.npz"))
    p.add_argument("--methods", nargs="+",
                   default=["mlp", "mlp_score_w2.5_cos0.5", "fm", "score_w2.5_cos0.5"])
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ep, prep = PushTEpisodes.load(args.episodes), PushTPrepared.load(args.prepared)
    test = _demos(ep, prep.test_ids)
    val = per_phase_mmd(_demos(ep, prep.val_ids), test)
    still = per_phase_mmd([d[:1].repeat(len(d), 1) for d in test], test)
    ref = {"bandwidth_px": MMD_BANDWIDTH_PX, "n_test": len(test),
           "validation_demos": {k: val[k] for k in ("mmd_mean", "mmd_final")},
           "no_motion": {k: still[k] for k in ("mmd_mean", "mmd_final")}}
    (args.runs / "mmd_reference.json").write_text(json.dumps(ref, indent=2))
    print(f"references: validation demos {val['mmd_mean']:.4f} / {val['mmd_final']:.4f}, "
          f"no motion {still['mmd_mean']:.4f} / {still['mmd_final']:.4f}")

    jobs = []
    for method in args.methods:
        for ckpt in sorted((args.runs / method).glob(f"seed*/{args.checkpoint}")):
            if (ckpt.parent / f"{OUT}.json").exists() and not args.overwrite:
                print(f"skip {ckpt.parent} (done)")
                continue
            jobs.append((str(ckpt), str(args.episodes), str(args.prepared)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_one, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
