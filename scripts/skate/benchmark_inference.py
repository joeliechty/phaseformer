"""Inference cost of each trained policy: latency per decision, per control step, and per
closed-loop rollout.

    python scripts/skate/benchmark_inference.py --runs runs/ --out output/skate/policy/ \
        --device cuda

The methods differ in how many network calls one decision takes and how many control steps that
decision covers:

* ``fm`` / ``mlp`` -- 1 call, 1 step: the policy is evaluated afresh at every control step.
* ``fm_chunk`` -- 1 call, 8 steps: FM's network predicting an action chunk.
* ``mip`` -- 2 calls (the two regression steps), 8 steps: one action chunk, executed before the
  next re-plan.
* ``dp`` -- 100 calls (the DDPM reverse chain), 8 steps.
* ``meanflow`` -- 1 call, 8 steps.

Latency is reported at batch 1 (one robot, where kernel-launch overhead dominates) and at the
evaluation's own batch, and as the wall-clock of a full ``--horizon``-step rollout. Writes
``inference.csv`` next to the other aggregates; ``evaluate_policies.py`` renders it into
SUMMARY.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate.baselines import ChunkPolicy, load_policy  # noqa: E402
from fm_bc.skate.dataset import TrajectoryDataset  # noqa: E402
from fm_bc.skate.evaluation import rollout_chunk_policy, rollout_policy  # noqa: E402
from fm_bc.skate.training import BASELINE_METHODS, CHUNK_METHOD  # noqa: E402

FIELDS = ["method", "params", "net_calls_per_decision", "steps_per_decision", "ms_decision_b1",
          "ms_step_b1", "ms_decision_bN", "ms_step_bN", "batch_n", "s_rollout", "horizon",
          "device", "hardware", "torch"]
# Network calls one decision costs; None: read it off the policy (DDPM's chain length).
_CALLS = {"fm": 1, "mlp": 1, CHUNK_METHOD: 1, "mip": 2, "meanflow": 1, "act": 1, "dp": None}


def hardware_name(device: str) -> str:
    """What the timings were measured on: the GPU's name, or the CPU model and thread count."""
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{model or 'CPU'} ({torch.get_num_threads()} threads)"


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _time(fn, repeats: int, device: str) -> float:
    """Median seconds per call of ``fn``, after 3 warm-up calls."""
    for _ in range(3):
        fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


@torch.no_grad()
def benchmark(model, poses: torch.Tensor, batches: tuple[int, int], repeats: int, horizon: int,
              dt: float, device: str) -> dict:
    """One row: decision latency at both batch sizes and a full rollout."""
    chunked = isinstance(model, ChunkPolicy)
    gen = torch.Generator(device=device).manual_seed(0)
    row = {}
    for tag, b in zip(("b1", "bN"), batches):
        obs = poses[:b] if poses.shape[0] >= b else poses[:1].expand(b, -1).contiguous()
        if chunked:
            window = obs[:, None, :].expand(b, model.obs_steps, 3).contiguous()
            fn = lambda: model.sample(window, gen)  # noqa: B023
        else:
            phase = obs.new_zeros(b)
            fn = lambda: model(obs, phase)  # noqa: B023
        row[f"ms_decision_{tag}"] = round(1e3 * _time(fn, repeats, device), 4)
    steps = model.action_steps  # 1 for FM and the MLP
    if chunked:
        roll = lambda: rollout_chunk_policy(model, poses, horizon, dt, None, gen)
    else:
        roll = lambda: rollout_policy(model, poses, horizon, dt, None)
    row["s_rollout"] = round(_time(roll, max(3, repeats // 5), device), 4)
    row["steps_per_decision"] = steps
    row["ms_step_b1"] = round(row["ms_decision_b1"] / steps, 4)
    row["ms_step_bN"] = round(row["ms_decision_bN"] / steps, 4)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--test", type=Path, default=Path("data/test"))
    parser.add_argument("--out", type=Path, default=Path("output/skate/policy"))
    parser.add_argument("--scenario", default="dual_path", help="which runs' checkpoints to time")
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1024, help="the evaluation's batch")
    parser.add_argument("--rollout-batch", type=int, default=None,
                        help="rollout width (default: the whole held-out set, as evaluation does)")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--append", action="store_true",
                        help="keep rows already in inference.csv, replacing this device's own")
    parser.add_argument("--methods", nargs="+", default=list(_CALLS), choices=list(_CALLS),
                        help="time only these (with --append, the other rows are kept)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.runs / args.scenario / f"n{args.n_train}"
    test = TrajectoryDataset.load(args.test / f"{args.scenario}.npz")
    poses = torch.as_tensor(test.poses[:, 0], dtype=torch.float32, device=args.device)
    if args.rollout_batch is not None:
        poses = poses[:args.rollout_batch]
    horizon = test.controls.shape[1]
    hardware = hardware_name(args.device)

    rows = []
    for method in [m for m in ["mlp", "fm", CHUNK_METHOD, *BASELINE_METHODS] if m in args.methods]:
        ckpt = root / method / f"seed{args.seed}" / "final.pt"
        if not ckpt.exists():
            print(f"skip {method}: no {ckpt}")
            continue
        model, _ = load_policy(ckpt, map_location=args.device)
        model = model.to(args.device).eval()
        calls = _CALLS[method]
        if calls is None:
            calls = model.num_train_timesteps
        row = {
            "method": method,
            "params": sum(p.numel() for p in model.parameters()),
            "net_calls_per_decision": calls,
            "batch_n": args.batch, "horizon": horizon, "device": args.device,
            "hardware": hardware, "torch": torch.__version__,
            **benchmark(model, poses, (1, args.batch), args.repeats, horizon, test.dt,
                        args.device),
        }
        rows.append(row)
        print(f"{method:>9}: {row['params'] / 1e6:6.2f}M params  {calls:>3} calls/decision  "
              f"{row['ms_decision_b1']:8.2f} ms/decision (b1)  {row['ms_step_b1']:8.2f} ms/step  "
              f"{row['ms_decision_bN']:8.2f} ms/decision (b{args.batch})  "
              f"{row['s_rollout']:6.2f} s/rollout")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "inference.csv"
    if args.append and path.exists():
        fresh = {(r["method"], r["device"]) for r in rows}
        with open(path) as f:
            kept = [r for r in csv.DictReader(f) if (r["method"], r["device"]) not in fresh]
        rows = kept + rows
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")
    (args.out / "inference.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
