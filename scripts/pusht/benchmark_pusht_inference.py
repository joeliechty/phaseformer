"""Inference cost of each trained Push-T policy: latency per decision and per control step.

    python scripts/pusht/benchmark_pusht_inference.py --runs runs/pusht/policy \
        --out output/pusht_9010 --devices cuda cpu

A backbone is added to an existing ``inference.csv`` without re-measuring what is already in it
-- ``--baselines`` with no names skips the chunk baselines, ``--append`` keeps every row this
run does not itself produce (see ``TFBIG_ARMS``)::

    python scripts/pusht/benchmark_pusht_inference.py --append --baselines \
        --methods mlp_tfbig mlp_score_w2.5_const_tfbig fm_tfbig score_w2.5_const_tfbig

The methods differ in how many network calls one decision takes and how many control steps that
decision covers:

* ``mlp`` / ``fm`` / the score arms -- 1 call, 1 step. A phase-conditioned arm rolled out on the
  learned clock also runs the phase estimator every step, so its decision is timed *with* that
  forward pass (``phase_estimator`` says so).
* ``mip`` -- 2 calls (the two regression steps), 8 steps open loop, 1 closed loop.
* ``dp`` -- 100 calls (the DDPM reverse chain), 8 steps open loop, 1 closed loop.
* ``act`` -- 1 call (a single encoder/decoder pass at ``z = 0``), 8 steps open loop.

Latency is the median of ``--repeats`` timed calls at batch 1 (one robot, where kernel-launch
overhead dominates), after three warm-up calls. Writes ``inference.csv``; ``plot_pusht.py``
renders it into the Pareto figure.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.pusht.baselines import load_pusht_baseline  # noqa: E402
from fm_bc.pusht.phase_mlp import load_phase_mlp  # noqa: E402
from fm_bc.pusht.policy import load_pusht_policy  # noqa: E402

FIELDS = ["method", "loop", "params", "net_calls_per_decision", "steps_per_decision",
          "ms_decision_b1", "ms_step_b1", "phase_estimator", "device", "hardware", "torch"]
# Default arms only; --methods names any run directory. Whether the phase estimator is timed
# with a decision comes from the loaded model's use_phase, not from a list that has to be
# kept in step with every new arm name.
MLP_ARMS = ("mlp", "mlp_score_w2.5_cos0.5", "fm", "score_w2.5_cos0.5")
# The same four arms on the two transformer backbones, which carry policy_training.ARCH_SUFFIX
# in their run names and train on the constant-lambda schedule. Named here so the invocation is
# discoverable rather than retyped, but deliberately *not* in the --methods default: inference.csv
# is built up one backbone at a time with --append, and a broader default would rewrite the
# published rows on any run without it.
TF_ARMS = ("mlp_tf", "mlp_score_w2.5_const_tf", "fm_tf", "score_w2.5_const_tf")
TFBIG_ARMS = ("mlp_tfbig", "mlp_score_w2.5_const_tfbig", "fm_tfbig", "score_w2.5_const_tfbig")


def hardware_name(device: str) -> str:
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


def _time(fn, repeats: int, device: str) -> float:
    """Median seconds per call, after three warm-ups."""
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for _ in range(3):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs/pusht/policy"))
    p.add_argument("--phase-runs", type=Path, default=Path("runs/pusht/phase_start"))
    p.add_argument("--out", type=Path, default=Path("output/pusht_9010"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=21)
    p.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    p.add_argument("--methods", nargs="*", default=list(MLP_ARMS),
                   help="policy run directories to time (default: the MLP arms); pass with no "
                        "names to skip them, as --baselines already allows")
    p.add_argument("--baselines", nargs="*", default=["dp", "mip"],
                   help="chunk baselines to time; pass with no names to skip them")
    p.add_argument("--append", action="store_true",
                   help="merge into the existing inference.csv instead of replacing it, "
                        "keeping rows this run does not re-measure (same method, loop and "
                        "device). Use it to add an arm without perturbing published numbers")
    args = p.parse_args()

    state = torch.rand(1, 2, 5) * 400
    rows = []
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        common = {"device": device, "hardware": hardware_name(device), "torch": torch.__version__}
        clock = None
        phase_ckpt = args.phase_runs / f"seed{args.seed}" / "model.pt"
        if phase_ckpt.exists():
            clock, _ = load_phase_mlp(phase_ckpt, map_location=device)
            clock = clock.to(device).eval()
        for method in args.methods:
            ckpt = args.runs / method / f"seed{args.seed}" / "best.pt"
            if not ckpt.exists():
                continue
            model, _ = load_pusht_policy(ckpt, map_location=device)
            model = model.to(device).eval()
            obs = state[:, -model.obs_steps:].to(device)
            rho = torch.zeros(1, device=device)
            with_clock = clock is not None and model.use_phase

            def call(model=model, obs=obs, rho=rho, with_clock=with_clock):
                if with_clock:
                    clock(obs[:, -1], obs[:, :1])
                return model(obs, rho)

            ms = 1e3 * _time(call, args.repeats, device)
            rows.append({**common, "method": method, "loop": "per step",
                         "params": sum(q.numel() for q in model.parameters())
                         + (sum(q.numel() for q in clock.parameters()) if with_clock else 0),
                         "net_calls_per_decision": 2 if with_clock else 1,
                         "steps_per_decision": 1, "ms_decision_b1": round(ms, 4),
                         "ms_step_b1": round(ms, 4),
                         "phase_estimator": int(with_clock)})
        for kind in args.baselines:
            ckpt = args.runs / kind / f"seed{args.seed}" / "best.pt"
            if not ckpt.exists():
                continue
            model, _ = load_pusht_baseline(ckpt, map_location=device)
            model = model.to(device).eval()
            window = state.to(device)
            gen = torch.Generator(device=device).manual_seed(0)
            ms = 1e3 * _time(lambda m=model, w=window, g=gen: m.sample(w, g), args.repeats, device)
            calls = {"dp": None, "act": 1}.get(kind, 2)  # mip: the two regression steps
            if calls is None:
                calls = model.num_train_timesteps
            for loop, steps in (("open", model.action_steps), ("closed", 1)):
                rows.append({**common, "method": kind, "loop": loop,
                             "params": sum(q.numel() for q in model.parameters()),
                             "net_calls_per_decision": calls, "steps_per_decision": steps,
                             "ms_decision_b1": round(ms, 4), "ms_step_b1": round(ms / steps, 4),
                             "phase_estimator": 0})
            print(f"{device} {kind}: {ms:.2f} ms/decision", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "inference.csv"
    if args.append and csv_path.exists():
        fresh = {(r["method"], r["loop"], r["device"]) for r in rows}
        with open(csv_path, newline="") as f:
            kept = [r for r in csv.DictReader(f)
                    if (r["method"], r["loop"], r["device"]) not in fresh]
        print(f"append: keeping {len(kept)} existing rows, writing {len(rows)} new")
        rows = kept + rows
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
    for r in rows:
        # rows kept from the existing csv come back as strings, so coerce before formatting
        print(f"  {r['device']:5s} {r['method']:24s} {r['loop']:8s} "
              f"{float(r['ms_decision_b1']):8.2f} ms/decision  "
              f"{float(r['ms_step_b1']):8.3f} ms/step")


if __name__ == "__main__":
    main()
