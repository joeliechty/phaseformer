"""Inference latency of every Push-T backbone x phase clock, and the chunk baselines.

    python scripts/pusht/benchmark_pusht_inference_grid.py \
        --out output/pusht_9010/inference_grid.csv

``benchmark_pusht_inference.py``'s protocol -- batch 1, a ``(1, 2, 5)`` state window, the clock's
forward pass timed with the policy's on every control step, three warm-up calls then the median of
``--repeats`` -- over the whole grid the scale study rolled out: the eight backbones (TF, TF-med
wide, TF-med deep, TF-large, MLP, MLP-med wide, MLP-med deep, MLP-large) with no clock (the
phase-free arm) and with each of the eight clocks (MLP small / med wide / med deep / large and their
transformer twins), on GPU and CPU, plus the same step replayed as a CUDA graph. The phase+score
and phase-free score arms are timed; the arms without score run the identical network.

Every number is the median of ``--passes`` passes, taken outermost so drift spreads evenly. The
policies take ``--repeats`` calls a pass (101: their sub-millisecond CPU times are too noisy at the
21 the other benchmark uses), the baselines ``--baseline-repeats`` (21; Diffusion Policy 5 on the
GPU and 3 on the CPU, at 0.45 s and 3.5 s a decision). Diffusion Policy's sampler builds its t = 0
constant from a Python float, which a CUDA graph cannot capture, so its graph replays an identical
copy with that constant built beforehand (same operations, same order, same outputs on the same
noise).

The committed ``inference_grid.csv`` was measured with these settings on the RTX 4090 / Ryzen 7
5700X workstation, the policies and baselines in two sittings.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.pusht.baselines import load_pusht_baseline  # noqa: E402
from fm_bc.pusht.phase_mlp import load_phase_mlp  # noqa: E402
from fm_bc.pusht.policy import load_pusht_policy  # noqa: E402

BACKBONES = {"tf": "_tf", "tfmed": "_tfmed", "tfdeep": "_tfdeep", "tflarge": "_tflarge",
             "mlp": "", "mlpmed": "_mlpmed", "mlpdeep": "_mlpdeep", "mlplarge": "_mlplarge"}
CLOCKS = {"512": "phase_start", "724": "phase_start_w724", "d3": "phase_start_d3",
          "large": "phase_start_w724_d3", "xf": "phase_start_xf", "xfwide": "phase_start_xf_wide",
          "xfdeep": "phase_start_xf_deep", "xflarge": "phase_start_xf_large"}
FIELDS = ["backbone", "clock", "params", "gpu", "gpu_graph", "cpu", "hardware"]


def median_ms(fn, device: str, repeats: int) -> float:
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
    return 1e3 * sorted(times)[len(times) // 2]


def graphed(fn) -> torch.cuda.CUDAGraph:
    """``fn`` captured as a CUDA graph, after warm-up on a side stream."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def dp_sample_graphsafe(m, obs: torch.Tensor, one: torch.Tensor) -> torch.Tensor:
    """``PushTDiffusionPolicy.sample`` with its t = 0 constant built before capture."""
    x = m._noise(obs, None)
    ac = m.alphas_cumprod
    for t in reversed(range(m.num_train_timesteps)):
        eps = m.eps(x, torch.full((x.shape[0],), t, device=x.device), obs)
        a_t = ac[t]
        a_prev = ac[t - 1] if t > 0 else one
        b_t, b_prev = 1 - a_t, 1 - a_prev
        alpha, beta = a_t / a_prev, 1 - a_t / a_prev
        x0 = ((x - b_t.sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        x = (a_prev.sqrt() * beta / b_t) * x0 + (alpha.sqrt() * b_prev / b_t) * x
        if t > 0:
            x = x + (b_prev / b_t * beta).clamp(min=1e-20).sqrt() * m._noise(obs, None)
    return m.act.invert(x)


def build(runs: Path, phase_root: Path, device: str):
    """``(key, fn, n_params, graph-safe fn)`` for every configuration on ``device``."""
    torch.manual_seed(0)
    state = torch.rand(1, 2, 5, device=device) * 400
    rho = torch.zeros(1, device=device)
    configs = []
    for b, suffix in BACKBONES.items():
        for run, clock in [(f"mlp{suffix}", None)] + [(f"score_w2.5_const{suffix}", c)
                                                     for c in CLOCKS]:
            pol = load_pusht_policy(runs / run / "seed0" / "best.pt")[0].to(device).eval()
            obs = state[:, -pol.obs_steps:]
            clk = (load_phase_mlp(phase_root / CLOCKS[clock] / "seed0" / "model.pt")[0]
                   .to(device).eval() if clock else None)

            def fn(pol=pol, obs=obs, clk=clk):
                if clk is not None:
                    clk(obs[:, -1], obs[:, :1])
                return pol(obs, rho)
            n = sum(p.numel() for p in pol.parameters()) + (
                sum(p.numel() for p in clk.parameters()) if clk else 0)
            configs.append(((b, clock or "none"), fn, n, fn))
    for kind in ("dp", "mip", "act"):
        m = load_pusht_baseline(runs / kind / "seed0" / "best.pt")[0].to(device).eval()
        safe = fn = (lambda m=m: m.sample(state, None))
        if kind == "dp":
            one = m.alphas_cumprod.new_tensor(1.0)
            safe = lambda m=m, one=one: dp_sample_graphsafe(m, state, one)  # noqa: E731
        configs.append(((kind, "chunk"), fn, sum(p.numel() for p in m.parameters()), safe))
    return configs


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs/pusht/policy"))
    p.add_argument("--phase-root", type=Path, default=Path("runs/pusht"))
    p.add_argument("--out", type=Path, default=Path("output/pusht_9010/inference_grid.csv"))
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--repeats", type=int, default=101, help="timed calls a pass, policies")
    p.add_argument("--baseline-repeats", type=int, default=21, help="timed calls a pass, MIP/ACT")
    args = p.parse_args()

    gpu, cpu = build(args.runs, args.phase_root, "cuda"), build(args.runs, args.phase_root, "cpu")
    graphs = {key: graphed(safe) for key, _, _, safe in gpu}
    params = {key: n for key, _, n, _ in gpu}
    reps = {"cuda": {"dp": 5}, "cpu": {"dp": 3}}
    results: dict = {}
    for i in range(args.passes):
        for device, configs in (("cuda", gpu), ("cpu", cpu)):
            for key, fn, _, _ in configs:
                n = reps[device].get(key[0], args.baseline_repeats if key[1] == "chunk"
                                     else args.repeats)
                results.setdefault((key, "gpu" if device == "cuda" else "cpu"), []).append(
                    median_ms(fn, device, n))
                if device == "cuda":
                    results.setdefault((key, "gpu_graph"), []).append(
                        median_ms(graphs[key].replay, device, n))
        print(f"pass {i + 1} of {args.passes} done", flush=True)
    hardware = f"{torch.cuda.get_device_name(0)} / {torch.get_num_threads()} CPU threads"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for key in params:
            w.writerow({"backbone": key[0], "clock": key[1], "params": params[key],
                        **{m: round(st.median(results[(key, m)]), 4)
                           for m in ("gpu", "gpu_graph", "cpu")},
                        "hardware": hardware})
    print(f"wrote {args.out} ({len(params)} rows)")


if __name__ == "__main__":
    main()
