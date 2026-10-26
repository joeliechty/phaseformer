"""Paper figures justifying kNN (k=16) over a fixed-bandwidth KDE for SE(2) divergence targets.

    python scripts/skate/paper_divergence_figures.py --data data/ --out output/skate/paper/ \
        --device cuda

Writes PDF (fonts embedded as TrueType, as IEEE requires), a PNG preview, and a CSV of the
plotted numbers for every figure:

  fig_div_bias       synthetic flows of known divergence.
                     (a) fraction of the true divergence recovered vs. the spread of the
                         contracting direction, with the closed-form KDE attenuation;
                     (b) per-sample bias vs. noise over k (kNN) and h (KDE).
  fig_div_datasets   the four estimators on the Fig. 6 datasets: mean divergence over phase
                     (top) and its spread across demonstrations (bottom).
  fig_div_corridor   scenario A colored by divergence, kNN k=16 vs. KDE h=0.2.
  table_div_estimators.tex   synthetic accuracy + dataset signal-to-spread, booktabs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate import SCENARIOS, TrajectoryDataset  # noqa: E402
from fm_bc.benchmarks import kde_attenuation, linear_flow  # noqa: E402
from fm_bc.divergence import divergence_kde, divergence_knn  # noqa: E402
from skate_planner.viz import draw_world  # noqa: E402

# Two families, two hues (validated categorical slots 1-2); the parameter is carried by line
# weight + dash so identity never rests on color alone.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#e34948"])
DASH = (0, (3.5, 2.0))
DOUBLE_COL = 7.16  # IEEE two-column text width, inches

# name -> (kind, parameter, color, linestyle, linewidth, marker)
ESTIMATORS = {
    "kNN k=16": ("knn", 16, BLUE, "-", 1.8, "o"),
    "kNN k=4": ("knn", 4, BLUE, DASH, 1.0, "o"),
    "KDE h=0.2": ("kde", 0.2, ORANGE, "-", 1.8, "s"),
    "KDE h=0.1": ("kde", 0.1, ORANGE, DASH, 1.0, "s"),
}

FLOWS = {  # name -> (tangent rates, true divergence)
    "isotropic expansion": ((0.4, 0.4, 0.4), 1.2),
    "y compression": ((0.0, -0.4, 0.0), -0.4),
    "heading compression": ((0.0, 0.0, -0.4), -0.4),
    "area-preserving strain": ((0.4, -0.4, 0.0), 0.0),
}
SWEEP_K = (2, 4, 8, 16, 32, 64)
SWEEP_H = (0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4)
SCENARIO_TITLES = {"single_path": "A) single path", "dual_goal": "B) dual goal",
                   "dual_path": "C) dual path"}


def _label(name: str) -> str:
    kind, param = name.split()
    sym, val = param.split("=")
    return f"{kind} ${sym}$={val}"


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": AXIS,
        "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "axes.titlecolor": INK, "legend.frameon": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.dpi": 300, "mathtext.default": "regular",
    })


def run(kind: str, param: float, poses: torch.Tensor, dt: float) -> torch.Tensor:
    if kind == "knn":
        return divergence_knn(poses, dt, k=int(param))
    return divergence_kde(poses, dt, bandwidth=float(param))


def estimate(name: str, poses: torch.Tensor, dt: float) -> torch.Tensor:
    kind, param = ESTIMATORS[name][:2]
    return run(kind, param, poses, dt)


def _save(fig, out: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out / stem}.pdf/.png")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


# -- experiments ------------------------------------------------------------------------------


def spread_sweep(device: str, seeds: int) -> list[dict]:
    """Compress one tangent direction at a fixed rate while sweeping its spread."""
    rate, t_end, n_steps = -0.3, 1.0, 11
    rows = []
    for s0 in np.geomspace(0.02, 1.0, 11):
        sigma_mid = float(s0 * np.exp(rate * t_end / 2))
        for seed in range(seeds):
            poses, dt = linear_flow((rate, 0.0, 0.0), sigma=(float(s0), 0.3, 0.15),
                                    t_end=t_end, n_steps=n_steps, seed=seed, device=device)
            for name in ESTIMATORS:
                ratio = estimate(name, poses, dt).mean().item() / rate
                rows.append({"sigma": sigma_mid, "estimator": name, "seed": seed,
                             "fraction_recovered": ratio})
    return rows


def tradeoff_sweep(device: str, seeds: int) -> list[dict]:
    rows = []
    for flow, (rates, truth) in FLOWS.items():
        for seed in range(seeds):
            poses, dt = linear_flow(rates, seed=seed, device=device)
            for kind, params in (("knn", SWEEP_K), ("kde", SWEEP_H)):
                for p in params:
                    d = run(kind, p, poses, dt)
                    rows.append({"flow": flow, "truth": truth, "kind": kind, "param": p,
                                 "seed": seed, "mean": d.mean().item(),
                                 "noise": d.std().item()})
    return rows


def summarize_tradeoff(rows: list[dict]) -> dict[tuple[str, float], dict]:
    """Per (kind, param): per-flow mean/noise over seeds, and RMS bias/noise over flows."""
    out: dict[tuple[str, float], dict] = {}
    for kind, params in (("knn", SWEEP_K), ("kde", SWEEP_H)):
        for p in params:
            per_flow = {}
            for flow, (_, truth) in FLOWS.items():
                sel = [r for r in rows if r["kind"] == kind and r["param"] == p
                       and r["flow"] == flow]
                per_flow[flow] = {"mean": np.mean([r["mean"] for r in sel]),
                                  "noise": np.mean([r["noise"] for r in sel]), "truth": truth}
            bias = np.array([v["mean"] - v["truth"] for v in per_flow.values()])
            noise = np.array([v["noise"] for v in per_flow.values()])
            out[(kind, p)] = {"per_flow": per_flow, "rms_bias": float(np.sqrt(np.mean(bias**2))),
                              "rms_noise": float(np.sqrt(np.mean(noise**2)))}
    return out


def dataset_curves(data_dir: Path, device: str):
    """Per scenario and estimator: mean over phase (mode means averaged) and within-mode spread."""
    curves, fields, datasets = {}, {}, {}
    for name in SCENARIOS:
        ds = TrajectoryDataset.load(data_dir / f"{name}.npz")
        datasets[name] = ds
        poses = torch.as_tensor(ds.poses, device=device)
        for est in ESTIMATORS:
            d = estimate(est, poses, ds.dt).cpu().numpy()
            modes = range(ds.spec.n_modes)
            mu = np.mean([d[ds.mode == m].mean(0) for m in modes], axis=0)
            band = np.mean([d[ds.mode == m].std(0) for m in modes], axis=0)
            curves[(name, est)] = (ds.phase, mu, band)
            fields[(name, est)] = d
    return curves, fields, datasets


def corridor_spread(ds: TrajectoryDataset) -> tuple[float, float]:
    """Narrowest minor-axis std of the xy marginal over phase, and the phase where it occurs."""
    minor = [np.sqrt(np.linalg.eigvalsh(np.cov(ds.poses[:, t, :2].T))[0])
             for t in range(ds.poses.shape[1])]
    t = int(np.argmin(minor))
    return float(minor[t]), float(ds.phase[t])


# -- figures ----------------------------------------------------------------------------------


def fig_bias(spread_rows, summary, pinch, out: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.7),
                                     gridspec_kw={"width_ratios": [1.05, 1]})

    # (a) fraction recovered vs spread
    grid = np.geomspace(0.012, 1.1, 200)
    for name, (kind, param, color, ls, lw, marker) in ESTIMATORS.items():
        sel = [r for r in spread_rows if r["estimator"] == name]
        sig = np.unique([r["sigma"] for r in sel])
        frac = np.array([np.mean([r["fraction_recovered"] for r in sel if r["sigma"] == s])
                         for s in sig])
        ax_a.plot(sig, frac, ls=ls, lw=lw, color=color, marker=marker, ms=3.5,
                  mfc="white" if ls == DASH else color, mec=color, mew=0.9,
                  label=_label(name), zorder=4)
        if kind == "kde":
            ax_a.plot(grid, kde_attenuation(param / grid), ls=ls, lw=0.7, color=INK2,
                      alpha=0.8, zorder=3)
    ax_a.plot([], [], lw=0.7, color=INK2, label="KDE theory, Gaussian")
    ax_a.axhline(1.0, color=INK2, lw=0.6, zorder=2)
    # Labels sit in the empty band at y ~ 0.3: left of the corridor line, right of the heading
    # line (every curve is above 0.5 there).
    ax_a.axvline(pinch, color=MUTED, lw=0.6, zorder=1)
    ax_a.text(pinch / 1.06, 0.30, "corridor\n(A)", fontsize=6.5, color=INK2, ha="right",
              va="center")
    ax_a.axvline(0.15, color=MUTED, lw=0.6, zorder=1)
    ax_a.text(0.15 * 1.06, 0.30, "start heading\nspread", fontsize=6.5, color=INK2,
              ha="left", va="center")
    ax_a.set_xscale("log")
    ax_a.set_xlim(grid[0], grid[-1])
    ax_a.set_ylim(0, 1.15)
    ax_a.set_xlabel(r"spread $\sigma$ of the contracting direction [m or rad]")
    ax_a.set_ylabel("estimated / true divergence")
    ax_a.set_title("(a) recovered fraction of a known compression")

    # (b) bias vs noise. Linear bias axis: kNN hugs zero, KDE lifts off it.
    for kind, params, color, marker in (("knn", SWEEP_K, BLUE, "o"),
                                        ("kde", SWEEP_H, ORANGE, "s")):
        x = [summary[(kind, p)]["rms_noise"] for p in params]
        y = [summary[(kind, p)]["rms_bias"] for p in params]
        ax_b.plot(x, y, "-", color=color, lw=1.0, marker=marker, ms=3.5, zorder=4,
                  label="kNN, varying $k$" if kind == "knn" else "KDE, varying $h$")
        for p, xi, yi in zip(params, x, y):
            chosen = (kind, p) in (("knn", 16), ("kde", 0.2))
            if kind == "knn" and not chosen and p not in (2, 64):
                continue  # the kNN points crowd near zero; label the ends and the pick
            if kind == "knn":
                offset, ha, va = (0, 5), "center", "bottom"
            elif xi > 1.0:
                offset, ha, va = (-5, 0), "right", "center"
            else:
                offset, ha, va = (5, 0), "left", "center"
            txt = f"$k$={p}" if kind == "knn" else f"$h$={p}"
            ax_b.annotate(txt, (xi, yi), textcoords="offset points", xytext=offset,
                          ha=ha, va=va, fontsize=6.5, color=INK if chosen else INK2,
                          fontweight="bold" if chosen else "normal")
            if chosen:
                ax_b.plot(xi, yi, marker=marker, ms=8, mfc="none", mec=color, mew=1.2,
                          zorder=5)
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.12, 2.5)
    ax_b.set_ylim(0, 0.62)
    ax_b.set_xlabel("per-sample noise (std) [1/s]")
    ax_b.set_ylabel("|bias|, RMS over flows [1/s]")
    ax_b.set_title("(b) bias vs. noise on four flows of known divergence")
    ax_b.legend(loc="upper center")

    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.9))
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.4, columnspacing=1.4)
    _save(fig, out, "fig_div_bias")


def fig_datasets(curves, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 3.7), sharex=True, sharey="row")
    for j, name in enumerate(SCENARIOS):
        for est, (_, _, color, ls, lw, _) in ESTIMATORS.items():
            phase, mu, band = curves[(name, est)]
            axes[0, j].plot(phase, mu, ls=ls, lw=lw, color=color, label=_label(est))
            axes[1, j].plot(phase, band, ls=ls, lw=lw, color=color)
        axes[0, j].axhline(0.0, color=INK2, lw=0.6, zorder=1)
        axes[0, j].set_title(SCENARIO_TITLES[name])
        axes[1, j].set_xlabel(r"phase $\rho$")
        axes[1, j].set_xlim(0, 1)
    axes[1, 0].set_ylim(bottom=0)
    axes[0, 0].set_ylabel(r"mean $\nabla\cdot u$ [1/s]")
    axes[1, 0].set_ylabel("spread across demos [1/s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03),
               handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out, "fig_div_datasets")


def fig_corridor(ds: TrajectoryDataset, fields, out: Path, n_paths: int = 350) -> None:
    pair = ("kNN k=16", "KDE h=0.2")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.75))
    idx = np.sort(np.random.default_rng(0).choice(len(ds), min(n_paths, len(ds)),
                                                  replace=False))
    vlim = float(np.percentile(np.abs(fields[("single_path", pair[0])]), 98))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
    xy = ds.poses[idx, :, :2]
    segs = np.stack([xy[:, :-1], xy[:, 1:]], axis=2).reshape(-1, 2, 2)
    xmin, ymin, xmax, ymax = ds.spec.world.bounds
    for ax, est in zip(axes, pair):
        d = fields[("single_path", est)][idx]
        lc = LineCollection(segs, cmap=DIV_CMAP, norm=norm, linewidths=0.8, alpha=0.8)
        lc.set_array((0.5 * (d[:, :-1] + d[:, 1:])).reshape(-1))
        ax.add_collection(lc)
        draw_world(ax, ds.spec.world)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_title(_label(est))
    axes[0].set_ylabel("y [m]")
    cb = fig.colorbar(lc, ax=axes, fraction=0.025, pad=0.02, extend="both")
    cb.set_label(r"$\nabla\cdot u$ [1/s]")
    cb.outline.set_visible(False)
    _save(fig, out, "fig_div_corridor")


def write_table(summary, curves, out: Path) -> None:
    flows = list(FLOWS)
    lines = [
        r"\begin{tabular}{l" + "c" * len(flows) + "ccc}",
        r"\toprule",
        r" & \multicolumn{%d}{c}{synthetic flows: mean estimate (noise) [1/s]} & "
        r"\multicolumn{3}{c}{datasets: $|\bar\mu|/\bar s$} \\" % len(flows),
        r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (1 + len(flows), 2 + len(flows),
                                                     4 + len(flows)),
        "estimator & " + " & ".join(
            f"{f.replace('area-preserving ', '')} ({FLOWS[f][1]:+.1f})" for f in flows
        ) + r" & A & B & C \\",
        r"\midrule",
    ]
    for est, (kind, param, *_) in ESTIMATORS.items():
        pf = summary[(kind, param)]["per_flow"]
        cells = [f"{pf[f]['mean']:+.2f} ({pf[f]['noise']:.2f})" for f in flows]
        ratios = []
        for name in SCENARIOS:
            _, mu, band = curves[(name, est)]
            ratios.append(f"{np.abs(mu).mean() / band.mean():.2f}")
        lines.append(f"{_label(est)} & " + " & ".join(cells + ratios) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = out / "table_div_estimators.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("output/skate/paper"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _style()

    curves, fields, datasets = dataset_curves(args.data, args.device)
    pinch, pinch_phase = corridor_spread(datasets["single_path"])
    print(f"scenario A narrowest xy minor-axis spread: {pinch:.3f} m at phase {pinch_phase:.2f}")

    spread_rows = spread_sweep(args.device, max(3, args.seeds // 2))
    trade_rows = tradeoff_sweep(args.device, args.seeds)
    summary = summarize_tradeoff(trade_rows)

    fig_bias(spread_rows, summary, pinch, args.out)
    fig_datasets(curves, args.out)
    fig_corridor(datasets["single_path"], fields, args.out)
    write_table(summary, curves, args.out)

    _write_csv(args.out / "fig_div_bias_spread.csv", spread_rows)
    _write_csv(args.out / "fig_div_bias_tradeoff.csv", [
        {"kind": k, "param": p, "rms_bias": v["rms_bias"], "rms_noise": v["rms_noise"]}
        for (k, p), v in summary.items()
    ])
    _write_csv(args.out / "fig_div_datasets.csv", [
        {"scenario": name, "estimator": est, "phase": f"{ph:.4f}", "mean": f"{m:.5f}",
         "spread": f"{b:.5f}"}
        for (name, est), (phase, mu, band) in curves.items()
        for ph, m, b in zip(phase, mu, band)
    ])

    print("\nsynthetic summary (RMS over flows):")
    for (k, p), v in summary.items():
        print(f"  {k} {p:>6}: |bias| {v['rms_bias']:.3f}  noise {v['rms_noise']:.3f}")


if __name__ == "__main__":
    main()
