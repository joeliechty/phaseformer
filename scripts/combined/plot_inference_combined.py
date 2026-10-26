"""Cross-task inference figures: the two tasks' panels side by side.

    python scripts/combined/plot_inference_combined.py

Recombines what :func:`evaluate_policies.fig_inference` (skate nav) and
:func:`plot_pusht.fig_inference` (Push-T) each draw as one three-panel figure, into two
cross-task figures:

* ``fig_inference_pareto_both``  -- both score-vs-cost panels, nav left and Push-T right.
* ``fig_inference_bars_both``    -- per decision and per executed control step, a row per task.

Nothing is re-measured: the costs come from the two published ``inference.csv`` files and the
accuracies from the same run directories the per-task figures read. This script only arranges
them, so a number here that disagrees with a per-task figure means one of the two is stale.

``--rollout-tag`` picks the Push-T rollout set (default ``best.pt``'s; pass ``--last`` for the
final-checkpoint set that ``fig_pusht_inference_backbones_last`` uses).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skate"))  # evaluate_policies
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pusht"))  # plot_pusht

import evaluate_policies as ep  # noqa: E402
import plot_pusht as pt  # noqa: E402

PALETTE = pt.PALETTE
TEXT_WIDTH = 7.16


# --------------------------------------------------------------------------- data
def nav_rows(out: Path) -> list[dict]:
    return ep.read_inference(out)


def nav_success(runs: Path, n: int = 1024) -> dict[str, list[float]]:
    """Per-scenario mean success for each arm, straight from the runs' metrics.json."""
    import json

    per: dict[str, list[float]] = {}
    for scenario_dir in sorted(runs.glob(f"*/n{n}")):
        for method_dir in sorted(scenario_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            vals = [json.loads(f.read_text())["success_rate"]
                    for f in sorted(method_dir.glob("seed*/metrics.json"))]
            if vals:
                per.setdefault(method_dir.name, []).append(100 * float(np.mean(vals)))
    return per


def pusht_scores(runs: Path) -> dict[str, tuple[float, float, float]]:
    out = {}
    for entry in pt._backbone_pairs():
        r = pt._rollouts(runs, entry[0])
        if r:
            seeds = [100 * float(x["score"].mean()) for x in r]
            out[entry[0]] = (float(np.mean(seeds)), min(seeds), max(seeds))
    return out


# --------------------------------------------------------------------------- shared helpers
def _axis(ax, xlabel=None, ylabel=None, size=8) -> None:
    ax.set_facecolor(PALETTE["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])
    ax.tick_params(labelsize=size - 1.5, colors=PALETTE["ink_muted"], length=2.5)
    ax.grid(True, color=PALETTE["grid"], lw=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=size, color=PALETTE["ink_muted"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=size, color=PALETTE["ink_muted"])


def _rate(ms: float) -> str:
    hz = 1000.0 / ms
    return f"{hz/1000:.0f} kHz" if hz >= 1000 else f"{hz:.0f} Hz"


# The latency budgets the per-task bar panels mark, on the Pareto panels' cost axis too: a
# point left of a line is fast enough to close that loop. Push-T's own 10 Hz line is left out --
# both panels share the convention here, and it is the task's rate, not a feedback budget.
CONTROL_BUDGETS = [(60.0, "60 Hz visual", "left"), (1000.0, "1 kHz tactile", "right")]


def _budget_lines(ax, bump=0.0) -> None:
    """``CONTROL_BUDGETS`` as dashed verticals, labelled along the top of the axis."""
    lo, hi = ax.get_xlim()
    for hz, what, ha in CONTROL_BUDGETS:
        ms = 1000.0 / hz
        if not lo < ms < hi:  # a budget off the panel would label an edge it does not mark
            continue
        ax.axvline(ms, ls=(0, (4, 3)), lw=0.9, color=PALETTE["ink_muted"], alpha=0.75, zorder=1)
        ax.annotate(what, (ms, 1.0), xycoords=ax.get_xaxis_transform(),
                    textcoords="offset points", xytext=(3 if ha == "left" else -3, -1),
                    ha=ha, va="top", fontsize=6 + bump, color=PALETTE["ink_muted"],
                    bbox=dict(facecolor=PALETTE["surface"], edgecolor="none", pad=0.4), zorder=3)


def _frontier(points):
    """Indices of the cost/accuracy Pareto front: cheapest-and-best, nothing dominating."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best = [], -np.inf
    for i in order:
        if points[i][1] > best:
            front.append(i)
            best = points[i][1]
    return front


def _devices(rows):
    return sorted({r["device"] for r in rows}, reverse=True)  # cuda first


# --------------------------------------------------------------------------- nav panel data
def nav_panel_data(runs: Path, out: Path, n: int = 1024):
    """(arms, cost(arm, device), accuracy, whiskers, colour, label) for the nav task."""
    rows = [r for r in nav_rows(out) if not ep.figure_excluded(r["method"])]
    devices = _devices(rows)
    by = {(r["method"], r["device"]): r for r in rows}

    def timing(method, device):
        r = by.get((ep.SAME_INFERENCE_AS.get(method, method), device))
        if r is not None and method in ep.CLOSED_LOOP_ONLY:
            r = {**r, "ms_step_b1": r["ms_decision_b1"], "steps_per_decision": "1"}
        return r

    per = nav_success(runs, n)
    measured = {r["method"] for r in rows}
    shared = [a for a, donor in ep.SAME_INFERENCE_AS.items()
              if not ep.figure_excluded(a) and donor in measured and a in per]
    bars = ep.table_order({*measured, *shared})[::-1]
    open_loop = {m for m in bars if int(timing(m, devices[0])["steps_per_decision"]) > 1}
    closed = sorted({m for m in per if ep._te_base(m)[1] is not None
                     and ep._te_base(m)[0] in (open_loop | ep.CLOSED_LOOP_ONLY)})
    points = ep.table_order({*bars} - ep.CLOSED_LOOP_ONLY)[::-1] + ep.table_order(closed)

    def cost(key, device):
        base = ep._te_base(key)[0]
        r = timing(base, device)
        if r is None:
            return None
        return float(r["ms_decision_b1" if ep._te_base(key)[1] is not None else "ms_step_b1"])

    return dict(
        devices=devices, timing=timing, bars=bars, open_loop=open_loop, points=points, cost=cost,
        accuracy={m: float(np.mean(per[m])) for m in per},
        whisker={m: (min(per[m]), max(per[m])) for m in per},
        colour=lambda m: ep.method_style(ep._te_base(m)[0], points)["color"],
        label=lambda m: ep.method_short(m, ep.FIGURE_FM),
        point_label=lambda m: (f"{ep._pareto_label(ep._te_base(m)[0])} (closed loop)"
                               if ep._te_base(m)[1] is not None
                               else ep._pareto_label(m) + (" (open loop)" if m in open_loop else "")),
        hardware={r["device"]: r["hardware"] for r in rows},
    )


# --------------------------------------------------------------------------- Push-T panel data
def pusht_panel_data(runs: Path, out: Path):
    """The same six fields for Push-T, read through plot_pusht's own tables."""
    import csv

    rows = list(csv.DictReader((out / "inference.csv").open()))
    devices = _devices(rows)
    by = {(r["method"], r["device"]): r for r in rows
          if (r["loop"] == "closed") == (r["method"] in pt.CLOSED_LOOP_ONLY)}
    twin = dict(zip(*pt.SCORE_ARMS.values()))
    twin.update({v: k for k, v in twin.items()})

    def timing(method, device):
        return by.get((method, device)) or by.get((twin.get(method, method), device))

    entries = pt._backbone_pairs()
    merge = pt._score_merge()
    drawn = [m for m in entries if not m[0].endswith("_te") and timing(m[0], devices[0])]
    bars = []
    for m, label, color, _ls in drawn:
        if m in merge:
            continue
        twins = [e for e in drawn if merge.get(e[0]) == m]
        names = [label] + [e[1][len(label):] if e[1].startswith(label) else e[1] for e in twins]
        bars.append((m, " / ".join(names), color))
    bars = bars[::-1]
    open_loop = {m for m, _, _ in bars if int(timing(m, devices[0])["steps_per_decision"]) > 1}

    scores = pusht_scores(runs)
    closed = [m for m in entries if m[0].endswith("_te") and m[0] in scores]
    points = [m for m in [*drawn, *closed]
              if m[0] in scores and m[0] not in pt.CLOSED_LOOP_ONLY]

    def cost(key, device):
        r = timing(key.removesuffix("_te"), device)
        if r is None:
            return None
        return float(r["ms_decision_b1" if key.endswith("_te") else "ms_step_b1"])

    bar_label = {"dp": "Diffusion Policy", "mip": "MIP", "act": "ACT"}
    return dict(
        devices=devices, timing=timing,
        bars=[(m, bar_label.get(m, lbl), c) for m, lbl, c in bars],
        open_loop=open_loop,
        points=[m[0] for m in points], cost=cost,
        accuracy={k: v[0] for k, v in scores.items()},
        whisker={k: (v[1], v[2]) for k, v in scores.items()},
        colour=lambda m, _c={e[0]: e[2] for e in entries}: _c[m],
        point_label=lambda m, _l={e[0]: e[1] for e in entries}: _l[m],
        hardware={r["device"]: r["hardware"] for r in rows},
    )


# --------------------------------------------------------------------------- figures
def _is_closed(m: str) -> bool:
    return m.endswith("_te") or "_te0.01" in m


def _label_points(ax, d, fronts, anchor, bump=0.0):
    """The per-task figures' scheme: costly arms label themselves on the right; the cheap
    cluster, too dense to label in place, stacks in a column beside it joined by hairlines."""
    label = dict(fontsize=5.8 + bump, color=PALETTE["ink"], ha="left", va="center")
    acc = d["accuracy"]
    # Only the far-right arms (Diffusion Policy) have room to label themselves in place; at half
    # width everything else shares one column, closed-loop arms included -- the per-task figures
    # can afford to label those beside their squares, but here the column would run over them.
    far = [m for m in anchor if anchor[m] > 100]
    for m in far:
        ax.annotate(d["point_label"](m), (anchor[m], acc[m]), textcoords="offset points",
                    xytext=(7, 0), **label)
    cluster = sorted((m for m in anchor if m not in far), key=lambda m: -acc[m])
    if not cluster:
        return
    x_col = 3.2 * max(anchor[m] for m in cluster)
    lo, hi = ax.get_ylim()
    # The column is spread evenly over the axis in accuracy order rather than tracking each
    # arm's own height: with a dozen arms in one half-width panel, following the points runs
    # the bottom labels off the axis, and a label that is not on the figure says nothing.
    top, bot = hi - 0.06 * (hi - lo), lo + 0.05 * (hi - lo)
    if far:  # start below the arms labelled in place, or the column's head runs into them
        top = min(top, min(acc[m] for m in far) - 0.06 * (hi - lo))
    n = len(cluster)
    ys = [top] if n == 1 else [top - i * (top - bot) / (n - 1) for i in range(n)]
    for m, y in zip(cluster, ys):
        ep._leader(ax, (anchor[m], acc[m]), (x_col / 1.09, y))
        ax.annotate(d["point_label"](m), (x_col, y), **label)


def _draw_pareto(ax, d, title, ylabel, bump=0.0):
    """One score-vs-cost panel: a point per arm per device, joined, with each device's front."""
    fronts = {}
    for device in d["devices"]:
        gpu = device == "cuda"
        pts = []
        for m in d["points"]:
            c = d["cost"](m, device)
            if c is None or m not in d["accuracy"]:
                continue
            y = d["accuracy"][m]
            lo, hi = d["whisker"][m]
            col = d["colour"](m)
            ax.vlines(c, lo, hi, color=col, lw=0.9, alpha=0.5, zorder=2)
            ax.plot(c, y, "s" if _is_closed(m) else "o", ms=5.5,
                    mfc=col if gpu else "none", mec=col, mew=1.3, zorder=3)
            pts.append((c, y))
        fronts[device] = [pts[i] for i in _frontier(pts)] if pts else []
        if fronts[device]:
            ax.plot([p[0] for p in fronts[device]], [p[1] for p in fronts[device]],
                    ls="--" if gpu else ":", lw=1.0, color=PALETTE["ink_muted"], zorder=1)
    # One accuracy, two costs: join a method's devices so the hardware penalty reads as a segment.
    anchor = {}
    for m in d["points"]:
        cs = [d["cost"](m, dev) for dev in d["devices"]]
        if any(c is None for c in cs) or m not in d["accuracy"]:
            continue
        ax.plot(cs, [d["accuracy"][m]] * 2, lw=0.7, color=d["colour"](m), alpha=0.35, zorder=2)
        anchor[m] = max(cs)
    ax.set_xscale("log")
    costs = [c for m in anchor for c in (d["cost"](m, dev) for dev in d["devices"])]
    ax.set_xlim(min(costs) / 2.5, max(costs) * 12)  # room for the labels beside the points
    # Whiskers may run off the bottom -- they span the scenarios, and an arm that fails one is
    # the point -- but a clipped *marker* would drop the arm out of the panel entirely.
    floor = min(60.0, 5 * np.floor((min(d["accuracy"][m] for m in anchor) - 2) / 5))
    ax.set_ylim(floor, 100)
    _label_points(ax, d, fronts, anchor, bump)
    _budget_lines(ax, bump)
    _axis(ax, "ms per control step (log)", ylabel, size=8 + bump)
    ax.set_title(title, fontsize=8.5 + bump, loc="left", color=PALETTE["ink"])


def _draw_bars(ax, d, key, title, with_rate, labels):
    """One bar panel: a row per arm, GPU filled above CPU hatched."""
    height = 0.34
    for i, (m, _label, color) in enumerate(labels):
        for j, device in enumerate(d["devices"]):
            r = d["timing"](m, device)
            if r is None:
                continue
            v = float(r[key])
            y = i + (0.5 - j) * height
            gpu = device == "cuda"
            col = color[0] if isinstance(color, list) else color
            ax.barh(y, v, height=height * 0.92, color=col if gpu else "none",
                    edgecolor=col, linewidth=0 if gpu else 1.0, hatch=None if gpu else "////")
            text = f"{v:.2f}" if v < 10 else f"{v:,.0f}"
            if with_rate:
                text = f"{text} ({_rate(v)})"
                if m in d["open_loop"] and j == 0:
                    text += f"  ← {int(r['steps_per_decision'])} steps open loop"
            elif m in d["open_loop"]:
                text = f"{text} ({_rate(v)})" + ("  ← closed loop" if j == 0 else "")
            ax.text(v * 1.3, y, text, va="center", fontsize=5.6, color=PALETTE["ink_muted"],
                    bbox=dict(facecolor=PALETTE["surface"], edgecolor="none", pad=0.25), zorder=3)
    ax.set_xscale("log")
    vals = [float(d["timing"](m, dev)[key]) for m, _l, _c in labels for dev in d["devices"]
            if d["timing"](m, dev) is not None]
    ax.set_xlim(min(vals) / 2, max(vals) * (60 if with_rate else 25))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([lbl for _m, lbl, _c in labels], fontsize=6.8)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    _axis(ax, "milliseconds (log)")
    ax.yaxis.grid(False)
    ax.set_title(title, fontsize=8.5, loc="left", color=PALETTE["ink"])


def _shared_legend(fig, nav, pusht, y=0.995, frontier=True, bump=0.0):
    """Device (fill vs hatch), loop (circle vs square) and the two frontiers, in neutral ink --
    colour is the arm, so it can carry nothing else."""
    hardware = {**pusht["hardware"], **nav["hardware"]}
    handles = [Patch(facecolor=PALETTE["ink_muted"] if dev == "cuda" else "none",
                     edgecolor=PALETTE["ink_muted"], hatch=None if dev == "cuda" else "////",
                     label=hardware[dev]) for dev in nav["devices"]]
    handles += [Line2D([], [], ls="", marker=mk, ms=5, mfc=PALETTE["ink_muted"],
                       mec=PALETTE["ink_muted"], label=text)
                for mk, text in [("o", "as trained (open-loop chunks)"),
                                 ("s", "closed loop (temporal ensembling)")]]
    if frontier:
        handles += [Line2D([], [], ls=ls, lw=1.0, color=PALETTE["ink_muted"],
                           label=f"Pareto frontier, {dev}")
                    for ls, dev in (("--", "GPU"), (":", "CPU"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=3,
               frameon=False, fontsize=6.8 + bump, labelcolor=PALETTE["ink_muted"],
               handlelength=1.8, columnspacing=1.8, ncols=3 if frontier else 2)


def _save(fig, out: Path, name: str, rect, pad: float | None = None, **layout) -> None:
    """``pad``: crop to what is drawn, leaving this many inches around it (no crop if None).
    ``layout`` goes to ``tight_layout`` (``w_pad`` and ``h_pad``, to close the gaps up)."""
    fig.tight_layout(rect=rect, **layout)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, facecolor=PALETTE["surface"],
                    bbox_inches="tight" if pad is not None else None, pad_inches=pad or 0.1)
    plt.close(fig)
    print(f"  wrote {out}/{name}.{{png,pdf}}")


def fig_pareto_both(nav, pusht, out: Path, n: int, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH * 1.35, 3.6), facecolor=PALETTE["surface"])
    bump = 2.0  # points added to every font in this figure
    _draw_pareto(axes[0], nav, f"skate nav: success vs cost, mean of A, B, C (n = {n})",
                 "success [%]", bump)
    _draw_pareto(axes[1], pusht, "Push-T: score vs cost (50 envs, DP protocol)",
                 "rollout score [%]", bump)
    _shared_legend(fig, nav, pusht, bump=bump)
    # The paper is short of space: cropped to the ink, with the gutter and the gap under the
    # legend closed up as far as the text allows.
    _save(fig, out, name, (0, 0, 1, 0.905), pad=0.02, w_pad=0.4)


def fig_bars_both(nav, pusht, out: Path, name: str) -> None:
    shared = set(ep.SAME_INFERENCE_AS)
    nav_labels = [(m, ep._pareto_label(m) if m in shared else ep.figure_label(m),
                   ep.method_style(m, nav["bars"])["color"]) for m in nav["bars"]]
    pt_labels = pusht["bars"]
    heights = [len(nav_labels), len(pt_labels)]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH * 1.45, 0.34 * sum(heights) + 2.4),
                             facecolor=PALETTE["surface"],
                             gridspec_kw={"height_ratios": heights})
    for row, (d, labels, task) in enumerate([(nav, nav_labels, "skate nav"),
                                             (pusht, pt_labels, "Push-T")]):
        _draw_bars(axes[row, 0], d, "ms_decision_b1", f"{task} — per decision", False, labels)
        _draw_bars(axes[row, 1], d, "ms_step_b1", f"{task} — per executed control step", True,
                   labels)
        axes[row, 1].set_yticklabels([])
    _shared_legend(fig, nav, pusht, frontier=False)
    _save(fig, out, name, (0, 0, 1, 0.92))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--nav-runs", type=Path, default=Path("runs"))
    p.add_argument("--nav-out", type=Path, default=Path("output/skate/policy"))
    p.add_argument("--pusht-runs", type=Path, default=Path("runs/pusht/policy"))
    p.add_argument("--pusht-out", type=Path, default=Path("output/pusht_9010"))
    p.add_argument("--out", type=Path, default=Path("output/combined"))
    p.add_argument("--n-train", type=int, default=1024)
    p.add_argument("--last", action="store_true",
                   help="use Push-T's final-checkpoint rollouts (the _last figures' basis)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    suffix = ""
    if args.last:  # the globals plot_pusht.main sets for --rollout-tag *_last
        pt.ROLLOUT_TAG = f"{pt.ROLLOUT_TAG}_last"
        pt.BASELINE_TAGS = {k: f"{v}_last" for k, v in pt.BASELINE_TAGS.items()}
        pt.CKPT_TAG = "_last"
        suffix = "_last"

    nav = nav_panel_data(args.nav_runs, args.nav_out, args.n_train)
    pusht = pusht_panel_data(args.pusht_runs, args.pusht_out)
    fig_pareto_both(nav, pusht, args.out, args.n_train, f"fig_inference_pareto_both{suffix}")
    fig_bars_both(nav, pusht, args.out, f"fig_inference_bars_both{suffix}")


if __name__ == "__main__":
    main()
