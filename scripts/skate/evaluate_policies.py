"""Aggregate policy runs over seeds: summary table, significance tests and figures.

    python scripts/skate/evaluate_policies.py --runs runs/ --test data/test/ \
        --out output/skate/policy/

Reads every ``{runs}/{scenario}/n{n}/{method}/seed{s}/metrics.json`` (written by
``scripts/skate/run_experiments.py``) and writes, per training-set size ``n``:

* ``summary.csv`` -- mean and std over seeds of every scalar metric, per scenario x n x method;
* ``significance.csv`` -- Welch t-tests of every other method (the regularized arms, the
  state-only MLP) against FM;
* ``table_policy.tex`` -- booktabs table for the paper (every scenario x n x method);
* ``table_policy_summary.tex`` -- compact paper table: MMD and success per scenario for FM,
  FM+CDM and the MLP, one row per training-set size, best mean in bold;
* ``table_policy_baselines.tex`` -- every method at the largest ``n`` (where the generative
  baselines are trained), a row each with the baselines below a rule: MMD and success per
  scenario, best mean in bold;
* ``SUMMARY.md`` -- the same numbers as Markdown tables with the figures inline (renders on
  GitHub);
* ``fig_policy_vs_ntrain``       every summary metric against training-set size (all sizes);
* ``fig_policy_violin_n{n}``     the same metrics at the largest ``n`` only: one violin per method
  over its seeds, the seeds drawn on top, daggers against ``VIOLIN_BASE`` -- the MLPs, MLP+phase+
  score and the generative baselines open and closed loop (``VIOLIN_METHODS``);
* ``fig_policy_violin_n{n}_ablation`` the same for every regularizer arm, without the baselines;
* ``fig_policy_rollouts_n{n}``   rows: held-out demos, then one per method; columns: A, B, C
  (seed 0);
* ``fig_policy_rollouts_n{n}_{C,AB}`` the same at the largest ``n`` for ``COMPACT_METHODS`` only,
  turned round: a row per scenario, C alone and then A and B;
* ``fig_policy_mmd_phase_n{n}``  per-phase MMD to the held-out demos, mean +/- std over seeds;
* ``fig_policy_training_n{n}``   a row per scenario: MMD and velocity loss against training step,
  sized for one column of a two-column paper;
* ``fig_policy_training_n{n}_row`` the same at the largest ``n`` turned round, a column per
  scenario, sized for the paper's full text width;
* ``fig_policy_training_n{n}_single_row`` the row version's two rows side by side, all six
  panels in one row across the text width, for the least height;
* ``fig_policy_divergence_n{n}`` one row per method: demo divergence vs. the rollouts' kNN
  divergence and the model's own divergence, over phase.

Methods are the run-directory names of ``fm_bc.skate.training.method_name``. The tables call the
phase-conditioned policy FM; the figures call it MLP+phase (``FIGURE_FM``), its arms
MLP+phase+CDM and so on, and put the state-only MLP first. Each kind of arm keeps one
categorical colour in every figure, and annealed arms are dashed. The generative baselines
(Diffusion Policy, MIP, MeanFlow: ``scripts/skate/train_baselines.py``) come last and are dotted;
they are left out of the training-curve figures, which share a step axis with the 20k-step arms.
FM with action chunking (``fm_chunk``, FM+chunk / MLP+phase+chunk) trains like FM, so it keeps
FM's blue, dash-dotted, and sits between the FM arms and the baselines; with the score
regularizer along the chunk (``chunk_score_w<λ>...``, FM+chunk+score) it is dash-dotted in the
annealed score arm's pink, right after FM+chunk. A chunk policy re-evaluated with temporal
ensembling (``{method}_te<m>``, ``scripts/skate/evaluate_temporal_ensemble.py``) keeps its
original's colour, follows it, and is dash-dot-dotted; it is left out of the training-curve
figures, whose curve it shares with the original.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FuncFormatter, LogLocator, MaxNLocator, NullFormatter, NullLocator,
)
from matplotlib.transforms import blended_transform_factory  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate.dataset import TrajectoryDataset  # noqa: E402
from fm_bc.skate.evaluation import collision_breakdown  # noqa: E402
from fm_bc.skate.plots import plot_rollouts  # noqa: E402
from fm_bc.skate.scenarios import SCENARIOS  # noqa: E402
from fm_bc.skate.training import (  # noqa: E402
    BASELINE_METHODS, CHUNK_METHOD, TrainConfig, method_name, parse_method,
)
from skate_planner.viz import PALETTE  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})

SCALARS = [
    "mmd_mean", "mmd_final", "collision_rate", "obstacle_collision_rate",
    "boundary_collision_rate", "reach_rate", "success_rate", "mode_tv", "undecided_rate",
    "saturation_rate", "saturated_step_fraction", "div_curve_mae",
]
TESTED = ["mmd_mean", "mmd_final", "success_rate", "obstacle_collision_rate", "mode_tv"]
LETTER = {"single_path": "A", "dual_goal": "B", "dual_path": "C"}
TITLES = {"single_path": "A · single path", "dual_goal": "B · dual goal",
          "dual_path": "C · dual path"}
# (metric, axis label, scale) for the summary-vs-n figure.
SUMMARY_METRICS = [
    ("mmd_mean", r"MMD$^2\times10^3$", 1e3),
    ("obstacle_collision_rate", "obstacle hits [%]", 100.0),
    ("success_rate", "success [%]", 100.0),
]
# The compact rollout figures (fig_rollouts_by_scenario): a row per scenario -- C alone, then A
# and B -- and a column each for the held-out demos, the state-only MLP, MLP+phase, the
# constant-λ score-only arm and the three baselines, which run open loop here (as trained).
COMPACT_METHODS = ["mlp", "fm", "score_w0.5", "dp", "mip", "act"]
COMPACT_GROUPS = [["dual_path"], ["single_path", "dual_goal"]]
# Inches of figure per compact rollout panel. A panel is a world, so its aspect is the world's;
# the height leaves room for the two-line titles (:func:`_compact_title`).
COMPACT_PANEL_W, COMPACT_PANEL_H = 2.75, 2.72
# Type sizes in the compact figures: the panel titles, the legend inside a panel, and the
# scenario letter down the left edge. These are what the figure is read at -- it is printed at
# a whole page's width and a column of it is only 2.75 in, so the type has to carry that far.
# The letter and the qualifier line of every title moved off the title's first line to make the
# room (see _compact_title), and the legend labels are abbreviated (LEGEND_SHORT).
COMPACT_TITLE_PT, COMPACT_LEGEND_PT, COMPACT_ROW_PT = 13.5, 11.0, 13.5
# Mode names as the compact legends abbreviate them: at COMPACT_LEGEND_PT the full words push
# the legend box across the panel it sits in, over the rollouts it is there to label.
LEGEND_SHORT = {"undecided": "undec.", "collision": "coll."}
# The violin figure's daggers test every arm against this one: the constant-λ score-only arm.
VIOLIN_BASE = "score_w0.5"
# The main violin figure's arms: the MLPs, the constant-λ score arm and the generative baselines,
# which it shows open and closed loop (their temporally ensembled re-evaluations too).
VIOLIN_METHODS = {"mlp", "mlp_score_w0.5", "fm", "score_w0.5", "dp", "mip", "act"}
# Metrics that figure draws against a log y axis. Its arms' MMD spans two and a half decades
# at n=1024 -- on a linear axis the worst arm sets the scale and every other one is pressed
# flat against zero, which is the comparison the figure exists to make.
VIOLIN_LOG_METRICS = {"mmd_mean"}
# Width of a violin body in slots. Well under a full slot: the figure is read at the size of its
# tick labels, and the space the bodies give up is what lets those be set large.
VIOLIN_WIDTH = 0.68
# Inches of figure per violin, and per scenario row. A slot need only be wide enough that the
# rotated labels underneath clear one another, which is a fraction of what a label is long.
VIOLIN_SLOT, VIOLIN_ROW = 0.54, 2.15
# Degrees the x tick labels are rotated. Every arm is named in full under its own violin, which
# is what lets the figure carry no legend at all.
VIOLIN_ROTATION = 32
# The panel titles, above the top row: the metric, and what a violin of it is taken over. The
# y label carries the same quantity in short, with its unit, so a panel read alone is not bare.
VIOLIN_TITLES = {
    "mmd_mean": r"mean MMD$^2$ per seed [$\times 10^3$, log scale]",
    "obstacle_collision_rate": "obstacle hits per seed [%]",
    "success_rate": "success rate per seed [%]",
}
# The two arms the paper's argument rests on -- MLP+phase and its constant-λ score arm. The
# violin figure sets their tick labels in bold and the rollout figures their panel titles.
BOLD_METHODS = {"fm", VIOLIN_BASE}
# The single-row training figure's arms: the two networks and their constant-λ score arms, the
# comparison the paper makes. Every arm that trains like FM goes to the _ablation version of the
# same figure, which is what that figure used to be.
TRAINING_METHODS = ["mlp", "mlp_score_w0.5", "fm", "score_w0.5"]
# Categorical slots 1-6 of the dataviz reference palette in figure order: the state-only MLP,
# MLP+phase (the tables' FM), its constant-λ CDM arm, then the annealed div, score and CDM arms.
# The first two are swapped against the reference order so MLP+phase keeps slot 1's blue; every
# adjacent pair still passes the validator. Annealed arms are also dashed (method_style): within
# each dash group every pair clears the all-pairs CVD and normal-vision gates, and the dash
# separates the cross-group pairs that don't (the MLP's orange vs the annealed CDM's green is
# Delta E 3.2 under protanopia).
# Slot 6 is FM with action chunking: FM's own blue, told apart by its dash-dot (method_style).
# Slots 7-9 are the generative baselines (Diffusion Policy, MIP, MeanFlow), all dotted: the
# palette's slots 7 (violet) and 8 (red), then -- it has only eight hues -- MeanFlow reuses the
# constant-λ CDM arm's aqua, which that arm draws solid, so the dots tell the two apart.
# Slot 10 is ACT, the deterministic chunk baseline. The palette is out of hues, so it takes the
# same aqua a third time and is told apart by a dash of its own (method_style): MeanFlow draws it
# dotted and the constant-λ CDM arm solid. The three never share a figure today -- MeanFlow is in
# neither violin nor inference figure -- so the reuse costs nothing a reader can trip on.
METHOD_COLORS = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                 "#2a78d6", "#4a3aa7", "#e34948", "#1baf7a", "#1baf7a"]
_CHUNK_SLOT = 6
_BASELINE_SLOT = {"dp": 7, "mip": 8, "meanflow": 9, "act": 10}
_BASELINE_LABEL = {"dp": "Diffusion Policy", "mip": "MIP", "meanflow": "MeanFlow", "act": "ACT"}
_BASELINE_SHORT = {"dp": "DP", "mip": "MIP", "meanflow": "MF", "act": "ACT"}
# ACT's dash: long, so it reads against MeanFlow's dots at the same hue.
ACT_LS = (0, (5, 1))
# Chunk baselines reported closed loop only: the inference figure quotes their closed-loop cost
# (a whole decision per executed step) and carries only their `_te` point, not an open-loop one,
# and the violin figure draws the one arm rather than a pair. ACT is here because its open-loop
# per-step figure is an artefact of executing 8 steps per decision -- it reads as the arm being
# 8x cheaper than it is under the loop it is evaluated in.
CLOSED_LOOP_ONLY = {"act"}
# The figures name FM after its network, next to "MLP (no phase)"; the tables keep FM.
FIGURE_FM = "MLP+phase"
# IEEE's figure widths for a two-column paper, in inches: one column and the full text width.
# fig_training draws at these and saves uncropped, so its text prints at its own size.
COLUMN_WIDTH, TEXT_WIDTH = 3.5, 7.16
# Type sizes as fractions of the figure's width, by role, taken from fig_inference_pareto_both
# (plot_inference_combined): a figure sized from these prints its text at that figure's size
# whatever width it is drawn at, once both are scaled to the same width on the page.
TYPE_RATIOS = {"title": 0.0151, "label": 0.0144, "legend": 0.0126, "tick": 0.0122,
               "note": 0.0115, "data": 0.0112}


def type_pt(width_in: float, role: str) -> float:
    """Points for ``role``'s text in a figure ``width_in`` inches wide (``TYPE_RATIOS``)."""
    return TYPE_RATIOS[role] * 72 * width_in
# The tables list FM first: FM, its constant-λ arms, the state-only MLP, then the annealed arms.
_TABLE_RANK = {1: 0, 2: 1, 0: 2}
# Line styles within a colour slot, one pool per schedule so the two can never collide:
# a constant-lambda arm is solid (then dash-dot), an annealed one dashed (then dotted).
CONSTANT_STYLES = ["-", "-."]
ANNEALED_STYLES = ["--", ":"]
_TERM_LABEL = {"both": "CDM", "div": "div", "score": "score"}
_ANNEALED_SLOT = {"div": 3, "score": 4, "both": 5}
# Line style of a temporally ensembled re-evaluation: dash-dot-dot, unlike any trained arm's.
TE_LS = (0, (4, 1.5, 1, 1.5, 1, 1.5))


def _te_base(method: str) -> tuple[str, float | None]:
    """The run a temporally ensembled arm re-evaluates, and its ``m`` (``None`` if not one)."""
    p = parse_method(method)
    return method_name(**{**p, "temporal_ensemble": None}), p["temporal_ensemble"]


def load_runs(root: Path) -> dict:
    """``{(scenario, n, method): [run, ...]}`` with each run's metrics, log and directory."""
    runs: dict = {}
    for path in sorted(root.glob("*/n*/*/seed*/metrics.json")):
        seed_dir = path.parent
        if not (seed_dir / "rollouts.npz").exists():  # still being written by a live grid
            continue
        method_dir = seed_dir.parent
        key = (method_dir.parent.parent.name, int(method_dir.parent.name[1:]), method_dir.name)
        log = np.genfromtxt(seed_dir / "log.csv", delimiter=",", names=True)
        runs.setdefault(key, []).append({
            "metrics": json.loads(path.read_text()), "log": log, "dir": seed_dir,
            "seed": int(seed_dir.name[4:]),
        })
    for group in runs.values():
        group.sort(key=lambda r: r["seed"])
    return runs


def fill_collision_breakdown(runs: dict, tests: dict) -> None:
    """Runs trained before the obstacle / boundary split was logged get it from their saved
    rollouts (the same dense check, so nothing is re-trained or re-rolled)."""
    for (scenario, _, _), group in runs.items():
        test = tests[scenario]
        for r in group:
            m = r["metrics"]
            if "obstacle_collision_rate" in m:
                continue
            arr = np.load(r["dir"] / "rollouts.npz")
            to = lambda a: torch.as_tensor(a, dtype=torch.double)
            obstacle, boundary = collision_breakdown(
                to(arr["poses"]), to(arr["controls"]), test.dt, test.spec.world,
                float(test.config.get("radius", 0.3)),
            )
            m["obstacle_collision_rate"] = float(obstacle.mean())
            m["boundary_collision_rate"] = float(boundary.mean())
    for group in runs.values():
        for r in group:
            # Unclamped runs logged the same quantity under its old name.
            if "limit_violation_rate" in r["metrics"]:
                r["metrics"].setdefault("saturation_rate", r["metrics"]["limit_violation_rate"])


def _slot(method: str) -> int:
    """Colour slot and figure position of an arm: the state-only MLP (and any arm of it) 0,
    MLP+phase 1, its constant-λ CDM arm 2, the div / score / CDM arms 3-5 whatever their schedule,
    FM with action chunking 6, the generative baselines 7-9. Colour says which regularizer; the
    line style (:func:`method_style`) says whether its λ is annealed."""
    p = parse_method(method)
    if p["baseline"] is not None:
        return _BASELINE_SLOT[p["baseline"]]
    if p["horizon"] > 1:
        return _CHUNK_SLOT
    if not p["use_phase"]:
        return 0
    if p["cdm_weight"] == 0:
        return 1
    if p["cdm_schedule"] == "constant" and p["cdm_terms"] == "both":
        return 2
    return _ANNEALED_SLOT[p["cdm_terms"]]


def figure_order(methods) -> list[str]:
    """Slot order (see :func:`_slot`), by weight within a slot -- so every legend walks the
    palette's validated adjacent pairs."""
    def key(m):
        p = parse_method(m)
        te = p["temporal_ensemble"]
        return _slot(m), p["cdm_weight"], p["cdm_anneal_frac"], -1.0 if te is None else te
    return sorted(methods, key=key)


def table_order(methods) -> list[str]:
    """The tables' order (see ``_TABLE_RANK``), by weight within a slot."""
    return sorted(figure_order(methods), key=lambda m: _TABLE_RANK.get(_slot(m), _slot(m)))


def _arch_base(method: str) -> tuple[str, str]:
    """The same arm on the MLP backbone, and this one's ``arch`` -- the backbone suffix peeled off
    the way :func:`_te_base` peels the temporal-ensemble one."""
    p = parse_method(method)
    return method_name(**{**p, "arch": "mlp"}), p["arch"]


def method_label(method: str, fm: str = "FM") -> str:
    """Display name of an arm; ``fm`` names the phase-conditioned policy (figures: FIGURE_FM)."""
    base, te = _te_base(method)
    if te is not None:
        return f"{method_label(base, fm)}+TE"
    # Without this a transformer arm and its MLP twin are one label in the legend.
    base, arch = _arch_base(method)
    if arch != "mlp":
        return f"{method_label(base, fm)} [TF]"
    p = parse_method(method)
    if p["baseline"] is not None:
        return _BASELINE_LABEL[p["baseline"]]
    if p["cdm_weight"] == 0:
        if p["horizon"] > 1:
            return f"{fm}+chunk"
        return fm if p["use_phase"] else "MLP (no phase)"
    w = f"{p['cdm_weight']:g}"
    if p["cdm_schedule"] == "constant":
        lam = f"λ={w}"
    else:
        lam = f"λ {w}→0"
        if p["cdm_anneal_frac"] != TrainConfig.cdm_anneal_frac:
            lam += f" by {p['cdm_anneal_frac']:.0%}"
    net = f"{fm}+chunk" if p["horizon"] > 1 else fm if p["use_phase"] else "MLP"
    return f"{net}+{_TERM_LABEL[p['cdm_terms']]} ({lam})"


def figure_label(method: str) -> str:
    return method_label(method, FIGURE_FM)


def method_short(method: str, fm: str = "FM") -> str:
    """Column header for the compact paper table, tick label in the violin figure."""
    base, te = _te_base(method)
    if te is not None:
        return f"{method_short(base, fm)}+TE"
    base, arch = _arch_base(method)
    if arch != "mlp":
        return f"{method_short(base, fm)} TF"
    if method in _BASELINE_SHORT:
        return _BASELINE_SHORT[method]
    if method == CHUNK_METHOD:
        return "+AC"
    if method in ("fm", "mlp"):
        return fm if method == "fm" else "MLP"
    p = parse_method(method)
    net = "+AC" if p["horizon"] > 1 else "" if p["use_phase"] else "MLP"
    return (net + f"+{_TERM_LABEL[p['cdm_terms']]}"
            + ("→0" if p["cdm_schedule"] == "cosine" else ""))


def _tex(text: str) -> str:
    return text.replace("λ", r"$\lambda$").replace("→", r"$\to$")


# Symbols a figure sets directly and LaTeX needs told about. Longest keys first: " · " is a
# clause break in a figure header and a comma in caption prose, where a lone "·" is not.
CAPTION_TEX = {
    " · ": ", ", "·": r"$\cdot$", "λ": r"$\lambda$", "→": r"$\to$", "†": r"$\dagger$",
    "×": r"$\times$", "≥": r"$\ge$", "≤": r"$\le$", "°": r"$^\circ$", "–": "--",
    "<": "$<$", ">": "$>$", "%": r"\%",
}


def _caption_tex(notes: list[str]) -> str:
    """The figure's notes as one LaTeX paragraph, to paste into a ``\\caption``.

    A note is a clause in the figure and a sentence here, so each is capitalised and stopped.
    Whatever a note already writes as math -- between ``$`` -- is left as it stands: those are
    the spans where "<" and "%" mean what LaTeX means by them.
    """
    out = []
    for note in notes:
        parts = (note[0].upper() + note[1:]).split("$")
        for i in range(0, len(parts), 2):  # the even spans are the ones outside math
            for src, dst in CAPTION_TEX.items():
                parts[i] = parts[i].replace(src, dst)
        out.append("$".join(parts).rstrip(".") + ".")
    return " ".join(out)


def method_style(method: str, methods: list[str]) -> dict:
    """The arm's slot colour; annealed arms are dashed, so the schedule reads without the hue.
    Arms sharing a slot and schedule (other weights) step on through that schedule's own style
    pool, so a constant arm can never take an annealed one's pattern; the generative
    baselines are dotted and FM with action chunking dash-dotted; a temporally ensembled
    re-evaluation keeps its original's colour, dash-dot-dotted."""
    base, te = _te_base(method)
    if te is not None:
        return {"color": method_style(base, [base])["color"], "ls": TE_LS}
    if parse_method(method)["baseline"] is not None:
        ls = ACT_LS if parse_method(method)["baseline"] == "act" else ":"
        return {"color": METHOD_COLORS[_slot(method)], "ls": ls}
    if _slot(method) == _CHUNK_SLOT:
        p = parse_method(method)
        slot = _CHUNK_SLOT if p["cdm_weight"] == 0 else _ANNEALED_SLOT[p["cdm_terms"]]
        return {"color": METHOD_COLORS[slot], "ls": "-."}
    # Counted within the slot *and* schedule, so an annealed arm sharing the MLP's slot is dashed
    # like every other annealed arm rather than stepping on to the baselines' dots.
    schedule = parse_method(method)["cdm_schedule"]
    same = [m for m in methods
            if _slot(m) == _slot(method) and parse_method(m)["cdm_schedule"] == schedule]
    pool = CONSTANT_STYLES if schedule == "constant" else ANNEALED_STYLES
    return {"color": METHOD_COLORS[_slot(method)], "ls": pool[same.index(method) % len(pool)]}


def _tint(color, alpha: float):
    """``color`` at ``alpha`` over the surface, returned opaque -- a violin fill.

    Blending here rather than drawing the fill translucent keeps the arm's colour off the
    artist's alpha, which would otherwise wash the outline out with it (violinplot leaves a
    body at alpha 0.3), and keeps a hatched fill legible in the PDF, whose backend paints a
    hatch as a tiling pattern and fills the tile with the face colour's RGB alone.
    """
    from matplotlib.colors import to_rgb  # noqa: PLC0415
    fg, bg = np.array(to_rgb(color)), np.array(to_rgb(PALETTE["surface"]))
    return tuple(alpha * fg + (1 - alpha) * bg)


def _violin(ax, vals: np.ndarray, pos: float, width: float, style: dict, log: bool):
    """One violin body at ``pos``: a Gaussian KDE of ``vals`` over their own range, scaled so the
    densest point is ``width`` across. Drawn here rather than by ``ax.violinplot`` because that
    always smooths in data space: on a log axis it would spread the top decade over the same
    width as the two below it and pinch every arm that lives low on the scale into a spike. The
    KDE is taken in the axis's own units instead -- log10 of the values where ``log``.

    Returns the body, for a caller that hatches it.
    """
    t = np.log10(vals) if log else vals
    grid = np.linspace(t.min(), t.max(), 100)
    dens = stats.gaussian_kde(t)(grid)
    half = 0.5 * width * dens / dens.max()
    return ax.fill_betweenx(10.0 ** grid if log else grid, pos - half, pos + half,
                            facecolor=_tint(style["color"], 0.30), edgecolor=style["color"],
                            linestyle=style["ls"], linewidth=1.0)


def _style(ax, xlabel=None, ylabel=None, size=8) -> None:
    """Surface, grid and spines; axis labels at ``size`` pt, tick labels a point smaller."""
    ax.set_facecolor(PALETTE["surface"])
    ax.grid(True, color=PALETTE["grid"], lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=PALETTE["ink_muted"], labelsize=size - 1)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=size, color=PALETTE["ink"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=size, color=PALETTE["ink"])


def _save(fig, out: Path, name: str, tight: bool = True, caption: str | None = None,
          pad: float = 0.1) -> None:
    """``tight``: crop to what is drawn, ``pad`` inches around it; otherwise keep the figure's
    own size.

    ``caption`` is written beside the figure as ``<name>.caption.txt``, LaTeX ready.
    """
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, facecolor=PALETTE["surface"],
                    bbox_inches="tight" if tight else None, pad_inches=pad)
    plt.close(fig)
    exts = "pdf,png"
    if caption is not None:
        (out / f"{name}.caption.txt").write_text(caption + "\n")
        exts += ",caption.txt"
    print(f"  wrote {out / name}.{{{exts}}}")


def _band(ax, x, series: np.ndarray, style: dict, label: str | None, lw=1.6) -> None:
    mu = np.nanmean(series, axis=0)
    ax.plot(x, mu, lw=lw, label=label, **style)
    if series.shape[0] > 1:
        sd = np.nanstd(series, axis=0)
        ax.fill_between(x, mu - sd, mu + sd, color=style["color"], alpha=0.15, lw=0)


# -- tables ----------------------------------------------------------------------------------


def summarize(runs: dict) -> list[dict]:
    rows = []
    for (scenario, n, method), group in sorted(runs.items()):
        row = {"scenario": scenario, "n_train": n, "method": method, "n_seeds": len(group),
               "clamped": all(r["metrics"].get("clamped", False) for r in group)}
        for key in SCALARS:
            # Runs from before a metric existed contribute NaN rather than failing the table.
            vals = np.array([r["metrics"].get(key, np.nan) for r in group], dtype=float)
            ok = np.isfinite(vals)
            row[f"{key}_mean"] = float(vals[ok].mean()) if ok.any() else np.nan
            row[f"{key}_std"] = float(vals[ok].std(ddof=1)) if ok.sum() > 1 else 0.0
        fracs = np.array([r["metrics"]["mode_fractions"] for r in group])
        for m, f in enumerate(fracs.mean(axis=0)):
            row[f"mode{m}_frac_mean"] = f
        rows.append(row)
    return rows


def significance(runs: dict, base: str = "fm") -> list[dict]:
    """Welch t-tests over seeds of every other method against ``base``, per scenario x n."""
    out = []
    for (scenario, n, method), group in sorted(runs.items()):
        ref = runs.get((scenario, n, base))
        if method == base or not ref:
            continue
        row = {"scenario": scenario, "n_train": n, "method": method,
               "n_seeds": f"{len(group)} vs {len(ref)}"}
        for key in TESTED:
            a = np.array([r["metrics"][key] for r in group])
            b = np.array([r["metrics"][key] for r in ref])
            p = stats.ttest_ind(a, b, equal_var=False).pvalue if min(len(a), len(b)) > 1 else np.nan
            row[f"{key}_diff"] = a.mean() - b.mean()
            row[f"{key}_p"] = p
        out.append(row)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def write_table(summary: list[dict], sig: list[dict], path: Path) -> None:
    """MMD in units of 1e-3; a dagger marks p < 0.05 against FM (Welch, over seeds)."""
    p = {(r["scenario"], r["n_train"], r["method"]): r for r in sig}

    def cell(r, key, scale=1.0, fmt="{:.2f}"):
        mu, sd = r[f"{key}_mean"] * scale, r[f"{key}_std"] * scale
        s = f"{fmt.format(mu)} $\\pm$ {fmt.format(sd)}"
        t = p.get((r["scenario"], r["n_train"], r["method"]))
        if t is not None and t.get(f"{key}_p", 1.0) < 0.05:
            s += "$^\\dagger$"
        return s

    lines = [
        r"\begin{tabular}{lllllllll}",
        r"\toprule",
        r"scen. & $n$ & method & MMD$\times10^3$ & final MMD$\times10^3$ & obstacle [\%] "
        r"& wall [\%] & success [\%] & mode split [\%] \\",
        r"\midrule",
    ]
    prev = None
    for r in _sorted_rows(summary):
        if prev is not None and prev != r["scenario"]:
            lines.append(r"\midrule")
        prev = r["scenario"]
        split = "/".join(f"{100 * r[k]:.0f}" for k in sorted(r) if k.startswith("mode") and
                         k.endswith("_frac_mean"))
        lines.append(" & ".join([
            LETTER.get(r["scenario"], r["scenario"]), str(r["n_train"]),
            _tex(method_label(r["method"])),
            cell(r, "mmd_mean", 1e3), cell(r, "mmd_final", 1e3),
            cell(r, "obstacle_collision_rate", 100, "{:.1f}"),
            cell(r, "boundary_collision_rate", 100, "{:.1f}"),
            cell(r, "success_rate", 100, "{:.1f}"), split,
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def _sorted_rows(rows: list[dict]) -> list[dict]:
    order = table_order({r["method"] for r in rows})
    return sorted(rows, key=lambda r: (SCENARIOS.index(r["scenario"]), -r["n_train"],
                                       order.index(r["method"])))


def _compact_methods(methods: list[str]) -> list[str]:
    """FM, the first constant-λ FM+CDM arm and the state-only MLP -- whichever of them exist."""
    cdm = [m for m in methods if _slot(m) == 2]
    return [m for m in ["fm", *cdm[:1], "mlp"] if m in methods]


def write_pivot_table(summary: list[dict], sig: list[dict], methods: list[str], path: Path) -> None:
    """Rows: training-set size. Per scenario: MMD and success for FM, the first FM+CDM arm and
    the state-only MLP. Best mean in bold; a dagger marks p < 0.05 against FM (Welch, over
    seeds)."""
    shown = _compact_methods(methods)
    if "fm" not in shown or len(shown) < 2:
        return
    k = len(shown)
    scenarios = [s for s in SCENARIOS if any(r["scenario"] == s for r in summary)]
    sizes = sorted({r["n_train"] for r in summary}, reverse=True)
    idx = {(r["scenario"], r["n_train"], r["method"]): r for r in summary}
    p = {(r["scenario"], r["n_train"], r["method"]): r for r in sig}
    cols = [("mmd_mean", 1e3, "{:.1f}", min), ("success_rate", 100.0, "{:.0f}", max)]

    head1 = " & ".join(rf"\multicolumn{{{2 * k}}}{{c}}{{{TITLES[s]}}}" for s in scenarios)
    rules = "".join(rf"\cmidrule(lr){{{2 + 2 * k * i}-{1 + 2 * k * (i + 1)}}}"
                    for i in range(len(scenarios)))
    head2 = " & ".join([rf"\multicolumn{{{k}}}{{c}}{{MMD$\times10^3$}} & "
                        rf"\multicolumn{{{k}}}{{c}}{{success [\%]}}"] * len(scenarios))
    names = " & ".join(_tex(method_short(m)) for m in shown)
    head3 = " & ".join([f"{names} & {names}"] * len(scenarios))
    lines = [rf"\begin{{tabular}}{{r{'c' * 2 * k * len(scenarios)}}}", r"\toprule",
             rf" & {head1} \\", rules, rf" & {head2} \\", rf"$n$ & {head3} \\", r"\midrule"]
    for n in sizes:
        cells = [str(n)]
        for s in scenarios:
            for key, scale, fmt, better in cols:
                vals = [idx[(s, n, m)][f"{key}_mean"] if (s, n, m) in idx else None for m in shown]
                present = [v for v in vals if v is not None]
                best = better(present) if len(present) == len(shown) else None
                for m, v in zip(shown, vals):
                    if v is None:
                        cells.append("--")
                        continue
                    txt = fmt.format(v * scale)
                    if v == best:
                        txt = rf"\textbf{{{txt}}}"
                    t = p.get((s, n, m))
                    if t is not None and t.get(f"{key}_p", 1.0) < 0.05:
                        txt += r"$^\dagger$"
                    cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def write_baseline_table(summary: list[dict], sig: list[dict], methods: list[str], n: int,
                         path: Path) -> None:
    """Every method at one training-set size -- the one the generative baselines are trained
    at -- a row each, below a rule for the baselines; MMD and success per scenario. Best mean in
    bold; a dagger marks p < 0.05 against FM (Welch, over seeds)."""
    shown = [m for m in methods if any(r["n_train"] == n and r["method"] == m for r in summary)]
    if not any(parse_method(m)["baseline"] for m in shown):
        return
    scenarios = [s for s in SCENARIOS if any(r["scenario"] == s for r in summary)]
    idx = {(r["scenario"], r["n_train"], r["method"]): r for r in summary}
    p = {(r["scenario"], r["n_train"], r["method"]): r for r in sig}
    cols = [("mmd_mean", 1e3, "{:.1f}", min), ("success_rate", 100.0, "{:.0f}", max)]
    best = {}
    for s in scenarios:
        for key, _, _, better in cols:
            present = [idx[(s, n, m)][f"{key}_mean"] for m in shown if (s, n, m) in idx]
            best[(s, key)] = better(present) if present else None

    head1 = " & ".join(rf"\multicolumn{{2}}{{c}}{{{TITLES[s]}}}" for s in scenarios)
    rules = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(scenarios)))
    head2 = " & ".join([r"MMD$\times10^3$ & success [\%]"] * len(scenarios))
    lines = [rf"\begin{{tabular}}{{l{'c' * 2 * len(scenarios)}}}", r"\toprule",
             rf" & {head1} \\", rules, rf"method ($n = {n}$) & {head2} \\", r"\midrule"]
    first_baseline = next(m for m in shown if parse_method(m)["baseline"])
    for m in shown:
        if m == first_baseline and m != shown[0]:
            lines.append(r"\midrule")
        cells = [_tex(method_label(m))]
        for s in scenarios:
            r = idx.get((s, n, m))
            for key, scale, fmt, _ in cols:
                if r is None:
                    cells.append("--")
                    continue
                v = r[f"{key}_mean"]
                txt = fmt.format(v * scale)
                if v == best[(s, key)]:
                    txt = rf"\textbf{{{txt}}}"
                t = p.get((s, n, m))
                if t is not None and t.get(f"{key}_p", 1.0) < 0.05:
                    txt += r"$^\dagger$"
                cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def _glance_lines(summary: list[dict], sig: list[dict], methods: list[str]) -> list[str]:
    """Markdown pivots, one per metric: a row per scenario x n, a column per method, best mean in
    bold, a dagger for p < 0.05 against FM."""
    idx = {(r["scenario"], r["n_train"], r["method"]): r for r in summary}
    p = {(r["scenario"], r["n_train"], r["method"]): r for r in sig}
    cols = [("mmd_mean", "MMD×10³", 1e3, "{:.1f}", min),
            ("success_rate", "success %", 100.0, "{:.0f}", max),
            ("obstacle_collision_rate", "obstacle %", 100.0, "{:.0f}", min)]
    scenarios = [s for s in SCENARIOS if any(r["scenario"] == s for r in summary)]
    lines = []
    for key, label, scale, fmt, better in cols:
        lines += [f"**{label}**", "",
                  "| scen. | n | " + " | ".join(method_label(m) for m in methods) + " |",
                  "|---|---:|" + "---:|" * len(methods)]
        for s in scenarios:
            for n in sorted({r["n_train"] for r in summary if r["scenario"] == s}, reverse=True):
                vals = [idx[(s, n, m)][f"{key}_mean"] if (s, n, m) in idx else None
                        for m in methods]
                present = [v for v in vals if v is not None]
                best = better(present) if len(present) > 1 else None
                cells = [LETTER[s], str(n)]
                for m, v in zip(methods, vals):
                    if v is None:
                        cells.append("—")
                        continue
                    txt = fmt.format(v * scale)
                    t = p.get((s, n, m))
                    if t is not None and t.get(f"{key}_p", 1.0) < 0.05:
                        txt += "†"
                    cells.append(f"**{txt}**" if v == best else txt)
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _method_lines(methods: list[str]) -> list[str]:
    """The methods bullet list; the CDM ablation's entry only once an annealed arm exists."""
    items = [
        "**FM** -- phase-conditioned, `(pose, phase) → [v, ω]`, trained with the flow-matching "
        "loss `E|u − v|²`",
        "**FM+CDM (λ=…)** -- FM plus the conditional divergence matching regularizer at a "
        "constant weight, `L_FM + λ E|a + b|`, with `a = ∇·u − ∇·v` (the demos' kNN divergence "
        "minus the model's SE(2) divergence) and `b = (u − v)·∇log p_ρ` (the twist error "
        "against the demos' kernel score)",
        "**MLP (no phase)** -- the same regression loss with no phase / flow-time input: a plain "
        "state-feedback policy `pose → [v, ω]` (behaviour cloning)",
    ]
    annealed = [parse_method(m) for m in methods if parse_method(m)["cdm_schedule"] == "cosine"]
    if annealed:
        w, frac = annealed[0]["cdm_weight"], annealed[0]["cdm_anneal_frac"]
        items.append(
            f"**FM+div, FM+score, FM+CDM (λ {w:g}→0)** -- the CDM ablation, each term alone and "
            f"both together: the regularizer is `E|a|`, `E|b|` or `E|a + b|`, and λ follows a "
            f"cosine from {w:g} to 0 over the first {frac:.0%} of training, then pure FM. `∇·v` "
            "is the trace over the forward and yaw body directions, the only ones a skate can "
            "drive, and `b` never sees the score's lateral row: both twists' lateral slot is zero")
    no_phase = [parse_method(m) for m in methods
                if not parse_method(m)["use_phase"] and parse_method(m)["cdm_weight"] > 0
                and parse_method(m)["cdm_schedule"] == "cosine"
                and parse_method(m)["baseline"] is None]
    if no_phase:
        q = no_phase[0]
        items.append(
            f"**MLP+{_TERM_LABEL[q['cdm_terms']]} (λ {q['cdm_weight']:g}→0)** -- the phase "
            "ablation of the annealed regularizer: the MLP's network, which never sees the "
            "phase, trained with FM+score's regularizer and schedule. The score is still that of "
            "the demos at the row's phase -- a training-time signal only")
    if CHUNK_METHOD in methods:
        items.append(
            "**FM+chunk** -- FM with action chunking: the same network, loss, optimizer and "
            "steps, but from `(pose, phase)` it predicts the next 16 controls and executes the "
            "first 8 before re-planning, like Diffusion Policy and MIP; trained at the largest n "
            "only, with no regularizer. Its divergence is that of the chunk's first control")
    chunk_reg = [parse_method(m) for m in methods
                 if parse_method(m)["horizon"] > 1 and parse_method(m)["cdm_weight"] > 0]
    if chunk_reg:
        w, frac = chunk_reg[0]["cdm_weight"], chunk_reg[0]["cdm_anneal_frac"]
        items.append(
            f"**FM+chunk+score (λ {w:g}→0)** -- FM+chunk with the score regularizer on every "
            "control of the chunk: the predicted chunk is integrated open loop from the demo "
            "pose (the exact, saturated arc a rollout drives), giving states `x̂⁽ʲ⁾`, and control "
            "`j` pays `|(u_{k+j} − v⁽ʲ⁾)·∇log p_{ρ_{k+j}}(x̂⁽ʲ⁾)|` -- its demo control's twist error "
            "against the kernel score of the training demos at its own phase, evaluated at the "
            "state it would actually be executed from. The terms are averaged over the 16 "
            "controls (as the regression loss is), `x̂` is not backpropagated through, and λ "
            f"follows FM+score's cosine to 0 over the first {frac:.0%} of training")
    ensembled = [m for m in methods if parse_method(m)["temporal_ensemble"] is not None]
    if ensembled:
        m_te = parse_method(ensembled[0])["temporal_ensemble"]
        names = ", ".join(method_label(_te_base(m)[0]) for m in ensembled)
        items.append(
            f"**…+TE** ({names}) -- the same trained policies, re-evaluated with ACT's temporal "
            "ensembling instead of executing 8 controls per chunk: a chunk is predicted at every "
            "control step, and the control executed at step `t` is the weighted mean of every "
            f"prediction made so far for `t`, weights `exp(−m·i)` with `i = 0` the oldest and "
            f"m = {m_te:g} (ACT's default). Nothing is retrained; the held-out starts, clamping "
            "and sampling seeds are the original evaluation's")
    if any(parse_method(m)["baseline"] for m in methods):
        items.append(
            "**Diffusion Policy, MIP, MeanFlow** -- the generative baselines, trained at the "
            "largest n only. From the last two poses (no phase) each samples a chunk of 16 "
            "controls and executes the first 8 before re-planning. Diffusion Policy is a 100-step "
            "DDPM on its 1D temporal UNet; MIP (Much Ado About Noising) is two regression steps "
            "on its residual MLP, the second from the first's output with noise injected in "
            "training; MeanFlow is a one-step average-velocity model on the Diffusion Policy "
            "UNet. Each keeps its own paper's optimizer, batch size and EMA (evaluated with the "
            "EMA weights) over a shared step budget. They have no velocity field of their own, "
            "so the divergence figure shows only their rollouts")
    return ["* **methods** -- every method trains on the same demos and training subsets. FM, "
            "the MLP and the CDM arms share the network (3×256 SiLU MLP → skate controls "
            "`[v, ω]`) and the optimizer. The figures call FM "
            f"**{FIGURE_FM}**, after its network (FM+CDM is {FIGURE_FM}+CDM, and so on):"] + [
        f"  * {text}{';' if i < len(items) - 1 else '.'}" for i, text in enumerate(items)]


def _inference_lines(out: Path) -> list[str]:
    """The inference-cost table from ``inference.csv`` (scripts/skate/benchmark_inference.py), if it
    has been measured. A decision is one action for the per-step policies and one chunk for the
    generative baselines, which amortize it over the steps they execute before re-planning."""
    rows = read_inference(out)
    if not rows:
        return []
    lines = [
        "## Inference cost", "",
        "Median of repeated calls on the trained policies. A **decision** is one action for FM "
        "and the MLP, and one 8-step action chunk for FM+chunk and the generative baselines, "
        "which re-plan "
        "only once the chunk has been executed — so the per-step cost is what a robot pays at "
        "each control tick, and the rate in brackets is what that latency allows (the "
        "demonstrations run at 6.7 Hz); the dashed lines mark two latency budgets, a 60 Hz "
        "visual-feedback loop and a 1 kHz tactile one, and a bar that ends left of a "
        "line fits inside that budget. The last panel puts success against that cost — the "
        "point is a method's mean over **all three scenarios** at the largest training-set size "
        "(each already a mean over its seeds) and the whisker spans the individual scenarios, so "
        "a method that carries its average on two tasks and fails the third shows a long "
        "whisker. The policy is the same on both devices, so a method's two points differ only "
        "in what the hardware costs it. Diffusion Policy and MIP appear twice there: as trained, "
        "running each 8-step chunk open loop between re-plans (circles), and closed loop with "
        "ACT's temporal ensembling (their +TE re-evaluations, squares), which predicts a chunk "
        "at every control step and so pays a whole decision per step. MLP+phase+score (and MLP+phase+chunk+score) is not "
        "timed on its own: its regularizer only acts in training, so at inference it runs "
        "MLP+phase's (MLP+phase+chunk's) network unchanged and is plotted at that measured "
        "cost.", "",
        "![inference cost](fig_policy_inference.png)", "",
    ]
    order = table_order({r["method"] for r in rows})
    for device in sorted({r["device"] for r in rows}, reverse=True):  # cuda first
        here = sorted((r for r in rows if r["device"] == device),
                      key=lambda r: order.index(r["method"]))
        r0 = here[0]
        lines += [
            f"**{r0['hardware']}** (torch {r0['torch']})", "",
            "| method | params | net calls per decision | ms per decision (batch 1) | ms per "
            f"control step (batch 1) | ms per decision (batch {r0['batch_n']}) | s per "
            f"{r0['horizon']}-step rollout (batch {r0['batch_n']}) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in here:
            lines.append("| " + " | ".join([
                method_label(r["method"]), f"{int(r['params']) / 1e6:.2f} M",
                r["net_calls_per_decision"], f"{float(r['ms_decision_b1']):.2f}",
                f"{float(r['ms_step_b1']):.2f}", f"{float(r['ms_decision_bN']):.1f}",
                f"{float(r['s_rollout']):.2f}",
            ]) + " |")
        lines.append("")
    return lines


def read_inference(out: Path) -> list[dict]:
    """Rows of ``inference.csv`` (scripts/skate/benchmark_inference.py), or ``[]`` if not
    measured."""
    path = out / "inference.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f)
                if r["method"] in {"fm", "mlp", CHUNK_METHOD, *BASELINE_METHODS}]


def write_markdown(summary: list[dict], sig: list[dict], sizes: list[int], methods: list[str],
                   limits, path: Path, note: str | None = None) -> None:
    p = {(r["scenario"], r["n_train"], r["method"]): r for r in sig}
    box = (f"v ∈ [{limits.v_min:g}, {limits.v_max:g}] m/s, |ω| ≤ {limits.omega_max:g} rad/s")
    if all(r["clamped"] for r in summary):
        execution = (f"* **execution** -- commanded controls are saturated to the planner's "
                     f"`SkateLimits` ({box}) before the robot executes them; collisions and all "
                     f"metrics are measured on the path actually driven. **saturated** is the "
                     f"share of rollouts whose commands left that box (beyond the demos' 0.05 "
                     f"tolerance) at least once.")
    else:
        execution = (f"* **execution** -- some or all runs executed raw, *unclamped* commands; "
                     f"the planner's limits are {box}. **saturated** is the share of rollouts "
                     f"whose commands left that box at least once.")

    def pm(r, key, scale, fmt="{:.1f}"):
        return f"{fmt.format(r[f'{key}_mean'] * scale)} ± {fmt.format(r[f'{key}_std'] * scale)}"

    def pval(t, key):
        if t is None or not np.isfinite(t.get(f"{key}_p", np.nan)):
            return "—"
        v = t[f"{key}_p"]
        return f"**{v:.3f}**" if v < 0.05 else f"{v:.3f}"

    lines = [
        "# Policy evaluation on the SE(2) skate scenarios",
        "",
        *([f"> {note}", ""] if note else []),
        "Generated by `scripts/skate/evaluate_policies.py`. Mean ± std over seeds.",
        "",
        *_method_lines(methods),
        "* **MMD** -- squared MMD ×10³ between rollouts and held-out demos, averaged over phase "
        "(lower is better).",
        "* **obstacle / wall** -- rollouts that come within the robot radius of an obstacle / "
        "the world boundary.",
        "* **success** -- reaches a goal (within 3σ) with no collision of either kind.",
        "* **modes** -- fraction of rollouts per homotopy class (demos are 50/50).",
        execution,
        "* **p** -- Welch t-test against FM over seeds; bold if < 0.05.",
        "",
        "![summary vs. training-set size](fig_policy_vs_ntrain.png)",
        "",
        f"![summary at n = {max(sizes)}, per seed](fig_policy_violin_n{max(sizes)}.png)",
        "",
        f"![regularizer ablation at n = {max(sizes)}, per seed]"
        f"(fig_policy_violin_n{max(sizes)}_ablation.png)",
        "",
        "## At a glance",
        "",
        "Means over seeds; best of the methods in bold; † p < 0.05 against FM; — not trained.",
        "",
        *_glance_lines(summary, sig, methods),
        *_inference_lines(path.parent),
        "## All metrics",
        "",
        "| scen. | n | method | seeds | MMD×10³ | obstacle % | wall % | success % | modes % "
        "| saturated % | div MAE [1/s] | p (MMD) | p (success) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in _sorted_rows(summary):
        t = p.get((r["scenario"], r["n_train"], r["method"]))
        modes = "/".join(f"{100 * r[k]:.0f}" for k in sorted(r) if k.endswith("_frac_mean"))
        lines.append(" | ".join([
            f"| {LETTER[r['scenario']]}", str(r["n_train"]), method_label(r["method"]),
            str(r["n_seeds"]), pm(r, "mmd_mean", 1e3, "{:.2f}"),
            pm(r, "obstacle_collision_rate", 100), pm(r, "boundary_collision_rate", 100),
            pm(r, "success_rate", 100), modes, pm(r, "saturation_rate", 100),
            pm(r, "div_curve_mae", 1, "{:.3f}"),
            pval(t, "mmd_mean"), pval(t, "success_rate") + " |",
        ]))
    scenarios = {r["scenario"] for r in summary}
    for n in sizes:
        lines += ["", f"## n = {n} training demos", ""]
        for name in ("rollouts", "mmd_phase", "training", "divergence"):
            lines.append(f"![{name} n={n}](fig_policy_{name}_n{n}.png)")
            if name == "rollouts" and n == max(sizes) and set(COMPACT_METHODS) & set(methods):
                lines += [f"![rollouts {tag} n={n}](fig_policy_rollouts_n{n}_{tag}.png)"
                          for tag, _ in _compact_groups(scenarios)]
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# -- figures ---------------------------------------------------------------------------------


def _fig_legend(fig, axes, ncol: int | None = None, fontsize=7, rows_first: bool = False,
                **kw) -> None:
    """One figure-level legend gathered from every panel -- a panel can be empty when a
    scenario has no finished runs at some training-set size. ``rows_first``: the entries read
    across the rows rather than down the columns. ``kw`` goes to ``fig.legend``."""
    seen: dict = {}
    for ax in np.ravel(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    if seen:
        labels = list(seen)
        ncol = min(ncol or len(labels), len(labels))
        if rows_first:  # matplotlib fills the columns in turn
            labels = [labels[i] for c in range(ncol) for i in range(c, len(labels), ncol)]
        fig.legend([seen[s] for s in labels], labels, loc="upper center", ncol=ncol,
                   fontsize=fontsize, frameon=False, **kw)


def fig_vs_ntrain(summary: list[dict], scenarios, methods, out: Path) -> None:
    """Every summary metric against training-set size -- the data-efficiency view."""
    sizes = sorted({r["n_train"] for r in summary})
    if len(sizes) < 2:
        return
    idx = {(r["scenario"], r["n_train"], r["method"]): r for r in summary}
    rows = len(SUMMARY_METRICS)
    fig, axes = plt.subplots(rows, len(scenarios), figsize=(3.3 * len(scenarios), 1.85 * rows + 0.5),
                             sharex=True, squeeze=False, facecolor=PALETTE["surface"])
    for c, scenario in enumerate(scenarios):
        for r, (key, label, scale) in enumerate(SUMMARY_METRICS):
            ax = axes[r, c]
            for i, method in enumerate(methods):
                pts = [idx[(scenario, n, method)] for n in sizes if (scenario, n, method) in idx]
                if not pts:
                    continue
                x = np.array([q["n_train"] for q in pts])
                mu = np.array([q[f"{key}_mean"] for q in pts]) * scale
                sd = np.array([q[f"{key}_std"] for q in pts]) * scale
                st = method_style(method, methods)
                ax.errorbar(x, mu, yerr=sd, color=st["color"], ls=st["ls"], lw=1.4, marker="o",
                            ms=5, capsize=2, label=figure_label(method))
            ax.set_xscale("log", base=2)
            ax.set_xlim(sizes[0] / 1.4, sizes[-1] * 1.4)
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(s) for s in sizes])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_ylim(bottom=0)
            _style(ax, "training demos" if r == rows - 1 else None, label if c == 0 else None)
            if r == 0:
                ax.set_title(TITLES.get(scenario, scenario), fontsize=9, loc="left",
                             color=PALETTE["ink"])
    _fig_legend(fig, axes)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    _save(fig, out, "fig_policy_vs_ntrain")


def _loop(method: str) -> str | None:
    """"open" for a chunk baseline as trained (8 controls open loop per re-plan), "closed" for
    its temporally ensembled re-evaluation (a chunk every step), ``None`` for any other arm."""
    base, te = _te_base(method)
    if parse_method(base)["baseline"] is None:
        return None
    return "open" if te is None else "closed"


def _violin_notes(n: int, seeds: str, base: str, log: bool, loops: bool,
                  lam: list[str]) -> list[str]:
    """What :func:`fig_violin` is a figure of: the conditions behind its violins and daggers.

    Caption prose, not figure furniture -- it goes in the ``.caption.txt`` beside the file
    rather than into a header block, which cost the panels the better part of an inch and set
    the same sentences at a size nobody reads them at.
    """
    notes = [f"{n} training demos; each violin: its {seeds} seeds (dots) and their mean (bar)"]
    if lam:
        notes.append("regularizer arms: " + ", ".join(lam))
    if loops:
        notes.append("the chunk baselines run open loop, as trained, and closed loop "
                     "(temporal ensembling)")
    if log:
        notes.append("MMD$^2$ is drawn on a log axis")
    notes.append(f"† marks $p < 0.05$ against {figure_label(base)} (Welch over seeds)")
    return notes


def fig_violin(runs: dict, scenarios, n: int, methods, out: Path, base: str = "fm",
               keep=None, name: str | None = None, font_scale: float = 1.0) -> None:
    """The summary metrics at one training-set size, one violin per method over its seeds -- the
    spread behind fig_vs_ntrain's error bars. A violin over a handful of seeds is only a smoothed
    outline, so the seeds are drawn as dots and their mean as a bar; a dagger marks p < 0.05
    against ``base`` (Welch, as in the tables). FM with action chunking is left out; the chunk
    baselines are shown open loop (as trained) and closed loop (temporal ensembling). The
    metrics in ``VIOLIN_LOG_METRICS`` take a log y axis, KDE included (:func:`_violin`).

    A row per scenario, named down the left; a column per metric, titled across the top. Every
    arm is named in full under its own violin, rotated, so the figure carries no legend -- which
    is what leaves the room to set the rest of it at a size that reads. ``keep(method)`` narrows
    the arms further, ``name`` is the file name, and ``font_scale`` multiplies every type size
    for a figure meant to be read at less than the full width of a page.
    """
    def pt(size: float) -> float:
        return size * font_scale

    shown = [m for m in methods if parse_method(m)["horizon"] == 1
             and (keep is None or keep(m)) and any(runs.get((s, n, m)) for s in scenarios)]

    # How many arms each base contributes: a baseline drawn open and closed loop has two, and
    # only then does a label need to say which loop it is.
    drawn = Counter(_te_base(m)[0] for m in shown)

    def tick(m: str) -> str:
        arm, loop = _te_base(m)[0], _loop(m)
        p = parse_method(arm)
        if p["baseline"] is not None:
            short = _BASELINE_SHORT[p["baseline"]]
        else:  # method_short writes a phase arm as its modifiers alone, under an FM column
            short = method_short(arm, FIGURE_FM)
            if p["use_phase"] and not short.startswith(FIGURE_FM):
                short = FIGURE_FM + short
        return f"{short} ({loop})" if loop and drawn[arm] > 1 else short

    if not shown:
        return
    p = {(r["scenario"], r["n_train"], r["method"]): r for r in significance(runs, base)}
    counts = sorted({len(runs[(s, n, m)]) for s in scenarios for m in shown if (s, n, m) in runs})
    cols = len(SUMMARY_METRICS)
    # ``{column: [min, max]}`` of what a log column actually plots -- see the y-range loop below.
    log_span: dict[int, list[float]] = {}
    width = cols * max(3.6, VIOLIN_SLOT * len(shown))
    seeds = str(counts[0]) if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
    # Built first: it wraps to the figure's width, and a narrower figure takes more lines and so
    # a taller header.
    lam = sorted({(q["cdm_schedule"], q["cdm_weight"]) for m in shown
                  if (q := parse_method(m))["baseline"] is None and q["cdm_weight"] > 0})
    caption = _caption_tex(_violin_notes(
        n, seeds, base, bool(VIOLIN_LOG_METRICS & {k for k, _, _ in SUMMARY_METRICS}),
        any(_loop(m) for m in shown),
        [rf"$\lambda = {w:g}$ held constant" if sch == "constant"
         else rf"$\lambda\,{w:g} \to 0$ annealed" for sch, w in lam]))
    # Header: the column titles alone -- the conditions are in the caption file beside the
    # figure. Footer: the rotated labels. Both are type, so both scale with it; the left margin
    # holds the row names, y label and tick labels.
    head = pt(0.42)
    foot = pt(1.15)
    height = head + foot + VIOLIN_ROW * len(scenarios)
    fig = plt.figure(figsize=(width, height), facecolor=PALETTE["surface"])
    grid = fig.add_gridspec(len(scenarios), cols, hspace=0.16, wspace=0.09 * cols,
                            top=1 - head / height, bottom=foot / height,
                            left=pt(1.05) / width, right=0.995)
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(cols)]
                     for r in range(len(scenarios))], dtype=object)
    for r, scenario in enumerate(scenarios):
        groups = [runs.get((scenario, n, m), []) for m in shown]
        for c, (key, label, scale) in enumerate(SUMMARY_METRICS):
            ax = axes[r, c]
            _style(ax, None, None, size=pt(11))
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            ax.set_ylabel(label, fontsize=pt(10.5), color=PALETTE["ink"])
            if r == 0:
                ax.set_title(VIOLIN_TITLES.get(key, key), fontsize=pt(12), loc="left",
                             color=PALETTE["ink"])
            log = key in VIOLIN_LOG_METRICS
            if log:
                # Ticks written out ("10", not "$10^1$"); which ones is settled with the y range,
                # below. Unlabelled marks on the decade's 2-9 in between, so a violin's height
                # can still be read where the labels are a whole decade apart.
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_locator(
                    LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=99))
                ax.yaxis.set_minor_formatter(NullFormatter())
            dagger_y = blended_transform_factory(ax.transData, ax.transAxes)
            span = log_span.setdefault(c, [np.inf, 0.0]) if log else None
            for i, (method, group) in enumerate(zip(shown, groups)):
                vals = np.array([g["metrics"].get(key, np.nan) for g in group], dtype=float) * scale
                # A log panel has no room for a zero, which an arm that matched the demos
                # exactly would give; none does, so this only guards the axis.
                vals = vals[np.isfinite(vals) & (vals > 0)] if log else vals[np.isfinite(vals)]
                if not len(vals):
                    continue
                if span is not None:
                    span[:] = [min(span[0], vals.min()), max(span[1], vals.max())]
                st = method_style(method, methods)
                if np.ptp(vals) > 0:  # the KDE of identical values is singular: dots only
                    _violin(ax, vals, i, VIOLIN_WIDTH, st, log)
                ax.hlines(vals.mean(), i - 0.25, i + 0.25, color=PALETTE["ink"], lw=1.2, zorder=3)
                ax.scatter(i + np.linspace(-0.1, 0.1, len(vals)), vals, s=16, color=st["color"],
                           edgecolors=PALETTE["surface"], linewidths=0.6, zorder=4)
                t = p.get((scenario, n, method))
                if t is not None and t.get(f"{key}_p", 1.0) < 0.05:
                    ax.text(i, 0.98, "†", transform=dagger_y, ha="center", va="top",
                            fontsize=pt(12), color=PALETTE["ink"])
            ax.set_xlim(-0.6, len(shown) - 0.4)
            ax.set_xticks(np.arange(len(shown)))
            if r == len(scenarios) - 1:
                # Rotated and right-anchored at the tick: a label may be as long as its arm's
                # name without crowding its neighbour, which is what lets the slots be narrow.
                ax.set_xticklabels([tick(m) for m in shown], fontsize=pt(10),
                                   rotation=VIOLIN_ROTATION, ha="right",
                                   rotation_mode="anchor")
                for text, m in zip(ax.get_xticklabels(), shown):
                    if m in BOLD_METHODS:
                        text.set_fontweight("bold")
            else:
                ax.set_xticklabels([])
    # One y range per column, so a metric can be read across scenarios at one scale. Set
    # explicitly rather than through a shared axis, which would also hide the tick labels of
    # every column but the first -- and each column here is its own metric and scale.
    for c in range(cols):
        if c in log_span:
            # Straight off the data rather than off the axis: a log axis's own autoscale pads by
            # a margin that reads as a fraction of the decades it already has, which on a column
            # this tall costs the better part of one at each end and squeezes every arm into the
            # middle. The margins below are a fraction of the *data*'s decades, the top one the
            # headroom the daggers need.
            lo, hi = log_span[c]
            lo, hi = lo * (lo / hi) ** 0.05, hi * (hi / lo) ** 0.15
        else:
            lo = min(axes[r, c].get_ylim()[0] for r in range(len(scenarios)))
            hi = max(axes[r, c].get_ylim()[1] for r in range(len(scenarios)))
            hi += 0.15 * (hi - lo)  # headroom for the daggers
        for r in range(len(scenarios)):
            axes[r, c].set_ylim(lo, hi)
            if c in log_span:
                # These panels are two inches high: four labels fit, so a column spanning more
                # than two decades is labelled by the decade and a shorter one also on its 2 and
                # 5. ``numticks`` is given because LogLocator's default is read off the axis
                # height, and at this one's it asks for every other decade -- a stride it answers
                # with no ticks at all whenever ``subs`` asks for a decade's subdivisions too.
                subs = (1.0,) if np.log10(hi / lo) > 2 else (1.0, 2.0, 5.0)
                axes[r, c].yaxis.set_major_locator(
                    LogLocator(base=10.0, subs=subs, numticks=99))
    # The scenario names the row, down the figure's left edge: a title would have to sit above
    # the leftmost panel, where the metric's own title belongs.
    for r, scenario in enumerate(scenarios):
        box = axes[r, 0].get_position()
        fig.text(0.004, (box.y0 + box.y1) / 2, TITLES.get(scenario, scenario), rotation=90,
                 ha="left", va="center", fontsize=pt(13), fontweight="bold",
                 color=PALETTE["ink"])
    _save(fig, out, name or f"fig_policy_violin_n{n}", tight=False, caption=caption)


def _compact_groups(scenarios) -> list[tuple[str, list[str]]]:
    """``(tag, scenarios)`` of each compact rollout figure whose scenarios all have runs."""
    return [("".join(LETTER[s] for s in group), group) for group in COMPACT_GROUPS
            if all(s in scenarios for s in group)]


def _compact_title(name: str) -> str:
    """A compact panel title over two lines: the arm's name, then its qualifier in brackets.

    Every title takes both lines, the second one empty where there is no qualifier, so that the
    panels' titles start at the same height and their tops stay in line. Splitting this way is
    what lets the titles be set large: no line has to fit a whole name plus its qualifier, and
    the longest that remains (Diffusion Policy, MLP+phase+score) fits a panel's width.
    """
    head, bracket, qualifier = name.partition(" (")
    return f"{head}\n{bracket.lstrip()}{qualifier}" if bracket else f"{name}\n"


def _scenario_head(scenario: str) -> str:
    """The scenario as the width-sized rollout figure heads its row -- "A) Single Path" -- set
    bold italic in mathtext, so the "held-out demos" line under it keeps the title's style."""
    words = TITLES.get(scenario, scenario).split(" · ")[-1].title().replace(" ", r"\ ")
    return rf"$\mathbfit{{{LETTER[scenario]})\ {words}}}$"


def _rollout_panels(grid, runs, tests, scenarios, n, methods, compact: bool = False,
                    sizes: tuple[float, float, float] | None = None) -> None:
    """``grid[i][0]``: scenario i's held-out demos; ``grid[i][j]``: method j's seed-0 rollouts.
    The callers lay the grid out either way round. ``compact``: larger text, no axis labels, ticks
    or grid, and the chunk baselines named as run open loop. ``sizes``: the compact figure's
    (title, legend, row) points, in place of the ``COMPACT_*_PT`` defaults."""
    # A grid of one scenario needs no letter on every panel; any grid of several does -- C's
    # included, which used to go unlettered wherever it appeared. A compact grid is the exception:
    # its rows are the scenarios, so the letter is drawn once down the left edge instead, which
    # takes it off the front of every title and leaves those the room to be set large.
    lettered = len(scenarios) > 1 and not compact
    for i, scenario in enumerate(scenarios):
        test = tests[scenario]
        if lettered:
            plot_title = f"{LETTER[scenario]} · held-out demos"
        else:
            plot_title = "Held-out demos"
        plot_rollouts(test, test.poses, test.mode, np.zeros(len(test), bool), ax=grid[i][0],
                      title=plot_title)
        for j, method in enumerate(methods, start=1):
            group = runs.get((scenario, n, method))
            ax = grid[i][j]
            if not group:
                ax.set_axis_off()
                continue
            run = group[0]
            arr = np.load(run["dir"] / "rollouts.npz")
            name = figure_label(method)
            if compact and _loop(method):
                name += f" ({_loop(method)} loop)"
            if not compact:
                name += f" (seed {run['seed']})"
            plot_title = f"{LETTER[scenario]} · {name}" if lettered else name
            plot_rollouts(test, arr["poses"], arr["labels"], arr["collided"], ax=ax,
                          title=plot_title)
    # The compact figures carry a handful of methods and are printed large, so their titles and
    # legends take more than the full grid's, which has a column per method and no room to spare.
    title_pt, legend_pt, row_pt = sizes or (COMPACT_TITLE_PT, COMPACT_LEGEND_PT, COMPACT_ROW_PT)
    # Type set from the width is too large to share a panel with the rollouts: the legend goes
    # under the panel, the scenario into its first panel's title rather than down the edge (a
    # row is shorter than its name at that size), and a name too long for a column breaks at
    # its last "+".
    below = sizes is not None
    if not compact:
        title_pt, legend_pt = 10, 6
    for i, scenario in enumerate(scenarios):
        # grid[i][0] is the held-out demos; column j past it is methods[j - 1].
        for j, ax in enumerate(grid[i]):
            # plot_rollouts sets the title left-aligned, so the drawn Text is not ``ax.title``
            # (which stays empty and centered) -- re-set it through the same loc instead.
            title = ax.get_title(loc="left")
            if compact:
                title = _compact_title(title)
            if below:
                name, _, rest = title.partition("\n")
                if j == 0:
                    title = f"{_scenario_head(scenario)}\n{name.lower()}"
                elif len(name) > 12 and "+" in name:
                    cut = name.rindex("+")
                    title = f"{name[:cut]}\n{name[cut:]} {rest}".rstrip()
            ax.set_title(title, loc="left",
                         fontsize=title_pt, color=PALETTE["ink"], linespacing=1.25,
                         fontweight="bold" if j and methods[j - 1] in BOLD_METHODS else "normal")
            leg = ax.get_legend()
            if not compact:
                if leg is not None:
                    for t in leg.get_texts():
                        t.set_fontsize(legend_pt)
                continue
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if leg is None:
                continue
            # Every panel shows the same number of rollouts, so no legend title; the legend sits
            # in the corner the scenario leaves empty, clear of the start lane: below A's lower
            # corridor wall, on B's and C's lower-left obstacle.
            handles, labels = ax.get_legend_handles_labels()
            labels = [t.replace(" goal", "") for t in labels]  # B's modes as C's: left / right
            for long, short in LEGEND_SHORT.items():
                labels = [t.replace(long, short) for t in labels]
            leg.remove()
            if below:
                ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(-0.03, 0.0),
                          fontsize=legend_pt, handlelength=1.0, handletextpad=0.4,
                          labelspacing=0.2, borderpad=0.1, borderaxespad=0.1, frameon=False)
                continue
            ax.legend(handles, labels,
                      loc="lower right" if scenario == "single_path" else "lower left",
                      fontsize=legend_pt, handlelength=1.0, handletextpad=0.4,
                      borderaxespad=0.2, labelspacing=0.25, framealpha=0.92,
                      facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"]).set_zorder(10)
    if compact and len(scenarios) > 1 and not below:
        # The scenario names its row down the left edge, where its panels no longer carry it:
        # its letter and what that letter is, as fig_violin names its own rows.
        for i, scenario in enumerate(scenarios):
            grid[i][0].set_ylabel(TITLES.get(scenario, scenario), fontsize=row_pt,
                                  fontweight="bold", color=PALETTE["ink"], labelpad=8)


def fig_rollouts(runs, tests, scenarios, n, methods, out):
    rows = 1 + len(methods)
    fig, axes = plt.subplots(rows, len(scenarios), figsize=(3.6 * len(scenarios), 2.95 * rows),
                             facecolor=PALETTE["surface"], squeeze=False)
    _rollout_panels(axes.T, runs, tests, scenarios, n, methods)
    fig.tight_layout()
    _save(fig, out, f"fig_policy_rollouts_n{n}")


def fig_rollouts_by_scenario(runs, tests, scenarios, n, methods, out, tag, by_width=False):
    """fig_rollouts turned round for a few methods: a row per scenario, a column for its
    held-out demos and one per method, each the rollouts of the method's first seed.
    ``by_width``: type sized from the figure's width (``TYPE_RATIOS``), not ``COMPACT_*_PT``."""
    cols = 1 + len(methods)
    width = COMPACT_PANEL_W * cols
    sizes = ((type_pt(width, "title"), type_pt(width, "legend"), type_pt(width, "label"))
             if by_width else None)
    fig, axes = plt.subplots(len(scenarios), cols,
                             figsize=(width, COMPACT_PANEL_H * len(scenarios)),
                             facecolor=PALETTE["surface"], squeeze=False)
    _rollout_panels(axes, runs, tests, scenarios, n, methods, compact=True, sizes=sizes)
    if by_width:
        # For a page short of space, placed by hand rather than by tight_layout, which shrinks
        # the aspect-locked panels and leaves the difference as gutter: every panel its column's
        # full width less a hairline gap, and every row exactly as tall as its two-line titles,
        # its world at that width and its longest legend under it. The save crops to the ink.
        pitch = width / cols
        panel_w = pitch - 0.1
        title_h = 2 * 1.3 * sizes[0] / 72 + 0.04
        blocks = []
        for i in range(len(scenarios)):
            aspect = np.ptp(axes[i, 0].get_ylim()) / np.ptp(axes[i, 0].get_xlim())
            lines = max(len(ax.get_legend().get_texts()) for ax in axes[i]
                        if ax.get_legend() is not None)
            blocks.append((panel_w * aspect, lines * 1.2 * sizes[1] / 72 + 0.05))
        height = sum(title_h + ph + lh for ph, lh in blocks)
        fig.set_figheight(height)
        top = height
        for i, (ph, lh) in enumerate(blocks):
            top -= title_h + ph
            for j, ax in enumerate(axes[i]):
                ax.set_position([j * pitch / width, top / height, panel_w / width, ph / height])
            top -= lh
    else:
        fig.tight_layout(h_pad=1.2, w_pad=1.0)
    _save(fig, out, f"fig_policy_rollouts_n{n}_{tag}", pad=0.02 if by_width else 0.1)


# fig_rollouts_pair: the two arms the compact figure's comparison is about, stacked and stripped
# to the picture -- the skate twin of plot_pusht's fig_final_poses_pair, down to its frame.
PAIR_SCENARIO = "dual_path"
PAIR_METHODS = ["mlp", VIOLIN_BASE]
# Panel width in inches; the height follows from the world's aspect, which the panels keep.
PAIR_PANEL_W = 3.2
# With no titles to tell one panel from the next, the frame does the separating.
PAIR_FRAME_LW = 3.5


def fig_rollouts_pair(runs, tests, n, out) -> None:
    """The MLP above MLP+phase+score on scenario C: their two compact panels, stacked and alone.

    The figure the compact grid cannot be -- one arm directly over the other, at the size a slide
    or a two-column inset wants, where the surrounding text names the two panels. Stripped to the
    rollouts: no titles, no legend, no axes, and a heavy black border in place of the titles that
    would otherwise say where one panel ends and the next begins.
    """
    groups = [runs.get((PAIR_SCENARIO, n, m)) for m in PAIR_METHODS]
    if not all(groups):
        return
    test = tests[PAIR_SCENARIO]
    xmin, ymin, xmax, ymax = test.spec.world.bounds
    panel_h = PAIR_PANEL_W * (ymax - ymin) / (xmax - xmin)  # the panels are the world's shape
    fig, axes = plt.subplots(len(PAIR_METHODS), 1, squeeze=False,
                             figsize=(PAIR_PANEL_W, panel_h * len(PAIR_METHODS)),
                             facecolor=PALETTE["surface"])
    for ax, group in zip(axes[:, 0], groups):
        arr = np.load(group[0]["dir"] / "rollouts.npz")
        plot_rollouts(test, arr["poses"], arr["labels"], arr["collided"], ax=ax, legend=False)
        ax.set_title("", loc="left")  # the drawn title is the left-aligned one, not ax.title
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(True)  # _setup_axes drops the top and right of a world panel
            side.set_color("black")
            side.set_linewidth(PAIR_FRAME_LW)
    fig.subplots_adjust(hspace=0.06, left=0.01, right=0.99, top=0.99, bottom=0.01)
    _save(fig, out, f"fig_policy_rollouts_pair_n{n}_{LETTER[PAIR_SCENARIO]}")


def fig_mmd_phase(runs, scenarios, n, methods, out):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.4 * len(scenarios), 2.5),
                             facecolor=PALETTE["surface"], squeeze=False)
    for c, scenario in enumerate(scenarios):
        ax = axes[0, c]
        for i, method in enumerate(methods):
            group = runs.get((scenario, n, method))
            if not group:
                continue
            series = np.array([r["metrics"]["mmd_per_phase"] for r in group]) * 1e3
            _band(ax, np.linspace(0, 1, series.shape[1]), series, method_style(method, methods),
                  figure_label(method))
        _style(ax, "phase", r"MMD$^2\times10^3$" if c == 0 else None)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{LETTER[scenario]}", fontsize=9, loc="left", color=PALETTE["ink"])
    _fig_legend(fig, axes)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, out, f"fig_policy_mmd_phase_n{n}")


def fig_training(runs, scenarios, n, methods, out, layout: str = "column",
                 name: str | None = None):
    """MMD and the velocity loss against training step, drawn at a two-column paper's own widths
    (``COLUMN_WIDTH``, ``TEXT_WIDTH``) and saved uncropped, so that the text prints at its own
    size. ``layout="column"``: a row per scenario, one column wide; ``"row"``: a column per
    scenario across the full text width, saved with the suffix ``_row``; ``"single_row"``: the
    row layout's two rows side by side -- every MMD panel, then every velocity-loss panel --
    across the text width at the least height, saved with the suffix ``_single_row``. ``name``
    overrides the file name, for a second cut of the same layout over a different set of arms."""
    suffix = {"column": "", "row": "_row", "single_row": "_single_row"}
    if layout not in suffix:
        raise ValueError(f"layout must be one of {', '.join(suffix)}, not {layout!r}")
    column, single = layout == "column", layout == "single_row"
    k = len(scenarios)
    # The legend wraps to as many rows as the drawn arms need; reserve a row's height for each
    # instead of a fixed two rows' worth, which the panel titles used to be written over once the
    # figure carried more than six arms. Four arms still give two rows -- the paper figure's own
    # size is unchanged.
    drawn = [m for m in methods if any(runs.get((sc, n, m)) for sc in scenarios)]
    # Four arms or fewer go on a single legend row across the text width; more wrap to three
    # columns, since the labels carry a λ and would not fit side by side.
    legend_ncol = 2 if column else (len(drawn) if 0 < len(drawn) <= 4 else 3)
    legend_rows = max(1, -(-len(drawn) // legend_ncol))
    legend_h = (0.21 if column else 0.18) * legend_rows
    if column:
        fig, axes = plt.subplots(k, 2, figsize=(COLUMN_WIDTH, 1.25 * k + 0.23 + legend_h),
                                 facecolor=PALETTE["surface"], squeeze=False, sharex=True)
    elif single:
        # Constrained layout sizes each gap for what is in it -- an axis label before the
        # velocity-loss panels, no labels at all between them -- where tight_layout would make
        # every gap as wide as the widest.
        fig, axes = plt.subplots(1, 2 * k, figsize=(TEXT_WIDTH, 1.44 + legend_h),
                                 facecolor=PALETTE["surface"], squeeze=False, sharex=True,
                                 layout="constrained")
        # Side by side, the velocity-loss panels share one (log) y scale, labelled on the first
        # alone, which hands the room the others' tick labels would take to the data. The MMD
        # scales differ too much to share: A's is about an eighth of B's and C's.
        for ax in axes[0, k + 1:]:
            ax.sharey(axes[0, k])
            ax.tick_params(labelleft=False)
    else:
        fig, axes = plt.subplots(2, k, figsize=(TEXT_WIDTH, 2.94 + legend_h),
                                 facecolor=PALETTE["surface"], squeeze=False, sharex=True)
    # grid[i]: scenario i's MMD and velocity-loss panels
    grid = axes if column else axes.reshape(2, k).T
    for i, scenario in enumerate(scenarios):
        for method in methods:
            group = runs.get((scenario, n, method))
            if not group:
                continue
            log0 = group[0]["log"]
            evald = np.isfinite(log0["mmd_mean"])
            mmd = np.array([g["log"]["mmd_mean"][evald] for g in group]) * 1e3
            fm = np.array([g["log"]["velocity_loss"] for g in group])
            style = method_style(method, methods)
            _band(grid[i, 0], log0["step"][evald], mmd, style, figure_label(method), lw=1.2)
            _band(grid[i, 1], log0["step"], fm, style, figure_label(method), lw=0.8)
        for m, ax in enumerate(grid[i]):
            # "training step" under the bottom row; in one row, under each metric's middle panel.
            bottom = {"column": i == k - 1, "row": m == 1, "single_row": i == k // 2}[layout]
            _style(ax, "training step" if bottom else None, size=9)
        grid[i, 0].set_ylim(bottom=0)
        grid[i, 1].set_yscale("log")
        letter = dict(fontsize=10, fontweight="bold", color=PALETTE["ink"])
        if column:
            grid[i, 0].set_ylabel(LETTER[scenario], rotation=0, ha="right", va="center", **letter)
        else:  # in two rows the velocity-loss panel sits under the lettered MMD one
            for ax in grid[i] if single else grid[i, :1]:
                ax.set_title(LETTER[scenario], loc="left", **letter)
    for m, metric in enumerate((r"MMD$^2\times10^3$", "velocity loss")):
        if column:
            grid[0, m].set_title(metric, fontsize=9, color=PALETTE["ink"])
        else:
            grid[0, m].set_ylabel(metric, fontsize=9, color=PALETTE["ink"])
    # Shared by every panel: every 5k on the row layout's wide panels, else 0/10k/20k.
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(5 if layout == "row" else 3,
                                                   steps=[1, 2, 5, 10]))
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0"))
    # With every arm present, the constant arms (solid) and the annealed ones (dashed) are two
    # groups: a legend column each one column wide (at 7 pt the six labels just fit), a legend
    # row each across the page.
    if column:
        _fig_legend(fig, axes, ncol=legend_ncol, fontsize=7, handlelength=1.8, handletextpad=0.5,
                    columnspacing=0.8)
    else:
        _fig_legend(fig, axes, ncol=legend_ncol, fontsize=8, rows_first=True, handlelength=1.8,
                    handletextpad=0.5, columnspacing=1.5)
    top = 1 - legend_h / fig.get_figheight()  # room for the legend
    if single:
        fig.get_layout_engine().set(rect=(0, 0, 1, top))  # (left, bottom, width, height)
    else:
        fig.tight_layout(rect=(0, 0, 1, top))
    # Uncropped: the page is exactly the paper's width, so the text prints at its own size.
    _save(fig, out, name or f"fig_policy_training_n{n}{suffix[layout]}", tight=False)


def _rate(ms: float) -> str:
    """The control rate a per-step latency allows."""
    hz = 1000.0 / ms
    if hz >= 1000:
        return f"{hz / 1000:.0f} kHz"
    return f"{hz:.0f} Hz" if hz >= 10 else f"{hz:.1f} Hz"


# Control rates the bar panels mark: a visual servo loop, and a tactile one. Each is a latency
# budget -- 1000 / hz milliseconds -- that a decision (or an executed step) has to fit inside.
# ``ha`` keeps the two labels off each other: the faster one reads left of its line.
CONTROL_BUDGETS = [(60.0, "60 Hz visual", "left"), (1000.0, "1 kHz tactile", "right")]


def _pareto(points: list[tuple[float, float]]) -> list[int]:
    """Indices of the non-dominated points of ``(cost, success)``: nothing else is both cheaper
    and at least as successful. Sorted by cost."""
    keep = [i for i, (x, y) in enumerate(points)
            if not any(xx <= x and yy >= y and (xx, yy) != (x, y) for xx, yy in points)]
    return sorted(keep, key=lambda i: points[i][0])


# Arms whose inference is a timed arm's network, unchanged: a training-time regularizer changes
# the weights, not the forward pass, so the success panel plots them at that arm's measured cost.
SAME_INFERENCE_AS = {"score_w0.5": "fm", "mlp_score_w0.5": "mlp",
                     "score_w0.05_cos0.5": "fm", "mlp_score_w0.05_cos0.5": "mlp",
                     "chunk_score_w0.05_cos0.5": CHUNK_METHOD}


def figure_excluded(method: str) -> bool:
    """Arms left out of the inference figure (still in the tables and the other figures): FM with
    action chunking, with or without a regularizer; the temporally ensembled re-evaluations, which
    the figure places itself (closed loop); and the annealed (cosine) score arms -- a regularizer
    changes the weights and not the forward pass, so an annealed arm's bars are identical to its
    constant-λ twin's and the figure would carry the same row twice under the same label."""
    p = parse_method(method)
    return (p["horizon"] > 1 or p["temporal_ensemble"] is not None
            or p["cdm_schedule"] == "cosine")


def _pareto_label(method: str) -> str:
    """Point label on the success panel: a regularized arm is named for its network and term,
    FM+chunk as in the bar panels."""
    p = parse_method(method)
    if p["cdm_weight"] > 0:
        net = f"{FIGURE_FM}+chunk" if p["horizon"] > 1 else FIGURE_FM if p["use_phase"] else "MLP"
        return f"{net}+{_TERM_LABEL[p['cdm_terms']]}"
    if method == CHUNK_METHOD:
        return figure_label(method)
    return method_short(method, FIGURE_FM)


def _leader(ax, point, end) -> None:
    """A hairline from ``point`` to ``end``, the label's left edge at mid height (data coords).

    Drawn as its own annotation with no text: an arrow attached to the label instead leaves the
    text box at whichever corner the target lies past, so a label stacked above or below its
    point is joined at a corner and reads as pointing nowhere in particular.
    """
    ax.annotate("", point, xytext=end, textcoords="data",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=PALETTE["ink_muted"],
                                shrinkA=0, shrinkB=2), zorder=1)


def _frontier_y(front: list[tuple[float, float]], x: float) -> float:
    """Height of a Pareto frontier (points sorted by cost) at cost ``x``, linear in log cost and
    flat past its ends."""
    return float(np.interp(np.log(x), np.log([p[0] for p in front]), [p[1] for p in front]))


def fig_inference(out: Path, summary: list[dict], n: int) -> None:
    """What inference costs, per method, on both devices -- and what that cost buys.

    Bars: one decision, and one executed control step (with the control rate it allows). Both are
    the batch-1 measurement, so the devices stand on identical conditions; an arm that runs a timed
    arm's network unchanged (a regularizer changes the weights, not the forward pass) gets its own
    row at that measurement; the log axis is
    unavoidable, the span being four orders of magnitude from an MLP forward pass to Diffusion
    Policy's 100-step DDPM chain on a CPU. A chunk policy's per-step cost is its decision spread
    over the steps it executes open loop before re-planning, which the bars say.

    The third panel puts success (mean over the scenarios at ``n``) against the per-step cost: a
    method's two points sit at the same height, since the device changes the cost and not the
    policy, so the gap between them *is* the hardware penalty. The chunk baselines appear twice:
    as trained and evaluated (open loop between re-plans), and closed loop with ACT's temporal
    ensembling (their ``_te<m>`` re-evaluations), which predicts a whole chunk at every step --
    real-time feedback, at a full decision per step. Each
    method keeps its figure colour; the device is fill vs hatch (bars) and filled vs hollow
    (points), never hue alone, and closed loop is a square.
    """
    timed = read_inference(out)
    devices = sorted({r["device"] for r in timed}, reverse=True)  # cuda first
    if not timed or len(devices) < 2:
        return
    by = {(r["method"], r["device"]): r for r in timed}
    rows = [r for r in timed if not figure_excluded(r["method"])]
    idx = {(r["scenario"], r["method"]): r for r in summary if r["n_train"] == n}
    # The arms that run a timed arm's network unchanged: a training-time regularizer changes the
    # weights, not the forward pass, so they are shown at their donor's measurement -- as bar rows
    # of their own (the reader looks for them there) and as points on the success panel.
    shared = [a for a, donor in SAME_INFERENCE_AS.items()
              if not figure_excluded(a) and any(r["method"] == donor for r in timed)
              and any(mm == a for _, mm in idx)]
    methods = table_order({*{r["method"] for r in rows}, *shared})[::-1]  # first method at the top

    def timing(method: str, device: str) -> dict | None:
        r = by.get((SAME_INFERENCE_AS.get(method, method), device))
        if r is not None and method in CLOSED_LOOP_ONLY:
            # One decision per executed step, so the per-step cost is the whole decision and
            # steps_per_decision is 1 -- which also drops the "N steps open loop" annotation.
            r = {**r, "ms_step_b1": r["ms_decision_b1"], "steps_per_decision": "1"}
        return r

    open_loop = {m for m in methods if int(timing(m, devices[0])["steps_per_decision"]) > 1}
    # The temporally ensembled re-evaluations of the chunk arms: closed loop. A CLOSED_LOOP_ONLY
    # arm is not in `open_loop` (its bars are already closed loop), but its `_te` point is the
    # only one it gets, so it has to be admitted here explicitly.
    closed = sorted({mm for _, mm in idx if _te_base(mm)[1] is not None
                     and _te_base(mm)[0] in (open_loop | CLOSED_LOOP_ONLY)})

    def base(key: str) -> str:
        return _te_base(key)[0]

    def is_closed(key: str) -> bool:
        return _te_base(key)[1] is not None

    def cost(key: str, device: str) -> float | None:
        """Per-step cost on ``device``: its donor's measurement for a shared network, a whole
        decision for a closed-loop arm."""
        r = by.get((SAME_INFERENCE_AS.get(base(key), base(key)), device))
        if r is None:
            return None
        return float(r["ms_decision_b1" if is_closed(key) else "ms_step_b1"])

    def point_label(key: str) -> str:
        if is_closed(key):
            return f"{_pareto_label(base(key))} (closed loop)"
        return _pareto_label(key) + (" (open loop)" if key in open_loop else "")

    # Success per scenario (each already a mean over seeds); the panel plots the mean over the
    # scenarios and whiskers over their range, so the averaging never hides a task it fails.
    per_scenario = {m: [100 * r["success_rate_mean"] for (_, mm), r in idx.items() if mm == m]
                    for m in {*methods, *shared, *closed} if any(mm == m for _, mm in idx)}
    success = {m: float(np.mean(v)) for m, v in per_scenario.items()}
    # Points on the success panel. CLOSED_LOOP_ONLY arms keep their bar row (a cost measurement)
    # but contribute only the `_te` point that `closed` carries.
    pareto = table_order({*methods, *shared} - CLOSED_LOOP_ONLY)[::-1] + table_order(closed)
    letters = ", ".join(LETTER[s] for s in SCENARIOS if any(s == sc for sc, _ in idx))
    panels = [("ms_decision_b1", "per decision", False),
              ("ms_step_b1", "per executed control step", True)]
    # Wider than the paper's text block and flat: three panels side by side, each needing room
    # for the value labels its bars carry.
    fig, axes = plt.subplots(1, 3, figsize=(1.45 * TEXT_WIDTH, 0.40 * len(methods) + 1.55),
                             facecolor=PALETTE["surface"])
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    height = 0.34
    for ax, (key, title, with_rate) in zip(axes, panels):
        for i, method in enumerate(methods):
            st = method_style(method, methods)
            for d, device in enumerate(devices):
                r = timing(method, device)
                if r is None:
                    continue
                v = float(r[key])
                y = i + (0.5 - d) * height
                gpu = device == "cuda"
                ax.barh(y, v, height=height * 0.92, color=st["color"] if gpu else "none",
                        edgecolor=st["color"], linewidth=0 if gpu else 1.0,
                        hatch=None if gpu else "////")
                text = f"{v:.2f}" if v < 10 else f"{v:,.0f}"
                if with_rate:
                    text = f"{text} ({_rate(v)})"
                    # A chunk policy's step cost is amortized: say so on its (upper) GPU bar.
                    if method in open_loop and d == 0:
                        steps = int(r["steps_per_decision"])
                        text += f"  ← {steps} steps open loop"
                elif method in open_loop:
                    # Re-planned every step, a decision per step: the rate closed loop allows.
                    text = f"{text} ({_rate(v)})" + ("  ← closed loop" if d == 0 else "")
                ax.text(v * 1.25, y, text, va="center", fontsize=6.5, color=PALETTE["ink_muted"],
                        bbox=dict(facecolor=PALETTE["surface"], edgecolor="none", pad=0.3),
                        zorder=3)
        ax.set_xscale("log")
        vals = [float(timing(m, d)[key]) for m in methods for d in devices
                if timing(m, d) is not None]
        ax.set_xlim(min(vals) / 2, max(vals) * (150 if with_rate else 40 if open_loop else 6))
        # The two latency budgets a robot is held to: a bar that ends left of a line is fast
        # enough to close that loop, whatever else it costs.
        for hz, what, ha in CONTROL_BUDGETS:
            ax.axvline(1000.0 / hz, ls=(0, (4, 3)), lw=0.9, color=PALETTE["ink_muted"],
                       alpha=0.75, zorder=1)
            ax.annotate(what, (1000.0 / hz, 1.0), xycoords=ax.get_xaxis_transform(),
                        textcoords="offset points", xytext=(3 if ha == "left" else -3, -1),
                        ha=ha, va="top", fontsize=6, color=PALETTE["ink_muted"],
                        bbox=dict(facecolor=PALETTE["surface"], edgecolor="none", pad=0.4),
                        zorder=3)
        _style(ax, "milliseconds (log)", size=8)
        ax.yaxis.grid(False)
        ax.set_title(title, fontsize=9, loc="left", color=PALETTE["ink"])
    axes[0].set_yticks(range(len(methods)))
    # A shared arm's row is named for its network and term, not its schedule: the rows are the
    # forward passes, and its λ is in the tables.
    axes[0].set_yticklabels([_pareto_label(m) if m in shared else figure_label(m)
                             for m in methods], fontsize=7.5)
    axes[0].set_ylim(-0.6, len(methods) - 0.4)

    ax = axes[2]
    fronts: dict[str, list[tuple[float, float]]] = {}
    for device in devices:
        gpu = device == "cuda"
        names = [m for m in pareto if cost(m, device) is not None and m in success]
        pts = [(cost(m, device), success[m]) for m in names]
        for (x, y), m in zip(pts, names):
            color = method_style(base(m), pareto)["color"]
            ax.vlines(x, min(per_scenario[m]), max(per_scenario[m]), color=color, lw=0.9,
                      alpha=0.5, zorder=2)
            # An arm plotted at its donor's cost sits on the donor's point: draw it on top.
            ax.plot(x, y, "s" if is_closed(m) else "o", ms=6,
                    mfc=color if gpu else "none", mec=color, mew=1.4,
                    zorder=4 if m in SAME_INFERENCE_AS else 3)
        fronts[device] = [pts[i] for i in _pareto(pts)]
        ax.plot([p[0] for p in fronts[device]], [p[1] for p in fronts[device]],
                ls="--" if gpu else ":", lw=1.0, color=PALETTE["ink_muted"], zorder=1,
                label=f"Pareto frontier, {'GPU' if gpu else 'CPU'}")
    # The same policy on both devices -- one accuracy, two costs -- joined by a faint line.
    anchor = {m: max(cost(m, d) for d in devices) for m in pareto
              if m in success and all(cost(m, d) is not None for d in devices)}
    for m in anchor:
        ax.plot([cost(m, d) for d in devices], [success[m]] * 2, lw=0.7,
                color=method_style(base(m), pareto)["color"], alpha=0.35, zorder=2)
    # Labels. The costly methods, apart on the right, keep their labels beside their points. The
    # rest sit in a cluster too dense to label in place -- FM and its score twin even share a
    # cost -- so their labels stack in a column in the empty band right of the cluster and under
    # both frontiers, in success order, each joined to its method's slower point by a hairline.
    label = dict(fontsize=6.5, color=PALETTE["ink"], ha="left", va="center")
    far = [m for m in anchor if anchor[m] > 100]  # ms: Diffusion Policy, far right
    for m in far:
        ax.annotate(point_label(m), (anchor[m], success[m]),
                    textcoords="offset points", xytext=(8, 0), **label)
    # A closed-loop arm short of those sits right of the cluster, clear of the others:
    # it labels itself beside its slower square.
    under = [m for m in anchor if m not in far and is_closed(m)]
    for m in under:
        ax.annotate(point_label(m), (anchor[m], success[m]),
                    textcoords="offset points", xytext=(8, 0), **label)
    cluster = sorted((m for m in anchor if m not in far and m not in under),
                     key=lambda m: -success[m])
    if cluster:
        x_col = 3 * max(anchor[m] for m in cluster)
        step = 3.5  # success points between stacked labels
        y = min(_frontier_y(front, x_col) for front in fronts.values() if front)
        for m in cluster:
            y = min(success[m], y - step)
            for u in under:  # step below a labelled closed-loop pair rather than onto it
                if abs(y - success[u]) < step:
                    y = success[u] - step
            _leader(ax, (anchor[m], success[m]), (x_col / 1.09, y))
            ax.annotate(point_label(m), (x_col, y), **label)
    ax.set_xscale("log")
    step_ms = [c for m in anchor for c in (cost(m, d) for d in devices)]
    ax.set_xlim(min(step_ms) / 2.5, max(step_ms) * 15)  # room for the labels beside the points
    # 60 unless an arm's mean sits below it -- the whiskers may run off the bottom (they span the
    # scenarios, and a method that fails one is the point), but a clipped *marker* would drop the
    # arm out of the panel with only a stray whisker to show it was ever there.
    floor = min(60.0, 5 * np.floor((min(success[m] for m in pareto) - 2) / 5))
    ax.set_ylim(floor, 100)
    _style(ax, "ms per control step (log)", "success [%]", size=8)
    ax.set_title(f"success vs cost: mean of {letters} (n = {n})", fontsize=9, loc="left",
                 color=PALETTE["ink"])
    # Upper left: the point labels run down the right-hand side, and the lowest of them now
    # reaches the bottom-right corner the legend used to sit in.
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")

    # Colour is the method here, so the device and loop legend is drawn in neutral ink: fill vs
    # hatch, and the success panel's point shape.
    hardware = {r["device"]: r["hardware"] for r in timed}
    # Name the chunk baselines actually drawn, rather than a hard-coded pair that goes stale.
    _open_loop_names = ", ".join(method_short(m, FIGURE_FM)
                                 for m in table_order(open_loop - CLOSED_LOOP_ONLY))
    shapes = [Line2D([], [], ls="", marker=mk, ms=5, mfc=PALETTE["ink_muted"],
                     mec=PALETTE["ink_muted"], label=text)
              # "ACT temporal ensembling" named the technique by its paper, which reads as the
              # ACT *arm* now that one is in the figure; the attribution stays in the docstring.
              for mk, text in [("o", f"as trained ({_open_loop_names}: open-loop chunks)"),
                               ("s", "closed loop (temporal ensembling)")]]
    fig.legend(handles=[
        Patch(facecolor=PALETTE["ink_muted"] if d == "cuda" else "none",
              edgecolor=PALETTE["ink_muted"], hatch=None if d == "cuda" else "////",
              label=hardware[d]) for d in devices
    ] + (shapes if closed else []), loc="upper center", ncol=len(devices) + 2, fontsize=7.5,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.34 / fig.get_figheight()))
    _save(fig, out, "fig_policy_inference")


def fig_divergence(runs, scenarios, n, methods, out):
    """One row per method -- overlaid, every method's rollout and model curves are unreadable:
    the held-out demos' kNN divergence, the method's rollouts (kNN, mean +/- std over seeds) and
    the model's own divergence, over phase. Each column shares its y axis across methods."""
    shown = [m for m in methods if any(runs.get((s, n, m)) for s in scenarios)]
    if not shown:
        return
    rows = len(shown)
    fig, axes = plt.subplots(rows, len(scenarios), figsize=(3.4 * len(scenarios), 1.5 * rows + 0.5),
                             facecolor=PALETTE["surface"], squeeze=False, sharex=True,
                             sharey="col")
    for r, method in enumerate(shown):
        style = method_style(method, methods)
        for c, scenario in enumerate(scenarios):
            ax = axes[r, c]
            group = runs.get((scenario, n, method))
            if group:
                demo = np.array(group[0]["metrics"]["demo_div_per_phase"])
                roll = np.array([g["metrics"]["rollout_div_per_phase"] for g in group])
                ax.plot(np.linspace(0, 1, len(demo)), demo, color=PALETTE["ink"], lw=1.1,
                        zorder=5)
                _band(ax, np.linspace(0, 1, roll.shape[1]), roll, style, None)
                # The generative baselines have no velocity field, hence no model divergence.
                if all("model_div_per_phase" in g["metrics"] for g in group):
                    model = np.array([g["metrics"]["model_div_per_phase"] for g in group])
                    ax.plot(np.arange(model.shape[1]) / model.shape[1], model.mean(axis=0),
                            lw=0.9, color=style["color"], ls=":")
            else:
                ax.text(0.5, 0.5, "not trained", transform=ax.transAxes, ha="center",
                        va="center", fontsize=7, color=PALETTE["ink_muted"])
            ax.axhline(0, color=PALETTE["ink_muted"], lw=0.6)
            _style(ax, "phase" if r == rows - 1 else None,
                   r"$\nabla\cdot v$ [1/s]" if c == 0 else None)
            ax.set_xlim(0, 1)
            ax.set_title(f"{LETTER[scenario]} · {figure_label(method)}", fontsize=8, loc="left",
                         color=PALETTE["ink"])
    handles = [
        Line2D([], [], color=PALETTE["ink"], lw=1.1, label="held-out demos (kNN)"),
        Line2D([], [], color=PALETTE["ink_muted"], lw=1.6,
               label="rollouts (kNN), mean ± std over seeds"),
        Line2D([], [], color=PALETTE["ink_muted"], lw=0.9, ls=":",
               label=r"model's own $\nabla\cdot v$, mean over seeds"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
    _save(fig, out, f"fig_policy_divergence_n{n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--test", type=Path, default=Path("data/test"))
    parser.add_argument("--out", type=Path, default=Path("output/skate/policy"))
    parser.add_argument("--note", default=None,
                        help="Markdown note placed as a callout under the SUMMARY.md title")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"No finished runs under {args.runs}")
    scenarios = [s for s in SCENARIOS if any(k[0] == s for k in runs)]
    tests = {s: TrajectoryDataset.load(args.test / f"{s}.npz") for s in scenarios}
    methods = table_order({k[2] for k in runs})
    figure_methods = figure_order(methods)
    sizes = sorted({k[1] for k in runs}, reverse=True)
    fill_collision_breakdown(runs, tests)

    summary, sig = summarize(runs), significance(runs)
    write_csv(summary, args.out / "summary.csv")
    write_csv(sig, args.out / "significance.csv")
    write_table(summary, sig, args.out / "table_policy.tex")
    write_pivot_table(summary, sig, methods, args.out / "table_policy_summary.tex")
    write_baseline_table(summary, sig, methods, max(sizes),
                         args.out / "table_policy_baselines.tex")
    write_markdown(summary, sig, sizes, methods, tests[scenarios[0]].limits,
                   args.out / "SUMMARY.md", args.note)
    fig_inference(args.out, summary, max(sizes))
    fig_vs_ntrain(summary, scenarios, figure_methods, args.out)
    violin_base = VIOLIN_BASE if VIOLIN_BASE in figure_methods else "fm"
    fig_violin(runs, scenarios, max(sizes), figure_methods, args.out, violin_base,
               keep=lambda m: _te_base(m)[0] in VIOLIN_METHODS
               and not (_loop(m) == "open" and _te_base(m)[0] in CLOSED_LOOP_ONLY))
    fig_violin(runs, scenarios, max(sizes), figure_methods, args.out, violin_base,
               keep=lambda m: _loop(m) is None, name=f"fig_policy_violin_n{max(sizes)}_ablation")

    print(f"\n{'scen':<5}{'n':>6}  {'method':<20}{'seeds':>6}{'MMD e3':>14}{'obst %':>13}"
          f"{'wall %':>13}{'succ %':>13}  modes %")
    for r in _sorted_rows(summary):
        modes = "/".join(f"{100 * r[k]:.0f}" for k in sorted(r) if k.endswith("_frac_mean"))
        print(f"{LETTER[r['scenario']]:<5}{r['n_train']:>6}  {method_label(r['method']):<20}"
              f"{r['n_seeds']:>6}{1e3 * r['mmd_mean_mean']:>8.2f}±{1e3 * r['mmd_mean_std']:<5.2f}"
              f"{100 * r['obstacle_collision_rate_mean']:>7.1f}"
              f"±{100 * r['obstacle_collision_rate_std']:<5.1f}"
              f"{100 * r['boundary_collision_rate_mean']:>7.1f}"
              f"±{100 * r['boundary_collision_rate_std']:<5.1f}"
              f"{100 * r['success_rate_mean']:>7.1f}±{100 * r['success_rate_std']:<5.1f}  {modes}")
    if sig:
        print("\nWelch t-test vs FM (p-values):")
        for r in sig:
            print(f"  {LETTER[r['scenario']]} n{r['n_train']} {method_label(r['method'])}: "
                  + "  ".join(f"{k} {r[f'{k}_diff']:+.4f} (p={r[f'{k}_p']:.3f})" for k in TESTED))

    # The generative baselines train for their own (longer) step budget and log no velocity
    # loss; on a shared step axis they would squash every other arm's curves. A temporally
    # ensembled re-evaluation shares its original's training curve, so it would only overdraw it.
    trained_like_fm = [m for m in figure_methods if parse_method(m)["baseline"] is None
                       and parse_method(m)["temporal_ensemble"] is None]
    for n in sizes:
        fig_rollouts(runs, tests, scenarios, n, figure_methods, args.out)
        fig_mmd_phase(runs, scenarios, n, figure_methods, args.out)
        fig_training(runs, scenarios, n, trained_like_fm, args.out)
        fig_divergence(runs, scenarios, n, figure_methods, args.out)
    for layout in ("row", "single_row"):
        # Every arm: the _ablation cut of the single-row figure (the row layout keeps its name).
        fig_training(runs, scenarios, max(sizes), trained_like_fm, args.out, layout=layout,
                     name=(f"fig_policy_training_n{max(sizes)}_single_row_ablation"
                           if layout == "single_row" else None))
    training_main = [m for m in trained_like_fm if m in TRAINING_METHODS]
    if training_main:
        fig_training(runs, scenarios, max(sizes), training_main, args.out, layout="single_row")
    fig_rollouts_pair(runs, tests, max(sizes), args.out)
    compact = [m for m in figure_methods if m in COMPACT_METHODS]
    if compact:
        for tag, group in _compact_groups(scenarios):
            fig_rollouts_by_scenario(runs, tests, group, max(sizes), compact, args.out, tag)
        # The same panels as the _AB and _C figures in one grid, a row per scenario in letter
        # order, for where one figure has to carry every scenario.
        every = [s for s in scenarios if any(s in group for group in COMPACT_GROUPS)]
        if len(every) > 1:
            fig_rollouts_by_scenario(runs, tests, every, max(sizes), compact, args.out,
                                     "".join(LETTER[s] for s in every), by_width=True)


if __name__ == "__main__":
    main()
