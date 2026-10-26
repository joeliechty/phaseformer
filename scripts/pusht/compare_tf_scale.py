"""Does the phase-former get better with more parameters? The published TF against three resizings.

    python scripts/pusht/compare_tf_scale.py --runs runs/pusht/policy \
        --out output/pusht_9010/tf_scale

The published transformer (``_tf``, ``hidden_dim = phase_dim = 128``, two blocks, 0.9M) against:

* **TF-med wide** (``--arch transformer_medium``, ``_tfmed``): width 128 -> 180, 1.8M;
* **TF-med deep** (``--arch transformer_medium_deep``, ``_tfdeep``): two blocks -> four, 1.7M;
* **TF-large** (``--arch transformer_large``, ``_tflarge``): both at once, 3.4M.

Heads, tokenization and training are the published ones throughout. The rollout clock is resized
the same three ways -- the paper's phase estimator has two hidden layers of 512, and the resized
ones two of 724 (wide, ``runs/pusht/phase_start_w724``), three of 512 (deep, ``phase_start_d3``)
and three of 724 (large, ``phase_start_w724_d3``), at 2x, 2x and 3.9x its parameters, each trained
exactly as the paper's is.

Three comparisons, on 10 seeds and the paper's protocol (``best.pt``, 50 test envs, the phase-MLP
clock from the start pose with eps = 1/245 and the fixed step):

* **Backbone size** on the paper's clock, all four arms. p is Welch's t-test over per-seed values
  against the published TF of the same arm, as in the paper's violins.
* **Clock size** on the two phase arms of every backbone: the same checkpoint rolled out with the
  paper's clock and with each resized one, so the difference is paired by seed and p is a paired
  t-test. The phase-free arms are left out on purpose: at a fixed step they ignore the clock
  entirely, so their rollouts would repeat bit for bit.
* **Matched systems**: each backbone with the clock resized the same way (wide with wide, deep with
  deep, large with large) against the published TF with the paper's clock, Welch.

Writes ``SUMMARY.md``, ``scores.csv``, ``fig_tf_scale.{png,pdf}`` (mean score per seed) and
``fig_tf_scale_success.{png,pdf}`` (success rate per seed: the share of the 50 episodes whose
coverage passed 0.95).

``--family mlp`` runs the same study on the MLP backbone -- the published MLP (4 x 512) against
MLP-med wide (``--arch mlp_medium``, 4 x 768), MLP-med deep (``--arch mlp_medium_deep``, 8 x 512)
and MLP-large (``--arch mlp_large``, 7 x 768), each parameter-matched to the transformer of its
size -- into ``output/pusht_9010/mlp_scale`` as
``fig_mlp_scale*``, and adds a table setting each MLP against the transformer of its size.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel  # noqa: E402

# key -> (rollout tag, phase estimators, label, shape, parameters). "512" is the paper's.
CLOCKS = {
    "512": ("phase_mlp_start_fixedstep_e1o245_inverse_pd", "runs/pusht/phase_start",
            "MLP 512 (paper)", "MLP 512, 512", 268_801),
    "724": ("phase_mlp_startw724_fixedstep_e1o245_inverse_pd", "runs/pusht/phase_start_w724",
            "MLP wide", "MLP 724, 724", 533_589),
    "d3": ("phase_mlp_startd3_fixedstep_e1o245_inverse_pd", "runs/pusht/phase_start_d3",
           "MLP deep", "MLP 512, 512, 512", 531_457),
    "large": ("phase_mlp_startw724d3_fixedstep_e1o245_inverse_pd",
              "runs/pusht/phase_start_w724_d3", "MLP large", "MLP 724, 724, 724", 1_058_489),
    # The transformer clocks (PhaseClockTransformer): robot, object now and object at the start
    # as separate tokens, each sized to the MLP clock it is twinned with in XF_TWIN.
    "xf": ("phase_mlp_startxf_fixedstep_e1o245_inverse_pd", "runs/pusht/phase_start_xf",
           "transformer base", "transformer d=104, 2 layers", 263_849),
    "xfwide": ("phase_mlp_startxfwide_fixedstep_e1o245_inverse_pd",
               "runs/pusht/phase_start_xf_wide", "transformer wide",
               "transformer d=148, 2 layers", 531_765),
    "xfdeep": ("phase_mlp_startxfdeep_fixedstep_e1o245_inverse_pd",
               "runs/pusht/phase_start_xf_deep", "transformer deep",
               "transformer d=104, 4 layers", 526_137),
    "xflarge": ("phase_mlp_startxflarge_fixedstep_e1o245_inverse_pd",
                "runs/pusht/phase_start_xf_large", "transformer large",
                "transformer d=148, 4 layers", 1_061_309),
}
RESIZED_CLOCKS = ("724", "d3", "large")
XF_CLOCKS = ("xf", "xfwide", "xfdeep", "xflarge")
XF_TWIN = {"xf": "512", "xfwide": "724", "xfdeep": "d3", "xflarge": "large"}
SIZES = ("base", "wide", "deep", "large")  # each XF_CLOCKS / XF_TWIN pair, in order
# (run name stem, label, colour) -- plot_pusht.METHODS' colours for the same four arms.
ARMS = [
    ("mlp", "TF", "#eb6834"),
    ("mlp_score_w2.5_const", "TF+score", "#eb6834"),
    ("fm", "TF+phase", "#2a78d6"),
    ("score_w2.5_const", "TF+phase+score", "#e87ba4"),
]
PHASE_ARMS = ("fm", "score_w2.5_const")
# key -> (run suffix, label, shape, marker, filled). The published TF is the hollow one.
BACKBONES = {
    "tf": ("_tf", "TF 0.9M", "hidden 128, 2 blocks, phase_dim = hidden", "o", False),
    "tfmed": ("_tfmed", "TF-med wide 1.8M", "hidden 180, 2 blocks, phase_dim = hidden", "s", True),
    "tfdeep": ("_tfdeep", "TF-med deep 1.7M", "hidden 128, 4 blocks, phase_dim = hidden", "D",
               True),
    "tflarge": ("_tflarge", "TF-large 3.4M", "hidden 180, 4 blocks, phase_dim = hidden", "^",
                True),
}
RESIZED = ("tfmed", "tfdeep", "tflarge")
# Each backbone with the clock resized the same way.
MATCHED = {"tf": "512", "tfmed": "724", "tfdeep": "d3", "tflarge": "large"}
BASE, FAMILY = "tf", "TF"  # the published backbone of the family and its name in the text
# Everything above that is per family. --family rebinds these globals in main(), as
# plot_pusht.py's --arch does; every function reads them at call time.
FAMILIES = {
    "tf": {"ARMS": ARMS, "BACKBONES": BACKBONES, "RESIZED": RESIZED, "MATCHED": MATCHED,
           "BASE": "tf", "FAMILY": "TF", "TITLE": "Phase-former at more parameters: wide, deep "
           "and large", "OUT": "output/pusht_9010/tf_scale", "FIG": "fig_tf_scale",
           "SHAPE_NOTE": "4 heads throughout", "PUBLISHED": "`_tf`",
           "READING": "*The phase-former at twice the parameters*"},
    "mlp": {"ARMS": [("mlp", "MLP", "#eb6834"), ("mlp_score_w2.5_const", "MLP+score", "#eb6834"),
                     ("fm", "MLP+phase", "#2a78d6"),
                     ("score_w2.5_const", "MLP+phase+score", "#e87ba4")],
            "BACKBONES": {
                "mlp": ("", "MLP 0.8M", "4 hidden layers of 512", "o", False),
                "mlpmed": ("_mlpmed", "MLP-med wide 1.8M", "4 hidden layers of 768", "s", True),
                "mlpdeep": ("_mlpdeep", "MLP-med deep 1.8M", "8 hidden layers of 512", "D", True),
                "mlplarge": ("_mlplarge", "MLP-large 3.6M", "7 hidden layers of 768", "^", True)},
            "RESIZED": ("mlpmed", "mlpdeep", "mlplarge"),
            "MATCHED": {"mlp": "512", "mlpmed": "724", "mlpdeep": "d3", "mlplarge": "large"},
            "BASE": "mlp", "FAMILY": "MLP", "TITLE": "The MLP at more parameters: wide, deep and "
            "large", "OUT": "output/pusht_9010/mlp_scale", "FIG": "fig_mlp_scale",
            "SHAPE_NOTE": "SiLU, no residuals", "PUBLISHED": "published MLP",
            "READING": "*The MLP at more parameters*"},
}
FAM = FAMILIES["tf"]
# Each MLP against the transformer of its size (parameter-matched), for --family mlp.
SIZE_PAIRS = [("base", "mlp", "tf"), ("medium wide", "mlpmed", "tfmed"),
              ("medium deep", "mlpdeep", "tfdeep"), ("large", "mlplarge", "tflarge")]


def seeds_of(runs: Path, run: str, tag: str) -> dict[int, dict]:
    """Per seed: mean score and success of the rollouts, and the run's offline numbers."""
    out = {}
    for f in (runs / run).glob(f"seed*/rollout_{tag}.json"):
        r = json.loads(f.read_text())
        m = json.loads((f.parent / "metrics.json").read_text())
        out[int(f.parent.name[4:])] = {
            "score": r["mean_score"], "success": r["success_rate"],
            "phase_model": r.get("phase_model", ""), "best_epoch": m["best"]["epoch"],
            "val_rmse": m["best"]["val_split"]["velocity_rmse"], "n_params": m["n_params"]}
    return dict(sorted(out.items()))


def fmt(values: list[float]) -> str:
    return f"{st.mean(values):.3f} ± {st.stdev(values):.3f}" if len(values) > 1 else "—"


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def welch(a: list[float], b: list[float]) -> str:
    return fmt_p(ttest_ind(a, b, equal_var=False).pvalue) if min(len(a), len(b)) > 1 else "—"


def clock_rmse(phase_runs: Path) -> tuple[str, int]:
    rmse = [json.loads(f.read_text())["rmse"]
            for f in sorted(phase_runs.glob("seed*/metrics.json"))]
    if len(rmse) < 2:
        return "—", len(rmse)
    return f"{st.mean(rmse):.4f} ± {st.stdev(rmse):.5f}", len(rmse)


def _style(ax, title: str) -> None:
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", color="#e4e4e4", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _strip(ax, x: float, v: list[float], colour: str, marker: str, filled: bool, rng) -> None:
    """One condition: its seeds, jittered, and their mean with a 95% interval beside them."""
    face = colour if filled else "white"
    ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, s=14, marker=marker, facecolor=face,
               edgecolor=colour, linewidth=0.8, alpha=0.75, zorder=2)
    half = 1.96 * np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    ax.errorbar(x + 0.3, np.mean(v), yerr=half, fmt=marker, ms=5, color="#222222",
                mfc=face, mec="#222222", lw=1.2, capsize=0, zorder=3)


def _groups(ax, groups, rng) -> None:
    """``groups``: ``(tick label, colour, values(backbone key))``, one strip per backbone each."""
    x, ticks = 0.0, []
    for tick, colour, values in groups:
        start = x
        for key, (_, _, _, marker, filled) in BACKBONES.items():
            v = values(key)
            if v:
                _strip(ax, x, v, colour, marker, filled, rng)
            x += 1
        ticks.append((start + x - 1) / 2 + 0.15)
        x += 0.6
    ax.set_xticks(ticks, [g[0] for g in groups], fontsize=7.5)


def figure(data: dict, out: Path, metric: str, ylabel: str, name: str) -> None:
    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    a = fig.add_subplot(grid[0, :])
    rng = np.random.default_rng(0)

    def at(stem, key, clock):
        return [r[metric] for r in data.get((stem, key, clock), {}).values()]

    _groups(a, [(label, colour, lambda key, stem=stem: at(stem, key, "512"))
                for stem, label, colour in ARMS], rng)
    _style(a, "(a) backbone size, paper's 512 clock")
    a.set_ylabel(ylabel, fontsize=8)

    def clock_delta(stem, key, clock):
        base, other = data.get((stem, key, "512"), {}), data.get((stem, key, clock), {})
        return [other[s][metric] - base[s][metric] for s in base if s in other]

    panels = [(stem, label, colour) for stem, label, colour in ARMS if stem in PHASE_ARMS]
    for i, (stem, label, colour) in enumerate(panels):
        ax = fig.add_subplot(grid[1, i], sharey=None if i == 0 else fig.axes[1])
        _groups(ax, [(f"{CLOCKS[c][2]} clock", colour,
                      lambda key, c=c: clock_delta(stem, key, c)) for c in RESIZED_CLOCKS], rng)
        ax.axhline(0.0, color="#888888", lw=0.8, zorder=1)
        _style(ax, f"({'bc'[i]}) {label}: resized clock − 512, per checkpoint")
        if i == 0:
            ax.set_ylabel(f"change in {ylabel}", fontsize=8)

    handles = [plt.Line2D([], [], ls="", marker=marker, mfc="#555555" if filled else "white",
                          mec="#555555", ms=5, label=label)
               for _, label, _, marker, filled in BACKBONES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=1.0, h_pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# The two clock families in fig_tf_scale_clocks: colour and marker, then the clock of each size.
CLOCK_FAMILIES = {
    "MLP clock": ("#4a3aa7", "o", ("512", "724", "d3", "large")),
    "transformer clock": ("#1baf7a", "s", XF_CLOCKS),
}


def clock_fits(clock: str) -> list[float]:
    """Each seed's RMSE of rho over the demonstrations, for one clock."""
    return [json.loads(f.read_text())["rmse"]
            for f in sorted(Path(CLOCKS[clock][1]).glob("seed*/metrics.json"))]


def clock_figure(data: dict, out: Path) -> None:
    """MLP against transformer clocks, size by size: (a) the fit, (b) what it does to rollouts."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    rng = np.random.default_rng(0)
    panels = [
        (lambda c: clock_fits(c), "(a) fit to the demonstrations", "RMSE of ρ per clock seed"),
        (lambda c: list(per_clock_seed(data, c, "512", "score").values()) if c != "512" else [],
         "(b) rollout score vs the paper's clock", "Δ mean score per clock seed"),
    ]
    for ax, (values, title, ylabel) in zip(axes, panels):
        x, ticks = 0.0, []
        for i, size in enumerate(SIZES):
            start = x
            for colour, marker, clocks in CLOCK_FAMILIES.values():
                v = values(clocks[i])
                if v:
                    _strip(ax, x, v, colour, marker, True, rng)
                x += 1
            ticks.append((start + x - 1) / 2 + 0.15)
            x += 0.6
        sizes = [f"{size}\n{CLOCKS[XF_TWIN[c]][4] / 1e6:.2f}M" for size, c in zip(SIZES, XF_CLOCKS)]
        ax.set_xticks(ticks, sizes, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=8)
        _style(ax, title)
    axes[1].axhline(0.0, color="#888888", lw=0.8, zorder=1)
    handles = [plt.Line2D([], [], ls="", marker=marker, mfc=colour, mec=colour, ms=5, label=name)
               for name, (colour, marker, _) in CLOCK_FAMILIES.items()]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{FAM['FIG']}_clocks.{ext}", dpi=200, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)


FULL_ARM = "score_w2.5_const"  # TF+phase+score, the arm the study is about


def per_clock_seed(data: dict, clock: str, ref: str, metric: str) -> dict[int, float]:
    """``clock`` minus ``ref`` on the same phase-arm checkpoints, averaged per clock seed: the
    eight checkpoints of a seed (four backbones, two phase arms) all run that seed's clock."""
    per_seed = defaultdict(list)
    for (stem, k, c), base in data.items():
        other = data.get((stem, k, clock), {})
        if c != ref or stem not in PHASE_ARMS or k not in BACKBONES:
            continue
        for seed, r in base.items():
            if seed in other:
                per_seed[seed].append(other[seed][metric] - r[metric])
    return {seed: st.mean(v) for seed, v in sorted(per_seed.items())}


TOP = 10  # rows of the ranked headline table


def headline(data: dict) -> list[str]:
    """The full arm (+phase+score) in every backbone x clock combination, best mean first."""
    base = [r["score"] for r in data.get((FULL_ARM, BASE, "512"), {}).values()]
    combos = []
    for k, (_, name, *_rest) in BACKBONES.items():
        for c, (_, _, clock, *_r) in CLOCKS.items():
            rows = data.get((FULL_ARM, k, c), {})
            if len(rows) > 1:
                combos.append((k, c, name, clock, rows))
    if not combos:
        return []
    combos.sort(key=lambda t: -st.mean(r["score"] for r in t[4].values()))
    k, c, name, clock, rows = combos[0]
    scores = [r["score"] for r in rows.values()]
    gain = st.mean(scores) - st.mean(base)
    lines = [
        f"## Headline: the best {FAMILY}+phase+score",
        "",
        f"**{name} with the {clock} clock: mean score {st.mean(scores):.3f} ± "
        f"{st.stdev(scores):.3f} over {len(scores)} seeds, against {st.mean(base):.3f} ± "
        f"{st.stdev(base):.3f} for the published {FAMILY} on the paper's clock ({gain:+.3f}, "
        "Welch p "
        f"{welch(scores, base)}).** Its best seed scores {max(scores):.3f}.",
        "",
        f"The top {min(TOP, len(combos))} of {len(combos)} backbone x clock combinations, best "
        f"mean first, and the published configuration. p is Welch against the published {FAMILY} "
        "on the "
        f"paper's clock. With {len(combos)} combinations the top one is the maximum of "
        f"{len(combos)} noisy means, so its lead over the next few is within seed noise; the gap "
        "between the published configuration and the top cluster is the result.",
        "",
        "| rank | backbone | clock | mean score | p | best seed | success rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (k, c, name, clock, rows) in enumerate(combos, 1):
        if i > TOP and (k, c) != (BASE, "512"):
            continue
        v = [r["score"] for r in rows.values()]
        sr = [r["success"] for r in rows.values()]
        tag = " (published)" if (k, c) == (BASE, "512") else ""
        bold = "**" if i == 1 else ""
        lines.append(f"| {i} | {bold}{name}{tag}{bold} | {bold}{clock}{bold} | "
                     f"{bold}{fmt(v)}{bold} | {welch(v, base) if tag == '' else '—'} | "
                     f"{max(v):.3f} | {fmt(sr)} |")
    # The whole grid: mean score of every combination, backbones down, clocks across.
    clocks = [c for c in CLOCKS if any(cc == c for _, cc, *_ in combos)]
    lines += ["", f"Mean {FAMILY}+phase+score of every combination:", "",
              "| backbone | " + " | ".join(CLOCKS[c][2] for c in clocks) + " |",
              "|---|" + "---|" * len(clocks)]
    means = {(k, c): st.mean(r["score"] for r in rows.values()) for k, c, _, _, rows in combos}
    best = max(means.values())
    for k, (_, name, *_rest) in BACKBONES.items():
        cells = []
        for c in clocks:
            m = means.get((k, c))
            cells.append("—" if m is None else f"**{m:.3f}**" if m == best else f"{m:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines + [""]


def cross_family(data: dict) -> list[str]:
    """Each MLP against the transformer of its size on the full arm: on the paper's clock, and
    each on its own best clock (the best mean over every clock it was rolled out on)."""
    mlp_name = {k: v[1] for k, v in FAMILIES["mlp"]["BACKBONES"].items()}
    tf_name = {k: v[1] for k, v in FAMILIES["tf"]["BACKBONES"].items()}

    def v(k, c, metric="score"):
        return [r[metric] for r in data.get((FULL_ARM, k, c), {}).values()]

    def best(k):
        return max((c for c in CLOCKS if len(v(k, c)) > 1), key=lambda c: st.mean(v(k, c)))

    lines = [
        "", "## Against the transformer of the same size", "",
        "Each MLP against the transformer it is parameter-matched to, +phase+score arm: on the "
        "paper's clock, and each on its own best clock. p is Welch over seeds.", "",
        "| size | MLP | TF | paper clock: MLP | TF | p | best clock: MLP | TF | p | "
        "success at best: MLP | TF |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for size, m, t in SIZE_PAIRS:
        if len(v(m, "512")) < 2 or len(v(t, "512")) < 2:
            continue
        cm, ct = best(m), best(t)
        lines.append(
            f"| {size} | {mlp_name[m]} | {tf_name[t]} | {fmt(v(m, '512'))} | {fmt(v(t, '512'))} | "
            f"{welch(v(m, '512'), v(t, '512'))} | {fmt(v(m, cm))} ({CLOCKS[cm][2]}) | "
            f"{fmt(v(t, ct))} ({CLOCKS[ct][2]}) | {welch(v(m, cm), v(t, ct))} | "
            f"{st.mean(v(m, cm, 'success')):.3f} | {st.mean(v(t, ct, 'success')):.3f} |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs/pusht/policy"))
    p.add_argument("--family", default="tf", choices=sorted(FAMILIES),
                   help="the transformer ladder (tf) or the MLP ladder (mlp)")
    p.add_argument("--out", type=Path, default=None,
                   help="default: output/pusht_9010/tf_scale or mlp_scale, by --family")
    args = p.parse_args()
    fam = FAMILIES[args.family]
    globals().update({k: fam[k] for k in ("ARMS", "BACKBONES", "RESIZED", "MATCHED", "BASE",
                                          "FAMILY")}, FAM=fam)
    args.out = args.out or Path(fam["OUT"])
    args.out.mkdir(parents=True, exist_ok=True)

    # Every family's runs, so --family mlp can set each MLP against the transformer of its size;
    # the tables and figures below show this family's backbones only.
    every = {k: v for f in FAMILIES.values() for k, v in f["BACKBONES"].items()}
    data = {}  # (arm stem, backbone key, clock) -> {seed: row}
    for stem, _, _ in ARMS:
        for key, (suffix, *_rest) in every.items():
            for clock, (tag, *_rest) in CLOCKS.items():
                if clock != "512" and stem not in PHASE_ARMS:
                    continue
                rows = seeds_of(args.runs, stem + suffix, tag)
                if rows:
                    data[(stem, key, clock)] = rows

    with (args.out / "scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "backbone", "clock", "seed", "mean_score", "success_rate", "best_epoch",
                    "val_velocity_rmse", "n_params", "phase_model"])
        for (stem, key, clock), rows in data.items():
            if key not in BACKBONES:
                continue
            for seed, r in rows.items():
                w.writerow([stem, key, clock, seed, f"{r['score']:.4f}", f"{r['success']:.3f}",
                            r["best_epoch"], f"{r['val_rmse']:.4f}", r["n_params"],
                            r["phase_model"]])

    def vals(stem, key, metric, clock="512"):
        return [r[metric] for r in data.get((stem, key, clock), {}).values()]

    # The phase arms carry the phase embedding and adaLN modulation, the phase-free ones do not.
    def n_params(key, phase):
        return next((next(iter(rows.values()))["n_params"] for (stem, k, _), rows in data.items()
                     if k == key and (stem in PHASE_ARMS) == phase), 0)

    shown = [k for k in BACKBONES if any(kk == k for _, kk, _ in data)]
    resized = [k for k in RESIZED if k in shown]
    label = {k: BACKBONES[k][1] for k in BACKBONES}

    lines = [
        f"# {FAM['TITLE']}",
        "",
        "Generated by `scripts/pusht/compare_tf_scale.py`"
        + (f" `--family {args.family}`" if args.family != "tf" else "")
        + "; the method is in its docstring, the reading in `fm_bc/pusht/STATUS.md`, "
        f"{FAM['READING']}.",
        "",
    ]
    lines += headline(data)
    lines += [
        "## Setup",
        "",
        f"| model | run suffix | shape ({FAM['SHAPE_NOTE']}) | parameters |",
        "|---|---|---|---|",
    ]
    for k in shown:
        suffix, name, shape, *_ = BACKBONES[k]
        lines.append(f"| {name}{' (published)' if k == BASE else ''} | `{suffix or '(none)'}` | "
                     f"{shape} | {n_params(k, True):,} ({n_params(k, False):,} without phase) |")
    lines += [
        "",
        "| phase clock | shape | parameters | fit: RMSE of ρ, all 206 demos |",
        "|---|---|---|---|",
    ]
    for key, (_, runs, name, hidden, n) in CLOCKS.items():
        rmse, n_seeds = clock_rmse(Path(runs))
        lines.append(f"| {name} | {hidden} | {n:,} | {rmse} ({n_seeds} seeds) |")
    lines += [
        "",
        f"Training is the {FAM['PUBLISHED']} arms' protocol unchanged: 90/10 split, 500 epochs, "
        "batch 256, lr 1e-4, EMA, λ = 2.5 constant, bandwidth 32 px, seeds 0-9, `best.pt` (lowest "
        f"EMA validation loss). The {FAM['PUBLISHED']} seeds trained on CHPC (RTX 6000 Ada), the "
        "resized ones on the workstation's 4090. Every clock is fit to every demonstration, as "
        "the paper's is.",
        "",
        "## Backbone size, on the paper's clock",
        "",
        f"Per-seed values, mean ± sd over seeds, with Welch p against the published {FAMILY} of "
        "the same arm.",
    ]
    for metric, what in (("score", "Mean score"), ("success", "Success rate")):
        head = " | ".join(f"{label[k]} | p" for k in resized)
        lines += ["", f"{what} per seed:", "",
                  f"| arm | {label[BASE]} | {head} |",
                  "|---|---|" + "---|---|" * len(resized)]
        for stem, arm, _ in ARMS:
            base = vals(stem, BASE, metric)
            cells = [fmt(base)]
            for k in resized:
                cells += [fmt(vals(stem, k, metric)), welch(vals(stem, k, metric), base)]
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Phase-clock size, on the phase arms",
        "",
        "Each checkpoint rolled out with the paper's clock and with each resized one: the 512 "
        "column is the paper's clock, the others the change from it, mean ± sd over seeds, with p "
        "paired by seed.",
    ]
    for (metric, what), clocks in (
            ((m, w), cs) for cs in (RESIZED_CLOCKS, XF_CLOCKS)
            for m, w in (("score", "Mean score"), ("success", "Success rate"))):
        if not any((PHASE_ARMS[0], k, c) in data for k in shown for c in clocks):
            continue
        kind = "MLP" if clocks is RESIZED_CLOCKS else "transformer"
        lines += ["", f"{what} per seed, {kind} clocks:", "",
                  "| backbone | arm | 512 (paper) | "
                  + " | ".join(f"Δ {CLOCKS[c][2]} | p" for c in clocks) + " |",
                  "|---|---|---|" + "---|---|" * len(clocks)]
        for k in shown:
            for stem, arm, _ in ARMS:
                base = data.get((stem, k, "512"), {})
                if stem not in PHASE_ARMS or not base:
                    continue
                cells = [f"{st.mean(r[metric] for r in base.values()):.3f}"]
                for c in clocks:
                    other = data.get((stem, k, c), {})
                    seeds = [s for s in base if s in other]
                    a_ = [base[s][metric] for s in seeds]
                    b_ = [other[s][metric] for s in seeds]
                    d = [y - x for x, y in zip(a_, b_)]
                    p = fmt_p(ttest_rel(b_, a_).pvalue) if len(d) > 1 and any(d) else "—"
                    cells += [fmt(d), p]
                lines.append(f"| {label[k]} | {arm} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Pooled over every backbone and both phase arms, {20 * len(shown)} checkpoints per clock. "
        f"The {2 * len(shown)} checkpoints of a seed all share that seed's clock, so they are "
        "averaged per clock seed "
        "first and p is a one-sample t-test over the 10 clock seeds. The last rows compare each "
        "transformer clock with its same-size MLP twin on the same checkpoints (the base "
        "transformer's twin is the paper's clock, so its row above is that comparison).",
        "",
        "| clock | against | fit RMSE of ρ | Δ score | clock seeds > 0 | p | Δ success "
        "| clock seeds > 0 | p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    pairs = [(c, "512") for c in (*RESIZED_CLOCKS, *XF_CLOCKS)]
    pairs += [(c, XF_TWIN[c]) for c in XF_CLOCKS if XF_TWIN[c] != "512"]  # base: row above
    for c, ref in pairs:
        if not any((PHASE_ARMS[0], k, c) in data for k in shown):
            continue
        cells = [CLOCKS[ref][2], clock_rmse(Path(CLOCKS[c][1]))[0]]
        for metric in ("score", "success"):
            v = list(per_clock_seed(data, c, ref, metric).values())
            p = fmt_p(ttest_1samp(v, 0.0).pvalue) if len(v) > 1 and any(v) else "—"
            cells += [f"{st.mean(v):+.3f}" if v else "—", f"{sum(x > 0 for x in v)}/{len(v)}", p]
        lines.append(f"| {CLOCKS[c][2]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Matched systems",
        "",
        f"Each backbone with the clock resized the same way, against the published {FAMILY} on the "
        "paper's clock (Welch). The phase-free arms are not repeated here: they ignore the clock, "
        "so their matched numbers are the backbone-size table's.",
    ]
    for metric, what in (("score", "Mean score"), ("success", "Success rate")):
        resized_pairs = [k for k in resized if (PHASE_ARMS[0], k, MATCHED[k]) in data]
        lines += ["", f"{what} per seed:", "",
                  f"| arm | {label[BASE]} + 512 (paper) | "
                  + " | ".join(f"{label[k]} + {CLOCKS[MATCHED[k]][2]} clock | p"
                               for k in resized_pairs) + " |",
                  "|---|---|" + "---|---|" * len(resized_pairs)]
        for stem, arm, _ in ARMS:
            if stem not in PHASE_ARMS:
                continue
            base = vals(stem, BASE, metric)
            cells = [fmt(base)]
            for k in resized_pairs:
                v = vals(stem, k, metric, MATCHED[k])
                cells += [fmt(v), welch(v, base)]
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    if args.family == "mlp":
        lines += cross_family(data)
    lines += [
        "",
        "## Offline",
        "",
        "Best epoch (of 500) and validation velocity RMSE at `best.pt`, mean over seeds. "
        "Validation loss does not order these policies (STATUS.md, *Transformer backbone*); "
        "this is here to show where training stopped, not to rank.",
        "",
        "| arm | best epoch: " + " / ".join(label[k] for k in shown)
        + " | val u RMSE: " + " / ".join(label[k] for k in shown) + " |",
        "|---|---|---|",
    ]
    for stem, arm, _ in ARMS:
        ep = " / ".join(f"{st.mean(vals(stem, k, 'best_epoch')):.0f}"
                        if vals(stem, k, "best_epoch") else "—" for k in shown)
        rm = " / ".join(f"{st.mean(vals(stem, k, 'val_rmse')):.3f}"
                        if vals(stem, k, "val_rmse") else "—" for k in shown)
        lines.append(f"| {arm} | {ep} | {rm} |")
    lines += [
        "",
        "## Figures",
        "",
        "Each small marker is one seed, the large one the mean over seeds with a 95% interval. "
        f"The hollow circle is the published {FAMILY}. (a) compares backbones on the paper's "
        "clock; (b) "
        "and (c) are, per phase arm, the change from the paper's clock to each resized one for "
        "the same checkpoint, per seed.",
        "",
        "**Mean score per seed.**",
        "",
        f"![per-seed mean scores]({FAM['FIG']}.png)",
        "",
        "**Success rate per seed** -- the share of the 50 test episodes whose coverage passed "
        "0.95. Each seed's rate moves in steps of 0.02 (one episode), so the seeds stack on a few "
        "levels.",
        "",
        f"![per-seed success rates]({FAM['FIG']}_success.png)",
        "",
        "**MLP against transformer clocks, size by size** -- (a) each clock seed's fit to the "
        "demonstrations, (b) its rollout score against the paper's clock, averaged over the "
        "eight phase-arm checkpoints of that seed (the paper's clock is the zero line). Sizes "
        "are the MLP clocks' parameters; each transformer is within 3% of its twin.",
        "",
        f"![MLP and transformer clocks]({FAM['FIG']}_clocks.png)",
        "",
    ]
    (args.out / "SUMMARY.md").write_text("\n".join(lines))
    figure(data, args.out, "score", "mean score per seed", FAM["FIG"])
    figure(data, args.out, "success", "success rate per seed", f"{FAM['FIG']}_success")
    clock_figure(data, args.out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
