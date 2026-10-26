"""The start-spread study at a glance: each headline training set's demos beside the four MLP
arms' rollouts on scenario C's held-out starts.

    python scripts/skate/plot_start_spread_overview.py

A row per training set -- S0, V, S2, S4, S8, S12, T8, M10, M'10 (``start_spread_eval.SETS``) --
and a column for its training demos, then one per arm: MLP, MLP+score, MLP+phase,
MLP+phase+score. A rollout panel pools all 10 seeds: 30 of C's 1024 held-out starts per seed,
a different 30 for each (300 distinct starts), the same draw for every arm and set. Its title
gives the success rate over every seed's 1024 rollouts, the table's number in START_SPREAD.md
(``deploy.csv``). Demo panels draw at most 300 of their demos, as ``plot_rollouts``.

Writes ``output/skate_start_spread/fig_spread_overview.{pdf,png}``.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cross_start_eval import evaluate  # noqa: E402
from evaluate_policies import COMPACT_PANEL_W, _save, figure_label, type_pt  # noqa: E402
from evaluate_fixed_start import METHODS  # noqa: E402
from fm_bc.skate.baselines import load_policy  # noqa: E402
from fm_bc.skate.dataset import TrajectoryDataset  # noqa: E402
from fm_bc.skate.plots import COLLISION_COLOR, MODE_COLORS, plot_rollouts  # noqa: E402
from skate_planner.viz import PALETTE  # noqa: E402
from start_spread_eval import SETS, data_paths, read_csv  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

# key -> (row name, what its demos are); the name heads the demo panel, the rest goes under it.
ROWS = {"S0": ("S0", "one start, 2 distinct"), "V": ("V", "one start, via jitter"),
        "S2": ("S2", "2 cm, random sides"), "S4": ("S4", "4 cm, random sides"),
        "S8": ("S8", "8 cm, random sides"), "S12": ("S12", "12 cm, random sides"),
        "T8": ("T8", "8 cm, sorted sides"), "M10": ("M10", "102 noisy + 922 copies"),
        "M'10": ("M′10", "M10's 102 noisy alone")}
PER_SEED = 30  # held-out starts drawn per seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs/skate_deterministic"))
    parser.add_argument("--c-test", type=Path, default=Path("data/test/dual_path.npz"))
    parser.add_argument("--out", type=Path, default=Path("output/skate_start_spread"))
    args = parser.parse_args()

    starts = TrajectoryDataset.load(args.c_test).poses[:, 0]
    draw = np.random.default_rng(0).permutation(len(starts))  # seed s takes its 30 in turn
    deploy = read_csv(args.out / "deploy.csv")

    def success(key, method):
        v = [r["success"] for r in deploy if r["set"] == key and r["method"] == method]
        return 100 * np.mean(v) if v else np.nan

    rows, cols = len(ROWS), 1 + len(METHODS)
    width = COMPACT_PANEL_W * cols
    title_pt, legend_pt = type_pt(width, "title"), type_pt(width, "legend")
    fig, axes = plt.subplots(rows, cols, figsize=(width, width), facecolor=PALETTE["surface"],
                             squeeze=False)
    for i, (key, (name, what)) in enumerate(ROWS.items()):
        scen = SETS[key][0]
        ds = TrajectoryDataset.load(data_paths(scen)[0])
        plot_rollouts(ds, ds.poses, ds.mode, np.zeros(len(ds), bool), ax=axes[i, 0],
                      title=f"{name} · demos ({len(ds)})\n{what}")
        for j, method in enumerate(METHODS, start=1):
            pooled = {"poses": [], "labels": [], "collided": []}
            seed_dirs = sorted((args.runs / scen / f"n{len(ds)}" / method).glob("seed*"))
            for s, seed_dir in enumerate(seed_dirs):
                model = load_policy(seed_dir / "final.pt")[0].float().eval()
                r = evaluate(model, starts[draw[s * PER_SEED:(s + 1) * PER_SEED]], ds)
                for k in pooled:
                    pooled[k].append(r[k])
            pooled = {k: np.concatenate(v) for k, v in pooled.items()}
            plot_rollouts(ds, pooled["poses"], pooled["labels"], pooled["collided"],
                          ax=axes[i, j], max_paths=None,
                          title=f"{figure_label(method)}\n{success(key, method):.0f}% success")
            print(f"{key} {method}: {len(seed_dirs)} seeds", flush=True)
    for ax in axes.flat:
        ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=title_pt,
                     color=PALETTE["ink"], linespacing=1.25)
        ax.set_xlabel(""), ax.set_ylabel("")
        ax.grid(False)
        ax.set_xticks([]), ax.set_yticks([])
        ax.get_legend().remove()

    # Placed by hand, as the compact rollout figure: each panel its column's width less a
    # hairline, each row its two-line titles and its world; one legend under the grid.
    pitch = width / cols
    panel_w = pitch - 0.1
    panel_h = panel_w * np.ptp(axes[0, 0].get_ylim()) / np.ptp(axes[0, 0].get_xlim())
    title_h = 2 * 1.3 * title_pt / 72 + 0.06
    legend_h = 1.6 * legend_pt / 72 + 0.1
    height = rows * (title_h + panel_h) + legend_h
    fig.set_figheight(height)
    for i in range(rows):
        bottom = height - (i + 1) * (title_h + panel_h)
        for j in range(cols):
            axes[i, j].set_position([j * pitch / width, bottom / height, panel_w / width,
                                     panel_h / height])
    handles = [Line2D([], [], color=c, lw=1.5) for c in
               (MODE_COLORS[0], MODE_COLORS[1], PALETTE["ink_muted"], COLLISION_COLOR)]
    fig.legend(handles, ["left", "right", "undecided", "collision"], loc="lower left",
               bbox_to_anchor=(0.0, 0.0), ncol=4, fontsize=legend_pt, frameon=False,
               handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    fig.text(1.0, 0.0, f"Rollouts: {PER_SEED} of C's held-out starts per seed × 10 seeds; "
             "success over every seed's 1024", ha="right", va="bottom", fontsize=legend_pt,
             color=PALETTE["ink_muted"], transform=fig.transFigure)
    _save(fig, args.out, "fig_spread_overview", pad=0.02)


if __name__ == "__main__":
    main()
