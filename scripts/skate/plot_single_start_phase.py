"""Phase alone doesn't split the modes when every training demo shares one start: the two
single-start training sets beside MLP+phase's rollouts from that start.

    python scripts/skate/plot_single_start_phase.py

Four panels in a row, styled as the paper's compact rollout figure
(``evaluate_policies.fig_rollouts_by_scenario`` with ``by_width``):

1. the training demos of ``dual_path_fixed`` -- one start pose, no via jitter, so exactly two
   distinct demos (left and right, 512 copies each);
2. MLP+phase trained on them, rolled out from that same fixed start: a deterministic policy
   drives one path per seed, so the panel is the 10 seeds' paths, one rollout each;
3. the training demos of ``dual_path_fixed_via`` -- the same start pose, C's via jitter (0.3 m);
4. MLP+phase trained on those, likewise.

The rollouts are closed loop and clamped, as every evaluation (``cross_start_eval.evaluate``).
The demo panels draw at most 300 of their 1024 demos, as ``plot_rollouts``.
Writes ``output/skate_start_spread/fig_single_start_phase.{pdf,png}``.
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

from cross_start_eval import NOMINAL, evaluate  # noqa: E402
from evaluate_policies import COMPACT_PANEL_W, _save, type_pt  # noqa: E402
from fm_bc.skate.baselines import load_policy  # noqa: E402
from fm_bc.skate.dataset import TrajectoryDataset  # noqa: E402
from fm_bc.skate.plots import COLLISION_COLOR, MODE_COLORS, plot_rollouts  # noqa: E402
from skate_planner.viz import PALETTE  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

METHOD = "fm"  # MLP+phase
# (training set, title of its demos panel)
SETS = [("dual_path_fixed", "Demos: one start,\nno via jitter"),
        ("dual_path_fixed_via", "Demos: one start,\nvia jitter (0.3 m)")]
PAIR_GAP = 0.25  # in, extra space between the two (demos, rollouts) pairs


def seed_paths(runs: Path, scenario: str, world: TrajectoryDataset) -> dict[str, np.ndarray]:
    """Every seed's one rollout from the fixed training start, stacked."""
    out = {"poses": [], "labels": [], "collided": []}
    for seed_dir in sorted((runs / scenario / "n1024" / METHOD).glob("seed*")):
        model = load_policy(seed_dir / "final.pt")[0].float().eval()
        r = evaluate(model, NOMINAL[None], world)
        for k in out:
            out[k].append(r[k][0])
    return {k: np.stack(v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs/skate_deterministic"))
    parser.add_argument("--data", type=Path, default=Path("data/skate_deterministic"))
    parser.add_argument("--out", type=Path, default=Path("output/skate_start_spread"))
    args = parser.parse_args()

    cols = 2 * len(SETS)
    width = COMPACT_PANEL_W * cols
    title_pt, legend_pt = type_pt(width, "title"), type_pt(width, "legend")
    fig, axes = plt.subplots(1, cols, figsize=(width, COMPACT_PANEL_W),
                             facecolor=PALETTE["surface"])
    legends = []
    for i, (scenario, demo_title) in enumerate(SETS):
        ds = TrajectoryDataset.load(args.data / f"{scenario}.npz")
        roll = seed_paths(args.runs, scenario, ds)
        n = len(roll["poses"])
        plot_rollouts(ds, ds.poses, ds.mode, np.zeros(len(ds), bool), ax=axes[2 * i],
                      title=demo_title)
        legends.append([(MODE_COLORS[m], f"{name}: {int(np.sum(ds.mode == m))} demos")
                        for m, name in enumerate(ds.spec.mode_names)])
        ax = axes[2 * i + 1]
        plot_rollouts(ds, roll["poses"], roll["labels"], roll["collided"], ax=ax,
                      title=f"MLP+phase rollouts:\n{n} seeds, from that start")
        # Ten paths, not hundreds: drawn solid, so each seed reads on its own.
        for line in ax.lines:
            line.set_alpha(0.9)
            line.set_linewidth(1.0)
        ok = ~roll["collided"]
        outcomes = [(MODE_COLORS[0], "left", ok & (roll["labels"] == 0)),
                    (MODE_COLORS[1], "right", ok & (roll["labels"] == 1)),
                    (PALETTE["ink_muted"], "undecided", ok & (roll["labels"] < 0)),
                    (COLLISION_COLOR, "collision", roll["collided"])]
        legends.append([(c, f"{name}: {int(sel.sum())} of {n} seeds")
                        for c, name, sel in outcomes if sel.any()])
    for ax, entries in zip(axes, legends):
        title = ax.get_title(loc="left")
        ax.set_title(title, loc="left", fontsize=title_pt, color=PALETTE["ink"],
                     linespacing=1.25)
        ax.set_xlabel(""), ax.set_ylabel("")
        ax.grid(False)
        ax.set_xticks([]), ax.set_yticks([])
        ax.get_legend().remove()
        handles = [Line2D([], [], color=c, lw=1.5) for c, _ in entries]
        # One column: side by side, "right: 512 demos" ran into the next panel's legend.
        ax.legend(handles, [t for _, t in entries], loc="upper left",
                  bbox_to_anchor=(-0.03, 0.0), fontsize=legend_pt, handlelength=1.0,
                  handletextpad=0.4, labelspacing=0.2, borderpad=0.1, borderaxespad=0.1,
                  frameon=False)

    # Placed by hand, as the compact rollout figure: each panel its column's width less a
    # hairline, the pairs a little further apart, every row as tall as its two-line titles,
    # its world and the legend under it. The save crops to the ink.
    pitch = (width - PAIR_GAP) / cols
    panel_w = pitch - 0.1
    aspect = np.ptp(axes[0].get_ylim()) / np.ptp(axes[0].get_xlim())
    panel_h = panel_w * aspect
    title_h = 2 * 1.3 * title_pt / 72 + 0.04
    legend_h = 2 * 1.2 * legend_pt / 72 + 0.05  # the demo panels' two rows
    height = title_h + panel_h + legend_h
    fig.set_figheight(height)
    for j, ax in enumerate(axes):
        x = j * pitch + (PAIR_GAP if j >= 2 else 0.0)
        ax.set_position([x / width, legend_h / height, panel_w / width, panel_h / height])
    _save(fig, args.out, "fig_single_start_phase", pad=0.02)


if __name__ == "__main__":
    main()
