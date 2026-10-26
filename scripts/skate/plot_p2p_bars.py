"""Bar chart of the point-to-point results: success per training set and arm, on each set's own
evaluation and on the two shared-start cross-evaluations.

    python scripts/skate/plot_p2p_bars.py

Reads the runs' ``metrics.json`` (own evaluation) and ``output/skate_deterministic/cross_start.csv``
(``scripts/skate/cross_start_eval.py``), writes ``output/skate_deterministic/fig_p2p_success_bars``.
Bars are means over seeds; the dots are the seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_fixed_start import METHODS, ORIG, SETS  # noqa: E402
from evaluate_policies import _save, figure_label, method_style  # noqa: E402
from skate_planner.viz import PALETTE  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})

N = 1024
# The two state-only arms share the MLP's colour in every figure (line style tells them apart
# there); as bars, the score arm gets a hatch instead.
HATCH = {"mlp_score_w0.5": "////"}
GROUP_LABEL = {"dual_path_fixed": "fixed\n(2 demos)", "dual_path_fixed_via": "+via jitter\n(one start)",
               "dual_path_noisy_start": "+start noise\n(no jitter)", ORIG: "C\n(paper)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs/skate_deterministic"))
    parser.add_argument("--orig-runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("output/skate_deterministic"))
    args = parser.parse_args()

    own = {}
    for scen in SETS:
        root = args.orig_runs if scen == ORIG else args.runs
        for m in METHODS:
            own[(scen, m)] = np.array([
                100 * json.loads(p.read_text())["success_rate"]
                for p in sorted((root / scen / f"n{N}" / m).glob("seed*/metrics.json"))])
    cross: dict = {}
    with (args.out / "cross_start.csv").open() as f:
        for r in csv.DictReader(f):
            key = (r["scenario"], r["method"])
            cross.setdefault(("single",) + key, []).append(100 * float(r["single_success"]))
            cross.setdefault(("dist",) + key, []).append(100 * float(r["dist_success"]))

    # (title, subtitle, values, draw the seeds): from the single start each seed is 0 or 100,
    # so its dots would say nothing the bar does not.
    panels = [
        ("Own evaluation", "each set's held-out starts; dots are seeds",
         lambda s, m: own[(s, m)], True),
        ("From the single start", "(5.0, 0.9, π/2): share of the 10 seeds",
         lambda s, m: np.array(cross[("single", s, m)]), False),
        ("From C's start distribution", "1024 shared starts; dots are seeds",
         lambda s, m: np.array(cross[("dist", s, m)]), True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), sharey=True)
    fig.subplots_adjust(wspace=0.06)
    width, rng = 0.2, np.random.default_rng(0)
    for ax, (title, sub, get, dots) in zip(axes, panels):
        for j, m in enumerate(METHODS):
            color = method_style(m, METHODS)["color"]
            for i, scen in enumerate(SETS):
                v = get(scen, m)
                x = i + (j - 1.5) * width
                ax.bar(x, v.mean(), width * 0.9, color=color, hatch=HATCH.get(m),
                       edgecolor=PALETTE["surface"], lw=0.0, zorder=2,
                       label=figure_label(m) if (ax is axes[0] and i == 0) else None)
                if dots:
                    ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=4,
                               color=PALETTE["ink"], alpha=0.35, lw=0, zorder=3)
                top = v.max() if dots else v.mean()
                ax.text(x, top + 1.5, f"{v.mean():.0f}", ha="center", va="bottom",
                        fontsize=5.8, color=PALETTE["ink_muted"], zorder=4)
        ax.set_xticks(range(len(SETS)), [GROUP_LABEL[s] for s in SETS], fontsize=7)
        ax.set_title(f"{title}\n", fontsize=8.5, loc="left", color=PALETTE["ink"])
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7, color=PALETTE["ink_muted"])
        ax.set_ylim(0, 110)
        ax.set_xlim(-0.55, len(SETS) - 0.45)
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.6, zorder=0)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(PALETTE["grid"])
    axes[0].set_ylabel("success [%]")
    axes[0].set_yticks([0, 25, 50, 75, 100])
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(METHODS), fontsize=7.5,
               frameon=False, handlelength=1.6)
    _save(fig, args.out, "fig_p2p_success_bars")


if __name__ == "__main__":
    main()
