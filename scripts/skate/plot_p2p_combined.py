"""One figure for every point-to-point training set: bars for the single start, violins for
scenario C's held-out starts.

    python scripts/skate/plot_p2p_combined.py

Reads ``output/skate_deterministic/start_mix.csv`` (``scripts/skate/start_mix_eval.py``), where
every policy of every set was rolled out the same way: once from the single nominal start
(5.0, 0.9, π/2), and from C's 1024 held-out starts. The training sets, left to right:

* one start -- ``fixed`` (the two trajectories repeated) and ``+via`` (via jitter);
* start-noise mixtures -- 5, 10 and 15% of the demos with C's start noise, the rest ``fixed``;
* every start noisy -- ``+start`` (100%, no via jitter) and C itself (the paper's runs).

Top: the share of the 10 seeds whose one path from the single start reaches the goal. Bottom: per
arm, a violin of the seeds' success from C's starts, the seeds as dots and their mean as a tick.
Writes ``output/skate_deterministic/fig_p2p_combined``.
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
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_fixed_start import METHODS  # noqa: E402
from evaluate_policies import _save, _tint, figure_label, method_style  # noqa: E402
from skate_planner.viz import PALETTE  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})

# (start_mix.csv key, tick label), left to right, and the bracket each run of groups sits under.
GROUPS = [("0", "fixed\n(2 demos)"), ("via", "+via\njitter"),
          ("5", "5%"), ("10", "10%"), ("15", "15%"),
          ("100", "+start\n(100%)"), ("C", "C\n(paper)")]
BRACKETS = [(0, 1, "one start"), (2, 4, "noisy-start mixtures"), (5, 6, "all starts noisy")]
# The two state-only arms share the MLP's colour in every figure; the score arm is hatched.
HATCH = {"mlp_score_w0.5": "////"}
WIDTH = 0.2


def violin(ax, vals: np.ndarray, x: float, color: str, hatch: str | None) -> None:
    """A Gaussian-KDE violin spanning exactly the seeds' range (so never past 0 or 100); a
    flat tick when the seeds (nearly) agree, where a KDE has nothing to smooth."""
    half_max = 0.45 * WIDTH
    if np.ptp(vals) < 1.0:
        ax.plot([x - half_max, x + half_max], [vals.mean()] * 2, color=color, lw=1.6,
                solid_capstyle="butt", zorder=3)
        return
    grid = np.linspace(vals.min(), vals.max(), 100)
    dens = stats.gaussian_kde(vals)(grid)
    half = half_max * dens / dens.max()
    ax.fill_betweenx(grid, x - half, x + half, facecolor=_tint(color, 0.35), edgecolor=color,
                     lw=0.9, hatch=hatch, zorder=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("output/skate_deterministic"))
    args = parser.parse_args()

    data: dict = {}
    with (args.out / "start_mix.csv").open() as f:
        for r in csv.DictReader(f):
            k = (r["pct_noisy"], r["method"])
            data.setdefault(k, {"single": [], "c": []})
            data[k]["single"].append(100 * float(r["single_success"]))
            data[k]["c"].append(100 * float(r["c_success"]))

    fig, (top, bot) = plt.subplots(2, 1, figsize=(8.6, 5.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.2})
    rng = np.random.default_rng(0)
    for j, m in enumerate(METHODS):
        color = method_style(m, METHODS)["color"]
        for i, (key, _) in enumerate(GROUPS):
            d = data.get((key, m))
            if d is None:
                continue
            x = i + (j - 1.5) * WIDTH
            single, c = np.array(d["single"]), np.array(d["c"])
            top.bar(x, single.mean(), WIDTH * 0.88, color=color, hatch=HATCH.get(m),
                    edgecolor=PALETTE["surface"], lw=0, zorder=2,
                    label=figure_label(m) if i == 0 else None)
            top.text(x, single.mean() + 2, f"{round(single.mean() / 10):.0f}", ha="center",
                     va="bottom", fontsize=6, color=PALETTE["ink_muted"])
            violin(bot, c, x, color, HATCH.get(m))
            bot.scatter(x + rng.uniform(-0.04, 0.04, len(c)), c, s=3.5, color=PALETTE["ink"],
                        alpha=0.4, lw=0, zorder=3)
            bot.plot([x - 0.07, x + 0.07], [c.mean()] * 2, color=PALETTE["ink"], lw=1.0,
                     zorder=4)

    top.set_ylabel("seeds reaching\nthe goal [%]")
    top.set_title("From the single start (5.0, 0.9, π/2): one path per seed; the number is "
                  "seeds of 10", fontsize=8.5, loc="left")
    bot.set_ylabel("success [%]")
    bot.set_title("From scenario C's 1024 held-out starts: a violin over the 10 seeds, "
                  "dots = seeds, tick = mean", fontsize=8.5, loc="left")
    for ax in (top, bot):
        ax.set_ylim(-4, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.6, zorder=0)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(PALETTE["grid"])
        for a, b, _ in BRACKETS[:-1]:  # a faint rule between the kinds of training set
            ax.axvline(b + 0.5, color=PALETTE["grid"], lw=0.8, zorder=0)
    bot.set_xticks(range(len(GROUPS)), [lab for _, lab in GROUPS], fontsize=7.5)
    bot.set_xlim(-0.55, len(GROUPS) - 0.45)
    for a, b, name in BRACKETS:
        bot.annotate("", xy=(a - 0.4, -0.2), xytext=(b + 0.4, -0.2),
                     xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                     arrowprops=dict(arrowstyle="-", color=PALETTE["ink_muted"], lw=0.8))
        bot.text((a + b) / 2, -0.24, name, transform=bot.get_xaxis_transform(), ha="center",
                 va="top", fontsize=7.5, color=PALETTE["ink_muted"])
    fig.legend(*top.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(METHODS), fontsize=7.5, frameon=False, handlelength=1.6)
    _save(fig, args.out, "fig_p2p_combined")


if __name__ == "__main__":
    main()
