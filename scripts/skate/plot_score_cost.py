"""Cost of the kernel score on SE(2) vs SE(3): (a) precompute time against the number of
demonstrations, (b) the demonstrations needed for equal accuracy against the dimension.

(b) is the rate argument: with the best bandwidth the mean squared error of a kernel estimate of
a density gradient decays as M^(-4/(D+6)) (Chacon, Duong & Wand 2011), so matching the error of
M0 demonstrations at D = 3 takes M0^((D+6)/9) at dimension D. It ignores the rate's constants,
so it is an order of magnitude, not a prediction.

The timings are hard-coded from a one-off benchmark (51 phase steps; GPU: RTX 4090, float32,
both kernels fused with torch.compile; CPU: 16 threads, float64). The SE(3) kernel has the
structure of fm_bc.divergence.score_kernel with a 6-D body twist and matches it to 1e-11 on
planar poses. The CPU points at M = 10240 are extrapolated from the per-pair cost at M = 4096,
a lower bound since the CPU per-pair cost grows with M.

    python scripts/skate/plot_score_cost.py [--out output/skate/paper]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Surface / ink / type sizes of scripts/pusht/plot_pusht.py.
PALETTE = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink_muted": "#52514e", "grid": "#e8e7e4"}
TYPE_RATIOS = {"label": 0.0144, "legend": 0.0126, "tick": 0.0122}
SE2, SE3 = "#2a78d6", "#e34948"
# (b): reference demo counts on SE(2), light -> dark ink; the paper's largest (1024) darkest.
REFERENCE_M = [(256, "#b4b2ad"), (4096, "#85837e"), (1024, "#0b0b0b")]

# (M, seconds); None marks a point not measured.
SERIES = [
    # label, colour, linestyle, points, extrapolated points
    ("SE(2), GPU", SE2, "-", [(1024, 0.009), (2048, 0.016), (4096, 0.055), (10240, 0.35)], []),
    ("SE(3), GPU", SE3, "-", [(1024, 0.013), (2048, 0.023), (4096, 0.091), (10240, 0.59)], []),
    ("SE(2), CPU", SE2, "--", [(512, 0.36), (1024, 2.1), (2048, 11.0), (4096, 66.8)],
     [(10240, 417.0)]),
    ("SE(3), CPU", SE3, "--", [(512, 1.10), (1024, 5.9), (2048, 41.5), (4096, 176.0)],
     [(10240, 1100.0)]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("output/skate/paper"))
    ap.add_argument("--width", type=float, default=5.5, help="figure width, inches (ICLR text width)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    w = args.width
    pt = {k: v * 72 * w for k, v in TYPE_RATIOS.items()}
    ink, grey = PALETTE["ink"], PALETTE["ink_muted"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(w, 0.42 * w), facecolor=PALETTE["surface"],
                                 gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.28})

    # (a) precompute time against M.
    for label, color, ls, pts, extra in SERIES:
        m, s = zip(*pts)
        ax.plot(m, s, ls, color=color, lw=1.3, marker="o", ms=3.2)
        if extra:
            em, es = zip(*extra)
            ax.plot([m[-1], *em], [s[-1], *es], ":", color=color, lw=1.3)
            ax.plot(em, es, "o", ms=3.2, mfc=PALETTE["surface"], mec=color, mew=1.0)
    # One minute, for scale.
    ax.axhline(60, color=grey, lw=0.8, alpha=0.5, zorder=0)
    ax.text(12300, 60 / 1.2, "1 min", color=grey, fontsize=pt["tick"], va="top", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([512, 1024, 2048, 4096, 10240])
    ax.set_xticklabels(["512", "1024", "2048", "4096", "10240"])
    ax.minorticks_off()
    ax.set_xlim(430, 12500)
    ax.set_xlabel("demonstrations $M$", fontsize=pt["label"], color=ink, labelpad=2)
    ax.set_ylabel("score precompute [s]", fontsize=pt["label"], color=ink, labelpad=2)
    ax.set_title("(a) precompute time", fontsize=pt["label"], color=ink, loc="left", pad=3)
    # One legend entry per encoding: colour is the manifold, line style the device.
    handles = [
        Line2D([], [], color=SE2, lw=1.3, marker="o", ms=3.2, label="SE(2)"),
        Line2D([], [], color=SE3, lw=1.3, marker="o", ms=3.2, label="SE(3)"),
        Line2D([], [], color=grey, lw=1.3, ls="-", label="GPU"),
        Line2D([], [], color=grey, lw=1.3, ls="--", label="CPU"),
        Line2D([], [], color=grey, lw=1.3, ls=":", marker="o", ms=3.2, mfc=PALETTE["surface"],
               mec=grey, mew=1.0, label="extrapolated"),
    ]
    leg = ax.legend(handles=handles, fontsize=pt["legend"], frameon=False, ncol=5,
                    loc="lower left", bbox_to_anchor=(-0.01, 1.07), borderaxespad=0.0,
                    handlelength=2.0, columnspacing=1.1, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_color(ink)

    # (b) demonstrations for the accuracy of M0 on SE(2), against the dimension D.
    dims = np.arange(1, 8)
    for d, color, name in ((3, SE2, "SE(2)"), (6, SE3, "SE(3)")):
        bx.axvline(d, color=color, lw=0.9, alpha=0.6, zorder=0)
        bx.text(d + 0.1, 0.03, name, color=color, fontsize=pt["legend"], ha="left",
                va="bottom", transform=bx.get_xaxis_transform())
    for m0, color in REFERENCE_M:
        need = m0 ** ((dims + 6) / 9)
        bx.plot(dims, need, "-", color=color, lw=1.3)
        bx.plot([3, 6], [m0, m0 ** (12 / 9)], "o", color=color, ms=3.2)
        bx.text(dims[-1] + 0.15, need[-1], f"$M_0$={m0}", color=grey if color != ink else ink,
                fontsize=pt["legend"], va="center", ha="left")
    m6 = 1024 ** (12 / 9)
    bx.annotate(r"$\approx 10^4$", (6, m6), xytext=(-4, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=pt["legend"], color=ink)
    bx.set_yscale("log")
    bx.set_xticks(dims)
    bx.minorticks_off()
    bx.set_xlim(0.7, 7.3)
    bx.set_xlabel("dimension $D$", fontsize=pt["label"], color=ink, labelpad=2)
    bx.set_ylabel("demonstrations needed", fontsize=pt["label"], color=ink, labelpad=2)
    bx.set_title("(b) data for equal accuracy", fontsize=pt["label"], color=ink, loc="left",
                 pad=3)

    for a in (ax, bx):
        a.set_facecolor(PALETTE["surface"])
        a.tick_params(labelsize=pt["tick"], colors=grey, length=2.5, pad=1.5)
        a.grid(True, which="major", color=PALETTE["grid"], lw=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(grey)
            a.spines[side].set_linewidth(0.8)

    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"fig_score_cost.{ext}", dpi=200, facecolor=PALETTE["surface"],
                    bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {args.out / 'fig_score_cost'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
