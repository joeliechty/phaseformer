"""Animate a scenario's demos and their x, y and theta marginals building up over phase.

    python scripts/skate/animate_phase_evolution.py --scenario dual_goal

Writes ``{scenario}_phase_evolution.gif`` to --out: the training demos drawn up to phase rho on
the left (as the held-out demos in ``fig_policy_rollouts``), and beside them the 3D surfaces of
``{scenario}_phase_evolution`` -- opaque up to rho, faded beyond, the marginal at rho in red --
all three sweeping rho from 0 to 1 together.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate import TrajectoryDataset  # noqa: E402
from fm_bc.skate.plots import (  # noqa: E402
    COMPONENT_LABELS, MODE_COLORS, PALETTE, _phase_histogram, _resample_phase, _subsample,
    _world_axes)

UPSAMPLE, BANDWIDTH, N_BINS = 6, 1.0, 60  # as plot_phase_evolution_overview
MAX_PATHS = 300  # as plot_rollouts


def _draw_surface(ax3, grid, edges, cut, label):
    """One snapshot of ``_draw_phase_evolution``: the whole surface faded, the part with phase
    at or below column ``cut`` opaque, and the marginal there outlined in red."""
    centers = 0.5 * (edges[1:] + edges[:-1])
    phase = np.linspace(0.0, 1.0, grid.shape[1])
    vmax = grid.max()
    P, V = np.meshgrid(phase, centers)
    ax3.clear()
    # Faded only past rho: the opaque part hides the rest, and each frame draws half as much.
    rest = slice(cut - 1, None)
    if grid.shape[1] - cut >= 1:
        ax3.plot_surface(P[:, rest], V[:, rest], grid[:, rest], cmap="viridis", vmin=0,
                         vmax=vmax, alpha=0.12, edgecolor="none", rstride=1, cstride=1)
    if cut >= 2:
        ax3.plot_surface(P[:, :cut], V[:, :cut], grid[:, :cut], cmap="viridis", vmin=0,
                         vmax=vmax, alpha=0.9, edgecolor="none", rstride=1, cstride=1,
                         antialiased=False)
    rho_c = phase[cut - 1]
    VY, VZ = np.meshgrid(edges[[0, -1]], [0, vmax])
    ax3.plot_surface(np.full_like(VY, rho_c), VY, VZ, color="white", alpha=0.25,
                     edgecolor="none")
    profile = grid[:, cut - 1]
    at = np.full_like(centers, rho_c)
    ax3.plot(at, centers, profile, color="red", lw=1.8, zorder=10)
    ax3.plot(at, centers, np.zeros_like(centers), color="red", lw=0.9, alpha=0.7, zorder=10)
    for i in (0, -1):
        ax3.plot([rho_c, rho_c], [centers[i]] * 2, [0, profile[i]], color="red", lw=0.9,
                 alpha=0.7, zorder=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(edges[0], edges[-1])
    ax3.set_zlim(0, vmax)
    ax3.set_xlabel("phase", fontsize=11, labelpad=6)
    ax3.set_ylabel(label, fontsize=11, labelpad=6)
    ax3.set_xticks([0, 0.5, 1])
    ax3.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax3.set_zticks([0, np.floor(vmax * 10) / 10])
    ax3.tick_params(labelsize=10, pad=1)
    ax3.view_init(elev=28, azim=-55)
    ax3.set_box_aspect((3, 2, 1), zoom=0.95)
    ax3.set_facecolor("none")
    for artist in (*ax3.collections, *ax3.lines):
        artist.set_clip_on(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", default="dual_goal")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("output/skate/divergence"))
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--hold", type=float, default=1.5, help="seconds on the last frame")
    parser.add_argument("--dpi", type=int, default=90)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ds = TrajectoryDataset.load(args.data / f"{args.scenario}.npz")
    grids = [_phase_histogram(ds, c, N_BINS, UPSAMPLE, BANDWIDTH) for c in range(3)]
    n_cols = grids[0][0].shape[1]
    # The demos on the same fine phase grid as the surfaces, so both move in step.
    idx = _subsample(len(ds), MAX_PATHS)
    xs = _resample_phase(ds.poses[idx, :, 0], UPSAMPLE)
    ys = _resample_phase(ds.poses[idx, :, 1], UPSAMPLE)
    colors = [MODE_COLORS[m % 2] for m in ds.mode[idx]]

    fig = plt.figure(figsize=(22, 5.0), dpi=args.dpi, facecolor=PALETTE["surface"])
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.25, 1.25, 1.25], wspace=0.0,
                          left=0.01, right=0.98, top=0.9, bottom=0.02)
    ax = fig.add_subplot(gs[0, 0])
    _world_axes(ax, ds)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    lines = LineCollection([], colors=colors, linewidths=0.7, alpha=0.45, zorder=4)
    ax.add_collection(lines)
    dots = ax.scatter(xs[:, 0], ys[:, 0], s=6, c=colors, zorder=5)
    names = [n.replace(" goal", "") for n in ds.spec.mode_names]
    ax.legend([Line2D([], [], color=MODE_COLORS[m % 2], lw=2) for m in range(len(names))],
              names, loc="lower right" if args.scenario == "single_path" else "lower left",
              fontsize=12, framealpha=0.92,
              facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"]).set_zorder(10)
    ax.set_title("training demos", fontsize=15, loc="left", color=PALETTE["ink"])
    axes3d = [fig.add_subplot(gs[0, c + 1], projection="3d") for c in range(3)]
    title = fig.suptitle("", fontsize=18, x=0.5, y=0.98, color=PALETTE["ink"])

    cuts = np.unique(np.linspace(1, n_cols, args.frames).round().astype(int))
    frames = []
    for cut in cuts:
        seg = np.stack([xs[:, :cut], ys[:, :cut]], axis=-1)
        lines.set_segments(list(seg))
        dots.set_offsets(seg[:, -1])
        for c, ax3 in enumerate(axes3d):
            _draw_surface(ax3, *grids[c], cut, COMPONENT_LABELS[c][1])
        title.set_text(rf"$\rho \leq {(cut - 1) / (n_cols - 1):.2f}$")
        fig.canvas.draw()
        rgb = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                               fig.canvas.buffer_rgba()).convert("RGB")
        frames.append(rgb)
    plt.close(fig)

    ms = int(round(1000 / args.fps))
    durations = [ms] * (len(frames) - 1) + [int(args.hold * 1000)]
    # One palette for every frame, so the colours do not shimmer from frame to frame: from the
    # first, middle and last side by side, which between them hold the faded surfaces too.
    picks = [frames[0], frames[len(frames) // 2], frames[-1]]
    sheet = Image.new("RGB", (frames[0].width, 3 * frames[0].height))
    for k, f in enumerate(picks):
        sheet.paste(f, (0, k * f.height))
    pal = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    save = args.out / f"{args.scenario}_phase_evolution.gif"
    frames[0].save(save, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True)
    print(f"wrote {save} ({len(frames)} frames, {save.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
