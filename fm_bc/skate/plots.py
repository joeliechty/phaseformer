"""Figures for demonstration distributions and their divergence / score.

Visual conventions follow ``skate_planner.viz``: same surface, obstacle styling and axes. Modes
(homotopy classes) get the two categorical accents; divergence uses a diverging map centered on
zero, since its sign is the whole story -- blue where the bundle contracts, red where it spreads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.quiver import QuiverKey
from matplotlib.text import Annotation, Text

from skate_planner.viz import PALETTE, _setup_axes, draw_world

from .dataset import TrajectoryDataset

__all__ = [
    "MODE_COLORS",
    "plot_distribution",
    "plot_rollouts",
    "plot_divergence_paths",
    "plot_phase_density",
    "plot_phase_density_evolution",
    "plot_phase_evolution_overview",
    "plot_divergence_vs_phase",
    "plot_divergence_estimators",
    "plot_score_snapshots",
    "plot_score_paths",
    "plot_scenario_summary",
    "plot_scenario_overview",
    "plot_demo_overview",
    "plot_field_overview",
]

MODE_COLORS = ["#2a78d6", "#eb6834"]
DIV_CMAP = "RdBu_r"
SCORE_CMAP = "PiYG"


def _axes(ax, figsize=(7, 5.6)):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, facecolor=PALETTE["surface"])
    return fig, ax


def _world_axes(ax, ds: TrajectoryDataset) -> None:
    spec = ds.spec
    _setup_axes(ax, spec.world)
    draw_world(ax, spec.world, inflate=ds.config.get("radius", 0.3))


def _subsample(n: int, max_n: Optional[int], seed: int = 0) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, max_n, replace=False))


def _div_norm(values: np.ndarray, vlim: Optional[float]) -> TwoSlopeNorm:
    vlim = vlim or float(np.nanpercentile(np.abs(values), 98)) or 1.0
    return TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)


def _save(fig, save):
    if save:
        fig.savefig(save, dpi=150, facecolor=PALETTE["surface"], bbox_inches="tight")
    return fig


def _enlarge_text(fig, factor: float) -> None:
    """Every text in ``fig`` ``factor`` times its size: titles, axis and tick labels, colorbars,
    legends, annotations and quiver keys. An annotation offset in points grows alongside, so a
    bigger label keeps its distance from what it points at."""
    tick_labels = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            ticks = axis.get_major_ticks() + axis.get_minor_ticks()
            tick_labels.update(t for tick in ticks for t in (tick.label1, tick.label2))
            if ticks:
                # Through the axis, so that the ticks it makes at draw time are enlarged too.
                axis.set_tick_params(which="both",
                                     labelsize=ticks[0].label1.get_fontsize() * factor)
    texts = set(fig.findobj(Text)) | {a.text for ax in fig.axes for a in ax.artists
                                      if isinstance(a, QuiverKey)}
    for text in texts - tick_labels:
        text.set_fontsize(text.get_fontsize() * factor)
        if isinstance(text, Annotation) and text.anncoords == "offset points":
            text.xyann = (text.xyann[0] * factor, text.xyann[1] * factor)


def _declutter_labels(fig, step: float = 15.0, margin: float = 2.0) -> None:
    """Keep annotation boxes apart, once the layout is final: an annotation whose box would
    overlap an earlier one's (or its axes' legend) turns its offset about its point, ``step``
    degrees at a time either way up to 90, until it is ``margin`` points clear -- or stays put
    if no turn clears it."""
    renderer = fig.canvas.get_renderer()
    pad = renderer.points_to_pixels(margin)
    # Equal-aspect axes shrink to their final box only when drawn; the labels' anchors need it
    # now, or the labels are placed in a stretched map.
    for ax in fig.axes:
        ax.apply_aspect()

    def extent(ann):
        ann.update_positions(renderer)
        ann.update_bbox_position_size(renderer)
        return (ann.get_bbox_patch() or ann).get_window_extent(renderer).padded(pad)

    turns = [0.0] + [sign * k * step for k in range(1, int(90 // step) + 1) for sign in (-1, 1)]
    for ax in fig.axes:
        legend = ax.get_legend()
        placed = [legend.get_window_extent(renderer)] if legend else []
        for ann in [t for t in ax.texts if isinstance(t, Annotation)]:
            home = ann.xyann
            for turn in turns:
                c, s = np.cos(np.radians(turn)), np.sin(np.radians(turn))
                ann.xyann = (c * home[0] - s * home[1], s * home[0] + c * home[1])
                box = extent(ann)
                if not any(box.overlaps(p) for p in placed):
                    break
            else:
                ann.xyann = home
                box = extent(ann)
            placed.append(box)


def plot_distribution(ds: TrajectoryDataset, ax=None, max_paths: Optional[int] = 400,
                      save=None, title: Optional[str] = None):
    """Every kept trajectory, one hue per mode, over the start and goal sample clouds."""
    fig, ax = _axes(ax)
    _world_axes(ax, ds)
    spec = ds.spec
    for i in _subsample(len(ds), max_paths):
        ax.plot(ds.poses[i, :, 0], ds.poses[i, :, 1], "-", lw=0.6, alpha=0.35,
                color=MODE_COLORS[ds.mode[i] % 2], zorder=4)
    for m, name in enumerate(spec.mode_names):
        ax.plot([], [], "-", color=MODE_COLORS[m % 2], lw=1.5,
                label=f"{name} ({int((ds.mode == m).sum())})")
    ax.scatter(ds.starts[:, 0], ds.starts[:, 1], s=4, color=PALETTE["start"], alpha=0.5,
               zorder=5, label="start samples")
    ax.scatter(ds.goals[:, 0], ds.goals[:, 1], s=4, color=PALETTE["ink"], alpha=0.35,
               zorder=5, label="goal samples")
    ax.set_title(title or spec.title, fontsize=10, color=PALETTE["ink"], loc="left")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.92, facecolor=PALETTE["surface"],
              edgecolor=PALETTE["grid"]).set_zorder(10)
    return _save(fig, save)


COLLISION_COLOR = "#e34948"


def plot_rollouts(ds: TrajectoryDataset, poses: np.ndarray, labels: np.ndarray,
                  collided: np.ndarray, ax=None, max_paths: Optional[int] = 300,
                  legend: bool = True, save=None, title: Optional[str] = None):
    """Policy rollouts over the world, one hue per mode label.

    Undecided / wrong-mode paths are muted and collisions red, drawn last so a failure is
    never hidden under a bundle of successes. Legend counts are over all rollouts, not just the
    ``max_paths`` drawn.
    """
    fig, ax = _axes(ax)
    _world_axes(ax, ds)
    spec = ds.spec
    idx = _subsample(len(poses), max_paths)
    n = len(poses)
    groups = [
        (f"{name} ({100 * np.mean((labels == m) & ~collided):.0f}%)", MODE_COLORS[m % 2],
         (labels == m) & ~collided, 4)
        for m, name in enumerate(spec.mode_names)
    ]
    groups.append((f"undecided ({100 * np.mean((labels < 0) & ~collided):.0f}%)",
                   PALETTE["ink_muted"], (labels < 0) & ~collided, 5))
    groups.append((f"collision ({100 * collided.mean():.0f}%)", COLLISION_COLOR, collided, 6))
    for label, color, sel, z in groups:
        for i in idx[sel[idx]]:
            ax.plot(poses[i, :, 0], poses[i, :, 1], "-", lw=0.6, alpha=0.4, color=color,
                    zorder=z)
        if sel.any() or z == 4:
            ax.plot([], [], "-", color=color, lw=1.5, label=label)
    ax.set_title(title or spec.title, fontsize=10, color=PALETTE["ink"], loc="left")
    if legend:
        ax.legend(loc="lower right", fontsize=7, framealpha=0.92, facecolor=PALETTE["surface"],
                  edgecolor=PALETTE["grid"], title=f"{n} rollouts",
                  title_fontsize=7).set_zorder(10)
    return _save(fig, save)


def plot_divergence_paths(ds: TrajectoryDataset, ax=None, key: str = "divergence",
                          vlim: Optional[float] = None, max_paths: Optional[int] = 400,
                          colorbar: bool = True, save=None, title: Optional[str] = None):
    """Trajectories colored segment-by-segment by the flow divergence at that state."""
    div = getattr(ds, key)
    if div is None:
        raise ValueError(f"Dataset has no {key!r}; run add_div_and_score first.")
    fig, ax = _axes(ax)
    _world_axes(ax, ds)
    idx = _subsample(len(ds), max_paths)
    xy = ds.poses[idx, :, :2]
    segs = np.stack([xy[:, :-1], xy[:, 1:]], axis=2).reshape(-1, 2, 2)
    seg_div = (0.5 * (div[idx, :-1] + div[idx, 1:])).reshape(-1)
    norm = _div_norm(div, vlim)
    lc = LineCollection(segs, cmap=DIV_CMAP, norm=norm, linewidths=0.9, alpha=0.7, zorder=4)
    lc.set_array(seg_div)
    ax.add_collection(lc)
    if colorbar:
        cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(r"$\nabla\cdot u$ [1/s]", fontsize=8, color=PALETTE["ink_muted"])
        cb.ax.tick_params(labelsize=7)
    metric = "SE(2)" if key == "divergence" else "xy"
    ax.set_title(title or f"{ds.spec.title} - {metric} divergence", fontsize=10,
                 color=PALETTE["ink"], loc="left")
    return _save(fig, save)


COMPONENT_LABELS = [("x", "x [m]"), ("y", "y [m]"), (r"$\theta$", r"$\theta$ [rad]")]


def _resample_phase(values: np.ndarray, upsample: int, angle: bool = False) -> np.ndarray:
    """``values`` ``(M, T)`` at the waypoints, linearly interpolated to ``(T-1)*upsample + 1``
    evenly spaced phases: every waypoint kept, each gap split into ``upsample`` steps. Angles go
    the shorter way round and come back wrapped to (-pi, pi]."""
    if angle:
        values = np.unwrap(values, axis=1)
    t = np.arange((values.shape[1] - 1) * upsample + 1) / upsample
    i = np.minimum(t.astype(int), values.shape[1] - 2)
    w = t - i
    out = (1 - w) * values[:, i] + w * values[:, i + 1]
    return np.angle(np.exp(1j * out)) if angle else out


def _phase_histogram(ds: TrajectoryDataset, component: int, n_density_bins: int,
                     upsample: int = 1, bandwidth: float = 0.0):
    """``(grid, edges)``: grid ``(n_density_bins, T)`` holds each phase's normalized histogram of
    pose ``component`` (0=x, 1=y, 2=theta) over the data range padded by 5%. With ``upsample``
    > 1 the demos are first resampled between their waypoints (:func:`_resample_phase`), and T
    grows to match.

    With ``bandwidth`` > 0 each demo is instead spread by a Gaussian that many bins wide, and the
    grid holds that smoothed density times the bin width -- the histogram's scale -- at three
    points a bin (``edges`` to match). A histogram of a spread narrower than a bin flickers
    between one bin and two as the spread moves across them; this runs smoothly."""
    vals = ds.poses[:, :, component]
    lo, hi = vals.min(), vals.max()
    pad = 0.05 * (hi - lo + 1e-9)
    edges = np.linspace(lo - pad, hi + pad, n_density_bins + 1)
    if upsample > 1:
        vals = _resample_phase(vals, upsample, angle=component == 2)
    if bandwidth > 0:
        width = edges[1] - edges[0]
        sigma = bandwidth * width
        edges = np.linspace(edges[0], edges[-1], 3 * n_density_bins + 1)
        centers = 0.5 * (edges[1:] + edges[:-1])
        g = np.stack([np.exp(-0.5 * ((centers[:, None] - vals[:, t]) / sigma) ** 2).sum(axis=1)
                      for t in range(vals.shape[1])], axis=1)
        return g * width / (len(vals) * sigma * np.sqrt(2 * np.pi)), edges
    g = np.stack([np.histogram(vals[:, t], bins=edges)[0] for t in range(vals.shape[1])], axis=1)
    return g / np.maximum(g.sum(axis=0, keepdims=True), 1), edges


def plot_phase_density(ds: TrajectoryDataset, n_density_bins: int = 60, save=None):
    """Per-phase histograms of x, y and theta as heatmaps (Figs. 3-5 of the paper).

    Each phase column is normalized to sum to one, so the color is the marginal
    ``p_rho(component)`` and one can read off where it is unimodal, splits or merges.
    """
    labels = COMPONENT_LABELS
    phase = ds.phase
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, facecolor=PALETTE["surface"])
    grids, edges = zip(*(_phase_histogram(ds, c, n_density_bins) for c in range(3)))
    vmax = max(g.max() for g in grids)
    for c, ax in enumerate(axes):
        im = ax.pcolormesh(phase, 0.5 * (edges[c][1:] + edges[c][:-1]), grids[c],
                           cmap="viridis", vmin=0, vmax=vmax, shading="auto")
        ax.set_ylabel(labels[c][1], fontsize=9)
        ax.set_title(labels[c][0], fontsize=10, loc="left")
        fig.colorbar(im, ax=ax, pad=0.01).set_label("probability", fontsize=8)
    axes[-1].set_xlabel("phase", fontsize=9)
    fig.suptitle(f"{ds.spec.title} - state marginals over phase", fontsize=11, x=0.02,
                 ha="left")
    fig.tight_layout()
    return _save(fig, save)


def _draw_phase_evolution(fig, gs, ds: TrajectoryDataset, component: int, n_density_bins: int,
                          n_snapshots: int, upsample: int = 1, bandwidth: float = 0.0):
    """:func:`plot_phase_density_evolution`'s panels drawn into ``gs`` (1 x 2: the heatmap, then
    the snapshot stack); returns the heatmap axes, its colorbar and the 3D axes. ``upsample``
    and ``bandwidth`` go to :func:`_phase_histogram`, and either draws every grid line of the
    3D surfaces -- ``plot_surface`` otherwise keeps at most 50 a side -- rasterized, so a PDF
    stays small."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    grid, edges = _phase_histogram(ds, component, n_density_bins, upsample, bandwidth)
    centers = 0.5 * (edges[1:] + edges[:-1])
    phase = np.linspace(0.0, 1.0, grid.shape[1])
    vmax = grid.max()
    snapshots = np.linspace(0.0, 1.0, n_snapshots)
    label = COMPONENT_LABELS[component][1]
    smooth = upsample > 1 or bandwidth > 0
    faded = dict(rstride=1, cstride=1, rasterized=True) if smooth else {}
    # Antialiasing seams every pair of the fine grid's facets. The faint surface keeps it:
    # without it, its facets alias into a visible net.
    solid = dict(faded, antialiased=False) if smooth else {}

    ax = fig.add_subplot(gs[0, 0])
    im = ax.pcolormesh(centers, phase, grid.T, cmap="viridis", vmin=0, vmax=vmax,
                       shading="auto", rasterized=smooth)
    for rho in snapshots[1:-1]:
        ax.axhline(rho, color="white", lw=1.0, ls="--", alpha=0.6)
    ax.set_ylim(1, 0)  # phase 0 at the top
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("phase", fontsize=9)
    cb = fig.colorbar(im, ax=ax, location="top", pad=0.01, shrink=0.8)
    cb.set_label("probability", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    P, V = np.meshgrid(phase, centers)
    gs_3d = GridSpecFromSubplotSpec(n_snapshots, 1, subplot_spec=gs[0, 1], hspace=0.0)
    axes3d = []
    for s, rho in enumerate(snapshots):
        ax3 = fig.add_subplot(gs_3d[s, 0], projection="3d")
        cut = int(np.searchsorted(phase, rho + 1e-9))  # columns with phase <= rho
        ax3.plot_surface(P, V, grid, cmap="viridis", vmin=0, vmax=vmax, alpha=0.12,
                         edgecolor="none", **faded)
        if cut >= 2:
            ax3.plot_surface(P[:, :cut], V[:, :cut], grid[:, :cut], cmap="viridis", vmin=0,
                             vmax=vmax, alpha=0.9, edgecolor="none", **solid)
        rho_c = phase[cut - 1]
        VY, VZ = np.meshgrid(edges[[0, -1]], [0, vmax])
        ax3.plot_surface(np.full_like(VY, rho_c), VY, VZ, color="white", alpha=0.25,
                         edgecolor="none")
        profile = grid[:, cut - 1]
        at = np.full_like(centers, rho_c)
        ax3.plot(at, centers, profile, color="red", lw=1.5, zorder=10)
        ax3.plot(at, centers, np.zeros_like(centers), color="red", lw=0.8, alpha=0.7, zorder=10)
        for i in (0, -1):
            ax3.plot([rho_c, rho_c], [centers[i]] * 2, [0, profile[i]], color="red", lw=0.8,
                     alpha=0.7, zorder=10)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(edges[0], edges[-1])
        ax3.set_zlim(0, vmax)
        ax3.set_xlabel("phase", fontsize=6, labelpad=-4)
        ax3.set_ylabel(label, fontsize=6, labelpad=-4)
        ax3.set_title(rf"$\rho \leq {rho:.2f}$", fontsize=9, loc="left", y=0.88)
        ax3.view_init(elev=28, azim=-55)
        ax3.tick_params(labelsize=5, pad=-2)
        ax3.set_box_aspect((3, 2, 1), zoom=1.1)
        ax3.set_facecolor(PALETTE["surface"])
        # The zoomed box overhangs the axes, which would clip the surfaces short of phase 0
        # and 1; the box's own panes and edges are never clipped.
        for artist in (*ax3.collections, *ax3.lines):
            artist.set_clip_on(False)
        axes3d.append(ax3)
    return ax, cb, axes3d


def plot_phase_density_evolution(ds: TrajectoryDataset, component: int = 0,
                                 n_density_bins: int = 60, n_snapshots: int = 4,
                                 figsize=(9, 11), save=None):
    """One pose component's marginal over phase, drawn as it builds up (robomimic's
    ``visualize_state_probability_evolution_per_axis``).

    Left: the ``plot_phase_density`` heatmap turned upright, phase running down the page, with a
    dashed line at each snapshot phase. Right: a stack of 3D surfaces of the same grid, each
    opaque up to its snapshot phase ``rho`` and faded beyond it, with the marginal at ``rho``
    outlined in red.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize, facecolor=PALETTE["surface"])
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.6], wspace=0.05,
                  left=0.08, right=0.98, top=0.9, bottom=0.05)
    _draw_phase_evolution(fig, gs, ds, component, n_density_bins, n_snapshots)
    fig.suptitle(f"{ds.spec.title} - {COMPONENT_LABELS[component][0]} marginal over phase",
                 fontsize=11, x=0.02, ha="left")
    return _save(fig, save)


def plot_phase_evolution_overview(ds: TrajectoryDataset, n_density_bins: int = 60,
                                  n_snapshots: int = 4, upsample: int = 6,
                                  bandwidth: float = 1.0, figsize=(24, 12), save=None):
    """:func:`plot_phase_density_evolution` for x, y and theta side by side, with no title and
    text sized to stay legible at page width. The demos are resampled ``upsample`` times between
    their waypoints (linearly, theta the shorter way round) and spread over the bins by a
    Gaussian ``bandwidth`` bins wide, so the ridges run smoothly instead of stepping from
    waypoint to waypoint and flickering from bin to bin."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.ticker import MaxNLocator

    fig = plt.figure(figsize=figsize, facecolor=PALETTE["surface"])
    outer = GridSpec(1, 3, figure=fig, wspace=0.15, left=0.04, right=0.97, top=0.92,
                     bottom=0.06)
    for c in range(3):
        gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, c], width_ratios=[1, 1.5],
                                     wspace=0.05)
        ax, cb, axes3d = _draw_phase_evolution(fig, gs, ds, c, n_density_bins, n_snapshots,
                                               upsample, bandwidth)
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.tick_params(labelsize=19)
        cb.set_label("probability", fontsize=18)
        cb.locator = MaxNLocator(4, steps=[1, 2, 5, 10])
        cb.ax.tick_params(labelsize=17)
        # Narrower from the left, so its phase label clears the previous stack's tick labels,
        # and taller, up into the room the snapshot titles leave above the colorbar.
        pos, cpos = ax.get_position(), cb.ax.get_position()
        width, top = pos.width * 0.8, 0.86
        ax.set_position([pos.x1 - width, pos.y0, width, top - pos.y0])
        cb.ax.set_position([pos.x1 - width, top + cpos.y0 - pos.y1, width, cpos.height])
        for s, ax3 in enumerate(axes3d):
            # The snapshots overlap their neighbours' tick labels: no background to hide them.
            ax3.set_facecolor("none")
            ax3.set_title(ax3.get_title(loc="left"), loc="left", fontsize=18, y=0.92)
            # They share their axes, so only the bottom one labels them.
            bottom = s == len(axes3d) - 1
            ax3.set_xlabel(ax3.get_xlabel() if bottom else "", fontsize=14, labelpad=10)
            ax3.set_ylabel(ax3.get_ylabel() if bottom else "", fontsize=14, labelpad=10)
            ax3.set_xticks([0, 0.5, 1])
            ax3.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            ax3.set_zticks([0, np.floor(ax3.get_zlim()[1] * 10) / 10])
            ax3.tick_params(labelsize=14, pad=2)
            ax3.set_box_aspect((3, 2, 1), zoom=1.25)
    return _save(fig, save)


def plot_divergence_vs_phase(ds: TrajectoryDataset, ax=None, max_paths: Optional[int] = 150,
                             show_xy: bool = True, save=None, title: Optional[str] = None):
    """Per-trajectory divergence (faded) with each mode's mean +/- std; xy mean dashed."""
    if ds.divergence is None:
        raise ValueError("Dataset has no divergence; run add_div_and_score first.")
    fig, ax = _axes(ax, figsize=(7, 3.6))
    ax.set_facecolor(PALETTE["surface"])
    phase = ds.phase
    for i in _subsample(len(ds), max_paths):
        ax.plot(phase, ds.divergence[i], lw=0.4, alpha=0.15,
                color=MODE_COLORS[ds.mode[i] % 2])
    for m, name in enumerate(ds.spec.mode_names):
        sel = ds.mode == m
        mu, sd = ds.divergence[sel].mean(0), ds.divergence[sel].std(0)
        color = MODE_COLORS[m % 2]
        ax.fill_between(phase, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)
        ax.plot(phase, mu, color=color, lw=2.0, label=f"{name}: SE(2)")
        if show_xy and ds.divergence_xy is not None:
            ax.plot(phase, ds.divergence_xy[sel].mean(0), "--", color=color, lw=1.2,
                    label=f"{name}: xy")
    ax.axhline(0.0, color=PALETTE["ink_muted"], lw=0.7)
    ax.grid(True, color=PALETTE["grid"], lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("phase", fontsize=9)
    ax.set_ylabel(r"$\nabla\cdot u$ [1/s]", fontsize=9)
    ax.set_xlim(0, 1)
    lim = np.nanpercentile(np.abs(ds.divergence), 99.5)
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=7, ncol=2, framealpha=0.92, loc="upper right")
    method = ds.div_config.get("method", "knn")
    ax.set_title(title or f"{ds.spec.title} - {method} divergence over phase", fontsize=10,
                 loc="left")
    return _save(fig, save)


def plot_divergence_estimators(ds: TrajectoryDataset, estimates: dict[str, np.ndarray],
                               save=None):
    """Small multiples, one per estimator: per-mode mean +/- std over phase on a shared y axis.

    A tighter band only means less noise if the mean curve keeps its size; compare both.
    """
    n = len(estimates)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.2), sharey=True,
                             facecolor=PALETTE["surface"], squeeze=False)
    phase = ds.phase
    modes = range(ds.spec.n_modes)
    lim = 0.0
    for ax, (label, d) in zip(axes[0], estimates.items()):
        ax.set_facecolor(PALETTE["surface"])
        bands = []
        for m in modes:
            sel = ds.mode == m
            mu, sd = d[sel].mean(0), d[sel].std(0)
            bands.append(sd.mean())
            lim = max(lim, float(np.abs(mu).max() + sd.mean()))
            color = MODE_COLORS[m % 2]
            ax.fill_between(phase, mu - sd, mu + sd, color=color, alpha=0.2, lw=0)
            ax.plot(phase, mu, color=color, lw=1.8, label=ds.spec.mode_names[m])
        ax.axhline(0.0, color=PALETTE["ink_muted"], lw=0.7)
        ax.grid(True, color=PALETTE["grid"], lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_xlabel("phase", fontsize=9)
        ax.set_title(f"{label}\nmean band ±{np.mean(bands):.2f} 1/s", fontsize=9, loc="left")
    axes[0, 0].set_ylabel(r"$\nabla\cdot u$ [1/s]", fontsize=9)
    axes[0, 0].set_ylim(-lim, lim)
    if ds.spec.n_modes > 1:
        axes[0, -1].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{ds.spec.title} - SE(2) divergence estimators", fontsize=10, x=0.01,
                 ha="left")
    fig.tight_layout()
    return _save(fig, save)


def plot_score_snapshots(ds: TrajectoryDataset, phases: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                         max_arrows: int = 200, save=None):
    """States at a few phases with their estimated score (translational part, in the world
    frame) as arrows; arrow color is the rotational score component."""
    if ds.score is None:
        raise ValueError("Dataset has no score; run add_div_and_score first.")
    fig, axes = plt.subplots(1, len(phases), figsize=(3.6 * len(phases), 3.4),
                             facecolor=PALETTE["surface"])
    idx = _subsample(len(ds), max_arrows)
    n_steps = ds.poses.shape[1]
    s_rot = ds.score[idx, :, 2]
    rot_lim = float(np.nanpercentile(np.abs(s_rot), 98)) or 1.0
    for ax, rho in zip(np.atleast_1d(axes), phases):
        t = int(round(rho * (n_steps - 1)))
        _world_axes(ax, ds)
        p = ds.poses[idx, t]
        s = ds.score[idx, t]
        c, sn = np.cos(p[:, 2]), np.sin(p[:, 2])
        world = np.stack([c * s[:, 0] - sn * s[:, 1], sn * s[:, 0] + c * s[:, 1]], axis=1)
        ax.scatter(p[:, 0], p[:, 1], s=3, c=[MODE_COLORS[m % 2] for m in ds.mode[idx]],
                   alpha=0.6, zorder=5)
        q = ax.quiver(p[:, 0], p[:, 1], world[:, 0], world[:, 1], s[:, 2], cmap=SCORE_CMAP,
                      norm=TwoSlopeNorm(0.0, -rot_lim, rot_lim), angles="xy", width=0.004,
                      zorder=6)
        ax.set_title(rf"$\rho$ = {rho:.2f}", fontsize=9, loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.colorbar(q, ax=axes, fraction=0.015, pad=0.01).set_label(
        r"score $\theta$ [1/rad]", fontsize=8)
    fig.suptitle(f"{ds.spec.title} - score " + r"$\nabla\log p_\rho$"
                 f" (bandwidth {ds.div_config.get('bandwidth', '?')})",
                 fontsize=10, x=0.01, ha="left")
    return _save(fig, save)


def _phase_label_spots(poses: np.ndarray, modes: np.ndarray, center_x: float,
                       merge: float = 1.0, offset: float = 24.0) -> list:
    """Where to label one phase's poses: each mode's centroid, or one shared centroid while the
    modes still overlap, with a text offset (points) square to the cluster's mean heading, on
    the side facing away from the world's middle -- a cluster in the middle takes the left,
    clear of the legend."""
    groups = [poses[modes == m] for m in np.unique(modes)]
    centers = [g[:, :2].mean(axis=0) for g in groups]
    if max(np.linalg.norm(c - centers[0]) for c in centers) < merge:
        groups, centers = [poses], [poses[:, :2].mean(axis=0)]
    spots = []
    for g, center in zip(groups, centers):
        heading = np.arctan2(np.sin(g[:, 2]).mean(), np.cos(g[:, 2]).mean())
        normal = np.array([-np.sin(heading), np.cos(heading)])
        outward = 1.0 if center[0] > center_x + 0.25 else -1.0
        if normal[0] * outward < 0:
            normal = -normal
        spots.append((center, offset * normal))
    return spots


def plot_score_paths(ds: TrajectoryDataset, phases: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                     ax=None, max_arrows: int = 120, max_paths: Optional[int] = 300,
                     arrow_length: float = 0.6, save=None, title: Optional[str] = None,
                     pose_label: str = r"demo poses at the labelled phase $\rho$"):
    """:func:`plot_score_snapshots` on one plot: the demo paths and, at each phase, the demos'
    score arrows -- translational part in the world frame, colored by the rotational part --
    labelled with the phase. Every phase shares one arrow scale (``arrow_length`` metres for
    the 90th-percentile magnitude), so arrow lengths compare across the whole plot."""
    if ds.score is None:
        raise ValueError("Dataset has no score; run add_div_and_score first.")
    fig, ax = _axes(ax)
    _world_axes(ax, ds)
    for i in _subsample(len(ds), max_paths):
        ax.plot(ds.poses[i, :, 0], ds.poses[i, :, 1], "-", lw=0.5, alpha=0.25,
                color=PALETTE["ink_muted"], zorder=3)
    idx = _subsample(len(ds), max_arrows)
    n_steps = ds.poses.shape[1]
    rot_lim = float(np.nanpercentile(np.abs(ds.score[..., 2]), 98)) or 1.0
    norm = TwoSlopeNorm(0.0, -rot_lim, rot_lim)
    ref = float(np.nanpercentile(np.linalg.norm(ds.score[..., :2], axis=-1), 90)) or 1.0
    center_x = float(np.mean(ax.get_xlim()))
    for rho in phases:
        t = int(round(rho * (n_steps - 1)))
        p, s = ds.poses[idx, t], ds.score[idx, t]
        c, sn = np.cos(p[:, 2]), np.sin(p[:, 2])
        world = np.stack([c * s[:, 0] - sn * s[:, 1], sn * s[:, 0] + c * s[:, 1]], axis=1)
        ax.scatter(p[:, 0], p[:, 1], s=3, color=PALETTE["ink"], alpha=0.5, zorder=5)
        q = ax.quiver(p[:, 0], p[:, 1], world[:, 0], world[:, 1], s[:, 2], cmap=SCORE_CMAP,
                      norm=norm, angles="xy", scale_units="xy", scale=ref / arrow_length,
                      width=0.0035, zorder=6)
        for xy, offset in _phase_label_spots(p, ds.mode[idx], center_x):
            ax.annotate(rf"$\rho$ = {rho:g}", xy, xytext=offset, textcoords="offset points",
                        ha="center", va="center", fontsize=7, color=PALETTE["ink"], zorder=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc=PALETTE["surface"],
                                  ec=PALETTE["grid"], alpha=0.9),
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=PALETTE["ink_muted"]))
    handles = [
        Line2D([], [], color=PALETTE["ink_muted"], lw=0.8, alpha=0.6, label="demo paths"),
        Line2D([], [], ls="", marker="o", ms=3, color=PALETTE["ink"], label=pose_label),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.92,
              facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"]).set_zorder(10)
    # Top-right corner, clear of every scenario's demos; the key's anchor is the arrow's tail.
    ax.quiverkey(q, 0.905, 0.985, ref, rf"score $(x, y)$: {ref:.1f} 1/m", labelpos="W",
                 coordinates="axes", fontproperties={"size": 7})
    cb = fig.colorbar(q, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"score $\theta$ [1/rad]  (arrow colour)", fontsize=8,
                 color=PALETTE["ink_muted"])
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title or f"{ds.spec.title} - score " + r"$\nabla\log p_\rho$", fontsize=10,
                 color=PALETTE["ink"], loc="left")
    return _save(fig, save)


def plot_scenario_summary(datasets: Sequence[TrajectoryDataset], save=None):
    """One column per scenario: distribution, divergence-colored paths, divergence vs phase."""
    n = len(datasets)
    fig, axes = plt.subplots(
        3, n, figsize=(5.4 * n, 12.5), facecolor=PALETTE["surface"],
        gridspec_kw={"height_ratios": [1, 1, 0.7]}, squeeze=False,
    )
    vlim = max(float(np.nanpercentile(np.abs(d.divergence), 98)) for d in datasets)
    for j, ds in enumerate(datasets):
        plot_distribution(ds, ax=axes[0, j])
        plot_divergence_paths(ds, ax=axes[1, j], vlim=vlim, colorbar=(j == n - 1),
                              title="SE(2) divergence")
        plot_divergence_vs_phase(ds, ax=axes[2, j], title="divergence over phase")
    fig.tight_layout()
    return _save(fig, save)


# A/B/C for the figures: single_path's and dual_goal's spec titles carry their letter, but
# dual_path's does not, so the letter cannot simply be read off the title.
SCENARIO_LETTER = {"single_path": "A", "dual_goal": "B", "dual_path": "C"}


def _letter(ds: TrajectoryDataset, i: int) -> str:
    """The scenario's figure letter, falling back to its position for an unknown scenario."""
    return SCENARIO_LETTER.get(ds.scenario, chr(ord("A") + i))


def _scenario_heading(ds: TrajectoryDataset, i: int) -> str:
    """The spec's title under its letter, added where the title itself lacks one."""
    title = ds.spec.title
    body = title.split(") ", 1)[1] if len(title) > 1 and title[1] == ")" else title
    return f"{_letter(ds, i)}) {body}"


def plot_demo_overview(datasets: Sequence[TrajectoryDataset], save=None,
                       text_scale: float = 1.3, legend_scale: float = 1.3):
    """Every scenario's demonstrations side by side, one panel each -- :func:`plot_distribution`
    in a single row, with all text ``text_scale`` times its standalone size.

    The panels carry no axis labels, grid or ticks: the three worlds are the same 10 x 8 m box
    at the same scale, so the metric coordinates add nothing that the obstacles do not show.
    The legend names each scenario's modes and is the only key left on the panel, so it takes
    ``legend_scale`` on top of ``text_scale``."""
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.7), facecolor=PALETTE["surface"],
                             squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[0, i]
        plot_distribution(ds, ax=ax, title=_scenario_heading(ds, i))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        # Before _enlarge_text, so the legend takes both factors.
        for text in ax.get_legend().get_texts():
            text.set_fontsize(text.get_fontsize() * legend_scale)
    _enlarge_text(fig, text_scale)
    fig.tight_layout()
    return _save(fig, save)


def plot_field_overview(datasets: Sequence[TrajectoryDataset], save=None,
                        text_scale: float = 1.3):
    """The two fields over every scenario: the SE(2) divergence along the paths across the top
    row and the score along them across the bottom -- :func:`plot_divergence_paths` over
    :func:`plot_score_paths`, a column per scenario, each panel with its own colour scale."""
    n = len(datasets)
    fig, axes = plt.subplots(2, n, figsize=(5.67 * n, 9.4), facecolor=PALETTE["surface"],
                             squeeze=False)
    for i, ds in enumerate(datasets):
        letter = _letter(ds, i)
        plot_divergence_paths(ds, ax=axes[0, i], title=f"{letter}) SE(2) divergence")
        # Narrower than the standalone figure: the full legend label would cover the start lane.
        plot_score_paths(ds, ax=axes[1, i], title=rf"{letter}) score $\nabla\log p_\rho$",
                         pose_label=r"demo poses at phase $\rho$")
        axes[0, i].set_xlabel("")
        if i:
            axes[0, i].set_ylabel("")
            axes[1, i].set_ylabel("")
    _enlarge_text(fig, text_scale)
    fig.tight_layout()
    # The bigger phase labels can run into each other where the phases crowd (B's junction).
    _declutter_labels(fig)
    return _save(fig, save)


def plot_scenario_overview(datasets: Sequence[TrajectoryDataset], save=None,
                           text_scale: float = 1.3):
    """One row per scenario: its demonstrations, the SE(2) divergence along the paths and the
    score along the paths -- :func:`plot_distribution`, :func:`plot_divergence_paths` and
    :func:`plot_score_paths` side by side, each with its own colour scale, and all text
    ``text_scale`` times the size it has in those standalone figures."""
    n = len(datasets)
    fig, axes = plt.subplots(n, 3, figsize=(17.0, 4.7 * n), facecolor=PALETTE["surface"],
                             squeeze=False)
    for i, ds in enumerate(datasets):
        plot_distribution(ds, ax=axes[i, 0])
        # The other two panels each carry a colorbar: reserve the same room here, empty, so all
        # three maps come out the same size.
        fig.colorbar(plt.cm.ScalarMappable(), ax=axes[i, 0], fraction=0.04,
                     pad=0.02).ax.set_visible(False)
        plot_divergence_paths(ds, ax=axes[i, 1], title="SE(2) divergence")
        # These panels are narrower than the standalone figure: the full legend label would
        # cover the start lane.
        plot_score_paths(ds, ax=axes[i, 2], title=r"score $\nabla\log p_\rho$",
                         pose_label=r"demo poses at phase $\rho$")
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if j:
                ax.set_ylabel("")
            if i < n - 1:
                ax.set_xlabel("")
    _enlarge_text(fig, text_scale)
    fig.tight_layout()
    # The bigger phase labels can run into each other where the phases crowd (B's junction).
    _declutter_labels(fig)
    return _save(fig, save)
