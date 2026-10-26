"""Plots and animation for skate trajectories.

Design notes, since a motion plan is a *spatial* figure rather than a data chart:

* The obstacle field is drawn as muted fills plus a dashed contour of the inflated boundary
  (the ``radius + safety_margin`` level set the planner actually respects). A full-bleed SDF
  heatmap was rejected -- it would fight the trajectory for attention and force the path into a
  hue that clashes with the field's own ramp.
* Multi-start figures do **not** give each restart its own hue. Which restart is #5 carries no
  meaning; only best-vs-rest does. So the losers share one recessive neutral and the winner
  takes the single accent.
* Colors are the validated reference palette's categorical slots 1-3, used unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Rectangle

from .planner import SkateSolution
from .skate_model import rollout
from .world import ObstacleField

__all__ = ["PALETTE", "draw_world", "plot_solution", "plot_multistart", "animate"]

PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_muted": "#52514e",
    "path": "#2a78d6",  # categorical slot 1
    "goal": "#eb6834",  # slot 2
    "start": "#1baf7a",  # slot 3
    "infeasible": "#e34948",
    "obstacle": "#d5d4d0",
    "obstacle_edge": "#9a9994",
    "rest": "#c3c2b7",
    "grid": "#e8e7e4",
}


def _setup_axes(ax, world: ObstacleField) -> None:
    xmin, ymin, xmax, ymax = world.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_facecolor(PALETTE["surface"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8, length=3)
    ax.set_xlabel("x [m]", color=PALETTE["ink_muted"], fontsize=9)
    ax.set_ylabel("y [m]", color=PALETTE["ink_muted"], fontsize=9)


def draw_world(ax, world: ObstacleField, inflate: Optional[float] = None) -> None:
    """Obstacles, plus (if ``inflate`` is given) the boundary a disk robot must stay outside."""
    for circle in world.circles:
        ax.add_patch(
            CirclePatch((circle.cx, circle.cy), circle.r, facecolor=PALETTE["obstacle"],
                        edgecolor=PALETTE["obstacle_edge"], linewidth=1.0, zorder=2)
        )
    for box in world.boxes:
        ax.add_patch(
            Rectangle((box.cx - box.hx, box.cy - box.hy), 2 * box.hx, 2 * box.hy,
                      facecolor=PALETTE["obstacle"], edgecolor=PALETTE["obstacle_edge"],
                      linewidth=1.0, zorder=2)
        )
    if inflate:
        # One contour of the analytic SDF handles circles and boxes uniformly, including where
        # inflated obstacles merge into a single forbidden region.
        xmin, ymin, xmax, ymax = world.bounds
        gx, gy = np.meshgrid(
            np.linspace(xmin, xmax, 400), np.linspace(ymin, ymax, 300), indexing="xy"
        )
        field = world.analytic_sdf(np.stack([gx, gy], axis=-1))
        ax.contour(gx, gy, field, levels=[inflate], colors=[PALETTE["obstacle_edge"]],
                   linestyles="--", linewidths=0.8, zorder=3)


def _titles(ax, title: str, subtitle: str) -> None:
    """Two-line heading above the axes.

    Diagnostics are long enough to run off the figure if crammed into a single title, so the
    headline goes on top and the numbers sit under it at a smaller size.
    """
    ax.text(0.0, 1.075, title, transform=ax.transAxes, fontsize=10.5,
            color=PALETTE["ink"], va="bottom", ha="left")
    ax.text(0.0, 1.018, subtitle, transform=ax.transAxes, fontsize=8,
            color=PALETTE["ink_muted"], va="bottom", ha="left")


def _legend(ax) -> None:
    ax.legend(
        loc="upper left", fontsize=8, frameon=True, framealpha=0.92,
        facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["ink_muted"], borderpad=0.6,
    ).set_zorder(10)


def _draw_endpoints(ax, start: np.ndarray, goal: np.ndarray) -> None:
    for point, color, label in (
        (start, PALETTE["start"], "start"), (goal, PALETTE["goal"], "goal")
    ):
        ax.plot(point[0], point[1], "o", color=color, markersize=9, zorder=7, label=label)
        ax.annotate(
            label, (point[0], point[1]), textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=8, color=PALETTE["ink_muted"], zorder=7,
        )


def plot_solution(
    world: ObstacleField,
    solution: SkateSolution,
    radius: float,
    safety_margin: float = 0.0,
    index: Optional[int] = None,
    every: int = 4,
    save: Optional[str | Path] = None,
    title: Optional[str] = None,
):
    """Draw one trajectory with heading arrows and sampled robot footprints."""
    index = solution.best_index if index is None else index
    poses = solution.poses[index].detach().cpu().numpy()
    feasible = bool(solution.feasible[index]) if len(solution.feasible) else False
    accent = PALETTE["path"] if feasible else PALETTE["infeasible"]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=PALETTE["surface"])
    _setup_axes(ax, world)
    draw_world(ax, world, inflate=radius + safety_margin)

    ax.plot(poses[:, 0], poses[:, 1], "-", color=accent, linewidth=2.0, zorder=6,
            label="planned path" if feasible else "planned path (infeasible)")
    for pose in poses[::every]:
        ax.add_patch(
            CirclePatch((pose[0], pose[1]), radius, facecolor="none", edgecolor=accent,
                        linewidth=0.7, alpha=0.45, zorder=5)
        )
    ax.quiver(
        poses[::every, 0], poses[::every, 1],
        np.cos(poses[::every, 2]), np.sin(poses[::every, 2]),
        color=accent, width=0.0035, scale=28, zorder=6, alpha=0.9,
    )
    _draw_endpoints(ax, poses[0], poses[-1])

    status = "feasible" if feasible else "INFEASIBLE"
    _titles(ax, title or f"Skate plan - {status}", solution.diagnostics.compact(index))
    _legend(ax)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, facecolor=PALETTE["surface"])
    return fig


def plot_multistart(
    world: ObstacleField,
    solution: SkateSolution,
    radius: float,
    safety_margin: float = 0.0,
    save: Optional[str | Path] = None,
):
    """All restarts at once: losers recede into one neutral, the winner takes the accent."""
    poses = solution.poses.detach().cpu().numpy()
    best = solution.best_index

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=PALETTE["surface"])
    _setup_axes(ax, world)
    draw_world(ax, world, inflate=radius + safety_margin)

    for i, traj in enumerate(poses):
        if i == best:
            continue
        ax.plot(traj[:, 0], traj[:, 1], "-", color=PALETTE["rest"], linewidth=1.0,
                alpha=0.85, zorder=4)
    # One proxy handle so the legend says "restarts" once rather than N times.
    ax.plot([], [], "-", color=PALETTE["rest"], linewidth=1.0,
            label=f"restarts ({len(poses) - 1})")

    feasible = bool(solution.feasible[best]) if len(solution.feasible) else False
    accent = PALETTE["path"] if feasible else PALETTE["infeasible"]
    ax.plot(poses[best][:, 0], poses[best][:, 1], "-", color=accent, linewidth=2.4, zorder=6,
            label=f"best (#{best})" if feasible else f"best (#{best}, infeasible)")
    _draw_endpoints(ax, poses[best][0], poses[best][-1])

    n_feasible = int(solution.feasible.sum())
    _titles(
        ax,
        f"Batched multi-start - {n_feasible}/{len(poses)} restarts feasible",
        solution.diagnostics.compact(best),
    )
    _legend(ax)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, facecolor=PALETTE["surface"])
    return fig


def animate(
    world: ObstacleField,
    solution: SkateSolution,
    radius: float,
    dt: float,
    safety_margin: float = 0.0,
    index: Optional[int] = None,
    substeps: int = 4,
    save: Optional[str | Path] = None,
    fps: int = 20,
):
    """Animate the **rolled-out** trajectory -- what the robot would actually drive.

    Deliberately not the optimizer's pose variables: those are only collocation knots, and
    animating them would show a motion the skate model cannot produce.
    """
    index = solution.best_index if index is None else index
    start = solution.poses[index : index + 1, 0]
    controls = solution.controls[index : index + 1]
    path = rollout(start, controls, dt, substeps=substeps)[0].detach().cpu().numpy()

    feasible = bool(solution.feasible[index]) if len(solution.feasible) else False
    accent = PALETTE["path"] if feasible else PALETTE["infeasible"]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=PALETTE["surface"])
    _setup_axes(ax, world)
    draw_world(ax, world, inflate=radius + safety_margin)
    _draw_endpoints(ax, path[0], path[-1])

    trail, = ax.plot([], [], "-", color=accent, linewidth=1.8, zorder=6)
    body = CirclePatch((path[0, 0], path[0, 1]), radius, facecolor=accent, alpha=0.25,
                       edgecolor=accent, linewidth=1.2, zorder=7)
    ax.add_patch(body)
    heading = ax.quiver([path[0, 0]], [path[0, 1]], [np.cos(path[0, 2])], [np.sin(path[0, 2])],
                        color=accent, width=0.006, scale=18, zorder=8)
    _titles(ax, "Skate rollout", solution.diagnostics.compact(index))

    def update(frame: int):
        trail.set_data(path[: frame + 1, 0], path[: frame + 1, 1])
        body.center = (path[frame, 0], path[frame, 1])
        heading.set_offsets([[path[frame, 0], path[frame, 1]]])
        heading.set_UVC([np.cos(path[frame, 2])], [np.sin(path[frame, 2])])
        return trail, body, heading

    anim = FuncAnimation(fig, update, frames=len(path), interval=1000 / fps, blit=False)
    if save:
        anim.save(str(save), writer=PillowWriter(fps=fps))
    return anim
