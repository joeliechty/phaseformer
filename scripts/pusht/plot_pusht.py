"""Figures of the Push-T training data and of the closed-loop rollouts, in the skate figures' style.

    python scripts/pusht/plot_pusht.py                 # everything
    python scripts/pusht/plot_pusht.py --only data     # training-data figures only
    python scripts/pusht/plot_pusht.py --only rollouts
    python scripts/pusht/plot_pusht.py --only gifs     # the ranked-rollout GIFs only
    python scripts/pusht/plot_pusht.py --only pareto_gifs   # the same for the Pareto models
    python scripts/pusht/plot_pusht.py --arch transformer   # the _tf runs -> fig_pusht_*_tf
    python scripts/pusht/plot_pusht.py --rollout-tag <tag>_last   # overfit -> fig_pusht_*_last
    python scripts/pusht/plot_pusht.py --hold-at-rho1             # phase arms stop at rho = 1

Writes ``output/pusht_9010/figures/fig_pusht_*.{png,pdf}``, and the GIFs at the end of the list.
The defaults draw the paper's 90/10 runs (``runs/pusht/policy``); the earlier 70/20/10 figures take
``--runs runs/pusht/policy_702010 --prepared data/pusht/pusht_prepared_702010.npz
--out output/pusht/figures``.

Training data (the training episodes of the --prepared split)

* ``fig_pusht_demos``            eight example episodes: pusher path coloured by phase, the block at
                                 five phases (light to dark), the goal T
* ``fig_pusht_distribution``     every training pusher path, every block-centre path with its start
                                 pose, and the episode lengths with the nominal clock's median
* ``fig_pusht_phase_evolution``  per-phase density of pusher x/y, block x/y/θ and SE(2) distance to
                                 the goal, from the shared phase grid (the skate phase-evolution
                                 heatmaps)
* ``fig_pusht_velocity_score``   at ρ = 0, ¼, ½, ¾, 1: the demos' phase velocity ``u = dx/dρ`` and
                                 the agent-position kernel score the regularizer uses

Rollouts (``{--runs}/{method}/seed*/rollout_{ROLLOUT_TAG}.npz``, by default the
phase-MLP clock with the start pose, eps = 1/245 and the fixed step)

* ``fig_pusht_rollout_grid``     rows: test envs, columns: methods (seed 0); pusher path coloured by
                                 time with a marker where the clock's floor reaches rho = 1, the start and
                                 final block and the goal, the episode score in the title
* ``fig_pusht_final_poses_backbones`` the same, both backbones: a row of MLP arms over a row of
  their transformer twins, the demos and the open-loop chunk baselines sharing the two leading
  columns, two to a column (drawn once, from --arch mlp);
* ``fig_pusht_final_poses``      the final block pose of every env (seed 0) coloured by coverage,
                                 held-out demos first -- the skate rollout figures' layout
* ``fig_pusht_coverage``         coverage against step, mean over every seed's 50 envs with the
                                 spread over seeds, the clock saturation marked
* ``fig_pusht_inference``       inference cost per decision and per control step, and the score
                                 it buys, with the Pareto frontier
* ``fig_pusht_violin``           episode scores and per-seed rollout metrics per method
* ``fig_pusht_violin_backbones`` the same, every MLP arm then every transformer twin
* ``fig_pusht_inference_backbones``  the inference figure, both backbones' arms
* ``fig_pusht_filmstrip``        simulator frames of one env, a row per method
* ``fig_pusht_filmstrip_ranked_{method}``  simulator frames of a method's best, median and worst
                                 episode over all seeds and envs
* ``fig_pusht_rollout_ranked_{method}.gif``  every policy seed's best, median and worst episode
                                 on its own envs (best and worst by final coverage, median by
                                 score), a column per seed, every control step at --gif-speed (3x)
                                 real time; MIP and ACT too. ``--only pareto_gifs`` draws them for
                                 the four arms of each ``PARETO_MODELS`` model, its phase arms on
                                 its own clock (``fig_pusht_rollout_ranked_{arm}_{mlpmed,tfmed}``)

``--hold-at-rho1`` stops every phase arm's episode where its clock first reads rho = 1 and holds
the state from there -- a pusher that does nothing once its phase is done -- and redraws the
figures that changes as ``*_rho1``: the backbones' final poses, the backbone violins (score,
success, final coverage and errors at the stop; a stall is the pusher sitting still over the 25
steps up to it) and the filmstrips (the ranked ones for the phase arms only; the rest are
unchanged). With ``--only gifs`` or ``--only pareto_gifs`` it draws the phase arms' GIFs, each
panel frozen where its clock first reads rho = 1.

Plots are drawn in the simulator's screen frame (y down), so they match rendered frames.
"""

from __future__ import annotations

import argparse
import math
import re
import textwrap
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from fm_bc.pusht.constants import (  # noqa: E402
    GOAL_POSE, ROT_SCALE_PX_PER_RAD, SUCCESS_COVERAGE, TEE_VERTICES, WORKSPACE_PX,
)
from fm_bc.pusht.data import PushTEpisodes  # noqa: E402
from fm_bc.pusht.policy import agent_score  # noqa: E402
from fm_bc.pusht.prepared import PushTPrepared  # noqa: E402
from fm_bc.pusht.phase_mlp import MonotonePhase, load_phase_mlp  # noqa: E402
from fm_bc.pusht.policy_training import ARCH_SUFFIX as POLICY_SUFFIX  # noqa: E402
from fm_bc.pusht.sim import NominalClock  # noqa: E402
from fm_bc.se2 import se2_log, twist_norm  # noqa: E402

# skate_planner.viz.PALETTE, restated: importing skate_planner pulls in theseus.
PALETTE = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink_muted": "#52514e", "grid": "#e8e7e4",
           "block": "#8f9fb3", "block_edge": "#5d6b7c", "goal": "#1baf7a", "demo": "#52514e"}
# The hatch that marks a transformer twin in the violins, drawn in the arm's own colour at full
# strength over a tinted fill: stroked above the 1 pt default so it reads at a glance and in
# print, but left at three strokes a tile so the fill and the seed dots still show through.
HATCH, HATCH_LW = "///", 1.2
# The skate figures' method colours (evaluate_policies.METHOD_COLORS slots): the MLP's orange,
# MLP+phase's blue, the score arm's pink; the phase-free score arm shares the MLP's orange,
# dashed. The score arms are the constant-lambda runs (--score-arms annealed for the others).
METHODS = [
    ("mlp", "MLP", "#eb6834", "-"),
    ("mlp_score_w2.5_const", "MLP+score", "#eb6834", "--"),
    ("fm", "MLP+phase", "#2a78d6", "-"),
    ("score_w2.5_const", "MLP+phase+score", "#e87ba4", "--"),
    # The generative chunk baselines, in evaluate_policies' slots (DP purple, MIP red), open loop
    # solid and closed loop (ACT temporal ensembling) dotted.
    ("dp", "DP (open)", "#4a3aa7", "-"),
    ("dp_te", "DP (closed)", "#4a3aa7", ":"),
    ("mip", "MIP (open)", "#e34948", "-"),
    ("mip_te", "MIP (closed)", "#e34948", ":"),
    # ACT, the deterministic chunk baseline: evaluate_policies' slot 10, which reuses the aqua of
    # the constant-lambda CDM arm because the palette has only eight hues.
    ("act", "ACT", "#1baf7a", "-"),
    ("act_te", "ACT (closed)", "#1baf7a", ":"),
]
# Chunk baselines reported closed loop only: their bars quote the closed-loop cost (a whole
# decision per executed step), the score panel carries just their `_te` point, and the violin
# figures draw the one arm rather than a pair. ACT is here because its open-loop per-step figure
# is an artefact of executing 8 steps per decision -- it reads as the arm being 8x cheaper than
# it is under the loop it is actually evaluated in, and its open-loop rollouts are the same
# eight-steps-to-a-decision policy the closed-loop ones re-evaluate properly.
CLOSED_LOOP_ONLY = {"act"}
# The two score arms come in two schedules of the same lambda: annealed to 0 over the first half
# of training (the sweep's pick) and held constant for the whole run (`_const`). --score-arms
# swaps one family for the other everywhere in the figures.
SCORE_ARMS = {"annealed": ("mlp_score_w2.5_cos0.5", "score_w2.5_cos0.5"),
              "const": ("mlp_score_w2.5_const", "score_w2.5_const")}
# Baselines predict the env's action, so their rollouts carry their own tag and share a run dir.
BASELINE_TAGS = {"dp": "chunk", "mip": "chunk", "act": "chunk",
                 "dp_te": "chunk_te0.01", "mip_te": "chunk_te0.01", "act_te": "chunk_te0.01"}
BASELINE_DIR = {"dp_te": "dp", "mip_te": "mip", "act_te": "act"}


def _backbone_pairs():
    """``METHODS`` with each policy arm followed by its transformer twin; baselines unchanged.

    The pair shares the arm's colour and sits side by side, so the comparison reads as a
    within-pair difference and no new hue is spent.
    """
    pairs = []
    for entry in METHODS:
        if entry[0] in BASELINE_TAGS:
            continue
        pairs += [entry, _tf_arms([entry])[0]]
    return pairs + [e for e in METHODS if e[0] in BASELINE_TAGS]


def _backbone_groups():
    """``METHODS`` as all the MLP arms, then all their transformer twins, then the baselines.

    :func:`_backbone_pairs` interleaves the twins so each comparison is the neighbouring
    violin; this groups by backbone instead, so a whole backbone reads as a block and the
    arms keep their order within it. Colour still names the arm and the hatch the backbone,
    so a pair is found by hue across the two blocks rather than by adjacency.
    """
    arms = [e for e in METHODS if e[0] not in BASELINE_TAGS]
    return arms + _tf_arms(arms) + [e for e in METHODS if e[0] in BASELINE_TAGS]


def _score_merge():
    """Each score arm mapped to the arm whose network it shares, on both backbones.

    The score term is a training-time loss, not a second head: the two arms run the identical
    forward pass, and the timers separate them only by rounding. ``fig_inference`` draws such a
    pair as one bar row at their mean cost, split between the two arms' colours.
    """
    mlp_score, phase_score = SCORE_ARMS[SCORE_ARM]
    pairs = {mlp_score: "mlp", phase_score: "fm"}
    return pairs | {f"{k}{ARCH_SUFFIX}": f"{v}{ARCH_SUFFIX}" for k, v in pairs.items()}


def _tf_arms(methods):
    """``methods`` with every policy arm swapped for its transformer twin; baselines unchanged
    -- a chunk baseline is its own architecture and has no ``_tf`` run.

    ``ARCH_LABEL`` takes ``"MLP"``'s place, so the label names the backbone exactly as "MLP"
    does for the MLP arms and the two transformers never both read as plain "TF".
    """
    return [(m, l, c, ls) if m in BASELINE_TAGS
            else (f"{m}{ARCH_SUFFIX}", l.replace("MLP", ARCH_LABEL), c, ls)
            for m, l, c, ls in methods]
# The per-env grids stay readable with a column per method: the MLP arms and DP open loop.
GRID_METHODS = ("mlp", "mlp_score_w2.5_const", "fm", "score_w2.5_const", "dp")
# A GIF per method has no grid to crowd, so the ranked-rollout GIFs add the other chunk
# baselines: MIP open loop, as DP is, and ACT in the one loop it is reported in (closed).
GIF_BASELINES = ("mip", "act_te")
# fig_final_poses_backbones has a leading block rather than a leading column, so it can carry
# every open-loop chunk baseline against the arms, not just DP. Order sets where each lands:
# the block fills column by column down its two rows, the demos first.
BACKBONE_BASELINES = ("dp", "act", "mip")
# Type sizes as fractions of the figure's width, by role: evaluate_policies.TYPE_RATIOS, copied
# for the reason PALETTE is. The figures sized from them print their text at the size
# fig_inference_pareto_both does, once both are scaled to the same width on the page.
TYPE_RATIOS = {"title": 0.0151, "label": 0.0144, "legend": 0.0126, "tick": 0.0122,
               "note": 0.0115, "data": 0.0112}


def type_pt(width_in: float, role: str) -> float:
    """Points for ``role``'s text in a figure ``width_in`` inches wide (``TYPE_RATIOS``)."""
    return TYPE_RATIOS[role] * 72 * width_in
# Sequential ramps (one hue, light -> dark): time / phase along a path, and coverage.
TIME_CMAP = LinearSegmentedColormap.from_list("time", ["#cfe0f5", "#2a78d6", "#0d2f5c"])
COVER_CMAP = LinearSegmentedColormap.from_list("cover", ["#d7f0e5", "#1baf7a", "#0a4d34"])
# The clock the rollout figures show. The evaluation condition: the phase estimator that also sees
# the episode's start pose, under the monotone ratchet with eps = 1 / 245 (one step of the longest
# demonstration), the policy step fixed at the median demonstration's 1 / (L_med - 1). Pass
# --rollout-tag to plot another clock; {--runs}/*/seed*/rollout_*.json lists them.
ROLLOUT_TAG = "phase_mlp_start_fixedstep_e1o245_inverse_pd"
# Per-method overrides of ROLLOUT_TAG, for a figure whose arms ran on different clocks
# (fig_violin_pareto: each model's phase arms on its own clock). Empty for every other figure.
METHOD_TAGS: dict[str, str] = {}
# The phase estimators those rollouts ran under, one per seed (--phase-runs): the "at rho = 1"
# figures replay the clock on the saved states to find the step it first read 1.
PHASE_RUNS = Path("runs/pusht/phase_start")
# --hold-at-rho1: the rollouts run on to the step budget after the clock saturates; this stops
# every phase arm's episode where its clock first read rho = 1 and holds the state from there,
# a pusher that does nothing once its task is done (_hold_at_phase_one). The phase-free arms and
# the baselines are untouched. The figures it draws carry RHO_TAG.
HOLD_AT_RHO1 = False
# A held arm's stall is the pusher sitting still over this many steps up to the stop -- half the
# other arms' 50 (_stalled): the held episodes end at rho = 1, around step 170 on average, rather
# than at the 300-step budget, so the window covers a like share of the episode.
HOLD_STALL_WINDOW = 25
SCORE_ARM = "const"  # --score-arms; the key of SCORE_ARMS the figures are drawn from
# The same four policy arms on an AdaLN transformer backbone carry policy_training.py's
# ARCH_SUFFIX in their run names. --arch draws every figure from those runs and tags the file
# names with the suffix, so all three backbones' figure sets coexist; ARCH_LABEL takes "MLP"'s
# place in the labels, naming the backbone exactly as "MLP" does for the MLP arms.
#
# These two are what --arch rebinds; they are read as globals at call time, so rebinding them in
# main() is enough for every site below (_backbone_pairs, _score_merge, _tf_arms, fig_violin_
# backbones' `base=` and `hatched=`, fig_rollouts, _save). The suffixes never collide in either
# direction -- "..._tfbig".endswith("_tf") is False -- so a _tf figure run cannot pick up _tfbig
# runs, nor the reverse.
ARCH = {"transformer": ("_tf", "TF"), "transformer_big": ("_tfbig", "TF-67M")}
ARCH_SUFFIX, ARCH_LABEL = ARCH["transformer"]
# The file-name tags, so a figure set never overwrites another's: the backbone (--arch), the
# checkpoint (--rollout-tag ..._last, the final overfit model rather than the lowest validation
# loss) and the phase arms held at rho = 1 (--hold-at-rho1). A name carries them in that order,
# e.g. fig_pusht_violin_tf_last, fig_pusht_filmstrip_tf_rho1.
ARCH_TAG = ""
CKPT_TAG = ""
RHO_TAG = ""
AGENT_RADIUS = 15.0
# The T in its body frame (gym-pusht add_tee, scale 30), as one outline. Shared with the policies
# that tokenize points on the block (fm_bc.pusht.chunks.tee_points), so there is one polygon.
TEE = np.array(TEE_VERTICES, dtype=np.float64)


# -- geometry ------------------------------------------------------------------------------


def in_world(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """``(P, 2)`` body-frame points under ``(..., 3)`` block poses -> ``(..., P, 2)``."""
    poses = np.asarray(poses, dtype=np.float64)
    c, s = np.cos(poses[..., 2])[..., None], np.sin(poses[..., 2])[..., None]
    x = poses[..., None, 0] + c * points[:, 0] - s * points[:, 1]
    y = poses[..., None, 1] + s * points[:, 0] + c * points[:, 1]
    return np.stack([x, y], axis=-1)


def tee_vertices(poses: np.ndarray) -> np.ndarray:
    """``(..., 3)`` block poses -> ``(..., 8, 2)`` outline vertices in the world frame."""
    return in_world(TEE, poses)


_GOAL = shapely.polygons(tee_vertices(np.array(GOAL_POSE)))
# The body points ADD averages over: the T filled on a 4 px grid (~390 points), so the average
# runs over the block's area and not its corners, which would overweight the stem's tip.
_grid = np.stack(np.meshgrid(*(np.arange(TEE[:, k].min(), TEE[:, k].max() + 4.0, 4.0)
                               for k in (0, 1)), indexing="ij"), -1).reshape(-1, 2)
TEE_POINTS = _grid[shapely.contains_xy(shapely.polygons(TEE), _grid[:, 0], _grid[:, 1])]


def coverage(poses: np.ndarray) -> np.ndarray:
    """The env's reward coverage, ``area(goal ∩ block) / area(goal)``, for ``(..., 3)`` poses."""
    shape = np.asarray(poses).shape[:-1]
    polys = shapely.polygons(tee_vertices(np.asarray(poses).reshape(-1, 3)))
    return (shapely.area(shapely.intersection(_GOAL, polys)) / shapely.area(_GOAL)).reshape(shape)


def goal_error(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(..., 3)`` block poses -> (position error [px], orientation error [deg]) to the goal.

    Reported apart rather than blended into one SE(2) distance: blending needs a px-per-radian
    exchange rate (``ROT_SCALE_PX_PER_RAD`` = 100) that decides the answer, and the two are not
    interchangeable to the task -- the T can sit on the goal at the wrong angle, and a reader
    comparing against :func:`fig_final_poses` sees position and orientation separately.
    """
    q = np.asarray(poses, dtype=np.float64)
    gx, gy, gt = GOAL_POSE
    pos = np.hypot(q[..., 0] - gx, q[..., 1] - gy)
    rot = np.degrees(np.abs((q[..., 2] - gt + np.pi) % (2 * np.pi) - np.pi))
    return pos, rot


def add_error(poses: np.ndarray) -> np.ndarray:
    """``(..., 3)`` block poses -> the ADD distance to the goal pose [px].

    The 6D-pose-estimation metric (average distance of corresponding model points) on SE(2):
    carry the same body points through both poses and average the Euclidean distances. Unlike
    a blended SE(2) norm it needs no px-per-radian exchange rate -- the block's own geometry
    sets one, so a given angular error costs more out at the tip of the stem than at the
    crossbar. It is the single continuous number :func:`goal_error`'s two panels are the
    decomposition of, and it is read in the same unit as the position error beside it.
    """
    d = in_world(TEE_POINTS, poses) - in_world(TEE_POINTS, np.asarray(GOAL_POSE))
    return np.linalg.norm(d, axis=-1).mean(-1)


# -- drawing helpers -----------------------------------------------------------------------


def _arena(ax, title: str | None = None, weight: str = "normal", frame: float = 0.8,
           frame_color: str | None = None) -> None:
    """The workspace as one panel draws it: the simulator's square, no ticks, a hairline frame.

    ``frame`` and ``frame_color`` are for a grid of many small arenas, where the default hairline
    in the grid colour leaves the panels running into one another -- a heavier, darker frame tells
    a reader where one arena ends and the next begins.
    """
    ax.set_xlim(0, WORKSPACE_PX)
    ax.set_ylim(WORKSPACE_PX, 0)  # the simulator's screen frame: y down
    ax.set_aspect("equal")
    ax.set_facecolor(PALETTE["surface"])
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(frame_color or PALETTE["grid"])
        side.set_linewidth(frame)
    if title:
        ax.set_title(title, fontsize=9, color=PALETTE["ink"], loc="left", weight=weight)


def _goal(ax, lw: float = 1.4, shade: float = 0.0) -> None:
    """The goal pose as a dashed T outline, optionally shaded.

    ``shade`` is the fill's alpha. A shaded goal reads as the region the block has to reach rather
    than as four more lines in a panel, which is what it needs to be wherever it sits under a
    scatter of positions and arrows. The fill carries its alpha in the colour, not in the patch's,
    so that the outline stays at full strength over it.
    """
    ax.add_patch(plt.Polygon(tee_vertices(np.array(GOAL_POSE)), closed=True, fill=shade > 0,
                             fc=to_rgba(PALETTE["goal"], shade), ec=PALETTE["goal"], lw=lw,
                             ls="--", zorder=1))


def _tee(ax, pose, face=None, alpha=1.0, edge=None, lw=0.8, zorder=2) -> None:
    ax.add_patch(plt.Polygon(tee_vertices(pose), closed=True,
                             fc=face or PALETTE["block"], ec=edge or PALETTE["block_edge"],
                             alpha=alpha, lw=lw, zorder=zorder))


def _path(ax, xy: np.ndarray, values: np.ndarray, cmap, norm, lw=1.4, zorder=3) -> None:
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, norm=norm, lw=lw, zorder=zorder, capstyle="round")
    lc.set_array(values[:-1])
    ax.add_collection(lc)


def _tint(color, alpha: float):
    """``color`` at ``alpha`` over the surface, returned opaque.

    A hatched fill must not carry alpha: the PDF backend paints the hatch as a tiling pattern
    whose tile it fills with the face colour's *RGB*, dropping the alpha, so a translucent fill
    saves as a solid block of the arm's colour and the hatch disappears into it. Blending here
    keeps what the reader sees and leaves the pattern a light tile under saturated strokes.
    """
    from matplotlib.colors import to_rgb  # noqa: PLC0415
    fg, bg = np.array(to_rgb(color)), np.array(to_rgb(PALETTE["surface"]))
    return tuple(alpha * fg + (1 - alpha) * bg)


# Symbols a figure sets directly and LaTeX needs told about, restated here rather than shared
# with evaluate_policies for the reason PALETTE is: that module pulls in theseus. Longest keys
# first: " · " is a clause break in a figure header and a comma in caption prose, where a lone
# "·" is not.
CAPTION_TEX = {
    " · ": ", ", "·": r"$\cdot$", "λ": r"$\lambda$", "ε": r"$\varepsilon$", "ρ": r"$\rho$",
    "→": r"$\to$", "†": r"$\dagger$", "×": r"$\times$", "≥": r"$\ge$", "≤": r"$\le$",
    "∩": r"$\cap$", "°": r"$^\circ$", "–": "--", "<": "$<$", ">": "$>$", "%": r"\%",
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
        note = "$".join(parts)
        # A symbol the figure writes as "ε = 1/245" becomes "$\varepsilon$ = 1/245" above; the
        # relation and its value belong inside that math span, not beside it.
        note = re.sub(r"\$([^$]+)\$ = ([\w/.]+)", r"$\1 = \2$", note)
        out.append(note.rstrip(".") + ".")
    return " ".join(out)


def _tagged(name: str) -> str:
    """``name`` with the backbone and checkpoint tags: no figure set overwrites another's."""
    if ARCH_TAG and not name.endswith(ARCH_TAG):  # ranked filmstrips are named after a run
        name = f"{name}{ARCH_TAG}"          # that already ends in ARCH_SUFFIX
    return f"{name}{CKPT_TAG}{RHO_TAG}"


def _save(fig, out: Path, name: str, fixed_width: bool = False,
          caption: str | None = None, pad: float = 0.1) -> None:
    """``fixed_width`` trims only the top and bottom, keeping the figure's full width.

    A tight box trims to the content, and the content's left edge is the widest y tick label of
    the first panel -- so two cuts of one figure, laid out to the same figure width, still save a
    few pixels apart. Trimming in y alone makes the saved width the figure width exactly.

    ``caption`` is written beside the figure as ``<name>.caption.txt``, LaTeX ready -- under the
    figure's own final name, so it carries the backbone and checkpoint tags with it.

    ``pad``: inches of margin a tight box leaves around the content.
    """
    name = _tagged(name)
    bbox = "tight"
    if fixed_width:
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        bbox = Bbox([[0.0, tight.y0], [fig.get_figwidth(), tight.y1]])
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, facecolor=PALETTE["surface"],
                    bbox_inches=bbox, pad_inches=pad)
    plt.close(fig)
    if caption is not None:
        (out / f"{name}.caption.txt").write_text(caption + "\n")
    print(f"  wrote {name}" + (" (+ caption)" if caption is not None else ""))


def _legend(fig_or_ax, handles, **kw) -> None:
    leg = fig_or_ax.legend(handles=handles, frameon=False, fontsize=8, **kw)
    for t in leg.get_texts():
        t.set_color(PALETTE["ink"])


# -- training data -------------------------------------------------------------------------


def fig_demos(ep: PushTEpisodes, ids, out: Path) -> None:
    ids = list(ids)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.9), facecolor=PALETTE["surface"])
    norm = Normalize(0, 1)
    for ax, e in zip(axes.flat, ids):
        s = ep.state[ep.episode_slice(e)].double().numpy()
        L = len(s)
        _arena(ax, f"episode {e} · {L} steps")
        _goal(ax)
        for k, ph in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
            _tee(ax, s[round(ph * (L - 1)), 2:5], alpha=0.12 + 0.2 * k, zorder=2)
        _path(ax, s[:, :2], np.linspace(0, 1, L), TIME_CMAP, norm)
        ax.add_patch(plt.Circle(s[0, :2], AGENT_RADIUS * 0.6, fc="white", ec=PALETTE["ink"],
                                lw=0.8, zorder=4))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, TIME_CMAP), ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("phase ρ = t / (L − 1)  (pusher path)", color=PALETTE["ink_muted"], fontsize=8)
    handles = [Line2D([], [], color=PALETTE["goal"], ls="--", label="goal pose"),
               plt.Polygon([[0, 0]], fc=PALETTE["block"], ec=PALETTE["block_edge"],
                           label="block at ρ = 0, ¼, ½, ¾, 1 (light → dark)"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=PALETTE["ink"],
                      label="pusher start")]
    fig.subplots_adjust(top=0.9)
    _legend(fig, handles, loc="upper center", ncol=3, bbox_to_anchor=(0.45, 0.975))
    _save(fig, out, "fig_pusht_demos")


def fig_distribution(ep: PushTEpisodes, prep: PushTPrepared, out: Path) -> None:
    train = prep.train_ids.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), facecolor=PALETTE["surface"],
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    _arena(axes[0], f"pusher paths · {len(train)} training episodes")
    _arena(axes[1], "block-centre paths and start poses")
    for ax in axes[:2]:
        _goal(ax)
    for e in train:
        s = ep.state[ep.episode_slice(e)].numpy()
        axes[0].plot(s[:, 0], s[:, 1], color=PALETTE["demo"], lw=0.5, alpha=0.18)
        axes[1].plot(s[:, 2], s[:, 3], color=PALETTE["demo"], lw=0.6, alpha=0.3)
        _tee(axes[1], s[0, 2:5], alpha=0.06, lw=0.4, zorder=1)
        axes[1].plot(*s[-1, 2:4], "o", ms=2.5, color=PALETTE["ink"], zorder=3)
    _legend(axes[1], [Line2D([], [], color=PALETTE["goal"], ls="--", label="goal pose"),
                      Line2D([], [], marker="o", ls="", ms=3, color=PALETTE["ink"],
                             label="block centre at episode end")], loc="lower left")

    ax = axes[2]
    L = ep.lengths[prep.train_ids].numpy()
    ax.hist(L, bins=np.arange(40, 260, 10), color=PALETTE["block"], ec=PALETTE["surface"], lw=1)
    # The clock's own median (torch's lower median), so the figure names the length rollouts use.
    med = NominalClock.from_episodes(ep.lengths[prep.train_ids]).median_length
    ax.axvline(med, color=PALETTE["ink"], lw=1.2, ls="--")
    ax.text(med + 3, ax.get_ylim()[1] * 0.95, f"median {med:.0f} steps\n(nominal clock)",
            fontsize=8, color=PALETTE["ink"], va="top")
    ax.set_xlabel("episode length [steps at 10 Hz]", fontsize=9, color=PALETTE["ink_muted"])
    ax.set_ylabel("training episodes", fontsize=9, color=PALETTE["ink_muted"])
    ax.set_title("episode lengths", fontsize=9, loc="left")
    _plain(ax)
    _save(fig, out, "fig_pusht_distribution")


def _plain(ax) -> None:
    ax.set_facecolor(PALETTE["surface"])
    ax.grid(True, color=PALETTE["grid"], lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8)


def fig_phase_evolution(prep: PushTPrepared, out: Path) -> None:
    g = prep.grid.grid_state[prep.train_ids].double()  # (E, N, 5) theta in (-pi, pi]
    goal = torch.tensor(GOAL_POSE, dtype=torch.float64).expand_as(g[..., 2:5])
    d = twist_norm(se2_log(g[..., 2:5], goal), rot_scale=ROT_SCALE_PX_PER_RAD)
    series = [(g[..., 0], "pusher x [px]", (0, 512)), (g[..., 1], "pusher y [px]", (0, 512)),
              (g[..., 2], "block x [px]", (0, 512)), (g[..., 3], "block y [px]", (0, 512)),
              (g[..., 4], "block θ [rad]", (-np.pi, np.pi)),
              (d, "SE(2) distance to goal [px]", (0, float(d.max()) * 1.02))]
    phase = prep.grid.grid_phase.numpy()
    fig, axes = plt.subplots(1, 6, figsize=(14, 4.6), facecolor=PALETTE["surface"], sharey=True)
    for ax, (v, label, (lo, hi)) in zip(axes, series):
        v = v.numpy()
        bins = np.linspace(lo, hi, 61)
        # Density per phase row: every row sums to 1 over the value bins.
        H = np.stack([np.histogram(v[:, j], bins=bins)[0] for j in range(v.shape[1])]).astype(float)
        H /= H.sum(1, keepdims=True)
        im = ax.imshow(H, aspect="auto", origin="upper", cmap="viridis",
                       extent=(lo, hi, 1.0, 0.0), interpolation="bilinear")
        ax.plot(np.median(v, 0), phase, color="white", lw=1, ls="--", alpha=0.8)
        ax.set_xlabel(label, fontsize=9, color=PALETTE["ink_muted"])
        ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8)
    axes[0].set_ylabel("phase ρ", fontsize=9, color=PALETTE["ink_muted"])
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("fraction of episodes at this phase", fontsize=8, color=PALETTE["ink_muted"])
    fig.suptitle(f"Training episodes on the shared phase grid ({prep.grid.n_grid} points) · "
                 "white dashed: median", fontsize=10, color=PALETTE["ink"], x=0.45)
    _save(fig, out, "fig_pusht_phase_evolution")


def fig_velocity_score(ep: PushTEpisodes, prep: PushTPrepared, out: Path,
                       bandwidth_px: float = 32.0) -> None:
    pos = prep.normalizer("agent_pos", "minmax")
    grid = prep.grid
    xy = grid.agent_positions()[prep.train_ids].double()  # (E, N, 2) px
    N = grid.n_grid
    # Phase velocity on the grid: dx/dρ by forward difference, px per unit phase.
    u = torch.cat([xy[:, 1:] - xy[:, :-1], xy[:, -1:] - xy[:, -2:-1]], 1) * (N - 1)
    phases = (0.0, 0.25, 0.5, 0.75, 1.0)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.2), facecolor=PALETTE["surface"])
    for col, ph in enumerate(phases):
        j = round(ph * (N - 1))
        pts = xy[:, j]
        q = pos.apply(pts.float())
        sc = agent_score(q.double(), q.double()[None].expand(len(q), -1, -1),
                         bandwidth_px / float(pos.scale.mean()))  # 1 / normalized unit
        for row, (vec, name) in enumerate([(u[:, j], "phase velocity u = dx/dρ"),
                                           (sc, "score ∇ log p_ρ(x)")]):
            ax = axes[row, col]
            # Ten small arenas side by side: a heavy dark frame keeps each panel its own, and
            # the goal is shaded so it stays legible under the scatter and the arrows.
            _arena(ax, f"ρ = {ph:g} · {name}" if col == 0 else f"ρ = {ph:g}",
                   frame=2.0, frame_color=PALETTE["ink_muted"])
            _goal(ax, lw=2.4, shade=0.22)
            v = vec.numpy()
            ref = np.percentile(np.linalg.norm(v, axis=1), 90)
            ax.scatter(pts[:, 0], pts[:, 1], s=6, color=PALETTE["ink"], zorder=3, lw=0)
            ax.quiver(pts[:, 0], pts[:, 1], v[:, 0], v[:, 1], angles="xy", scale_units="xy",
                      scale=ref / 60.0, width=0.006, color=["#2a78d6", "#e87ba4"][row],
                      alpha=0.85, zorder=4)
    handles = [Line2D([], [], marker="o", ls="", ms=3, color=PALETTE["ink"],
                      label="pusher position of each training episode at that phase"),
               Line2D([], [], color="#2a78d6", lw=2, label="demo phase velocity (row 1)"),
               Line2D([], [], color="#e87ba4", lw=2,
                      label=f"kernel score, h = {bandwidth_px:g} px (row 2)"),
               Line2D([], [], color=PALETTE["goal"], ls="--", lw=2.4, label="goal pose")]
    fig.subplots_adjust(top=0.9, hspace=0.18)
    _legend(fig, handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.97))
    fig.text(0.5, -0.01, "Arrow lengths are scaled per panel (the 90th-percentile arrow is 60 px), "
             "so they show direction and relative size within a panel only.",
             ha="center", fontsize=8, color=PALETTE["ink_muted"])
    _save(fig, out, "fig_pusht_velocity_score")


# -- rollouts ------------------------------------------------------------------------------


def _saturation(meta: dict) -> int | None:
    """The step at which the clock's phase reaches 1 on its own (the nominal clock's length, or the
    step the ratchet's ``eps`` floor needs) -- what the rollout figures mark. ``None`` for a chunk
    baseline, which has no clock."""
    if "median_length" in meta:
        return round(meta.get("phase_length", meta["median_length"]) - 1)
    if "eps" in meta and meta.get("ratchet", True):  # without the ratchet there is no floor
        return round(1.0 / meta["eps"])
    return None


def _clock_label(meta: dict) -> str:
    if meta.get("clock") == "phase_mlp" and not meta.get("ratchet", True):
        return "phase-MLP clock, no ratchet (the raw estimate, fixed step)"
    if _saturation(meta) is None:
        return f"chunk baseline, {meta.get('loop', 'open')} loop (no phase clock)"
    if meta.get("clock") == "phase_mlp":
        step = "fixed step" if meta.get("step") else "estimate step"
        return (f"phase-MLP clock (ε = 1/{1 / meta['eps']:.0f}, {step}), "
                f"ρ = 1 by step {_saturation(meta)}")
    length = meta.get("phase_length", meta.get("median_length"))
    return f"nominal clock L = {length:g}, ρ = 1 at step {_saturation(meta)}"


def _rollouts(runs: Path, method: str, seed: int | None = None) -> list[dict]:
    pattern = f"seed{seed}" if seed is not None else "seed*"
    tag = BASELINE_TAGS.get(method, METHOD_TAGS.get(method, ROLLOUT_TAG))
    out = []
    for f in sorted((runs / BASELINE_DIR.get(method, method)).glob(f"{pattern}/rollout_{tag}.npz")):
        d = dict(np.load(f))
        d["meta"] = json.loads(f.with_suffix(".json").read_text())
        d["meta"]["_path"] = str(f)
        out.append(_hold_at_phase_one(d) if HOLD_AT_RHO1 and method in _phase_arms() else d)
    return out


def _phase_arms() -> set[str]:
    """The arms whose policy reads the phase: MLP+phase and MLP+phase+score, on every backbone
    (``policy_training.ARCH_SUFFIX``: the published two and their resized twins)."""
    arms = {"fm", SCORE_ARMS[SCORE_ARM][1]}
    return {f"{m}{suffix}" for m in arms for suffix in POLICY_SUFFIX.values()}


def _phase_one_steps(r: dict) -> np.ndarray:
    """Per env, the first step its clock read rho = 1, or its last step if it succeeded first.

    The clock is deterministic given the states, so the one the rollout ran (the same
    :class:`MonotonePhase` or :class:`NominalClock`, rebuilt from its json) is replayed on the
    saved ones. The estimator is the one the json records when that file exists -- a rollout on
    a resized clock names its own -- and otherwise the policy seed's from ``PHASE_RUNS``: the
    published rollouts record a ``runs_9010/`` path that has since moved there.
    """
    meta = r["meta"]
    if meta["clock"] == "nominal":
        clock = NominalClock(meta["median_length"], meta["phase_length"])
    else:
        recorded = Path(meta.get("phase_model", ""))
        fallback = PHASE_RUNS / f"seed{_policy_seed(r)}" / "model.pt"
        estimator, _ = load_phase_mlp(recorded if recorded.is_file() else fallback)
        clock = MonotonePhase(estimator, eps=meta["eps"], smooth=meta["smooth"],
                              max_rate=meta["max_rate"], step=meta["step"],
                              ratchet=meta.get("ratchet", True))
    stops = []
    for states, steps in zip(r["states"], r["steps"].astype(int)):
        clock.reset()
        stops.append(next((t for t in range(steps) if clock.advance(t, states[t])[0] >= 1.0),
                          steps))
    return np.array(stops)


_HELD: dict[str, dict] = {}  # rollout file -> its held arrays: the replay is per step, so once


def _hold_at_phase_one(r: dict, window: int = HOLD_STALL_WINDOW, tol: float = 1.0) -> dict:
    """``r`` stopped where each episode's clock first read rho = 1 (:func:`_phase_one_steps`),
    the state held from there to the end, as an episode that succeeds early holds its last.

    ``steps``, ``score`` (the max reward up to the stop) and ``success`` (coverage over 0.95 at
    the stop) are recomputed to match, and ``held`` marks the episodes the clock cut rather than
    success. ``stalled`` is measured against the clock: the pusher moving < ``tol`` px over the
    ``window`` steps up to the stop, sitting still while the phase counts up -- the hold after
    it is never a stall.
    """
    key = r["meta"]["_path"]
    if key not in _HELD:
        stops = _phase_one_steps(r)
        S = r["states"].copy()
        for e, t in enumerate(stops):
            S[e, t + 1:] = S[e, t]
        cov = coverage(S[..., 2:5])
        xy = [S[e, max(0, t - window):t + 1, :2] for e, t in enumerate(stops)]
        _HELD[key] = {
            "states": S, "steps": stops.astype(float),
            "score": np.clip(cov[:, 1:] / SUCCESS_COVERAGE, 0.0, 1.0).max(1),
            "success": (cov[np.arange(len(stops)), stops] > SUCCESS_COVERAGE).astype(float),
            "held": stops < r["steps"].astype(int),
            "stalled": np.array([np.abs(p - p[-1]).max() < tol for p in xy]),
        }
    return {**r, **_HELD[key]}


def _stop(r: dict, e: int) -> str:
    """What ended episode ``e`` early, for a frame's title: the clock or the env."""
    return "ρ = 1" if "held" in r and r["held"][e] else "done"


def _stalled(r: dict, window: int = 50, tol: float = 1.0) -> np.ndarray:
    """Which episodes of a rollout stalled: the pusher moves < ``tol`` px over the ``window``
    steps up to the episode's own end.

    The end is the episode's last real step (``steps``), not the step budget: the env ends an
    episode the step it succeeds and the rollout holds that state to the budget, so a window on
    the last stored steps would count every early success as a stall -- the held padding never
    moves. Measured up to ``steps`` instead, a success is judged on the pushing that got it there,
    and an episode that ran the whole budget on its last ``window`` steps, as before.

    A phase arm held at rho = 1 carries its own, measured the same way up to its stop
    (:func:`_hold_at_phase_one`).
    """
    if "stalled" in r:
        return r["stalled"]
    ends = r["steps"].astype(int)
    return np.array([np.abs(xy - xy[-1]).max() < tol for xy in
                     (r["states"][e, max(0, t - window):t + 1, :2] for e, t in enumerate(ends))])


def fig_rollout_grid(runs: Path, out: Path, envs=(0, 1, 2, 3, 4)) -> None:
    methods = [m for m in METHODS if m[0] in GRID_METHODS and _rollouts(runs, m[0], 0)]
    fig, axes = plt.subplots(len(envs), len(methods), figsize=(2.9 * len(methods), 2.9 * len(envs)),
                             facecolor=PALETTE["surface"], squeeze=False)
    norm = Normalize(0, 299)
    sat = None
    for c, (m, label, color, _) in enumerate(methods):
        r = _rollouts(runs, m, 0)[0]
        sat = _saturation(r["meta"]) or sat
        for row, i in enumerate(envs):
            ax = axes[row, c]
            S, T = r["states"][i], int(r["steps"][i])
            _arena(ax)
            _goal(ax)
            _tee(ax, S[0, 2:5], face="none", edge=PALETTE["block_edge"], lw=0.8, alpha=0.9, zorder=2)
            _tee(ax, S[T, 2:5], alpha=0.75, zorder=2)
            _path(ax, S[: T + 1, :2], np.arange(T + 1), TIME_CMAP, norm, lw=1.2)
            if _saturation(r["meta"]) is not None and T > sat:
                ax.plot(*S[sat, :2], marker="D", ms=7, mfc="white", mec=PALETTE["ink"], zorder=5)
            ax.plot(*S[T, :2], marker="o", ms=3.5, color=PALETTE["ink"], zorder=6)
            head = f"{label}\n" if row == 0 else ""
            ax.set_title(f"{head}env {int(r['seeds'][i])} · score {r['score'][i]:.2f}",
                         fontsize=8.5, loc="left", color=PALETTE["ink"])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, TIME_CMAP), ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("step (pusher path)", fontsize=8, color=PALETTE["ink_muted"])
    handles = [plt.Polygon([[0, 0]], fc="none", ec=PALETTE["block_edge"], label="block at start"),
               plt.Polygon([[0, 0]], fc=PALETTE["block"], ec=PALETTE["block_edge"],
                           label="block at the end"),
               Line2D([], [], color=PALETTE["goal"], ls="--", label="goal pose"),
               Line2D([], [], marker="D", ls="", mfc="white", mec=PALETTE["ink"],
                      label=f"pusher at step {sat} (the clock's floor reaches ρ = 1)"),
               Line2D([], [], marker="o", ls="", ms=3.5, color=PALETTE["ink"],
                      label="pusher at the end (dot inside ◇: it never moved again)")]
    fig.subplots_adjust(top=0.93, hspace=0.3)
    _legend(fig, handles, loc="upper center", ncol=3, bbox_to_anchor=(0.46, 0.985))
    _save(fig, out, "fig_pusht_rollout_grid")


def _demo_final(ep: PushTEpisodes, test) -> np.ndarray:
    """The block pose in the last frame of each held-out episode."""
    return np.stack([ep.state[ep.episode_ends[e] - 1, 2:5].double().numpy() for e in test])


def _run_final(runs: Path, method: str) -> np.ndarray:
    """The block pose each of seed 0's 50 envs ended at."""
    r = _rollouts(runs, method, 0)[0]
    return r["states"][np.arange(len(r["steps"])), r["steps"].astype(int), 2:5]


def _final_poses_panel(ax, poses: np.ndarray, norm, frame: float = 0.8,
                       frame_color: str | None = None) -> np.ndarray:
    """The picture alone -- the final Ts over the goal in an arena -- and their goal coverage.

    The text around it belongs to :func:`_draw_final_poses`; a figure that wants only the panel
    (:func:`fig_final_poses_pair`) draws through here and passes ``frame`` for its own border.
    """
    cov = coverage(poses)
    order = np.argsort(cov)  # best on top
    pc = PolyCollection(tee_vertices(poses[order]), array=cov[order], cmap=COVER_CMAP,
                        norm=norm, alpha=0.55, edgecolors=PALETTE["surface"], lw=0.3, zorder=2)
    ax.add_collection(pc)
    _goal(ax)
    _arena(ax, frame=frame, frame_color=frame_color)
    return cov


def _draw_final_poses(ax, poses: np.ndarray, norm, title: str, weight: str = "normal",
                      fs: float = 9, note_fs: float | None = None) -> None:
    """One panel of final block poses over the goal, each T shaded by its goal coverage.

    ``fs`` sizes the panel's text: the name and its sub-line at ``fs``, the coverage read-out a
    point under it, as the 9/8 pair the single-row figure has always used, unless ``note_fs``
    sets it.
    """
    cov = _final_poses_panel(ax, poses, norm)
    # The panel's name and what it was measured over, drawn as two texts rather than one
    # two-line title: `weight` then falls on the name alone and leaves the sub-line plain.
    lines = title.split("\n")
    if len(lines) == 3:  # a plain line over the name: the model, then its arm, then the seed
        head, name, sub = lines
        ax.set_title(head, fontsize=fs, color=PALETTE["ink"], loc="left", pad=2.9 * fs)
        ax.annotate(name, (0.0, 1.015), xycoords="axes fraction", xytext=(0, 1.2 * fs),
                    textcoords="offset points", fontsize=fs, color=PALETTE["ink"],
                    weight=weight, va="bottom")
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=fs, color=PALETTE["ink"],
                va="bottom")
        ax.text(0.03, 0.03, f"mean coverage {cov.mean():.2f}\n> 0.95: {np.mean(cov > 0.95):.0%}",
                transform=ax.transAxes, fontsize=note_fs or fs - 1, color=PALETTE["ink"],
                va="bottom")
        return
    name, _, sub = title.partition("\n")
    ax.set_title(name, fontsize=fs, color=PALETTE["ink"], loc="left", weight=weight,
                 pad=1.7 * fs if sub else 6)
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=fs, color=PALETTE["ink"],
                va="bottom")
    ax.text(0.03, 0.03, f"mean coverage {cov.mean():.2f}\n> 0.95: {np.mean(cov > 0.95):.0%}",
            transform=ax.transAxes, fontsize=note_fs or fs - 1, color=PALETTE["ink"],
            va="bottom")


def _coverage_bar(fig, axes, norm, fs: float = 8, tick_fs: float | None = None,
                  pad: float = 0.03, shrink: float = 0.8,
                  label: str = "goal coverage of the final block pose") -> None:
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, COVER_CMAP), ax=axes, fraction=0.012, pad=pad,
                      shrink=shrink)
    cb.set_label(label, fontsize=fs, color=PALETTE["ink_muted"])
    # The ticks have always sat two points over the label, unless tick_fs says otherwise.
    cb.ax.tick_params(labelsize=tick_fs or fs + 2)


def fig_final_poses(ep: PushTEpisodes, prep: PushTPrepared, runs: Path, out: Path) -> None:
    methods = [m for m in METHODS if m[0] in GRID_METHODS and _rollouts(runs, m[0], 0)]
    cols = 1 + len(methods)
    fig, axes = plt.subplots(1, cols, figsize=(3.0 * cols, 3.55), facecolor=PALETTE["surface"])
    norm = Normalize(0, 1)
    test = prep.held_out_ids.tolist()
    _draw_final_poses(axes[0], _demo_final(ep, test), norm,
                      f"held-out demos\nlast frame of {len(test)} episodes")
    for ax, (m, label, _, _) in zip(axes[1:], methods):
        _draw_final_poses(ax, _run_final(runs, m), norm, f"{label}\n50 envs, seed 0")
    _coverage_bar(fig, axes, norm)
    _save(fig, out, "fig_pusht_final_poses")


# fig_final_poses_pair: the bare backbone over its phase+score arm, by label so that --arch
# picks the transformer twins (the labels are the same names with the backbone's in front).
PAIR_SUFFIXES = ("", "+phase+score")
# The pair has no titles to tell one panel from the next, so its frame does the separating.
PAIR_FRAME_LW = 3.5


def fig_final_poses_pair(runs: Path, out: Path) -> None:
    """Two panels of :func:`fig_final_poses`, stacked: the bare backbone above its phase+score arm.

    The figure the paper's single row cannot be -- one arm directly over the other, at the size a
    slide or a two-column inset wants, where the surrounding text names the two panels. Stripped
    to the picture: no titles, no coverage read-out, no colour bar, and a heavy black border in
    place of the titles that would otherwise say where one panel ends and the next begins.
    """
    base = ARCH_LABEL if ARCH_TAG else "MLP"
    labels = {label: m for m, label, _, _ in METHODS}
    methods = [labels.get(f"{base}{suffix}") for suffix in PAIR_SUFFIXES]
    if all(m and _rollouts(runs, m, 0) for m in methods):
        _final_poses_pair(runs, out, methods, "fig_pusht_final_poses_pair")


def _final_poses_pair(runs: Path, out: Path, methods, name: str) -> None:
    """:func:`fig_final_poses_pair`'s stack for any ``methods``, top to bottom."""
    fig, axes = plt.subplots(len(methods), 1, figsize=(3.2, 3.2 * len(methods)),
                             facecolor=PALETTE["surface"])
    norm = Normalize(0, 1)
    for ax, m in zip(axes, methods):
        _final_poses_panel(ax, _run_final(runs, m), norm, frame=PAIR_FRAME_LW, frame_color="black")
    fig.subplots_adjust(hspace=0.06, left=0.01, right=0.99, top=0.99, bottom=0.01)
    _save(fig, out, name)


def fig_final_poses_pair_pareto(runs: Path, out: Path) -> None:
    """:func:`fig_final_poses_pair` for the violins' transformer (``PARETO_MODELS``): the bare
    TF-med wide above its phase+score arm on its own clock, as
    ``fig_pusht_final_poses_pair_pareto_tf`` (``_rho1`` with the phase arm stopped where that
    clock first reads rho = 1)."""
    suffix, _, tag, _, _ = next(e for e in PARETO_MODELS if e[0].startswith("_tf"))
    phase_score = f"{SCORE_ARMS[SCORE_ARM][1]}{suffix}"
    METHOD_TAGS[phase_score] = tag
    methods = [f"mlp{suffix}", phase_score]
    if all(_rollouts(runs, m, 0) for m in methods):
        _final_poses_pair(runs, out, methods, "fig_pusht_final_poses_pair_pareto_tf")


def fig_final_poses_backbones(ep: PushTEpisodes, prep: PushTPrepared, runs: Path,
                              out: Path) -> None:
    """:func:`fig_final_poses` over both backbones: a row of the MLP arms above a row of their
    transformer twins, the same arm in the same column.

    The panels both rows are read against -- the held-out demos and the open-loop chunk
    baselines -- share a leading block to the left of the arms, filled column by column down
    the two rows, rather than being drawn once per row.

    Under ``HOLD_AT_RHO1`` the phase arms are drawn where their clock first read rho = 1 and
    every other panel as before, as ``fig_pusht_final_poses_backbones_rho1``.
    """
    arms = [m for m in METHODS if m[0] in GRID_METHODS and m[0] not in BASELINE_TAGS]
    rows = [[(m, label) for m, label, _, _ in row if _rollouts(runs, m, 0)]
            for row in (arms, _tf_arms(arms))]
    if not all(rows):
        return
    _final_poses_grid(ep, prep, runs, out, rows, "fig_pusht_final_poses_backbones")


def _final_poses_grid(ep: PushTEpisodes, prep: PushTPrepared, runs: Path, out: Path, rows,
                      name: str, row_height: float = 4.0, top: float = 0.9) -> None:
    """:func:`fig_final_poses_backbones`' grid for any rows of ``(method, title)``: the leading
    block of held-out demos and open-loop chunk baselines, then one row per backbone. A title
    naming a phase arm is bold; ``row_height`` and ``top`` make room for longer titles."""
    test = prep.held_out_ids.tolist()
    labels = {m: label for m, label, _, _ in METHODS}
    # "last frame of" is dropped from the demos' sub-line: at type_pt's sizes it runs into the
    # next panel's title, and every panel here is a last frame.
    lead = [(f"held-out demos\n{len(test)} episodes", _demo_final(ep, test))]
    lead += [(f"{labels[m]}\n50 envs, seed 0", _run_final(runs, m))
             for m in BACKBONE_BASELINES if _rollouts(runs, m, 0)]
    n_lead = -(-len(lead) // len(rows))  # columns the leading block takes, two panels each
    cols = n_lead + max(len(row) for row in rows)
    # Wider columns than the single-row figure's 3.0 in: the panels are square, so the extra
    # width is gap between them -- room for type_pt's text that would otherwise run into the
    # next panel's title. Taller rows too, for the two-line titles at that size.
    fig, axes = plt.subplots(len(rows), cols, figsize=(3.4 * cols, row_height * len(rows)),
                             facecolor=PALETTE["surface"], squeeze=False)
    width = fig.get_figwidth()
    fs, note_fs = type_pt(width, "title"), type_pt(width, "note")
    # Narrow gutters and margins: the panels are square, so what the default spacing leaves
    # between them is width the panels, and the titles over them, could have had.
    fig.subplots_adjust(left=0.005, right=0.955, bottom=0.01, top=top, wspace=0.08, hspace=0.3)
    norm = Normalize(0, 1)
    for r, methods in enumerate(rows):
        for c in range(n_lead):
            i = c * len(rows) + r
            if i < len(lead):
                _draw_final_poses(axes[r, c], lead[i][1], norm, lead[i][0], fs=fs,
                                  note_fs=note_fs)
            else:
                axes[r, c].set_axis_off()
        for ax, (m, label) in zip(axes[r, n_lead:], methods):
            # The phase arms -- the arms the figure is making a case for -- carry a bold title.
            phase = "phase" in label
            # "seed 0, at ρ = 1" in place of "50 envs, seed 0": the longer line runs into the
            # next panel's title, and the 50 envs are the same in every panel.
            sub = "seed 0, at ρ = 1" if HOLD_AT_RHO1 and phase else "50 envs, seed 0"
            _draw_final_poses(ax, _run_final(runs, m), norm, f"{label}\n{sub}",
                              "bold" if phase else "normal", fs, note_fs)
        for ax in axes[r, n_lead + len(methods):]:
            ax.set_axis_off()
    _coverage_bar(fig, axes, norm, type_pt(width, "label"), type_pt(width, "tick"), pad=0.015,
                  shrink=1.0, **({"label": "goal coverage of the block pose"}
                                 if HOLD_AT_RHO1 else {}))
    _save(fig, out, name, pad=0.02)


def fig_final_poses_pareto(ep: PushTEpisodes, prep: PushTPrepared, runs: Path,
                           out: Path) -> None:
    """:func:`fig_final_poses_backbones` for the violins' models (``PARETO_MODELS``): a row of
    each model's four arms, its phase arms on its own clock, after the same leading block. A
    title is the model, the arm (named as the Pareto figure's legend names it) and the seed."""
    arm_names = {"mlp": "neither", SCORE_ARMS[SCORE_ARM][0]: "score", "fm": "phase",
                 SCORE_ARMS[SCORE_ARM][1]: "phase + score"}
    rows = []
    for suffix, name, tag, _, _ in PARETO_MODELS:
        row = []
        for m in arm_names:
            run = f"{m}{suffix}"
            if run in _phase_arms():
                METHOD_TAGS[run] = tag
            if _rollouts(runs, run, 0):
                row.append((run, f"{name}\n{arm_names[m]}"))
        rows.append(row)
    if all(rows):
        _final_poses_grid(ep, prep, runs, out, rows, "fig_pusht_final_poses_pareto",
                          row_height=4.6, top=0.88)


def fig_coverage(runs: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=PALETTE["surface"])
    sat = None
    for m, label, color, ls in METHODS:
        R = _rollouts(runs, m)
        if not R:
            continue
        sat = _saturation(R[0]["meta"]) or sat
        per_seed = []
        for r in R:
            cov = coverage(r["states"][..., 2:5])  # (50, 301); finished envs repeat their state
            per_seed.append(np.maximum.accumulate(cov, axis=1).mean(0))  # best coverage so far
        per_seed = np.stack(per_seed)
        t = np.arange(per_seed.shape[1])
        mean, sd = per_seed.mean(0), per_seed.std(0)
        ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)
        ax.plot(t, mean, color=color, ls=ls, lw=2, label=f"{label} ({len(R)} seeds)")
    if sat is not None:
        ax.axvline(sat, color=PALETTE["ink_muted"], lw=1, ls=":")
        ax.text(sat + 3, 0.02, f"clock floor ρ = 1\n(step {sat})", fontsize=8,
                color=PALETTE["ink_muted"], va="bottom")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, None)
    ax.set_xlabel("control step (10 Hz)", fontsize=9, color=PALETTE["ink_muted"])
    ax.set_ylabel("best goal coverage so far", fontsize=9, color=PALETTE["ink_muted"])
    ax.set_title("Closed-loop rollouts · mean over 50 envs per seed, band ±1 std over seeds",
                 fontsize=9.5, loc="left", color=PALETTE["ink"])
    _plain(ax)
    _legend(ax, ax.get_legend_handles_labels()[0], loc="upper left")
    _save(fig, out, "fig_pusht_coverage")


# The one panel that pools every episode (seeds x envs): the share reported above the violin,
# and where a never-moving pusher sits in its unit -- it scores 0.139, and since the score is
# ``coverage / 0.95`` unclipped down there, that is a coverage of 0.139 * 0.95.
POOLED = {"episodes": (0.9, 0.139)}
# The two terminal-error panels, and the start-pose error whose mean is their never-moving
# reference: a pusher that does not move leaves the block where it started. Both are the mean
# within a seed, read like the mean score beside them. Per episode they were a three-decade log
# cloud that only got harder to read as arms were added, and averaging is what makes the split
# legible -- the phase arms are the best on orientation and the worst on position.
SEED_ERRORS = {"final_pos": "start_pos", "final_rot": "start_rot", "final_add": "start_add",
               # Not an error, but read the same way: the block's goal coverage where it ended,
               # against where a never-moving pusher leaves it.
               "final_cov": "start_cov"}


def _violin_width(n_shown: int, n_cols: int) -> float:
    """Inches for a violin figure of ``n_cols`` panels, each holding ``n_shown`` violins.

    Pass it to :func:`fig_violin` to hold two cuts of the same arms to one width: the narrower
    one then stretches its panels instead of shrinking the figure.
    """
    return max(3.4, 0.42 * n_shown) * n_cols


def _violin_notes(runs: Path, shown, keys, hatched, base: str, n_ep: int, n_seeds: int,
                  reference: bool, extra: tuple[str, ...] = ()) -> list[str]:
    """What :func:`fig_violin` is a figure of, for the panels it actually draws.

    Caption prose, not figure furniture -- it goes in the ``.caption.txt`` beside the file
    rather than into a header block, which cost the panels the better part of an inch and set
    the same sentences at a size nobody reads them at.
    """
    notes = [f"Push-T · closed loop ({_clock_label(_rollouts(runs, shown[0][0])[0]['meta'])}, "
             "DP protocol)"]
    if HOLD_AT_RHO1:
        notes.append("the phase arms (bold) stop where their clock first reads ρ = 1 and hold "
                     "the state from there: score, success, final coverage and errors at that "
                     "stop, and a stall is the pusher moving $< 1$ px over the "
                     f"{HOLD_STALL_WINDOW} steps up to it (the other arms: the 50 up to the "
                     "episode's end)")
    if "episodes" in keys:
        notes.append(f"score per episode: {n_ep} episodes, percentile lines 25 / 50 / 75, median "
                     "and share ≥ 0.9 above the violin")
    if isinstance(hatched, dict):  # a hatch per backbone, which the legend names
        notes.append("hatch: the backbone (legend); every backbone's arms share the arm colours")
    elif hatched:  # the twins share a colour; the hatch is what separates the backbones
        notes.append("hatched: the transformer backbone; twins share the arm's colour")
    notes += list(extra)
    notes.append("score arms: λ held constant (no anneal)" if SCORE_ARM == "const"
                 else "score arms: λ annealed to 0")
    notes.append(f"per-seed panels: {n_seeds} seeds (dots), mean (bar)")
    if "stalled" in keys:
        notes.append("stalled: the pusher moves $< 1$ px over the 50 steps up to the episode's "
                     "end -- the step the env ends it on success, else the step budget; the "
                     "number over each arm is its mean, and a flat lens marks an arm whose seeds "
                     "all stalled equally often")
    if "success" in keys:  # it reads as IoU and is not; see the panel title
        notes.append("success: the env's coverage over 0.95 -- overlap with the goal over the "
                     "goal's own area, an IoU of 0.90")
    line = (["the score it earns"] if "episodes" in keys and reference else []) + (
        ["the block left at its start pose"]
        if reference and SEED_ERRORS.keys() & set(keys) else [])
    if line:
        notes.append("the dotted line is a never-moving pusher: " + " and ".join(line))
    labels = dict((m, l) for m, l, _, _ in list(METHODS) + list(shown))
    notes.append(f"† marks $p < 0.05$ against {labels.get(base, base)} (Welch over seeds)")
    return notes


def fig_violin(runs: Path, out: Path, base: str | None = None, methods=None, hatched=(),
               name: str = "fig_pusht_violin", keys: tuple[str, ...] = (),
               width: float = 0.0, font_scale: float = 1.0, reference: bool = True,
               by_width: bool = False, hatch_legend=(), notes: tuple[str, ...] = ()) -> None:
    """Per-method distributions, as the skate violin figure.

    **Score per episode** pools all seeds × envs: every episode is a dot, the violin its density,
    and the lines inside it the 25th / 50th / 75th percentiles; the numbers above give the median
    and the share of episodes at or over 0.9. The other five panels are per seed (dots, mean bar):
    closed-loop mean score, success, the number of stalled episodes, and three terminal errors
    of the block -- position, orientation, and the :func:`add_error` distance that folds the two
    into one length (the evaluation clock, DP protocol); the dotted line is a never-moving
    pusher, in each panel's unit. † marks p < 0.05 against ``base`` (Welch over seeds), which
    defaults to the plotted backbone's phase+score arm. The phase arms' labels are bold: phase
    is the axis this whole figure set is about.

    ``keys`` selects and orders the panels, for figures asking a narrower question than the
    default seven; they lay out in one row up to four, then a second. ``width`` overrides the
    figure's own :func:`_violin_width`, to match a sibling figure with a different panel count.
    ``font_scale`` multiplies every type size, for a figure meant to be read at less than the
    full width of a page; the header and footer grow with it. ``reference=False`` drops the
    never-moving-pusher line, along with its legend entry and its sentence in the caption: it
    is a floor the arms clear by so much that in some cuts it only stretches the axis.
    ``by_width`` sizes the type from the figure's width instead (``TYPE_RATIOS``, by role), and
    ``font_scale`` then follows the title's size, for the header and footer.

    ``hatched`` is a set of methods drawn with the transformer's hatch, or a dict giving each
    method its own; ``hatch_legend`` then lists ``(label, hatch or None)`` for the legend, and
    ``notes`` adds sentences to the caption.
    """
    from matplotlib.patches import Ellipse, Patch  # noqa: PLC0415
    from matplotlib.ticker import MaxNLocator  # noqa: PLC0415
    from matplotlib.transforms import blended_transform_factory  # noqa: PLC0415
    from scipy.stats import ttest_ind  # noqa: PLC0415

    # The full arm, whichever schedule is plotted -- and whichever backbone, or an --arch
    # transformer figure would test against an MLP arm it does not draw and show no daggers.
    def pt(size: float) -> float:
        return size * font_scale

    base = base or f"{SCORE_ARMS[SCORE_ARM][1]}{ARCH_TAG}"
    phase_arms = _phase_arms()
    # A baseline reported closed loop only is drawn once, as its closed-loop arm. Its label
    # then has no twin to be told apart from, so the loop comes off the name.
    shown = [(m, l.replace(" (closed)", "") if BASELINE_DIR.get(m) in CLOSED_LOOP_ONLY else l,
              c, ls)
             for m, l, c, ls in (methods or METHODS)
             if _rollouts(runs, m) and m not in CLOSED_LOOP_ONLY]
    data = {}
    for m, *_ in shown:
        R = _rollouts(runs, m)
        final = [r["states"][np.arange(len(r["steps"])), r["steps"].astype(int), 2:5] for r in R]
        start = [r["states"][:, 0, 2:5] for r in R]
        data[m] = {
            "episodes": np.concatenate([r["score"] for r in R]),
            "score": np.array([r["score"].mean() for r in R]),
            "success": 100 * np.array([r["success"].mean() for r in R]),
            "final_pos": np.array([goal_error(q)[0].mean() for q in final]),
            "final_rot": np.array([goal_error(q)[1].mean() for q in final]),
            "final_add": np.array([add_error(q).mean() for q in final]),
            "final_cov": np.array([coverage(q).mean() for q in final]),
            "start_pos": np.concatenate([goal_error(q)[0] for q in start]),
            "start_rot": np.concatenate([goal_error(q)[1] for q in start]),
            "start_add": np.concatenate([add_error(q) for q in start]),
            "start_cov": np.concatenate([coverage(q) for q in start]),
            "stalled": np.array([_stalled(r).sum() for r in R], dtype=float),
        }
    n_env = len(_rollouts(runs, shown[0][0])[0]["score"])
    titles = {
        "episodes": "score per episode (all seeds × envs)",
        "final_pos": "mean final position error per seed [px]",
        "final_rot": "mean final orientation error per seed [°]",
        "final_add": "mean final ADD per seed [px]",
        "final_cov": "mean final coverage per seed",
        "score": "mean score per seed",
        # Spelled out on the panel because it reads as IoU and is not: the denominator is the
        # goal's area alone, so the 0.95 threshold is an IoU of 0.95 / (2 - 0.95) = 0.90.
        "success": "success rate per seed (%)\ncoverage = area(goal ∩ block) / area(goal) > 0.95",
        "stalled": f"stalled episodes per seed (of {n_env})",
    }
    # The title names the panel and what it is averaged over; the y label names the quantity the
    # axis carries, with its unit, so a panel read on its own is not unitless.
    ylabels = {"episodes": "score", "final_pos": "position error [px]",
               "final_rot": "orientation error [°]", "final_add": "ADD [px]",
               "final_cov": "goal coverage",
               "score": "mean score", "success": "success rate [%]",
               "stalled": "stalled episodes"}
    keys = keys or ("episodes", "final_pos", "final_rot", "final_add",
                    "score", "success", "stalled")
    panels = [(k, titles[k], "score" if k == "episodes" else k) for k in keys]
    # One row of panels up to four, then a second; the grid has as many columns as both rows
    # divide, so the default's 4 + 3 is 12. A panel is wide enough for the rotated labels.
    rows = [list(keys)] if len(keys) <= 4 else [list(keys[:4]), list(keys[4:])]
    # Two rows sit on a grid of at least the default's 12, so a 4 + 4 split keeps its gutters:
    # wspace below scales with the column count, not exactly with the figure's width.
    ncol = math.lcm(*(len(r) for r in rows), *((12,) if len(rows) > 1 else ()))
    if len(rows) > 1 and len(rows[1]) == len(rows[0]):
        # A full second row has no slot clear of the first row's rotated labels, so a two-line
        # title runs into them: success drops its coverage definition, which the caption
        # carries, as it does in the cuts set from the width.
        panels = [(k, t.split("\n")[0], pk) for k, t, pk in panels]
    width = width or _violin_width(len(shown), max(len(r) for r in rows))
    # The size of each role, at font_scale unless by_width sets it from the width.
    fs = {"title": pt(9.5), "label": pt(9), "legend": pt(9), "tick": pt(8.5), "arm": pt(8),
          "data": pt(7.5), "dagger": pt(11)}
    if by_width:
        font_scale = type_pt(width, "title") / 9.5
        fs = {role: type_pt(width, role) for role in ("title", "label", "legend", "tick", "data")}
        fs |= {"arm": fs["tick"], "dagger": fs["title"] * 11 / 9.5}
    # Type set from the width runs larger per panel than the panels were drawn for, so the
    # titles wrap to their panel (at ~0.52 em a character), and the arm names turn steeper until
    # neighbouring labels clear each other (their spacing, across the slant, a line and a third).
    gutter = 0.065 if by_width else 0.10
    panel_pt = 0.94 * (1 - gutter) * 72 * width / max(len(r) for r in rows)
    rotation = 30.0
    if by_width:
        # Shortened to fit a line: "/seed" for " per seed", and success without its coverage
        # definition, which the caption carries.
        panels = [(k, t.split("\n")[0].replace(" per seed", "/seed"), pk) for k, t, pk in panels]
        chars = int(panel_pt / (0.52 * fs["title"]))
        panels = [(k, "\n".join(textwrap.fill(line, chars) for line in t.split("\n")), pk)
                  for k, t, pk in panels]
        pitch = panel_pt / len(shown)
        rotation = max(rotation, math.degrees(math.asin(min(1.0, 1.35 * fs["arm"] / pitch))))
    n_ep = len(data[shown[0][0]]["episodes"])
    n_seeds = len(data[shown[0][0]]["score"])
    caption = _caption_tex(
        _violin_notes(runs, shown, keys, hatched, base, n_ep, n_seeds, reference, notes))
    # Header: the legend, a gap, then the tallest panel title (``success`` carries its
    # definition on a second line) -- the conditions are in the caption file beside the figure.
    # Footer: the rotated labels. All of it is type, so all of it scales with it.
    head = ((pt(0.28) if by_width else pt(0.42) + 0.16)
            + pt(0.16) * max(t.count("\n") + 1 for k, t, _ in panels))
    foot = pt(0.78)
    fig = plt.figure(figsize=(width, head + foot + 2.4 * len(rows)),
                     facecolor=PALETTE["surface"])
    # hspace: the rotated x labels of a row are tall enough to reach the next row's titles
    # wspace is a fraction of a column, so scaling it by the column count keeps the gutter a
    # fixed share of the figure: a two-panel cut then stretches its panels, not its gutter.
    # Type set from the width takes a narrower gutter -- just the next panel's y label and ticks
    # -- to give its panels the room the larger text needs.
    grid = fig.add_gridspec(len(rows), ncol, hspace=0.52, wspace=gutter * ncol,
                            top=1 - head / fig.get_figheight(),
                            bottom=foot / fig.get_figheight(), left=0.05, right=0.99)
    spans = [grid[r, i * (ncol // len(row)):(i + 1) * (ncol // len(row))]
             for r, row in enumerate(rows) for i in range(len(row))]
    axes = np.array([fig.add_subplot(spans[i]) for i in range(len(panels))], dtype=object)
    rng = np.random.default_rng(0)
    for ax, (key, title, p_key) in zip(axes, panels):
        dagger_y = blended_transform_factory(ax.transData, ax.transAxes)
        # The stall counts pile up at zero, where a violin collapses onto the axis: so in that
        # panel every arm carries its mean over it, an arm whose seeds all stalled alike gets a
        # flat lens where its violin would be, and zero is lifted off the axis below.
        stalls = key == "stalled"
        span = max(float(np.max(data[m][key])) for m, *_ in shown) or 1.0
        for i, (m, label, color, ls) in enumerate(shown):
            drawn = vals = data[m][key]
            episodes = key in POOLED
            if np.ptp(drawn) > 0:
                body = ax.violinplot(drawn, [i], widths=0.8, showextrema=False,
                                     bw_method=0.12 if episodes else "scott")["bodies"][0]
                # violinplot leaves the body at alpha 0.3, which washes the outline and the
                # hatch out with the fill. The fill is tinted here instead, so drop it and let
                # the arm's colour draw the outline and the hatch at full strength.
                body.set_alpha(None)
                body.set_facecolor(_tint(color, 0.30))
                body.set_edgecolor(color)
                body.set_linestyle(ls)
                body.set_linewidth(1.0)
                if m in hatched:  # second backbone: a texture, not another hue
                    body.set_hatch(hatched[m] if isinstance(hatched, dict) else HATCH)
                    body.set_hatch_linewidth(HATCH_LW)
            elif stalls:  # no spread to draw a violin of: a flat lens, in the arm's own style
                lens = Ellipse((i, float(drawn[0])), 0.62, 0.045 * span, zorder=2,
                               facecolor=_tint(color, 0.30), edgecolor=color, linestyle=ls,
                               linewidth=1.0)
                if m in hatched:
                    lens.set_hatch(hatched[m] if isinstance(hatched, dict) else HATCH)
                    lens.set_hatch_linewidth(HATCH_LW)
                ax.add_patch(lens)
            if episodes:
                ax.scatter(i + rng.uniform(-0.28, 0.28, len(drawn)), drawn, s=5, color=color,
                           alpha=0.45, linewidths=0, zorder=3)
                q1, med, q3 = np.percentile(drawn, [25, 50, 75])
                for q, lw, qls in ((q1, 1.1, (0, (3, 2))), (med, 2.0, "-"), (q3, 1.1, (0, (3, 2)))):
                    ax.hlines(q, i - 0.34, i + 0.34, color=PALETTE["ink"], lw=lw, ls=qls, zorder=4)
                share = (vals >= POOLED[key][0]).mean()
                ax.text(i, 0.83, f"{np.median(vals):.2f}\n{100 * share:.0f} %",
                        transform=dagger_y,
                        ha="center", va="bottom", fontsize=fs["data"], color=PALETTE["ink"])
            else:
                ax.hlines(vals.mean(), i - 0.25, i + 0.25, color=PALETTE["ink"], lw=1.2, zorder=3)
                ax.scatter(i + np.linspace(-0.1, 0.1, len(vals)), vals, s=18, color=color,
                           edgecolors=PALETTE["surface"], linewidths=0.6, zorder=4)
                if stalls:  # the mean count, in a row of its own under the daggers
                    mean = float(vals.mean())
                    ax.text(i, 0.77, f"{mean:.0f}" if mean == 0 or mean >= 10 else f"{mean:.1f}",
                            transform=dagger_y, ha="center", va="center", fontsize=fs["data"],
                            color=PALETTE["ink"])
                if m != base and base in data:
                    pv = ttest_ind(data[m][p_key], data[base][p_key], equal_var=False).pvalue
                    if pv < 0.05:
                        ax.text(i, 0.98, "†", transform=dagger_y, ha="center", va="top",
                                fontsize=fs["dagger"], color=PALETTE["ink"])
        if key in POOLED:
            if reference:
                ax.axhline(POOLED[key][1], color=PALETTE["ink_muted"], lw=0.9, ls=":", zorder=1)
            ax.set_ylim(-0.04, 1.26)  # room above for the printed median and share
            ax.set_yticks(np.linspace(0, 1, 6))
        else:
            if reference and key in SEED_ERRORS:  # a never-moving pusher leaves the block
                ax.axhline(float(np.mean(data[shown[0][0]][SEED_ERRORS[key]])),  # where it was
                           color=PALETTE["ink_muted"], lw=0.9, ls=":", zorder=1)
            lo, hi = ax.get_ylim()
            if stalls:  # zero off the axis, the violins under the row of means, and ticks
                lo = -0.09 * span  # only as high as the data, clear of that row
                ax.set_ylim(lo, span + 0.55 * (span - lo))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= 1.05 * span])
                ax.set_ylim(lo, span + 0.55 * (span - lo))
            else:
                ax.set_ylim(lo, hi + 0.14 * (hi - lo))
        ax.set_title(title, fontsize=fs["title"], loc="left", color=PALETTE["ink"])
        ax.set_ylabel(ylabels[key], fontsize=fs["label"], color=PALETTE["ink"])
        ax.set_xticks(np.arange(len(shown)))
        # Every panel is two columns wide, too narrow for stacked labels: rotate them all.
        ax.set_xticklabels([e[1] for e in shown], fontsize=fs["arm"], rotation=rotation,
                           ha="right", rotation_mode="anchor")
        for tick, (m, *_) in zip(ax.get_xticklabels(), shown):
            tick.set_fontweight("bold" if m in phase_arms else "normal")
        ax.set_xlim(-0.6, len(shown) - 0.4)
        ax.tick_params(colors=PALETTE["ink_muted"], labelsize=fs["tick"])
        ax.yaxis.grid(True, color=PALETTE["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(PALETTE["grid"])
        ax.set_facecolor(PALETTE["surface"])
    if hatch_legend:  # a hatch per backbone: the legend names each one
        grey = _tint(PALETTE["ink_muted"], 0.30)
        handles = [Patch(facecolor=grey, edgecolor=PALETTE["ink_muted"], hatch=h, label=l)
                   for l, h in hatch_legend]
        for handle in handles:
            handle.set_hatch_linewidth(HATCH_LW)
    elif hatched:  # every violin is named on its x axis, so the legend need only decode the hatch
        grey = _tint(PALETTE["ink_muted"], 0.30)
        handles = [Patch(facecolor=grey, edgecolor=PALETTE["ink_muted"], label="MLP backbone"),
                   Patch(facecolor=grey, edgecolor=PALETTE["ink_muted"], hatch=HATCH,
                         label="transformer backbone")]
        handles[1].set_hatch_linewidth(HATCH_LW)
    else:
        handles = [Patch(facecolor=_tint(c, 0.30), edgecolor=c, linestyle=ls, label=l)
                   for _, l, c, ls in shown]
    if reference and ("episodes" in keys or SEED_ERRORS.keys() & set(keys)):  # if one is drawn
        # The key needs the swatch, so it stays; what the line means is caption prose.
        handles.append(Line2D([], [], color=PALETTE["ink_muted"], ls=":", lw=0.9,
                              label="never-moving pusher"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=fs["legend"],
               frameon=False)
    # By width, the figure is for a page short of space: cropped to the ink on every side.
    _save(fig, out, name, fixed_width=not by_width, caption=caption,
          pad=0.02 if by_width else 0.1)


def fig_violin_backbones(runs: Path, out: Path) -> None:
    """:func:`fig_violin` with both backbones: every MLP arm, then every transformer arm, then
    the chunk baselines. Three figures: all seven panels, and two narrower cuts --
    ``_score`` (mean score, stalls) and ``_goal`` (final position, final orientation, success) --
    both tested against the transformer's phase+score arm.

    Grouped by backbone (:func:`_backbone_groups`), so each backbone reads as a block. The twins
    still share the arm's colour and the transformer is still hatched, so a pair is found by hue
    across the two blocks -- eight policy violins plus the baselines would otherwise need more
    categorical colours than stay separable.
    """
    methods = _backbone_groups()
    # Held at rho = 1, the full figure also shows where the block ended up: the final coverage
    # beside the score, which is the best coverage reached on the way there.
    keys = (("episodes", "final_pos", "final_rot", "final_add",
             "score", "final_cov", "success", "stalled") if HOLD_AT_RHO1 else ())
    fig_violin(runs, out, methods=methods,
               hatched={m for m, *_ in methods if m.endswith(ARCH_SUFFIX)},
               name="fig_pusht_violin_backbones", keys=keys)
    # Two narrower cuts of the same twelve arms, tested against the transformer's full arm:
    # what the policy scored, and where it actually left the block.
    # One width for both, set by the wider cut, so they sit side by side in a document and the
    # two-panel one stretches rather than shrinking.
    common = dict(methods=methods, base=f"{SCORE_ARMS[SCORE_ARM][1]}{ARCH_SUFFIX}",
                  hatched={m for m, *_ in methods if m.endswith(ARCH_SUFFIX)},
                  width=_violin_width(len(methods), 3), by_width=True)
    fig_violin(runs, out, keys=("score", "stalled"),
               name="fig_pusht_violin_backbones_score", **common)
    # No never-moving-pusher line in this cut: it sits above every arm on both error panels,
    # so it buys nothing and costs the arms a third of the axis they are compared on.
    fig_violin(runs, out, keys=("final_pos", "final_rot", "success"), reference=False,
               name="fig_pusht_violin_backbones_goal", **common)


# An MLP and a transformer of the score-vs-latency Pareto figure
# (scripts/pusht/plot_pusht_pareto.py), each with the clock its phase arms run on:
# (run suffix, name, rollout tag, clock name, hatch).
# The transformer takes the backbone violins' hatch and the MLP none, as there, so the arm colours
# stay the paper's; the transformer's phase+score arm is what the daggers test against.
PARETO_MODELS = [
    ("_mlpmed", "MLP-med wide", "phase_mlp_startd3_fixedstep_e1o245_inverse_pd",
     "MLP clock med deep", None),
    ("_tfmed", "TF-med wide", "phase_mlp_startw724d3_fixedstep_e1o245_inverse_pd",
     "MLP clock large", HATCH),
]


def fig_violin_pareto(runs: Path, out: Path) -> None:
    """:func:`fig_violin_backbones`' two cuts for an MLP and a transformer of the Pareto figure
    (``PARETO_MODELS``): all four arms of each, then the chunk baselines. A third cut,
    ``fig_pusht_violin_pareto_score_goal``, puts the two in one row of four without the stalls:
    mean score, final position error, final orientation error, success.

    Each model's phase arms are its rollouts on its own clock (``PARETO_MODELS``); the phase-free
    arms ignore the clock, so they are the ones rolled out on the paper's. The arm colours are
    the paper's and the backbone is a hatch the legend names. † tests against the transformer's
    phase+score arm.
    """
    arms = [e for e in METHODS if e[0] not in BASELINE_TAGS]
    methods, hatched = [], {}
    for suffix, name, tag, _, hatch in PARETO_MODELS:
        for m, label, colour, ls in arms:
            run = f"{m}{suffix}"
            methods.append((run, label.replace("MLP", name), colour, ls))
            if run in _phase_arms():
                METHOD_TAGS[run] = tag
            if hatch:
                hatched[run] = hatch
    methods += [e for e in METHODS if e[0] in BASELINE_TAGS]
    legend = [(f"{name} ({clock})", hatch) for _, name, _, clock, hatch in PARETO_MODELS]
    notes = ("the phase arms run on each model's own clock, the phase-free arms on none: "
             + "; ".join(f"{name}, {clock}" for _, name, _, clock, _ in PARETO_MODELS),)
    tf_suffix = next(suffix for suffix, *_ in PARETO_MODELS if suffix.startswith("_tf"))
    common = dict(methods=methods, base=f"{SCORE_ARMS[SCORE_ARM][1]}{tf_suffix}", hatched=hatched,
                  hatch_legend=legend, notes=notes, width=_violin_width(len(methods), 3),
                  by_width=True)
    fig_violin(runs, out, keys=("score", "stalled"), name="fig_pusht_violin_pareto_score",
               **common)
    fig_violin(runs, out, keys=("final_pos", "final_rot", "success"), reference=False,
               name="fig_pusht_violin_pareto_goal", **common)
    # Four panels across leave each arm name less room, so this cut names the arms as the
    # backbone violins do (MLP, TF+phase, ...) and its legend says which MLP and TF they are.
    short = {suffix: "TF" if suffix.startswith("_tf") else "MLP" for suffix, *_ in PARETO_MODELS}
    methods_short = [(m, label.replace(name, short[suffix]), c, ls)
                     for (m, label, c, ls) in methods
                     for suffix, name, *_ in PARETO_MODELS
                     if m.endswith(suffix) and suffix and label.startswith(name)]
    methods_short += [e for e in methods if e[0] in BASELINE_TAGS]
    legend_short = [(f"{short[suffix]}: {name} ({clock})", hatch)
                    for suffix, name, _, clock, hatch in PARETO_MODELS]
    fig_violin(runs, out, keys=("score", "final_pos", "final_rot", "success"), reference=False,
               name="fig_pusht_violin_pareto_score_goal",
               **(common | {"methods": methods_short, "hatch_legend": legend_short,
                            "width": _violin_width(len(methods), 4)}))


def _pareto(points):
    """Indices of the non-dominated ``(cost, score)`` points, sorted by cost."""
    keep = [i for i, (x, y) in enumerate(points)
            if not any(xx <= x and yy >= y and (xx, yy) != (x, y) for xx, yy in points)]
    return sorted(keep, key=lambda i: points[i][0])


def _leader(ax, point, end, coords: str) -> None:
    """A hairline from ``point`` to ``end``, the label's left edge at mid height.

    Drawn as its own annotation with no text: an arrow attached to the label instead leaves the
    text box at whichever corner the target lies past, so a label above or below its point is
    joined at a corner and reads as pointing nowhere in particular.
    """
    ax.annotate("", point, xytext=end, textcoords=coords,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=PALETTE["ink_muted"],
                                shrinkA=0, shrinkB=2), zorder=1)


def _rate(ms: float) -> str:
    """The control rate a per-step latency allows (evaluate_policies._rate)."""
    hz = 1000.0 / ms
    if hz >= 1000:
        return f"{hz / 1000:.0f} kHz"
    return f"{hz:.0f} Hz" if hz >= 10 else f"{hz:.1f} Hz"


# The latency budgets the bar panels mark, as in the skate figure: a visual servo loop and a
# tactile one. Push-T itself runs at 10 Hz, which the third budget states.
CONTROL_BUDGETS = [(10.0, "10 Hz Push-T", "left"), (60.0, "60 Hz visual", "left"),
                   (1000.0, "1 kHz tactile", "right")]


def _style(ax, xlabel=None, ylabel=None, size=8) -> None:
    """evaluate_policies._style: surface, grid and spines, labels at ``size`` pt."""
    ax.set_facecolor(PALETTE["surface"])
    ax.grid(True, color=PALETTE["grid"], lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=PALETTE["ink_muted"], labelsize=size - 1)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=size, color=PALETTE["ink_muted"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=size, color=PALETTE["ink_muted"])


def fig_inference(runs: Path, out: Path, timings: Path, methods=None,
                  name: str = "fig_pusht_inference", merge=None) -> None:
    """What inference costs, per method, on both devices -- and what that cost buys.

    The skate inference figure (``evaluate_policies.fig_inference``) for Push-T. Bars: one
    decision, and one executed control step (with the control rate it allows), both at batch 1 so
    the devices stand on identical conditions (``methods`` overrides which arms are drawn); the
    log axis is unavoidable, the span being four
    orders of magnitude from an MLP forward pass to Diffusion Policy's 100-step DDPM chain on a
    CPU. A chunk policy's per-step cost is its decision spread over the steps it executes open
    loop before re-planning, which the bars say; closed loop it is a whole decision per step.

    ``merge`` maps an arm onto one whose network it shares (``_score_merge``): the pair takes one
    bar row, at their mean cost, sliced horizontally into their two colours and labelled with both
    names, since separate rows would draw nothing but the timer's rounding. The scatter still
    carries a point per arm -- the cost is shared, the score is not.

    The third panel puts the rollout score against the per-step cost, with the Pareto frontier of
    each device: a method's two points sit at the same height, since the device changes the cost
    and not the policy, so the gap between them *is* the hardware penalty. A phase arm's cost
    includes the phase estimator it runs every step. Colour is the method; the device is fill vs
    hatch (bars) and filled vs hollow (points), never hue alone, and closed loop is a square.
    """
    import csv  # noqa: PLC0415

    from matplotlib.lines import Line2D as _Line2D  # noqa: PLC0415
    from matplotlib.patches import Patch as _Patch  # noqa: PLC0415

    if not timings.exists():
        return
    rows = list(csv.DictReader(timings.open()))
    devices = sorted({r["device"] for r in rows}, reverse=True)  # cuda first
    if not rows or len(devices) < 2:
        return
    # inference.csv keys a chunk baseline by (method, loop); the figure shows one bar row per
    # method and reads the closed-loop cost off the same decision.
    # A CLOSED_LOOP_ONLY arm takes its closed row instead, so steps_per_decision is 1 and its
    # per-executed-step bar is a whole decision, with no "8 steps open loop" annotation.
    by = {(r["method"], r["device"]): r for r in rows
          if (r["loop"] == "closed") == (r["method"] in CLOSED_LOOP_ONLY)}
    # The two score schedules differ only in training, so an unmeasured one is drawn at the
    # measurement of the run whose network it is (annealed <-> const, identical architectures).
    twin = dict(zip(*SCORE_ARMS.values()))
    twin.update({v: k for k, v in twin.items()})

    def timing(method: str, device: str) -> dict | None:
        return by.get((method, device)) or by.get((twin.get(method, method), device))

    entries = list(methods or METHODS)
    merge = merge or {}
    drawn = [m for m in entries if not m[0].endswith("_te") and timing(m[0], devices[0])]
    # One row per bar, holding the arms it stands for: the arm itself and whatever `merge` folds
    # into it. The row's label names both, by the twin's suffix ("MLP+phase / +score"), and its
    # colours are theirs -- a single colour where the pair already shares one.
    methods = []
    for m, label, color, _ls in drawn:
        if m in merge:
            continue
        twins = [e for e in drawn if merge.get(e[0]) == m]
        names = [label] + [e[1][len(label):] if e[1].startswith(label) else e[1] for e in twins]
        colors = [color] + [e[2] for e in twins if e[2] != color]
        methods.append((m, " / ".join(names), colors, [m] + [e[0] for e in twins]))
    methods = methods[::-1]  # first method at the top
    open_loop = {m[0] for m in methods if int(timing(m[0], devices[0])["steps_per_decision"]) > 1}

    def score_of(method: str):
        R = _rollouts(runs, method)
        if not R:
            return None
        per_seed = [100 * float(r["score"].mean()) for r in R]
        return float(np.mean(per_seed)), min(per_seed), max(per_seed)

    scores = {m[0]: score_of(m[0]) for m in entries}
    scores = {k: v for k, v in scores.items() if v is not None}

    fig, axes = plt.subplots(1, 3, figsize=(1.45 * 7.16, 0.40 * len(methods) + 1.75),
                             facecolor=PALETTE["surface"])
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    height = 0.34
    panels = [("ms_decision_b1", "per decision", False),
              ("ms_step_b1", "per executed control step", True)]
    for ax, (field, title, with_rate) in zip(axes, panels):
        for i, (m, _label, colors, members) in enumerate(methods):
            for d, device in enumerate(devices):
                rs = [timing(k, device) for k in members]
                if any(r is None for r in rs):
                    continue
                r = rs[0]
                v = float(np.mean([float(x[field]) for x in rs]))
                y = i + (0.5 - d) * height
                gpu = device == "cuda"
                # One bar, sliced into equal horizontal bands: a band per colour the row carries.
                full = height * 0.92
                for j, c in enumerate(colors):
                    ax.barh(y + full * (0.5 - (j + 0.5) / len(colors)), v,
                            height=full / len(colors), color=c if gpu else "none",
                            edgecolor=c, linewidth=0 if gpu else 1.0,
                            hatch=None if gpu else "////")
                text = f"{v:.2f}" if v < 10 else f"{v:,.0f}"
                if with_rate:
                    text = f"{text} ({_rate(v)})"
                    if m in open_loop and d == 0:
                        text += f"  ← {int(r['steps_per_decision'])} steps open loop"
                elif m in open_loop:
                    text = f"{text} ({_rate(v)})" + ("  ← closed loop" if d == 0 else "")
                ax.text(v * 1.25, y, text, va="center", fontsize=6.5, color=PALETTE["ink_muted"],
                        bbox=dict(facecolor=PALETTE["surface"], edgecolor="none", pad=0.3),
                        zorder=3)
        ax.set_xscale("log")
        vals = [float(r[field]) for r in by.values()]
        ax.set_xlim(min(vals) / 2, max(vals) * (150 if with_rate else 40))
        for b, (hz, what, ha) in enumerate(CONTROL_BUDGETS):
            ax.axvline(1000.0 / hz, ls=(0, (4, 3)), lw=0.9, color=PALETTE["ink_muted"],
                       alpha=0.75, zorder=1)
            # 10 Hz and 60 Hz are close on a log axis: stagger their labels vertically.
            ax.annotate(what, (1000.0 / hz, 1.0), xycoords=ax.get_xaxis_transform(),
                        textcoords="offset points",
                        xytext=(3 if ha == "left" else -3, -1 - 9 * (b % 2)),
                        ha=ha, va="top", fontsize=6, color=PALETTE["ink_muted"],
                        bbox=dict(facecolor=PALETTE["surface"], edgecolor="none", pad=0.4),
                        zorder=3)
        _style(ax, "milliseconds (log)", size=8)
        ax.yaxis.grid(False)
        ax.set_title(title, fontsize=9, loc="left", color=PALETTE["ink"])
    axes[0].set_yticks(range(len(methods)))
    # The loop is stated by the bar annotations, so a baseline's row is just its name.
    bar_label = {"dp": "Diffusion Policy", "mip": "MIP", "act": "ACT"}
    axes[0].set_yticklabels([bar_label.get(m[0], m[1]) for m in methods], fontsize=7.5)
    # +0.55 at the top: the staggered budget label hangs ~10 pt below the axes top and would
    # otherwise sit on the first row's value label once the bars are long enough to reach it.
    axes[0].set_ylim(-0.6, len(methods) + 0.15)

    # -- what the cost buys ---------------------------------------------------------------
    ax = axes[2]
    label_style = dict(fontsize=6.5, color=PALETTE["ink"], ha="left", va="center")
    closed = [m for m in entries if m[0].endswith("_te") and m[0] in scores]

    def cost(method: str, device: str) -> float | None:
        r = timing(method.removesuffix("_te"), device)
        if r is None:
            return None
        return float(r["ms_decision_b1" if method.endswith("_te") else "ms_step_b1"])

    # CLOSED_LOOP_ONLY arms contribute their `_te` point and not their open-loop one; the bar row
    # stays (it is a cost measurement), so they are dropped here rather than from `drawn`.
    shown = [m for m in [*drawn, *closed] if m[0] in scores and m[0] not in CLOSED_LOOP_ONLY]
    fronts = {}
    for device in devices:
        gpu = device == "cuda"
        pts, keys = [], []
        for m, label, color, ls in shown:
            c = cost(m, device)
            if c is None:
                continue
            mean, lo, hi = scores[m]
            pts.append((c, mean))
            keys.append(m)
            ax.vlines(c, lo, hi, color=color, lw=0.9, alpha=0.5, zorder=2)
            ax.plot(c, mean, "s" if m.endswith("_te") else "o", ms=6,
                    mfc=color if gpu else "none", mec=color, mew=1.4, zorder=3)
        fronts[device] = [pts[i] for i in _pareto(pts)]
        ax.plot([p[0] for p in fronts[device]], [p[1] for p in fronts[device]],
                ls="--" if gpu else ":", lw=1.0, color=PALETTE["ink_muted"], zorder=1,
                label=f"Pareto frontier, {'GPU' if gpu else 'CPU'}")
    # One accuracy, two costs: join a method's devices with a faint line, and label at the slower.
    anchor = {m[0]: max(cost(m[0], d) for d in devices) for m in shown
              if all(cost(m[0], d) is not None for d in devices)}
    colors = {m[0]: m[2] for m in entries}
    labels = {m[0]: m[1] for m in entries}
    for m, x in anchor.items():
        ax.plot([cost(m, d) for d in devices], [scores[m][0]] * 2, lw=0.7, color=colors[m],
                alpha=0.35, zorder=2)
    far = sorted((m for m in anchor if anchor[m] > 50), key=lambda m: -scores[m][0])
    # The DDPM chain, far right. Two of its points can score within a label's height of each
    # other, so a pair that close is pushed apart and joined back to its point by a hairline.
    dy = [0.0] * len(far)
    for i in range(1, len(far)):
        if scores[far[i - 1]][0] - scores[far[i]][0] < 4:
            dy[i - 1], dy[i] = dy[i - 1] or 7.0, -7.0
    for m, off in zip(far, dy):
        if off:
            _leader(ax, (anchor[m], scores[m][0]), (6, off), "offset points")
        ax.annotate(labels[m], (anchor[m], scores[m][0]), textcoords="offset points",
                    xytext=(8, off), **label_style)
    # The cheap arms cluster too tightly to label in place: stack their labels in the empty band
    # right of the cluster, in score order, each joined to its slower point by a hairline.
    cluster = sorted((m for m in anchor if m not in far), key=lambda m: -scores[m][0])
    if cluster:
        x_col = 4 * max(anchor[m] for m in cluster)
        # Evenly, from the best-scoring arm down: the spacing closes up so that however many arms
        # the column holds, the last label still clears the axis floor. Following each label to
        # its own score instead would let a tight cluster walk the stack off the bottom.
        top = max(scores[m][0] for m in cluster)
        step = min(5.5, (top - 33.0) / max(len(cluster) - 1, 1))
        for i, m in enumerate(cluster):
            y = top - i * step
            _leader(ax, (anchor[m], scores[m][0]), (x_col / 1.09, y), "data")
            ax.annotate(labels[m], (x_col, y), **label_style)
    ax.set_xscale("log")
    costs = [c for m in anchor for c in (cost(m, d) for d in devices)]
    ax.set_xlim(min(costs) / 2.5, max(costs) * 15)
    ax.set_ylim(30, 105)
    _style(ax, "ms per control step (log)", "rollout score [%]", size=8)
    ax.set_title("score vs cost (50 envs, DP protocol)", fontsize=9, loc="left",
                 color=PALETTE["ink"])
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")

    hardware = {r["device"]: r["hardware"] for r in rows}
    # Name the chunk baselines actually drawn rather than a hard-coded pair, and drop "ACT" from
    # the temporal-ensembling label: it named the technique by its paper, which now reads as the
    # ACT *arm* with one in the figure. The attribution stays in METHODS' comment.
    # Reversed: `methods` runs bottom-to-top (first row at the top), and the legend should name
    # them in the order the reader meets the bars.
    open_names = ", ".join(bar_label.get(m[0], m[1]).split(" (")[0]
                           for m in reversed(methods)
                           if m[0] in BASELINE_TAGS and not m[0].endswith("_te")
                           and m[0] not in CLOSED_LOOP_ONLY)
    shapes = [_Line2D([], [], ls="", marker=mk, ms=5, mfc=PALETTE["ink_muted"],
                      mec=PALETTE["ink_muted"], label=text)
              for mk, text in [("o", f"as trained ({open_names}: open-loop chunks)"),
                               ("s", "closed loop (temporal ensembling)")]]
    fig.legend(handles=[
        _Patch(facecolor=PALETTE["ink_muted"] if d == "cuda" else "none",
               edgecolor=PALETTE["ink_muted"], hatch=None if d == "cuda" else "////",
               label=hardware[d]) for d in devices
    ] + (shapes if closed else []), loc="upper center", ncol=len(devices) + 2, fontsize=7.5,
        frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.34 / fig.get_figheight()))
    _save(fig, out, name)


def fig_inference_backbones(runs: Path, out: Path, timings: Path) -> None:
    """:func:`fig_inference` with both backbones: each arm's MLP and transformer twin adjacent.

    Fill vs hatch is already spent on the device here, so the backbone is carried by the row
    labels and by the pair sharing a colour -- the same pairing as
    :func:`fig_violin_backbones`, read down the bars instead of across the violins.
    """
    fig_inference(runs, out, timings, methods=_backbone_pairs(), merge=_score_merge(),
                  name="fig_pusht_inference_backbones")


def _render_env():
    import warnings  # noqa: PLC0415

    import gymnasium as gym  # noqa: PLC0415
    import gym_pusht  # noqa: F401, PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array",
                        disable_env_checker=True).unwrapped


def _film_frame(env, ax, S: np.ndarray, T: int, t: int, stop: str = "done") -> None:
    """Step ``t`` of the state sequence ``S`` (``T`` steps taken) in ``ax``: the rendered frame
    under its step and coverage. A second call redraws the same image, so an animation can
    step one set of axes through an episode. Past ``T`` the title says what stopped it
    (:func:`_stop`)."""
    import warnings  # noqa: PLC0415

    from fm_bc.pusht.sim import place_state  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        place_state(env, S[min(t, T)])
        img = env.render()
    if ax.images:
        ax.images[0].set_data(img)
    else:
        ax.imshow(img)
    cov = coverage(S[min(t, T), 2:5])
    ax.set_title(f"t = {t} · cov {cov:.2f}" + (f" ({stop} at {T})" if t > T else ""),
                 fontsize=7.5, color=PALETTE["ink_muted"])


def _film_panel(ax, ylabel: str | None = None) -> None:
    """A filmstrip panel's frame: no ticks, no spines, the row's label down its left side."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=8.5, color=PALETTE["ink"])


def _film_row(env, axes, S: np.ndarray, T: int, steps, ylabel: str, stop: str = "done") -> None:
    """One filmstrip row: rendered frames of the state sequence ``S`` (``T`` steps taken)."""
    for c, t in enumerate(steps):
        _film_frame(env, axes[c], S, T, t, stop)
        _film_panel(axes[c], ylabel if c == 0 else None)


def _held_label() -> str:
    """The filmstrip header's note that the phase arms stop at rho = 1 (``HOLD_AT_RHO1``)."""
    return " · phase arms held from ρ = 1" if HOLD_AT_RHO1 else ""


def fig_filmstrip(runs: Path, out: Path, env_index: int = 0, steps=None) -> None:
    env = _render_env()
    methods = [m for m in METHODS if m[0] in GRID_METHODS and _rollouts(runs, m[0], 0)]
    steps = steps or _film_steps(_rollouts(runs, methods[0][0], 0)[0]["meta"])
    fig, axes = plt.subplots(len(methods), len(steps), figsize=(1.9 * len(steps), 2.05 * len(methods)),
                             facecolor=PALETTE["surface"], squeeze=False)
    for r_i, (m, label, _, _) in enumerate(methods):
        r = _rollouts(runs, m, 0)[0]
        _film_row(env, axes[r_i], r["states"][env_index], int(r["steps"][env_index]), steps, label,
                  _stop(r, env_index))
    fig.suptitle(f"env {int(_rollouts(runs, methods[0][0], 0)[0]['seeds'][env_index])} · seed-0 "
                 f"policies · green T: goal · {_clock_label(_rollouts(runs, methods[0][0], 0)[0]['meta'])}"
                 f"{_held_label()}",
                 fontsize=9.5, color=PALETTE["ink"], y=0.995)
    fig.tight_layout()
    _save(fig, out, "fig_pusht_filmstrip")


def _film_steps(meta: dict, max_steps: int = 300) -> tuple[int, ...]:
    """Six frames: the start, three through the episode, the clock's saturation and the last step."""
    sat = min(_saturation(meta) or max_steps - 1, max_steps - 1)
    return tuple(sorted({0, round(sat / 4), round(sat / 2), round(3 * sat / 4), sat, max_steps - 1}))


def _ranked(R: list[dict], ends: str = "score") -> list[tuple[str, tuple[float, float, int, int]]]:
    """The best, median and worst of all the episodes in ``R`` (every seed × every env).

    The median is the lower-middle episode by score (max coverage over the episode), ties broken
    by final coverage (coverage at its last step). The best and worst are the ends of that same
    order, or with ``ends="final"`` of the order by final coverage, ties broken by score. Each
    pick is ``(name, (score, final coverage, seed index, env index))``.
    """
    eps = []
    for s_i, r in enumerate(R):
        for e in range(len(r["seeds"])):
            T = int(r["steps"][e])
            eps.append((float(r["score"][e]), float(coverage(r["states"][e, T, 2:5])), s_i, e))
    by_score = sorted(eps)
    ranked = by_score if ends == "score" else sorted(eps, key=lambda p: (p[1], p[0], *p[2:]))
    return [("best", ranked[-1]), ("median", by_score[(len(eps) - 1) // 2]), ("worst", ranked[0])]


def _policy_seed(r: dict) -> int:
    return int(re.search(r"seed(\d+)", r["meta"]["_path"]).group(1))


def fig_filmstrip_ranked(runs: Path, out: Path, steps=None) -> None:
    """Per method, the best, median and worst of all its episodes (every seed × every env).

    Ranked by score (max coverage over the episode), ties broken by final coverage; the median is
    the lower-middle episode of that order.
    """
    env = _render_env()
    for m, label, _, _ in METHODS:
        if m not in GRID_METHODS:
            continue
        if ARCH_TAG and m in BASELINE_TAGS:  # --arch leaves baselines alone; already drawn
            continue
        if HOLD_AT_RHO1 and m not in _phase_arms():  # the hold leaves them as they were drawn
            continue
        R = _rollouts(runs, m)
        if not R:
            continue
        film = steps or _film_steps(R[0]["meta"])
        fig, axes = plt.subplots(3, len(film), figsize=(1.9 * len(film), 2.15 * 3),
                                 facecolor=PALETTE["surface"], squeeze=False)
        for row, (name, (score, _, s_i, e)) in enumerate(_ranked(R)):
            r = R[s_i]
            _film_row(env, axes[row], r["states"][e], int(r["steps"][e]), film,
                      f"{name}\nenv {int(r['seeds'][e])} · policy seed {_policy_seed(r)}\n"
                      f"score {score:.2f}", _stop(r, e))
        fig.suptitle(f"{label} · best / median / worst of {sum(len(r['seeds']) for r in R)} "
                     f"episodes ({len(R)} seeds × {len(R[0]['seeds'])} envs) · green T: goal · "
                     # The header already spans the panels: the hold's note takes a line of its own.
                     f"{_clock_label(R[0]['meta'])}"
                     + ("\nepisodes held from where the clock first reads ρ = 1"
                        if HOLD_AT_RHO1 else ""),
                     fontsize=9.5, color=PALETTE["ink"], y=0.995)
        fig.tight_layout()
        _save(fig, out, f"fig_pusht_filmstrip_ranked_{m}")


def gif_rollout_ranked(runs: Path, out: Path, speed: float = 3.0, dpi: int = 120) -> None:
    """Per method, a GIF of every policy seed's best, median and worst episode: a column per
    seed, all stepping through the rollouts together.

    Each seed is ranked on its own test envs, so the method is shown seed by seed rather than by
    its luckiest and unluckiest seeds' episodes: the best and worst by final coverage and the
    median by score (see :func:`_ranked`), each panel labelled with the number it was ranked by.
    Every control step is a frame, played at ``speed`` × real time with a hold on the first and
    the last; an episode that finishes early stays on its last frame. The panels keep the
    filmstrip's 1.9 in and its type, so ``dpi`` scales the whole GIF. The filmstrip's methods,
    and MIP and ACT (``GIF_BASELINES``).
    """
    env = _render_env()
    for m, label, _, _ in METHODS:
        if m not in GRID_METHODS and m not in GIF_BASELINES:
            continue
        if ARCH_TAG and m in BASELINE_TAGS:  # --arch leaves baselines alone; already drawn
            continue
        if HOLD_AT_RHO1 and m not in _phase_arms():  # the hold leaves them as they were drawn
            continue
        _gif_ranked(env, runs, out, m, label, speed, dpi)


def gif_rollout_ranked_pareto(runs: Path, out: Path, speed: float = 3.0,
                              dpi: int = 120) -> None:
    """:func:`gif_rollout_ranked` for the violins' models (``PARETO_MODELS``): all four arms of
    each, named as :func:`fig_violin_pareto` names them, the phase arms on the model's own clock
    -- which the header names -- and under ``HOLD_AT_RHO1`` only those. The chunk baselines are
    the published GIFs'."""
    env = _render_env()
    arms = [e for e in METHODS if e[0] not in BASELINE_TAGS]
    for suffix, name, tag, clock, _ in PARETO_MODELS:
        for m, label, _, _ in arms:
            run = f"{m}{suffix}"
            phase = run in _phase_arms()
            if HOLD_AT_RHO1 and not phase:  # the hold leaves them as they were drawn
                continue
            if phase:
                METHOD_TAGS[run] = tag
            _gif_ranked(env, runs, out, run, label.replace("MLP", name), speed, dpi,
                        clock if phase else None)


def _gif_ranked(env, runs: Path, out: Path, m: str, label: str, speed: float, dpi: int,
                clock: str | None = None) -> None:
    """:func:`gif_rollout_ranked`'s GIF of one method, as ``fig_pusht_rollout_ranked_{m}``.
    ``clock`` names the phase-MLP clock in the header when it is not the paper's."""
    R = _rollouts(runs, m)
    if not R:
        return
    # Inches, around the filmstrip's 1.9 in panel: a column's seed over its first title, each
    # panel's title above it and its ranking number below.
    panel, label_w, seed_h, title_h, rank_h, gap, pad = 1.9, 0.45, 0.2, 0.24, 0.2, 0.08, 0.05
    head = 1.3 * (9.5 + 7) / 72
    span = len(R) * panel + (len(R) - 1) * gap
    W = pad + label_w + span + pad
    H = pad + head + seed_h + 3 * (title_h + panel + rank_h) + 2 * gap + pad
    fig = plt.figure(figsize=(W, H), dpi=dpi, facecolor=PALETTE["surface"])
    cx = (pad + label_w + span / 2) / W  # the header centres on the panels, not the labels
    fig.text(cx, 1 - pad / H, f"{label} · best / median / worst of each policy seed's "
             f"{len(R[0]['seeds'])} test envs", ha="center", va="top", fontsize=9.5,
             color=PALETTE["ink"])
    clock_label = _clock_label(R[0]["meta"])
    if clock:
        clock_label = clock_label.replace("phase-MLP clock", clock)
    fig.text(cx, 1 - (pad + 1.3 * 9.5 / 72) / H, f"green T: goal · played at {speed:g}× "
             f"real time · {clock_label}{_held_label()}", ha="center", va="top", fontsize=7,
             color=PALETTE["ink_muted"])
    panels = []  # (axes, states, steps taken, what stopped it)
    for c, r in enumerate(R):
        x = pad + label_w + c * (panel + gap)
        for row, (name, (score, final, _, e)) in enumerate(_ranked([r], ends="final")):
            y = pad + rank_h + (2 - row) * (rank_h + panel + title_h + gap)
            ax = fig.add_axes((x / W, y / H, panel / W, panel / H))
            key, value = ("score", score) if name == "median" else ("final cov", final)
            _film_panel(ax, f"{name}\nby {key}" if c == 0 else None)
            ax.set_xlabel(f"env {int(r['seeds'][e])} · {key} {value:.2f}", fontsize=7,
                          color=PALETTE["ink_muted"])
            panels.append((ax, r["states"][e], int(r["steps"][e]), _stop(r, e)))
        fig.text((x + panel / 2) / W, (H - pad - head - seed_h) / H,
                 f"policy seed {_policy_seed(r)}", ha="center", va="bottom", fontsize=8.5,
                 color=PALETTE["ink"])

    def draw(t: int) -> np.ndarray:
        for ax, S, T, stop in panels:
            _film_frame(env, ax, S, T, t, stop)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3]

    # Crop to the ink once, so every frame is the same size: only the panel titles change
    # after the first, and they stay centred and narrower than their panel.
    draw(0)
    box = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.03)
    crop = (slice(max(0, math.floor((H - box.y1) * dpi)), math.ceil((H - box.y0) * dpi)),
            slice(max(0, math.floor(box.x0 * dpi)), math.ceil(box.x1 * dpi)))
    n = max(T for _, _, T, _ in panels) + 1
    # Frame times on the GIF's 10 ms clock, rounded cumulatively so the rate stays exact.
    ticks = np.round(np.arange(n + 1) * 100 / (env.control_hz * speed))
    durations = 10 * np.diff(ticks).astype(int)
    durations[0] += 800
    durations[-1] += 2000
    sample = [draw(t)[crop].copy() for t in range(0, n, max(1, n // 8))]
    _save_gif((draw(t)[crop].copy() for t in range(n)), durations.tolist(), sample, out,
              f"fig_pusht_rollout_ranked_{m}")
    plt.close(fig)


def _save_gif(frames, durations: list[int], sample: list[np.ndarray], out: Path,
              name: str) -> None:
    """RGB ``frames`` as ``<name>.gif``, looping, each shown for its ``durations`` (ms).

    One palette for the whole animation, drawn from the ``sample`` frames across it: per-frame
    palettes can shift a colour between frames, and a shared one leaves Pillow's optimizer only
    the pixels that changed to store -- it writes the rest as transparent, which keeps a
    300-frame GIF small. ``frames`` can be a generator: each is quantized as it arrives, so the
    animation is never held in RGB.
    """
    from PIL import Image  # noqa: PLC0415

    name = _tagged(name)
    palette = Image.fromarray(np.concatenate(sample)).quantize(
        colors=128, method=Image.Quantize.MEDIANCUT)  # spare indices
    ims = (Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames)
    next(ims).save(out / f"{name}.gif", save_all=True, append_images=ims, duration=durations,
                   loop=0, optimize=True)
    print(f"  wrote {name}.gif ({(out / f'{name}.gif').stat().st_size / 1e6:.1f} MB)")



def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episodes", type=Path, default=Path("data/pusht/pusht_episodes.npz"))
    p.add_argument("--prepared", type=Path, default=Path("data/pusht/pusht_prepared_9010.npz"))
    p.add_argument("--runs", type=Path, default=Path("runs/pusht/policy"))
    p.add_argument("--out", type=Path, default=Path("output/pusht_9010/figures"))
    p.add_argument("--only", choices=["data", "rollouts", "gifs", "pareto", "pareto_gifs"],
                   default=None,
                   help="pareto: just the violins and final poses of the Pareto figure's "
                        "models (fig_pusht_{violin,final_poses}_pareto*), with --hold-at-rho1 "
                        "their rho = 1 cut; pareto_gifs: those models' ranked-rollout GIFs")
    p.add_argument("--timings", type=Path, default=None,
                   help="inference.csv from scripts/pusht/benchmark_pusht_inference.py "
                        "(default: <out>/../inference.csv)")
    p.add_argument("--rollout-tag", default=ROLLOUT_TAG,
                   help="which rollout files the figures read (default: the evaluation clock); a "
                        "tag ending in _last (the final, overfit checkpoints) also switches the "
                        "chunk baselines to their _last rollouts")
    p.add_argument("--phase-runs", type=Path, default=PHASE_RUNS,
                   help="the per-seed phase estimators the rollouts' clock ran (the rollout "
                        "script's --phase-runs), replayed for --hold-at-rho1")
    p.add_argument("--hold-at-rho1", action="store_true",
                   help="stop each phase arm's episodes where their clock first reads rho = 1 "
                        "and hold the state from there; draws the final-poses, violin and "
                        "filmstrip figures that change, as fig_pusht_*_rho1")
    p.add_argument("--score-arms", default=SCORE_ARM, choices=sorted(SCORE_ARMS),
                   help="which schedule of the score regularizer the figures show: held constant "
                        "(the default, `_const` runs) or annealed to 0 over the first half of "
                        "training")
    p.add_argument("--arch", default="mlp", choices=["mlp", *ARCH],
                   help="backbone the policy arms are drawn from; 'transformer' reads the _tf "
                        "runs and writes every figure as fig_pusht_*_tf, 'transformer_big' the "
                        "_tfbig runs of the 67M backbone as fig_pusht_*_tfbig")
    p.add_argument("--filmstrip-env", type=int, default=4,
                   help="index into the 50 test envs (4 = seed 100004)")
    p.add_argument("--gif-speed", type=float, default=3.0,
                   help="playback speed of the ranked-rollout GIFs, as a multiple of the env's "
                        "10 Hz control rate")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    globals()["ROLLOUT_TAG"] = args.rollout_tag
    globals()["PHASE_RUNS"] = args.phase_runs
    if args.hold_at_rho1:
        globals()["HOLD_AT_RHO1"], globals()["RHO_TAG"] = True, "_rho1"
    if args.score_arms != SCORE_ARM:  # swap both score arms for the chosen schedule's runs
        swap = dict(zip(SCORE_ARMS[SCORE_ARM], SCORE_ARMS[args.score_arms]))
        globals()["METHODS"] = [(swap.get(m, m), *rest) for m, *rest in METHODS]
        globals()["GRID_METHODS"] = tuple(swap.get(m, m) for m in GRID_METHODS)
    globals()["SCORE_ARM"] = args.score_arms
    if args.arch != "mlp":  # after the score-arms swap: the twins of those runs
        # Before _tf_arms and the GRID_METHODS swap below, both of which read these.
        globals()["ARCH_SUFFIX"], globals()["ARCH_LABEL"] = ARCH[args.arch]
        globals()["METHODS"] = _tf_arms(METHODS)
        globals()["GRID_METHODS"] = tuple(m if m in BASELINE_TAGS else f"{m}{ARCH_SUFFIX}"
                                          for m in GRID_METHODS)
        globals()["ARCH_TAG"] = ARCH_SUFFIX
    if args.rollout_tag.endswith("_last"):  # the final checkpoints: so are the baselines'
        globals()["BASELINE_TAGS"] = {k: f"{v}_last" for k, v in BASELINE_TAGS.items()}
        globals()["CKPT_TAG"] = "_last"
    args.timings = args.timings or args.out.parent / "inference.csv"

    if args.only == "gifs":  # they read nothing but the rollouts
        gif_rollout_ranked(args.runs, args.out, args.gif_speed)
        return
    if args.only == "pareto_gifs":
        gif_rollout_ranked_pareto(args.runs, args.out, args.gif_speed)
        return
    ep = PushTEpisodes.load(args.episodes)
    prep = PushTPrepared.load(args.prepared)
    if args.only == "pareto":
        fig_violin_pareto(args.runs, args.out)
        fig_final_poses_pareto(ep, prep, args.runs, args.out)
        fig_final_poses_pair_pareto(args.runs, args.out)
        return
    if HOLD_AT_RHO1:  # where the phase arms had got to by rho = 1: the figures that ask that
        if args.arch == "mlp":  # both backbones in one figure, so once and not per --arch
            fig_final_poses_backbones(ep, prep, args.runs, args.out)
            fig_violin_backbones(args.runs, args.out)
        fig_filmstrip(args.runs, args.out, args.filmstrip_env)
        fig_filmstrip_ranked(args.runs, args.out)
        return
    if args.only in (None, "data"):
        rng = np.random.default_rng(0)
        fig_demos(ep, sorted(rng.choice(prep.train_ids.numpy(), 8, replace=False)), args.out)
        fig_distribution(ep, prep, args.out)
        fig_phase_evolution(prep, args.out)
        fig_velocity_score(ep, prep, args.out)
    if args.only in (None, "rollouts"):
        fig_rollout_grid(args.runs, args.out)
        fig_final_poses(ep, prep, args.runs, args.out)
        fig_final_poses_pair(args.runs, args.out)
        if args.arch == "mlp":  # draws both backbones itself, so once and not per --arch
            fig_final_poses_backbones(ep, prep, args.runs, args.out)
        fig_coverage(args.runs, args.out)
        fig_violin(args.runs, args.out)
        if args.arch == "mlp":  # draws both backbones itself, so once and not per --arch
            fig_violin_backbones(args.runs, args.out)
        fig_filmstrip(args.runs, args.out, args.filmstrip_env)
        fig_filmstrip_ranked(args.runs, args.out)
        fig_inference(args.runs, args.out, args.timings)
        if args.arch == "mlp":  # draws both backbones itself, so once and not per --arch
            fig_inference_backbones(args.runs, args.out, args.timings)
    if args.only is None:
        gif_rollout_ranked(args.runs, args.out, args.gif_speed)


if __name__ == "__main__":
    main()
