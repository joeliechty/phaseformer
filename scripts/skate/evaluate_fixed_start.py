"""The MLP arms on the point-to-point dual-path datasets against the same arms on the paper's
scenario C.

    python scripts/skate/evaluate_fixed_start.py

The point-to-point sets (``fm_bc.skate.scenarios``, ``EXTRA_SCENARIOS``) are C's world and block
driven to one goal *pose* (5.0, 7.0, heading up), and differ from each other in one thing each:

* ``dual_path_fixed``        one start pose, no via jitter: the demos are exactly two
  trajectories, a left and a right one, each repeated 512 times;
* ``dual_path_fixed_via``    the same one start pose, with C's via jitter (0.3 m): the demos
  spread in the middle of the path and share both ends;
* ``dual_path_noisy_start``  C's start noise (8 cm, 0.05 rad), no via jitter: each demo is the
  planner's one optimum from its own start.

With one start pose every held-out start is the same, so a deterministic policy drives the *same*
trajectory 1024 times -- each seed is one path. With a noisy start the policy has to split the
start distribution between the modes.

Reads ``runs/skate_deterministic/{scenario}/n1024/{method}/seed*/`` (made by
``scripts/skate/run_deterministic_grid.sh``) and, for C, ``runs/dual_path/n1024/{method}/``, and
writes to ``output/skate_deterministic/``:

* ``summary.csv``, ``significance.csv`` and ``SUMMARY.md`` (figures inline, ``FINDINGS.md`` on top);
* ``fig_p2p_dataset``    the demos of every set;
* ``fig_p2p_rollouts``   a row per set, a column per arm: every seed's path where the start is
  fixed, seed 0's rollouts where it is not;
* ``fig_p2p_xphase``     x over phase for the one-start sets, one line per seed;
* ``fig_p2p_outcomes``   per arm and set, the share of rollouts that reach the goal left / right,
  drive into the block, or go round it and still collide;
* ``fig_p2p_mmd_phase``  per-phase MMD, mean +/- std over seeds, a panel per set;
* ``fig_p2p_violin``     MMD and success per seed, in the Push-T violins' style, with a LaTeX
  caption beside it;
* ``fig_p2p_training``   MMD and velocity loss against training step, a row per point-to-point set;
* ``fig_p2p_targets``    the demos' spread, SE(2) kNN divergence and score magnitude over phase;
* ``fig_p2p_restart``    each policy put back onto held-out demos at phase 0.2.
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
import torch  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_policies import (_save, _tint, _violin, figure_label, method_style,  # noqa: E402
                               type_pt)
from fm_bc.se2 import se2_exp_retract  # noqa: E402
from fm_bc.skate.baselines import load_policy  # noqa: E402
from fm_bc.skate.dataset import TrajectoryDataset  # noqa: E402
from fm_bc.skate.evaluation import collision_breakdown  # noqa: E402
from fm_bc.skate.plots import COLLISION_COLOR, MODE_COLORS, _world_axes, plot_rollouts  # noqa: E402
from fm_bc.skate.scenarios import get_scenario  # noqa: E402
from skate_planner.skate_model import body_twist  # noqa: E402
from skate_planner.viz import PALETTE  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})

METHODS = ["mlp", "mlp_score_w0.5", "fm", "score_w0.5"]
ORIG = "dual_path"
# Every set, in the order it is shown: the one-knob-at-a-time point-to-point sets, then C.
SETS = {"dual_path_fixed": "fixed start + goal pose",
        "dual_path_fixed_via": "fixed start/goal + via jitter",
        "dual_path_noisy_start": "noisy start, fixed goal pose",
        ORIG: "C (paper)"}
SET_SHORT = {"dual_path_fixed": "fixed", "dual_path_fixed_via": "+via",
             "dual_path_noisy_start": "+start", ORIG: "C"}
SET_COLORS = {"dual_path_fixed": "#444444", "dual_path_fixed_via": "#1baf7a",
              "dual_path_noisy_start": "#8e5ad6", ORIG: "#eda100"}
SET_HATCH = {"dual_path_fixed": None, "dual_path_fixed_via": "....",
             "dual_path_noisy_start": "\\\\\\\\", ORIG: "////"}
P2P = [s for s in SETS if s != ORIG]
N = 1024
BOX = get_scenario(ORIG).split_box  # the same block in every set
SCALARS = ["mmd_mean", "mmd_final", "success_rate", "reach_rate", "obstacle_collision_rate",
           "boundary_collision_rate", "collision_rate", "undecided_rate", "mode_tv",
           "saturation_rate", "into_block_rate", "heading_err_deg"]
TESTED = ["mmd_mean", "success_rate", "collision_rate", "heading_err_deg"]
# Short arm names for crowded tick labels.
SHORT = {"mlp": "MLP", "mlp_score_w0.5": "MLP+score", "fm": "MLP+phase",
         "score_w0.5": "MLP+ph.+score"}
# The restart diagnostic: phases at which each policy is put back onto a held-out demo, and how
# many demos per mode it is put back onto (a set with only two distinct demos uses those two).
RESTART_PHASES = (0.1, 0.2, 0.3, 0.5)
RESTART_DEMOS = 10
TRACK_TOL = 0.3  # m: a restarted rollout "tracks" its demo if it never strays further
OUTCOMES = ["left, reached", "right, reached", "undecided", "into block",
            "round, then collided"]


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def into_block(poses: np.ndarray, box) -> np.ndarray:
    """Per rollout: its median x while level with the block lies within the block's own x-span
    -- it drove at the block (the two modes' average) rather than round it. ``label_mode`` would
    still call such a path left or right, by which side of centre it is on."""
    level = np.abs(poses[..., 1] - box.cy) <= box.hy
    out = np.zeros(len(poses), dtype=bool)
    for i in range(len(poses)):
        if level[i].any():
            out[i] = abs(np.median(poses[i, level[i], 0]) - box.cx) <= box.hx
    return out


def load(root: Path, scenario: str) -> dict[str, list[dict]]:
    """``{method: [run, ...]}``; each run holds its metrics, log, rollouts, seed and directory."""
    runs = {}
    for method in METHODS:
        group = []
        for seed_dir in sorted((root / scenario / f"n{N}" / method).glob("seed*")):
            if not (seed_dir / "rollouts.npz").exists():
                continue
            m = json.loads((seed_dir / "metrics.json").read_text())
            with np.load(seed_dir / "rollouts.npz") as f:
                roll = {k: f[k] for k in ("poses", "labels", "collided")}
            m["into_block_rate"] = float(into_block(roll["poses"], BOX).mean())
            # The point-to-point goals are heading up; C's goal heading is free, so there it is
            # only reported.
            m["heading_err_deg"] = float(np.degrees(np.abs(
                _wrap(roll["poses"][:, -1, 2] - np.pi / 2))).mean())
            group.append({"metrics": m, "roll": roll, "seed": int(seed_dir.name[4:]),
                          "dir": seed_dir,
                          "log": np.genfromtxt(seed_dir / "log.csv", delimiter=",", names=True)})
        runs[method] = sorted(group, key=lambda r: r["seed"])
    return runs


def one_start(ds: TrajectoryDataset) -> bool:
    return bool(np.all(ds.poses[:, 0] == ds.poses[0, 0]))


def vals(group: list[dict], key: str) -> np.ndarray:
    return np.array([r["metrics"].get(key, np.nan) for r in group], dtype=float)


def outcome_fractions(group: list[dict]) -> np.ndarray:
    """``(seeds, 5)``: share of each seed's rollouts that are left, right, undecided (all
    collision-free), drove into the block, or went round it and still collided."""
    out = []
    for r in group:
        lab, col = r["roll"]["labels"], r["roll"]["collided"].astype(bool)
        blk = into_block(r["roll"]["poses"], BOX) & col
        out.append([np.mean((lab == 0) & ~col), np.mean((lab == 1) & ~col),
                    np.mean((lab < 0) & ~col), np.mean(blk), np.mean(col & ~blk)])
    return np.array(out)


def summarize(all_runs: dict) -> list[dict]:
    rows = []
    for scen, runs in all_runs.items():
        for method in METHODS:
            g = runs[method]
            if not g:
                continue
            row = {"scenario": scen, "method": method, "n_seeds": len(g)}
            for key in SCALARS:
                v = vals(g, key)
                row[f"{key}_mean"] = float(np.nanmean(v))
                row[f"{key}_std"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
            fr = outcome_fractions(g).mean(axis=0)
            for name, f in zip(["left", "right", "undecided", "block_collided",
                                "round_collided"], fr):
                row[f"{name}_frac"] = float(f)
            # The side each rollout passes the block on, whether or not it then collides.
            labs = np.concatenate([r["roll"]["labels"] for r in g])
            row["side_left_frac"] = float(np.mean(labs == 0))
            row["side_right_frac"] = float(np.mean(labs == 1))
            row["side_undecided_frac"] = float(np.mean(labs < 0))
            rows.append(row)
    return rows


def _welch(x: np.ndarray, y: np.ndarray) -> float:
    if min(len(x), len(y)) < 2 or (np.ptp(x) == 0 and np.ptp(y) == 0):
        return np.nan
    return float(stats.ttest_ind(x, y, equal_var=False).pvalue)


def significance(all_runs: dict) -> list[dict]:
    """Welch t-tests: every arm against MLP+phase within each point-to-point set, and each
    point-to-point set against C per arm."""
    out = []

    def test(label, a, b, scen, method):
        row = {"comparison": label, "scenario": scen, "method": method}
        for key in TESTED:
            x, y = vals(a, key), vals(b, key)
            row[f"{key}_diff"], row[f"{key}_p"] = float(np.mean(x) - np.mean(y)), _welch(x, y)
        out.append(row)

    for scen in P2P:
        runs = all_runs[scen]
        for method in METHODS:
            if method != "fm" and runs[method] and runs["fm"]:
                test(f"{SET_SHORT[scen]}: vs MLP+phase", runs[method], runs["fm"], scen, method)
    for scen in P2P:
        for method in METHODS:
            if all_runs[scen][method] and all_runs[ORIG][method]:
                test(f"{SET_SHORT[scen]} vs C", all_runs[scen][method], all_runs[ORIG][method],
                     scen, method)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------------- figures


def _demo_panel(ax, ds: TrajectoryDataset, title: str, n_paths: int = 200) -> None:
    _world_axes(ax, ds)
    distinct = len(np.unique(ds.poses.reshape(len(ds), -1), axis=0))
    if distinct <= 4:
        for m in range(ds.spec.n_modes):
            p = ds.poses[ds.mode == m][0]
            ax.plot(p[:, 0], p[:, 1], "-", color=MODE_COLORS[m % 2], lw=1.6,
                    label=f"{ds.spec.mode_names[m]} (×{int((ds.mode == m).sum())})")
            q = p[::5]
            ax.quiver(q[:, 0], q[:, 1], np.cos(q[:, 2]), np.sin(q[:, 2]),
                      color=MODE_COLORS[m % 2], scale=16, width=0.006, zorder=5)
        ax.legend(loc="lower right", fontsize=6.5)
        title += f"\nthe {distinct} distinct demos"
    else:
        idx = np.random.default_rng(0).choice(len(ds), n_paths, replace=False)
        for i in idx:
            ax.plot(ds.poses[i, :, 0], ds.poses[i, :, 1], "-", lw=0.5, alpha=0.35,
                    color=MODE_COLORS[ds.mode[i] % 2])
        title += f"\n{n_paths} of {len(ds)} demos"
    ax.set_title(title, fontsize=8.5, loc="left")


def fig_dataset(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(SETS), figsize=(3.0 * len(SETS), 2.6))
    for ax, scen in zip(axes, SETS):
        _demo_panel(ax, data[scen], SETS[scen])
        if scen != list(SETS)[0]:
            ax.set_ylabel("")
    _save(fig, out, "fig_p2p_dataset")


def _seed_paths(g: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One path per seed (all of a seed's rollouts from one start are the same pose sequence)."""
    return (np.stack([r["roll"]["poses"][0] for r in g]),
            np.array([r["roll"]["labels"][0] for r in g]),
            np.array([bool(r["roll"]["collided"][0]) for r in g]))


def fig_rollouts(all_runs: dict, data: dict, out: Path) -> None:
    fig, axes = plt.subplots(len(SETS), len(METHODS),
                             figsize=(2.6 * len(METHODS), 1.75 * len(SETS)), squeeze=False)
    for i, scen in enumerate(SETS):
        ds = data[scen]
        for j, method in enumerate(METHODS):
            ax, g = axes[i, j], all_runs[scen][method]
            if not g:
                ax.axis("off")
                continue
            if one_start(ds):
                poses, labels, collided = _seed_paths(g)
                note = f"{len(g)} seeds, one path each"
            else:
                r = g[0]
                poses, labels = r["roll"]["poses"], r["roll"]["labels"]
                collided = r["roll"]["collided"].astype(bool)
                note = "seed 0, 150 of its rollouts"
            plot_rollouts(ds, poses, labels, collided, ax=ax,
                          max_paths=None if one_start(ds) else 150, legend=False,
                          title=f"{SHORT[method]} · {SET_SHORT[scen]}")
            if one_start(ds):  # plot_rollouts draws paths thin and faint, for 1000s of them
                for line in ax.lines:
                    if line.get_alpha() == 0.4:
                        line.set_linewidth(1.1), line.set_alpha(0.85)
            ok = ~collided
            ax.text(0.02, 0.03, f"{note}\nsuccess {100 * np.mean(ok):.0f}% · into block "
                    f"{100 * np.mean(into_block(poses, BOX) & collided):.0f}%",
                    transform=ax.transAxes, fontsize=6, color=PALETTE["ink"], va="bottom",
                    bbox=dict(fc=PALETTE["surface"], ec="none", alpha=0.85))
            ax.set_xlabel(""), ax.set_ylabel("")
            ax.set_ylim(3.3, 8)
    _save(fig, out, "fig_p2p_rollouts")


def fig_xphase(all_runs: dict, data: dict, out: Path) -> None:
    scens = [s for s in P2P if one_start(data[s])]
    fig, axes = plt.subplots(len(scens), len(METHODS), figsize=(2.6 * len(METHODS),
                                                                2.0 * len(scens)),
                             sharex=True, sharey=True, squeeze=False)
    for i, scen in enumerate(scens):
        ds = data[scen]
        idx = np.random.default_rng(0).choice(len(ds), min(len(ds), 60), replace=False)
        for j, method in enumerate(METHODS):
            ax, g = axes[i, j], all_runs[scen][method]
            for k in idx:
                ax.plot(ds.phase, ds.poses[k, :, 0], "-", color=PALETTE["ink_muted"], lw=1.2,
                        alpha=0.15)
            if g:
                for p, lab, col in zip(*_seed_paths(g)):
                    color = COLLISION_COLOR if col else (MODE_COLORS[lab % 2] if lab >= 0
                                                         else PALETTE["ink_muted"])
                    ax.plot(ds.phase, p[:, 0], "-", color=color, lw=1.0, alpha=0.85)
            ax.axhspan(BOX.cx - BOX.hx, BOX.cx + BOX.hx, color=PALETTE["ink_muted"], alpha=0.08,
                       lw=0)
            ax.set_title(f"{SHORT[method]} · {SET_SHORT[scen]}", fontsize=8.5, loc="left")
            ax.set_ylim(2.5, 7.5)
            if i == len(scens) - 1:
                ax.set_xlabel("phase")
            if j == 0:
                ax.set_ylabel("x [m]")
    _save(fig, out, "fig_p2p_xphase")


def fig_outcomes(all_runs: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 2.7))
    colors = [MODE_COLORS[0], MODE_COLORS[1], PALETTE["ink_muted"], "#8c1d1d", COLLISION_COLOR]
    ticks, labels, groups = [], [], []
    x, legend_done = 0.0, False
    for method in METHODS:
        start = x
        for scen in SETS:
            g = all_runs[scen][method]
            if not g:
                continue
            bottom = 0.0
            for f, c, nm in zip(outcome_fractions(g).mean(axis=0), colors, OUTCOMES):
                ax.bar(x, 100 * f, bottom=bottom, color=c, width=0.8,
                       label=None if legend_done else nm, hatch=SET_HATCH[scen],
                       edgecolor=PALETTE["surface"], lw=0.5)
                bottom += 100 * f
            legend_done = True
            ticks.append(x)
            labels.append(SET_SHORT[scen])
            x += 1
        groups.append(((start + x - 1) / 2, SHORT[method]))
        x += 0.7
    ax.set_xticks(ticks, labels, fontsize=7)
    for gx, name in groups:
        ax.text(gx, -0.13, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8)
    ax.set_ylabel("rollouts [%] (mean over seeds)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=False)
    ax.set_ylim(0, 100)
    _save(fig, out, "fig_p2p_outcomes")


def fig_mmd_phase(all_runs: dict, phase: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(1, len(SETS), figsize=(2.4 * len(SETS), 2.4), sharex=True,
                             sharey=True)
    for ax, scen in zip(axes, SETS):
        for method in METHODS:
            g = all_runs[scen][method]
            if not g:
                continue
            curves = 1e3 * np.array([r["metrics"]["mmd_per_phase"] for r in g])
            st = method_style(method, METHODS)
            mu, sd = curves.mean(0), curves.std(0)
            ax.plot(phase, mu, color=st["color"], ls=st["ls"], lw=1.4, label=figure_label(method))
            ax.fill_between(phase, np.maximum(mu - sd, 0), mu + sd, color=st["color"],
                            alpha=0.12, lw=0)
        ax.set_title(SETS[scen], fontsize=8.5, loc="left")
        ax.set_xlabel("phase")
        ax.set_yscale("symlog", linthresh=1.0)
    axes[0].set_ylabel(r"MMD$^2\times10^3$")
    axes[-1].legend(fontsize=6.5, frameon=False)
    _save(fig, out, "fig_p2p_mmd_phase")


# The Push-T violins' look (scripts/pusht/plot_pusht.py, fig_violin), so the two read as one
# set: colour names the arm, in the paper's colours; the training set is named over its block.
ARM_STYLE = {"mlp": ("MLP", "#eb6834"), "mlp_score_w0.5": ("MLP+score", "#eb6834"),
             "fm": ("MLP+phase", "#2a78d6"), "score_w0.5": ("MLP+phase+score", "#e87ba4")}
PHASE_ARMS = {"fm", "score_w0.5"}  # bold on the axis, as in the Push-T figures
VIOLIN_BASE = "score_w0.5"  # the daggers test every other arm against it, on the same set


def violin_caption(all_runs: dict) -> str:
    """LaTeX caption for ``fig_p2p_violin``, its numbers read off the runs."""
    def success(scen, method):
        return 100 * vals(all_runs[scen][method], "success_rate")

    ns = "dual_path_noisy_start"
    scored = sorted(success(ns, m).mean() for m in ("mlp_score_w0.5", "score_w0.5"))
    plain = [success(ns, m) for m in ("mlp", "fm")]
    lo, hi = min(v.min() for v in plain), max(v.max() for v in plain)
    orig = " / ".join(f"{success(ORIG, m).mean():.0f}" for m in METHODS)
    seeds = len(all_runs[ORIG][METHODS[0]])
    return (
        r"\textbf{Spread in the training starts, not variation along the path, is what lets a "
        r"deterministic policy split between the two routes.} Four MLP policies (colour; the "
        r"score arms use $\lambda = 0.5$; the phase-conditioned ones are in bold) are each "
        r"trained on 1024 demonstrations from four datasets that differ in one factor (named "
        r"over each block). \textbf{fixed}: two mirror-image trajectories (left and right) from one exact "
        r"start pose, repeated. \textbf{+via}: the same start with 0.3\,m via-point jitter, so "
        r"the demonstrations vary mid-path. \textbf{+start}: start-pose noise (8\,cm, 0.05\,rad) "
        r"and no via jitter. \textbf{C}: the full scenario, with start noise, via jitter, "
        r"goal-position noise and a free goal heading. \textbf{Left}: $\mathrm{MMD}^2 \times "
        r"10^3$ to the set's held-out demonstrations (log scale). \textbf{Right}: success (within "
        r"0.6\,m of the goal with no collision). "
        f"Each dot is one of {seeds} seeds, the bar is their mean, and the violin is a kernel "
        r"density estimate over the seeds; a flat lens marks a policy whose seeds all scored the "
        r"same. $\dagger$ marks $p < 0.05$ against MLP+phase+score trained on the same set "
        r"(Welch's $t$-test over seeds). With a single start (fixed, +via), every seed of every "
        r"policy collides (0\% success), whether or not there is via jitter. Start noise alone "
        r"restores success. On +start, the "
        f"score-regularized policies reach {scored[0]:.0f}--{scored[1]:.0f}\\% with little "
        f"variation across seeds; the unregularized MLP and MLP+phase reach "
        f"{plain[0].mean():.0f}\\% and {plain[1].mean():.0f}\\% on average, with individual "
        f"seeds ranging from {lo:.0f}\\% to {hi:.0f}\\%. C shows the same ordering ({orig}\\%). "
        r"MMD is measured against each set's own demonstrations, so it is not comparable across "
        r"sets.")


def fig_violin(all_runs: dict, out: Path) -> None:
    """MMD and success per seed, side by side, in the Push-T violins' style (``ARM_STYLE``): a
    block of the four arms per training set, the set named over it. Sized as
    those are -- 0.42 in a violin, type from the width (``TYPE_RATIOS``) -- so the two print at
    one type size when set to the same width on the page."""
    from matplotlib.patches import Ellipse  # noqa: PLC0415
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: PLC0415
    from matplotlib.transforms import blended_transform_factory  # noqa: PLC0415

    panels = [("mmd_mean", r"MMD$^2$/seed ($\times10^3$)", r"MMD$^2\times10^3$", 1e3, True),
              ("success_rate", "success rate/seed (%)", "success rate [%]", 100, False)]
    gap = 0.6  # slots between one set's block of arms and the next
    xs = {(scen, m): b * (len(METHODS) + gap) + i
          for b, scen in enumerate(SETS) for i, m in enumerate(METHODS)}
    slots = len(xs) + (len(SETS) - 1) * gap
    width = 0.42 * slots * len(panels)
    fs = {role: type_pt(width, role) for role in ("title", "label", "tick")}
    fs["dagger"] = fs["title"] * 11 / 9.5
    em = fs["title"] / 9.5  # Push-T's header and footer heights are in units of its title size
    gutter = 0.065
    # The arm names turn steeper until neighbours clear each other (a line and a third apart).
    pitch = 0.94 * (1 - gutter) * 72 * width / len(panels) / slots
    rotation = max(30.0, float(np.degrees(np.arcsin(min(1.0, 1.35 * fs["tick"] / pitch)))))
    head, foot = 0.2 * em, 0.78 * em
    fig = plt.figure(figsize=(width, head + foot + 2.4), facecolor=PALETTE["surface"])
    grid = fig.add_gridspec(1, len(panels), wspace=gutter * len(panels),
                            top=1 - head / fig.get_figheight(),
                            bottom=foot / fig.get_figheight(), left=0.05, right=0.99)
    for c, (key, title, ylabel, scale, log) in enumerate(panels):
        ax = fig.add_subplot(grid[0, c])
        dagger_y = blended_transform_factory(ax.transData, ax.transAxes)
        v_all = {k: scale * vals(all_runs[k[0]][k[1]], key) for k in xs}
        span = max(float(v.max()) for v in v_all.values())
        for (scen, m), x in xs.items():
            v = v_all[scen, m]
            color = ARM_STYLE[m][1]
            if np.ptp(v) > 0:
                _violin(ax, v, x, 0.8, {"color": color, "ls": "-"}, log)
            else:  # every seed alike, which only the success panel has: a flat lens instead
                ax.add_patch(Ellipse((x, float(v[0])), 0.62, 0.045 * span, zorder=2,
                                     facecolor=_tint(color, 0.30), edgecolor=color,
                                     linewidth=1.0))
            ax.hlines(v.mean(), x - 0.25, x + 0.25, color=PALETTE["ink"], lw=1.2, zorder=3)
            ax.scatter(x + np.linspace(-0.1, 0.1, len(v)), v, s=18, color=color,
                       edgecolors=PALETTE["surface"], linewidths=0.6, zorder=4)
            if m != VIOLIN_BASE and _welch(v, v_all[scen, VIOLIN_BASE]) < 0.05:
                ax.text(x, 0.87, "†", transform=dagger_y, ha="center", va="top",
                        fontsize=fs["dagger"], color=PALETTE["ink"])
        # Each block named over its arms.
        for b, scen in enumerate(SETS):
            ax.text(b * (len(METHODS) + gap) + (len(METHODS) - 1) / 2, 0.98, SET_SHORT[scen],
                    transform=dagger_y, ha="center", va="top", fontsize=fs["label"],
                    color=PALETTE["ink_muted"])
        if log:  # ticks written out, unlabelled marks on each decade's 2-9
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                                  numticks=99))
            ax.yaxis.set_minor_formatter(NullFormatter())
            # Off the data, with room above for the set names and daggers: a fraction of its
            # decades.
            lo = min(float(v.min()) for v in v_all.values())
            ax.set_ylim(lo * (lo / span) ** 0.05, span * (span / lo) ** 0.36)
        else:  # zero, and the lenses on it, lifted off the axis
            ax.set_ylim(-0.09 * span, 1.36 * span)
            ax.set_yticks(np.arange(0, span + 1, 20))
        ax.set_title(title, fontsize=fs["title"], loc="left", color=PALETTE["ink"])
        ax.set_ylabel(ylabel, fontsize=fs["label"], color=PALETTE["ink"])
        ax.set_xticks(list(xs.values()))
        ax.set_xticklabels([ARM_STYLE[m][0] for _, m in xs], fontsize=fs["tick"],
                           rotation=rotation, ha="right", rotation_mode="anchor")
        for tick, (_, m) in zip(ax.get_xticklabels(), xs):
            tick.set_fontweight("bold" if m in PHASE_ARMS else "normal")
        ax.set_xlim(-0.6, slots - 0.4)
        ax.tick_params(colors=PALETTE["ink_muted"], labelsize=fs["tick"])
        ax.yaxis.grid(True, color=PALETTE["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(PALETTE["grid"])
        ax.set_facecolor(PALETTE["surface"])
    _save(fig, out, "fig_p2p_violin", caption=violin_caption(all_runs), pad=0.02)


def fig_training(all_runs: dict, out: Path) -> None:
    fig, axes = plt.subplots(len(P2P), 2, figsize=(7.2, 2.1 * len(P2P)), sharex=True,
                             squeeze=False)
    for i, scen in enumerate(P2P):
        for method in METHODS:
            g = all_runs[scen][method]
            if not g:
                continue
            st = method_style(method, METHODS)
            for ax, key, scale in ((axes[i, 0], "mmd_mean", 1e3), (axes[i, 1], "velocity_loss", 1.0)):
                logs = [r["log"] for r in g]
                ok = np.isfinite(logs[0][key])
                steps = logs[0]["step"][ok]
                ys = scale * np.array([lg[key][ok] for lg in logs])
                ax.plot(steps, ys.mean(0), color=st["color"], ls=st["ls"], lw=1.3,
                        label=figure_label(method))
                ax.fill_between(steps, ys.min(0), ys.max(0), color=st["color"], alpha=0.1, lw=0)
        axes[i, 0].set_ylabel(f"MMD$^2\\times10^3$\n{SET_SHORT[scen]}")
        axes[i, 0].set_yscale("log")
        axes[i, 1].set_ylabel("velocity loss")
        axes[i, 1].set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("training step")
    axes[0, 1].legend(fontsize=6.5, frameon=False)
    _save(fig, out, "fig_p2p_training")


def fig_targets(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.4), sharex=True)
    for scen in SETS:
        ds, c = data[scen], SET_COLORS[scen]
        spread = np.mean([np.linalg.norm(ds.poses[ds.mode == m][:, :, :2].std(0), axis=-1)
                          for m in range(ds.spec.n_modes)], axis=0)
        axes[0].plot(ds.phase, spread, color=c, lw=1.4, label=SETS[scen])
        axes[1].plot(ds.phase, ds.divergence.mean(0), color=c, lw=1.4)
        axes[2].plot(ds.phase, np.linalg.norm(ds.score, axis=-1).mean(0), color=c, lw=1.4)
    axes[0].set_ylabel("within-mode xy spread [m]")
    axes[1].set_ylabel("demo SE(2) kNN divergence [1/s]")
    axes[2].set_ylabel("|demo kernel score| (mean)")
    for ax in axes:
        ax.set_xlabel("phase")
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.subplots_adjust(wspace=0.4)
    _save(fig, out, "fig_p2p_targets")


# ------------------------------------------------------------------------ restart diagnostic


def restart_demos(ds: TrajectoryDataset) -> np.ndarray:
    """The held-out demos restarts are put onto: the first ``RESTART_DEMOS`` distinct demos of
    each mode (just the two, for the two-trajectory set)."""
    out = []
    for m in range(ds.spec.n_modes):
        p = ds.poses[ds.mode == m]
        _, first = np.unique(p.reshape(len(p), -1), axis=0, return_index=True)
        out.append(p[np.sort(first)[:RESTART_DEMOS]])
    return np.concatenate(out)


@torch.no_grad()
def restart(model, ds: TrajectoryDataset, demos: np.ndarray, k0: int) -> tuple[np.ndarray, np.ndarray]:
    """Put the policy on each demo at step ``k0`` (its pose and phase) and roll it out to the
    end, clamped like the evaluation. Returns ``(poses (K, N+1-k0, 3), collided (K,))``."""
    n = ds.controls.shape[1]
    dtype = next(model.parameters()).dtype
    pose = torch.as_tensor(demos[:, k0], dtype=dtype)
    poses, executed = [pose], []
    for k in range(k0, n):
        u = ds.limits.clamp(model(pose, pose.new_full((len(pose),), k / n)))
        pose = se2_exp_retract(pose, ds.dt * body_twist(u))
        poses.append(pose)
        executed.append(u)
    poses, executed = torch.stack(poses, 1), torch.stack(executed, 1)
    ob, bd = collision_breakdown(poses.double(), executed.double(), ds.dt, ds.spec.world,
                                 float(ds.config.get("radius", 0.3)))
    return poses.numpy(), (np.asarray(ob) | np.asarray(bd))


def restart_study(all_runs: dict, data: dict) -> dict:
    """``{scenario: {method: {phase: [(poses, collided, max_dev), ...] per seed}}}``."""
    out = {}
    for scen, runs in all_runs.items():
        ds = data[scen]
        n = ds.controls.shape[1]
        demos = restart_demos(ds)
        out[scen] = {}
        for method in METHODS:
            out[scen][method] = {}
            for r in runs[method]:
                model = load_policy(r["dir"] / "final.pt")[0].float().eval()
                for ph in RESTART_PHASES:
                    k0 = int(round(ph * n))
                    poses, collided = restart(model, ds, demos, k0)
                    dev = np.linalg.norm(poses[..., :2] - demos[:, k0:, :2], axis=-1).max(axis=1)
                    out[scen][method].setdefault(ph, []).append((poses, collided, dev))
    return out


def track_rate(entries: list) -> float:
    """Share of (seed, demo) restarts that stay within TRACK_TOL of their demo, collision-free."""
    if not entries:
        return np.nan
    ok = [(dev <= TRACK_TOL) & ~col for _, col, dev in entries]
    return float(np.mean(np.concatenate(ok)))


def fig_restart(study: dict, data: dict, out: Path, ph: float = 0.2) -> None:
    fig, axes = plt.subplots(len(SETS), len(METHODS),
                             figsize=(2.6 * len(METHODS), 1.75 * len(SETS)), squeeze=False)
    for i, scen in enumerate(SETS):
        ds = data[scen]
        demos = restart_demos(ds)
        k0 = int(round(ph * ds.controls.shape[1]))
        modes = np.repeat(np.arange(ds.spec.n_modes), len(demos) // ds.spec.n_modes)
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            _world_axes(ax, ds)
            for p in demos:
                ax.plot(p[:, 0], p[:, 1], "-", color=PALETTE["ink_muted"], lw=2, alpha=0.2)
            for poses, collided, _ in study[scen][method].get(ph, []):
                for m in range(len(poses)):
                    ax.plot(poses[m, :, 0], poses[m, :, 1], "-", lw=0.6, alpha=0.6, zorder=5,
                            color=COLLISION_COLOR if collided[m] else MODE_COLORS[modes[m]])
            ax.set_title(f"{SHORT[method]} · {SET_SHORT[scen]}", fontsize=8.5, loc="left")
            ax.text(0.02, 0.03, f"from phase {ph}: tracks "
                    f"{100 * track_rate(study[scen][method].get(ph, [])):.0f}%",
                    transform=ax.transAxes, fontsize=6.5,
                    bbox=dict(fc=PALETTE["surface"], ec="none", alpha=0.85))
            ax.set_xlabel(""), ax.set_ylabel("")
            ax.set_ylim(3.3, 8)
    _save(fig, out, "fig_p2p_restart")


# ---------------------------------------------------------------------------------- markdown


def _fmt(row: dict, key: str, scale: float = 1.0, digits: int = 1) -> str:
    return f"{scale * row[f'{key}_mean']:.{digits}f} ± {scale * row[f'{key}_std']:.{digits}f}"


def write_markdown(summary: list[dict], sig: list[dict], out: Path, findings: str,
                   study: dict) -> None:
    lines = ["# Point-to-point dual path (skate)", "",
             "Generated by `scripts/skate/evaluate_fixed_start.py`. Not part of the paper figures.",
             "", findings.strip(), "",
             "## Setup", "",
             "* **data** -- C's world and block, goal pose (5.0, 7.0, π/2) with no noise and the "
             "planner's goal constrained in heading (`goal_mode=\"pose\"`); 1024 training demos, "
             "a held-out set planned with another seed. The three point-to-point sets differ in "
             "one thing each:",
             "  * **fixed** (`dual_path_fixed`) -- start (5.0, 0.9, π/2) with no noise, no via "
             "jitter. Each mode is planned once and repeated: one left and one right "
             "trajectory ×512, exact mirror images; the held-out set is the same two.",
             "  * **+via** (`dual_path_fixed_via`) -- the same one start, with C's via jitter "
             "(0.3 m): the demos spread in the middle of the path (0.16 m at phase 0.5) and "
             "share both ends.",
             "  * **+start** (`dual_path_noisy_start`) -- C's start noise (8 cm, 0.05 rad), no via "
             "jitter: every demo is the planner's one optimum from its own start (0.11 m spread "
             "at the start, narrowing to the goal).",
             "  * **C** -- the paper's scenario: start noise, via jitter, goal position noise "
             "(0.2 m) and a free goal heading.",
             "* **arms** -- the paper's MLP arms at n = 1024, 10 seeds, 20k steps, commands "
             "clamped to SkateLimits: MLP (no phase), MLP+score (λ=0.5), MLP+phase, "
             "MLP+phase+score (λ=0.5). The C rows are the paper's runs of the same arms.",
             "* **success** -- within 0.6 m of the goal (C's 3σ) with no collision; **heading "
             "err.** is |θ_final − π/2| (the target in the point-to-point sets; C's goal "
             "heading is free).",
             "* In **fixed** and **+via** every held-out start is the same pose, so each seed's "
             "1024 rollouts are one path: success and mode are 0/1 per seed.",
             "", "![demos](fig_p2p_dataset.png)", "",
             "![spread and targets](fig_p2p_targets.png)", "",
             "## Results (mean ± std over seeds)", "",
             "| set | arm | seeds | MMD×10³ | success % | obstacle % | wall % | reached L/R % "
             "| side L/R/– % | into block % | heading err. ° |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(
            f"| {SET_SHORT[r['scenario']]} | {figure_label(r['method'])} | {r['n_seeds']} | "
            f"{_fmt(r, 'mmd_mean', 1e3)} | {_fmt(r, 'success_rate', 100, 0)} | "
            f"{_fmt(r, 'obstacle_collision_rate', 100, 0)} | "
            f"{_fmt(r, 'boundary_collision_rate', 100, 0)} | "
            f"{100 * r['left_frac']:.0f}/{100 * r['right_frac']:.0f} | "
            f"{100 * r['side_left_frac']:.0f}/{100 * r['side_right_frac']:.0f}/"
            f"{100 * r['side_undecided_frac']:.0f} | {100 * r['into_block_rate_mean']:.0f} | "
            f"{_fmt(r, 'heading_err_deg', 1, 1)} |")
    lines += ["", "**reached L/R** counts collision-free rollouts on each side. **side** is the "
              "side of the block each rollout passes on (`label_mode`: median x while level "
              "with the block), collided or not; – is undecided. **into block** is the share "
              "whose median x while level with the block lies within the block's x-span: they "
              "drove at it, the modes' average, rather than round it.",
              "", "![outcomes](fig_p2p_outcomes.png)", "",
              "![per seed](fig_p2p_violin.png)", "",
              "## Welch t-tests", "",
              "| comparison | arm | ΔMMD×10³ (p) | Δsuccess % (p) | Δcollision % (p) | Δheading ° (p) |",
              "|---|---|---:|---:|---:|---:|"]

    def cell(r, key, scale):
        p = r[f"{key}_p"]
        ps = "n/a" if not np.isfinite(p) else (f"**{p:.2g}**" if p < 0.05 else f"{p:.2g}")
        return f"{scale * r[f'{key}_diff']:+.1f} ({ps})"

    for r in sig:
        lines.append(f"| {r['comparison']} | {figure_label(r['method'])} | "
                     f"{cell(r, 'mmd_mean', 1e3)} | {cell(r, 'success_rate', 100)} | "
                     f"{cell(r, 'collision_rate', 100)} | {cell(r, 'heading_err_deg', 1)} |")
    lines += ["", "p is n/a where both samples are constant (e.g. every seed collides).", "",
              "## Restarted on a demo", "",
              "Each trained policy put back *onto* held-out demos at a given phase (the demo's "
              "exact pose, the phase input set to match) and rolled out to the end: "
              f"{RESTART_DEMOS} demos per mode (the two, for **fixed**) × 10 seeds. **tracks** = "
              f"share of restarts that stay within {TRACK_TOL} m of their demo with no "
              "collision. The demos fork at phase ≈ 0.15.", "",
              "| set | arm | " + " | ".join(f"from {p}" for p in RESTART_PHASES) + " |",
              "|---|---|" + "---:|" * len(RESTART_PHASES)]
    for scen in SETS:
        for method in METHODS:
            lines.append(f"| {SET_SHORT[scen]} | {figure_label(method)} | " + " | ".join(
                f"{100 * track_rate(study[scen][method].get(p, [])):.0f}%"
                for p in RESTART_PHASES) + " |")
    cross = out / "CROSS.md"  # written by scripts/skate/cross_start_eval.py
    lines += ["", "![restarts](fig_p2p_restart.png)", "",
              "C's restart rates are not comparable with the others: its demos end at scattered "
              "goals (σ = 0.2 m) with free headings, and a policy heads for their mean, so it "
              "leaves a single demo near the end whatever it does at the fork.", "",
              *([cross.read_text(), ""] if cross.exists() else []),
              "## Figures", "",
              "Rollouts: every seed's path where the start is fixed, seed 0's rollouts where it "
              "is not:", "", "![rollouts](fig_p2p_rollouts.png)", "",
              "x over phase for the one-start sets, a line per seed, against the demos (grey):",
              "", "![x over phase](fig_p2p_xphase.png)", "",
              "![MMD per phase](fig_p2p_mmd_phase.png)", "",
              "![training](fig_p2p_training.png)", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print(f"  wrote {out / 'SUMMARY.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs/skate_deterministic"))
    parser.add_argument("--orig-runs", type=Path, default=Path("runs"))
    parser.add_argument("--data", type=Path, default=Path("data/skate_deterministic/test"))
    parser.add_argument("--orig-data", type=Path, default=Path("data/test"))
    parser.add_argument("--out", type=Path, default=Path("output/skate_deterministic"))
    parser.add_argument("--findings", type=Path, default=None,
                        help="Markdown put at the top of SUMMARY.md (default: FINDINGS.md in --out)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = {s: TrajectoryDataset.load((args.orig_data if s == ORIG else args.data) / f"{s}.npz")
            for s in SETS}
    all_runs = {s: load(args.orig_runs if s == ORIG else args.runs, s) for s in SETS}
    for scen, runs in all_runs.items():
        print(f"{scen}: " + ", ".join(f"{m} {len(g)}" for m, g in runs.items()))

    summary, sig = summarize(all_runs), significance(all_runs)
    write_csv(summary, args.out / "summary.csv")
    write_csv(sig, args.out / "significance.csv")

    fig_dataset(data, args.out)
    fig_rollouts(all_runs, data, args.out)
    fig_xphase(all_runs, data, args.out)
    fig_outcomes(all_runs, args.out)
    fig_mmd_phase(all_runs, data[ORIG].phase, args.out)
    fig_violin(all_runs, args.out)
    fig_training(all_runs, args.out)
    fig_targets(data, args.out)
    study = restart_study(all_runs, data)
    fig_restart(study, data, args.out)

    findings_path = args.findings or args.out / "FINDINGS.md"
    findings = findings_path.read_text() if findings_path.exists() else ""
    write_markdown(summary, sig, args.out, findings, study)


if __name__ == "__main__":
    main()
