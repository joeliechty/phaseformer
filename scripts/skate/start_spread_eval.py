"""Start-spread experiment: every MLP-arm policy on the deployment starts and a lateral line sweep,
the restart control and the training MSE; the training sets' spread diagnostics; the figures and
``START_SPREAD.md``.

    python scripts/skate/start_spread_eval.py              # evaluate what is new, then write
    python scripts/skate/start_spread_eval.py --plot-only  # figures + markdown from the CSVs

The sets (``SETS``) are the start-spread sets of ``build_start_spread.py`` -- S2-S12 (random
sides), T2/T8 (sorted sides), M'5-M'15 (the mixtures' noisy demos alone) -- next to the existing
point-to-point sets re-evaluated: S0 (``dual_path_fixed``), R8 (``dual_path_noisy_start``, an
independent replicate of S8), M5-M15 (the mixtures), V (``dual_path_fixed_via``) and C (the
paper's ``dual_path``). Every policy (4 arms x 10 seeds per set) is rolled out closed loop,
clamped, for 50 steps of 0.15 s:

* **deployment** -- from C's 1024 held-out starts (``data/test/dual_path.npz``): success (within
  0.6 m of (5.0, 7.0), no collision), collision, undecided (neither) and the left share of the
  successes; and from the noisy-start held-out set's starts, the phase-mean MMD^2 to its demos
  (the same start distribution; C's own demos end at scattered goals with free headings);
* **line sweep** -- from (5.0 + δ, 0.9, π/2), δ ∈ {0} ∪ ±10^u m for 60 u evenly spaced in
  [-4, log10 0.24] (121 starts, 0.1 mm to 24 cm), each start's outcome (left, right, collision,
  undecided). From it: the pattern of the successful sides along δ (sorted: every left success
  left of every right one; reversed; one-sided; interleaved), the band width w between the
  innermost left and right successes when sorted, corr(sign δ, side) over the successes, and the
  sweep failures weighted by the deployment density N(0, 8 cm) (trapezoid rule);
* **restart control** -- put back onto the set's held-out demos at phase 0.3 and 0.5
  (``evaluate_fixed_start.restart``): the share that reaches the goal with no collision;
* **train MSE** -- the training loss without the score term, over every row of the training set.

The data diagnostics (no policies): per step k, the pooled within-side std σ_k and half the
left-right gap d_k of the demos' x; σ_onset = σ_1, where the sides begin to part; σ and d where
the demos cross the lane exit (y = 3.3), d where they reach the grown block's front (y = 4.6);
corr(start x, side) and corr(sign(start x - 5), side).

Writes to ``output/skate_start_spread/``: ``deploy.csv``, ``sweep.csv``, ``data_diag.csv``,
``data_diag_phase.csv``, ``fig_spread_*`` and ``START_SPREAD.md`` (``START_SPREAD_FINDINGS.md`` on
top). Policies already in ``deploy.csv`` are not re-evaluated unless ``--force``.
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
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cross_start_eval import GOAL, NOMINAL, REACH, evaluate  # noqa: E402
from evaluate_fixed_start import METHODS, SHORT, restart, restart_demos  # noqa: E402
from evaluate_policies import _save, figure_label, method_style  # noqa: E402
from fm_bc.skate.baselines import load_policy  # noqa: E402
from fm_bc.skate.dataset import TrajectoryDataset  # noqa: E402
from fm_bc.skate.evaluation import mmd  # noqa: E402
from fm_bc.skate.plots import MODE_COLORS, _world_axes  # noqa: E402
from fm_bc.skate.training import make_rows  # noqa: E402
from skate_planner.viz import PALETTE  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})

# key -> (scenario, runs root, group, start sigma in cm). Groups: S random sides, T sorted sides,
# R the replicate of S8, M mixtures, M' their noisy demos alone, V via jitter, C the paper's.
SETS = {
    "S0": ("dual_path_fixed", "runs", "S", 0),
    "S2": ("dual_path_spread02", "runs", "S", 2),
    "S4": ("dual_path_spread04", "runs", "S", 4),
    "S8": ("dual_path_spread08", "runs", "S", 8),
    "S12": ("dual_path_spread12", "runs", "S", 12),
    "T2": ("dual_path_sorted02", "runs", "T", 2),
    "T8": ("dual_path_sorted08", "runs", "T", 8),
    "R8": ("dual_path_noisy_start", "runs", "R", 8),
    "M5": ("dual_path_mix05", "runs", "M", 8),
    "M10": ("dual_path_mix10", "runs", "M", 8),
    "M15": ("dual_path_mix15", "runs", "M", 8),
    "M'5": ("dual_path_mixonly05", "runs", "M'", 8),
    "M'10": ("dual_path_mixonly10", "runs", "M'", 8),
    "M'15": ("dual_path_mixonly15", "runs", "M'", 8),
    "V": ("dual_path_fixed_via", "runs", "V", 0),
    "C": ("dual_path", "orig", "C", 8),
}
SET_NOTE = {
    "S0": "one start: 2 distinct demos, x512 each", "V": "one start + via jitter (0.3 m)",
    "R8": "dual_path_noisy_start (own draws)", "C": "the paper's scenario C",
    "M5": "52 noisy + 972 fixed copies", "M10": "102 noisy + 922 fixed copies",
    "M15": "154 noisy + 870 fixed copies", "M'5": "M5's 52 noisy demos alone",
    "M'10": "M10's 102 noisy demos alone", "M'15": "M15's 154 noisy demos alone",
}
SWEEP_U = np.linspace(-4.0, np.log10(0.24), 60)
SWEEP = np.concatenate([-(10.0 ** SWEEP_U[::-1]), [0.0], 10.0 ** SWEEP_U])  # m, 121 offsets
EVAL_SIGMA = 0.08  # m, the deployment start spread the sweep is weighted with
RESTART_AT = (0.3, 0.5)
# The counter-steer: the demos' left-goers first swing right of the right-goers (d_k < 0) over
# these steps. Post hoc, not in the spec: max |d_k| / σ_k there, how well the position predicts
# the side in that first window.
EARLY = range(1, 8)
Y_EXIT, Y_FRONT = 3.3, 4.6  # the lane exit; the block's front edge grown by the 0.3 m radius
GAP_G = 2.8  # m, the block's width grown by the robot radius: what separates the corridors
OUTCOMES = ("left", "right", "collision", "undecided")
FAIL_COLOR = "#52514e"
# The start spreads as an ordinal blue ramp (steps 250 / 400 / 550 / 700).
SIGMA_COLORS = {2: "#86b6ef", 4: "#3987e5", 8: "#1c5cab", 12: "#0d366b"}


def data_paths(scenario: str) -> tuple[Path, Path]:
    if scenario == "dual_path":
        return Path("data/dual_path.npz"), Path("data/test/dual_path.npz")
    return (Path(f"data/skate_deterministic/{scenario}.npz"),
            Path(f"data/skate_deterministic/test/{scenario}.npz"))


def ci95(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


# ------------------------------------------------------------------------------- the data


def x_at(poses: np.ndarray, y0: float) -> np.ndarray:
    """Each demo's x where it first crosses ``y = y0`` (linear between steps; NaN if never)."""
    out = np.full(len(poses), np.nan)
    for i, p in enumerate(poses):
        j = np.flatnonzero((p[:-1, 1] < y0) & (p[1:, 1] >= y0))
        if len(j):
            j = j[0]
            a = (y0 - p[j, 1]) / (p[j + 1, 1] - p[j, 1])
            out[i] = p[j, 0] + a * (p[j + 1, 0] - p[j, 0])
    return out


def pooled_sd(x: np.ndarray, mode: np.ndarray) -> float:
    return float(np.sqrt(0.5 * (np.nanvar(x[mode == 0]) + np.nanvar(x[mode == 1]))))


def half_gap(x: np.ndarray, mode: np.ndarray) -> float:
    return float(0.5 * (np.nanmean(x[mode == 1]) - np.nanmean(x[mode == 0])))


def data_diagnostics() -> tuple[list[dict], list[dict]]:
    rows, phase_rows = [], []
    for key, (scen, _, group, sigma_cm) in SETS.items():
        if not data_paths(scen)[0].exists():
            continue
        ds = TrajectoryDataset.load(data_paths(scen)[0])
        x, m = ds.poses[..., 0], ds.mode
        for k in range(x.shape[1]):
            phase_rows.append({"set": key, "step": k, "y_mean": float(ds.poses[:, k, 1].mean()),
                               "sigma_k": pooled_sd(x[:, k], m), "d_k": half_gap(x[:, k], m)})
        sig = np.array([pooled_sd(x[:, k], m) for k in EARLY])
        gap = np.array([half_gap(x[:, k], m) for k in EARLY])
        x_exit, x_front = x_at(ds.poses, Y_EXIT), x_at(ds.poses, Y_FRONT)
        sx = ds.starts[:, 0]
        rows.append({
            "set": key, "scenario": scen, "group": group, "start_sigma_cm": sigma_cm,
            "n_demos": len(ds), "n_distinct": len(np.unique(ds.poses.reshape(len(ds), -1), axis=0)),
            "sigma_onset": pooled_sd(x[:, 1], m), "sigma_exit": pooled_sd(x_exit, m),
            "early_signal": float(np.max(np.abs(gap) / np.maximum(sig, 1e-12))),
            "d_exit": half_gap(x_exit, m), "d_commit": half_gap(x_front, m), "g": GAP_G,
            "corr_start_x_side": float(np.corrcoef(sx, m)[0, 1]) if sx.std() > 0 else np.nan,
            "corr_start_sign_side": (float(np.corrcoef((sx > 5.0).astype(float), m)[0, 1])
                                     if np.ptp((sx > 5.0).astype(float)) else np.nan),
            "n_redraws": ds.stats.get("redraws", ""),
        })
    return rows, phase_rows


# ------------------------------------------------------------------------------ the policies


@torch.no_grad()
def train_mse(model, rows: dict) -> float:
    return float(((model(rows["poses"], rows["phase"]) - rows["controls"]) ** 2).mean())


def sweep_summary(outcome: np.ndarray) -> dict:
    """The sweep's pattern, band width, order and deployment-weighted failure."""
    succ = np.isin(outcome, ("left", "right"))
    d, side = SWEEP[succ], (outcome[succ] == "right").astype(int)
    switches = int(np.sum(side[1:] != side[:-1]))
    if not len(d):
        pattern = "none"
    elif switches == 0:
        pattern = "one-sided"
    elif switches == 1:
        pattern = "sorted" if side[0] == 0 else "reversed"
    else:
        pattern = "interleaved"
    w = centre = np.nan
    if switches == 1:
        i = int(np.flatnonzero(side[1:] != side[:-1])[0])
        w, centre = float(d[i + 1] - d[i]), float(0.5 * (d[i + 1] + d[i]))
    nz = d != 0
    order = (float(np.corrcoef(np.sign(d[nz]), side[nz])[0, 1])
             if nz.sum() > 1 and np.ptp(side[nz]) and np.ptp(np.sign(d[nz])) else np.nan)
    widths = np.gradient(SWEEP)  # trapezoid weights on the uneven grid
    dens = stats.norm.pdf(SWEEP, 0.0, EVAL_SIGMA) * widths
    return {"sweep_pattern": pattern, "band_w": w, "band_centre": centre,
            "n_switches": switches, "sweep_order": order, "sweep_success": float(succ.mean()),
            "sweep_pred_fail": float(np.sum(dens * ~succ) / np.sum(dens)),
            "centre_outcome": str(outcome[len(SWEEP) // 2])}


def outcomes(r: dict) -> np.ndarray:
    out = np.where(r["collided"], "collision", "undecided").astype(object)
    out[r["success"] & (r["labels"] == 0)] = "left"
    out[r["success"] & (r["labels"] == 1)] = "right"
    return out


def evaluate_all(args, done: set) -> tuple[list[dict], list[dict]]:
    world = TrajectoryDataset.load(Path("data/skate_deterministic/test/dual_path_fixed.npz"))
    deploy = TrajectoryDataset.load(Path("data/test/dual_path.npz")).poses[:, 0]
    ref = TrajectoryDataset.load(Path("data/skate_deterministic/test/dual_path_noisy_start.npz"))
    ref_demos = torch.as_tensor(ref.poses, dtype=torch.float32)
    sweep_starts = NOMINAL[None] + np.outer(SWEEP, [1.0, 0.0, 0.0])
    recs, sweeps = [], []
    for key, (scen, root, group, sigma_cm) in SETS.items():
        if args.sets and key not in args.sets:
            continue
        train_path, test_path = data_paths(scen)
        if not train_path.exists():
            print(f"{key}: no data ({train_path}), skipped")
            continue
        train_ds, test_ds = TrajectoryDataset.load(train_path), TrajectoryDataset.load(test_path)
        rows = make_rows(train_ds, "cpu")
        demos = restart_demos(test_ds)
        n = test_ds.controls.shape[1]
        runs = args.orig_runs if root == "orig" else args.runs
        for method in METHODS:
            seeds = sorted((runs / scen / f"n{len(train_ds)}" / method).glob("seed*"))
            for seed_dir in seeds:
                seed = int(seed_dir.name[4:])
                if not (seed_dir / "metrics.json").exists() or (key, method, seed) in done:
                    continue
                model = load_policy(seed_dir / "final.pt")[0].float().eval()
                dep = evaluate(model, deploy, world)
                poses = torch.as_tensor(evaluate(model, ref.poses[:, 0], world)["poses"],
                                        dtype=torch.float32)
                mmd2 = float(np.mean([mmd(poses[:, t], ref_demos[:, t])
                                      for t in range(poses.shape[1])]))
                sw = outcomes(evaluate(model, sweep_starts, world))
                rec = {"set": key, "scenario": scen, "group": group, "sigma_cm": sigma_cm,
                       "method": method, "seed": seed,
                       "success": float(dep["success"].mean()),
                       "collision": float(dep["collided"].mean()),
                       "undecided": float((~dep["success"] & ~dep["collided"]).mean()),
                       "left_share": (float(np.mean(dep["labels"][dep["success"]] == 0))
                                      if dep["success"].any() else np.nan),
                       "mmd2": mmd2, "train_mse": train_mse(model, rows)}
                for ph in RESTART_AT:
                    rp, col = restart(model, test_ds, demos, int(round(ph * n)))
                    reach = np.linalg.norm(rp[:, -1, :2] - GOAL, axis=-1) <= REACH
                    rec[f"restart_{ph}"] = float(np.mean(reach & ~col))
                rec.update(sweep_summary(sw))
                recs.append(rec)
                sweeps += [{"set": key, "method": method, "seed": seed, "delta": float(d),
                            "outcome": o} for d, o in zip(SWEEP, sw)]
            print(f"{key:>4} {SHORT[method]}: {len(seeds)} seeds", flush=True)
    return recs, sweeps


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if k not in ("seed", "step") else int(v)
            except (TypeError, ValueError):
                pass
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------- the figures


class Table:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def get(self, key: str, method: str, col: str) -> np.ndarray:
        return np.array([r[col] for r in self.rows if r["set"] == key and r["method"] == method],
                        dtype=float)

    def has(self, key: str) -> bool:
        return any(r["set"] == key for r in self.rows)


def _clean(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=PALETTE["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["obstacle_edge"])
    ax.tick_params(colors=PALETTE["ink_muted"], labelsize=7)


# x slots of the reference sets on the deployment axis, outside the 0-12 cm spread range.
REF_SLOTS = {"V": -2.4, "R8": 14.6, "C": 16.6}


def fig_deploy(t: Table, out: Path) -> None:
    """(a) Deployment success against the start spread, a panel per arm; V, R8 and C in their
    own slots either side."""
    fig, axes = plt.subplots(1, len(METHODS), figsize=(7.2, 2.0), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for ax, method in zip(axes, METHODS):
        color = method_style(method, METHODS)["color"]
        for group, ls, marker, mfc, dx in (("S", "-", "o", color, 0.0),
                                           ("T", "--", "s", PALETTE["surface"], 0.6)):
            keys = [k for k, s in SETS.items() if s[2] == group and t.has(k)]
            xs = np.array([SETS[k][3] + dx for k in keys], dtype=float)
            ys = [100 * t.get(k, method, "success") for k in keys]
            ax.errorbar(xs, [y.mean() for y in ys], yerr=[ci95(y) for y in ys], ls=ls, lw=1.4,
                        color=color, marker=marker, ms=4, mfc=mfc, mec=color, capsize=2,
                        elinewidth=0.8, zorder=3)
        for key, x in REF_SLOTS.items():
            if t.has(key):
                y = 100 * t.get(key, method, "success")
                ax.errorbar([x], [y.mean()], yerr=[ci95(y)], color=PALETTE["ink_muted"],
                            marker="D", ms=3.5, mfc=PALETTE["surface"], ls="none", capsize=2,
                            elinewidth=0.8, zorder=4)
        for x in (-1.2, 13.4):
            ax.axvline(x, color=PALETTE["grid"], lw=0.8, zorder=0)
        ax.set_xticks([REF_SLOTS["V"], 0, 2, 4, 8, 12, REF_SLOTS["R8"], REF_SLOTS["C"]],
                      ["V", "0", "2", "4", "8", "12", "R8", "C"])
        ax.set_xlim(-3.6, 17.8)
        ax.set_ylim(-3, 103)
        ax.set_title(figure_label(method), fontsize=8, loc="left", color=PALETTE["ink"])
        ax.set_xlabel("start spread σ [cm]", fontsize=7.5, color=PALETTE["ink_muted"])
        _clean(ax)
    axes[0].set_ylabel("deployment success [%]", fontsize=7.5, color=PALETTE["ink_muted"])
    ink = PALETTE["ink"]
    handles = [Line2D([], [], color=ink, marker="o", ms=4, lw=1.4, label="S: random sides"),
               Line2D([], [], color=ink, marker="s", ms=4, mfc=PALETTE["surface"], ls="--",
                      lw=1.4, label="T: sorted sides"),
               Line2D([], [], color=PALETTE["ink_muted"], marker="D", ms=3.5, ls="none",
                      mfc=PALETTE["surface"],
                      label="V: one start + via jitter;  R8: S8 replicate;  C: paper scenario")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3,
               fontsize=6.5, frameon=False, handlelength=2.2)
    _save(fig, out, "fig_spread_deploy")


def sweep_fractions(sweep: list[dict], key: str, method: str) -> dict[str, np.ndarray]:
    rows = [r for r in sweep if r["set"] == key and r["method"] == method]
    seeds = sorted({r["seed"] for r in rows})
    grid = {o: np.zeros(len(SWEEP)) for o in OUTCOMES}
    idx = {round(float(d), 12): i for i, d in enumerate(SWEEP)}
    for r in rows:
        grid[r["outcome"]][idx[round(float(r["delta"]), 12)]] += 1
    return {o: v / max(1, len(seeds)) for o, v in grid.items()}


SWEEP_TITLE = {"S0": "S0: one start", "S2": "S2: 2 cm, random sides",
               "S8": "S8: 8 cm, random sides", "T8": "T8: 8 cm, sorted sides"}


def fig_sweep(sweep: list[dict], out: Path, keys=("S0", "S2", "S8", "T8"), method="fm") -> None:
    """(b) The line sweep: the share of seeds ending left, right or failed at each offset."""
    keys = [k for k in keys if any(r["set"] == k for r in sweep)]
    fig = plt.figure(figsize=(1.85 * len(keys), 2.25))
    gs = fig.add_gridspec(2, len(keys), height_ratios=[4, 1], hspace=0.08, wspace=0.12)
    d_cm = 100 * SWEEP
    # On a log offset axis the deployment starts' share per log-interval is |δ| p(δ): it shows
    # how little of the deployment mass sits in a sub-centimetre central band.
    dens = np.abs(SWEEP) * stats.norm.pdf(SWEEP, 0.0, EVAL_SIGMA)
    for j, key in enumerate(keys):
        ax = fig.add_subplot(gs[0, j])
        strip = fig.add_subplot(gs[1, j], sharex=ax)
        f = sweep_fractions(sweep, key, method)
        failed = f["collision"] + f["undecided"]
        ax.stackplot(d_cm, 100 * f["left"], 100 * failed, 100 * f["right"],
                     colors=[MODE_COLORS[0], FAIL_COLOR, MODE_COLORS[1]], alpha=0.9, lw=0)
        ax.set_xscale("symlog", linthresh=0.01, linscale=0.4)
        ax.set_xlim(d_cm[0], d_cm[-1])
        ax.set_ylim(0, 100)
        ax.set_title(SWEEP_TITLE.get(key, key), fontsize=7.5, loc="left", color=PALETTE["ink"])
        ax.tick_params(labelbottom=False)
        _clean(ax)
        strip.fill_between(d_cm, dens / dens.max(), color=PALETTE["rest"], lw=0)
        strip.set_ylim(0, 1.1)
        strip.set_yticks([])
        strip.set_xticks([-10, -1, -0.1, 0, 0.1, 1, 10], ["-10", "-1", "", "0", "", "1", "10"])
        strip.set_xlabel("start offset δ [cm]", fontsize=7.5, color=PALETTE["ink_muted"])
        _clean(strip, grid_axis="x")
        strip.spines["left"].set_visible(False)
        if j == 0:
            ax.set_ylabel("seeds [%]", fontsize=7.5, color=PALETTE["ink_muted"])
            strip.set_ylabel("deploy.\nstarts", fontsize=6.5, color=PALETTE["ink_muted"],
                             rotation=0, ha="right", va="center")
        else:
            ax.tick_params(labelleft=False)
    handles = [Patch(color=MODE_COLORS[0], label="reached the goal, left"),
               Patch(color=FAIL_COLOR, label="failed (collision or undecided)"),
               Patch(color=MODE_COLORS[1], label="reached the goal, right"),
               Patch(color=PALETTE["rest"],
                     label="deployment starts per log-interval of δ, N(0, 8 cm)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2,
               fontsize=6.5, frameon=False)
    fig.suptitle(f"{figure_label(method)}, 10 seeds per set", fontsize=8, x=0.1, ha="left",
                 y=1.03, color=PALETTE["ink"])
    _save(fig, out, "fig_spread_sweep")


def fig_mix(t: Table, out: Path) -> None:
    """(c) The mixtures against their noisy demos alone, and against all-noisy S8 / R8."""
    pcts = [5, 10, 15]
    fig, axes = plt.subplots(1, len(METHODS), figsize=(7.2, 1.9), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for ax, method in zip(axes, METHODS):
        color = method_style(method, METHODS)["color"]
        for prefix, mfc, dx in (("M", color, -0.12), ("M'", PALETTE["surface"], 0.12)):
            ys = [100 * t.get(f"{prefix}{p}", method, "success") for p in pcts]
            ok = [len(y) > 0 for y in ys]
            xs = np.arange(len(pcts))[ok] + dx
            ax.errorbar(xs, [y.mean() for y, k in zip(ys, ok) if k],
                        yerr=[ci95(y) for y, k in zip(ys, ok) if k], ls="none", marker="o",
                        ms=4.5, color=color, mfc=mfc, mec=color, capsize=2, elinewidth=0.8,
                        zorder=3)
        for key, ls in (("S8", "-"), ("R8", ":")):
            if t.has(key):
                y = 100 * t.get(key, method, "success").mean()
                ax.axhline(y, color=PALETTE["ink_muted"], lw=0.9, ls=ls, zorder=2)
        ax.set_xticks(range(len(pcts)), [f"{p}%" for p in pcts])
        ax.set_xlim(-0.5, len(pcts) - 0.5)
        ax.set_ylim(-3, 103)
        ax.set_title(figure_label(method), fontsize=8, loc="left", color=PALETTE["ink"])
        ax.set_xlabel("noisy-start demos", fontsize=7.5, color=PALETTE["ink_muted"])
        _clean(ax)
    axes[0].set_ylabel("deployment success [%]", fontsize=7.5, color=PALETTE["ink_muted"])
    ink = PALETTE["ink"]
    handles = [Line2D([], [], color=ink, marker="o", ms=4.5, ls="none",
                      label="M: + copies of the 2 fixed paths (1024 demos)"),
               Line2D([], [], color=ink, marker="o", ms=4.5, ls="none", mfc=PALETTE["surface"],
                      label="M': the noisy demos alone"),
               Line2D([], [], color=PALETTE["ink_muted"], lw=0.9, label="S8 (all 1024 noisy)"),
               Line2D([], [], color=PALETTE["ink_muted"], lw=0.9, ls=":", label="R8")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4,
               fontsize=6.5, frameon=False)
    _save(fig, out, "fig_spread_mix")


def fig_rollouts(args, t: Table, out: Path, keys=("S0", "S2", "S8"), method="fm", seed=0) -> None:
    """(d) One seed's 1024 deployment rollouts, the whole world."""
    world = TrajectoryDataset.load(Path("data/skate_deterministic/test/dual_path_fixed.npz"))
    deploy = TrajectoryDataset.load(Path("data/test/dual_path.npz")).poses[:, 0]
    keys = [k for k in keys if t.has(k)]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.5 * len(keys), 2.25))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(wspace=0.06)
    for ax, key in zip(axes, keys):
        scen, root = SETS[key][:2]
        runs = args.orig_runs if root == "orig" else args.runs
        n = len(TrajectoryDataset.load(data_paths(scen)[0]))
        model = load_policy(runs / scen / f"n{n}" / method / f"seed{seed}" / "final.pt")[0]
        r = evaluate(model.float().eval(), deploy, world)
        oc = outcomes(r)
        _world_axes(ax, world)
        for o, color, ls, a, z in (("left", MODE_COLORS[0], "-", 0.12, 3),
                                   ("right", MODE_COLORS[1], "-", 0.12, 3),
                                   ("fail", FAIL_COLOR, "--", 0.25, 4)):
            sel = np.isin(oc, ("collision", "undecided")) if o == "fail" else oc == o
            for p in r["poses"][sel]:
                ax.plot(p[:, 0], p[:, 1], ls, color=color, lw=0.5, alpha=a, zorder=z)
        ax.add_patch(Circle(NOMINAL[:2], 2 * EVAL_SIGMA, fill=False, lw=0.9,
                            ec=PALETTE["start"], zorder=6))
        ax.plot(*NOMINAL[:2], "o", ms=2.5, color=PALETTE["start"], zorder=7)
        ax.plot(*GOAL, "*", ms=6, color=PALETTE["ink"], zorder=7)
        ax.set_title(f"{key} · seed {seed}: {100 * r['success'].mean():.0f}% success",
                     fontsize=7.5, loc="left", color=PALETTE["ink"])
        ax.set_xlabel(""), ax.set_ylabel("")
        ax.tick_params(labelsize=6)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    handles = [Line2D([], [], color=MODE_COLORS[0], lw=1.2, label="left, reached"),
               Line2D([], [], color=MODE_COLORS[1], lw=1.2, label="right, reached"),
               Line2D([], [], color=FAIL_COLOR, lw=1.0, ls="--", label="failed"),
               Line2D([], [], color=PALETTE["obstacle_edge"], lw=0.8, ls="--",
                      label="obstacles grown by the robot radius (0.3 m)"),
               Line2D([], [], color=PALETTE["start"], marker="o", ms=3, lw=0.9,
                      label="nominal start, 2σ of the deployment starts"),
               Line2D([], [], color=PALETTE["ink"], marker="*", ms=6, ls="none", label="goal")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3,
               fontsize=6.5, frameon=False)
    fig.suptitle(f"{figure_label(method)}: deployment rollouts from C's 1024 held-out starts",
                 fontsize=8, x=0.12, ha="left", y=0.98, color=PALETTE["ink"])
    _save(fig, out, "fig_spread_rollouts")


def fig_profile(phase_rows: list[dict], out: Path, k_max: int = 20) -> None:
    """The demos' within-side spread σ_k and half-separation d_k over the first steps."""
    def series(key, col):
        rs = sorted((r for r in phase_rows if r["set"] == key), key=lambda r: r["step"])
        return (np.array([r["step"] for r in rs])[: k_max + 1],
                100 * np.array([r[col] for r in rs])[: k_max + 1])

    styles = {**{f"S{c}": (SIGMA_COLORS[c], "-", f"S{c}: {c} cm, random sides") for c in (2, 4, 8, 12)},
              **{f"T{c}": (SIGMA_COLORS[c], "--", f"T{c}: {c} cm, sorted sides") for c in (2, 8)},
              "V": (PALETTE["infeasible"], "-.", "V: one start + via jitter"),
              "C": (PALETTE["ink_muted"], ":", "C: paper scenario")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.3))
    fig.subplots_adjust(wspace=0.25)
    for key, (color, ls, label) in styles.items():
        if not any(r["set"] == key for r in phase_rows):
            continue
        k, s = series(key, "sigma_k")
        pos = s > 0
        a1.plot(k[pos], s[pos], ls, color=color, lw=1.3, label=label)
        k, d = series(key, "d_k")
        a2.plot(k, d, ls, color=color, lw=1.3)
    a1.set_yscale("log")
    a1.set_ylabel("within-side spread σ_k [cm]", fontsize=7.5, color=PALETTE["ink_muted"])
    a2.set_ylabel("half-separation d_k [cm]", fontsize=7.5, color=PALETTE["ink_muted"])
    for ax in (a1, a2):
        ax.axvline(1, color=PALETTE["ink"], lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.set_xlabel("step k (phase k/50)", fontsize=7.5, color=PALETTE["ink_muted"])
        ax.set_xlim(0, k_max)
        _clean(ax, grid_axis="both")
    a1.annotate("onset of the split (k = 1)", (1, a1.get_ylim()[0]), xytext=(3, 30),
                textcoords="offset points", fontsize=6.5, color=PALETTE["ink"],
                bbox=dict(fc=PALETTE["surface"], ec="none", pad=1.0))
    k, s = series("V", "sigma_k")
    if len(s) > 5:
        for kk in (1, 5):
            a1.plot(kk, s[kk], "o", ms=3.5, color=PALETTE["infeasible"], zorder=5)
            a1.annotate(f"V {s[kk]:.2f} cm", (kk, s[kk]), xytext=(5, 0 if kk == 1 else 3),
                        textcoords="offset points", fontsize=6.5, color=PALETTE["ink_muted"])
    a1.text(0.98, 0.03, "S0: 0 at every step", transform=a1.transAxes, ha="right", fontsize=6.5,
            color=PALETTE["ink_muted"])
    a2.axhline(0, color=PALETTE["obstacle_edge"], lw=0.6)
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.17),
               ncol=4, fontsize=6.5, frameon=False)
    _save(fig, out, "fig_spread_profile")


# ------------------------------------------------------------------------------- markdown


def fmt(v, scale=100.0, d=0) -> str:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return "—"
    if len(v) == 1:
        return f"{scale * v[0]:.{d}f}"
    return f"{scale * v.mean():.{d}f} ± {scale * ci95(v):.{d}f}"


def welch(x: np.ndarray, y: np.ndarray) -> str:
    x, y = 100 * x, 100 * y
    if min(len(x), len(y)) < 2:
        return "—"
    if (np.ptp(x) == 0 and np.ptp(y) == 0) or max(x.mean(), y.mean()) < 1.0:
        return f"{x.mean() - y.mean():+.1f} (n/a)"
    p = stats.ttest_ind(x, y, equal_var=False).pvalue
    ps = f"**{p:.2g}**" if p < 0.05 else f"{p:.2g}"
    return f"{x.mean() - y.mean():+.1f} ({ps})"


def markdown(t: Table, diag: list[dict], out: Path) -> None:
    keys = [k for k in SETS if t.has(k)]
    lines = ["# Start spread: how the spread of the training starts sets success (skate, "
             "point-to-point dual path)", "",
             "Generated by `scripts/skate/start_spread_eval.py`; the sets by "
             "`scripts/skate/build_start_spread.py`. The design and the predictions, written "
             "before the runs: [SPEC.md](SPEC.md). Replaces the start-noise mixture study "
             "([START_MIX.md](../skate_deterministic/START_MIX.md)).", ""]
    lines += ["## Training sets", "",
              "C's world, the fixed goal pose (5.0, 7.0, π/2), no via jitter (except V and C), "
              "512 demos per side. **σ_onset** is the pooled within-side std of the demos' x at "
              "step 1, where the sides begin to part; σ and d at the lane exit (y = 3.3) and d "
              "at the grown block's front (y = 4.6) are descriptive (the lane funnels the "
              "spread). **early signal** (post hoc, not in the spec) is max |d_k| / σ_k over "
              "steps 1–7, the counter-steer, where the left-goers first swing right of the "
              "right-goers (d_k < 0): how well the position predicts the side in that first "
              "window (1000 = no spread). The S and T sets share one draw of standard normals "
              "in antithetic pairs (common random numbers); R8 is S8's planner set with its own "
              "draws.", "",
              "| set | demos | distinct | start σ [cm] | σ_onset [cm] | early signal | "
              "σ at exit [cm] | d exit [m] | d front [m] | corr(start x, side) | "
              "corr(sign x, side) | redraws | note |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in diag:
        def num(v, f):
            return "—" if v == "" or v is None or (isinstance(v, float) and np.isnan(v)) \
                else format(float(v), f)
        note = SET_NOTE.get(r["set"], "random sides" if r["group"] == "S" else "sorted sides")
        lines.append(
            f"| {r['set']} | {int(r['n_demos'])} | {int(r['n_distinct'])} | "
            f"{num(r['start_sigma_cm'], 'g')} | {num(100 * float(r['sigma_onset']), '.3f')} | "
            f"{num(min(float(r['early_signal']), 1e3), '.2f')} | "
            f"{num(100 * float(r['sigma_exit']), '.2f')} | {num(r['d_exit'], '.3f')} | "
            f"{num(r['d_commit'], '.3f')} | {num(r['corr_start_x_side'], '+.2f')} | "
            f"{num(r['corr_start_sign_side'], '+.2f')} | {num(r['n_redraws'], 'g')} | {note} |")
    lines += ["", "![the demos' spread and separation over the first steps](fig_spread_profile.png)",
              "", "## Results", "",
              "**Deployment** is C's 1024 held-out starts; the score arms use λ = 0.5. Means "
              "± 95% t-intervals over 10 seeds.", "",
              "![deployment success against the start spread](fig_spread_deploy.png)", "",
              "![line sweep](fig_spread_sweep.png)", "",
              "![mixtures](fig_spread_mix.png)", "",
              "![deployment rollouts](fig_spread_rollouts.png)", "",
              "| set | arm | success % | collision % | undecided % | left share % | MMD²×10³ | "
              "train MSE×10³ | sweep: sorted / one-sided / interleaved / reversed | band w [cm] "
              "median (IQR) | sweep order | restart 0.3 / 0.5 % |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for key in keys:
        for method in METHODS:
            pats = [r["sweep_pattern"] for r in t.rows if r["set"] == key and r["method"] == method]
            w = 100 * t.get(key, method, "band_w")
            w = w[~np.isnan(w)]
            wtxt = (f"{np.median(w):.2f} ({np.percentile(w, 25):.2f}–{np.percentile(w, 75):.2f})"
                    if len(w) else "—")
            order = t.get(key, method, "sweep_order")
            order = order[~np.isnan(order)]
            order_txt = f"{np.median(order):+.2f}" if len(order) else "—"
            lines.append(
                f"| {key} | {SHORT[method]} | {fmt(t.get(key, method, 'success'))} | "
                f"{fmt(t.get(key, method, 'collision'))} | "
                f"{fmt(t.get(key, method, 'undecided'))} | "
                f"{fmt(t.get(key, method, 'left_share'))} | "
                f"{fmt(t.get(key, method, 'mmd2'), 1e3, 1)} | "
                f"{fmt(t.get(key, method, 'train_mse'), 1e3, 2)} | "
                f"{pats.count('sorted')} / {pats.count('one-sided')} / "
                f"{pats.count('interleaved')} / {pats.count('reversed')} | {wtxt} | "
                f"{order_txt} | {100 * t.get(key, method, 'restart_0.3').mean():.0f} / "
                f"{100 * t.get(key, method, 'restart_0.5').mean():.0f} |")
    lines += ["", "**Sweep against deployment (P7).** Deployment failure against the sweep's "
              "failures weighted by N(0, 8 cm); the sweep varies x only, deployment also y and "
              "heading.", "",
              "| set | " + " | ".join(f"{SHORT[m]}: deploy / sweep-pred. %" for m in METHODS)
              + " |", "|---|" + "---:|" * len(METHODS)]
    for key in keys:
        cells = [f"{100 * (1 - t.get(key, m, 'success').mean()):.1f} / "
                 f"{100 * t.get(key, m, 'sweep_pred_fail').mean():.1f}" for m in METHODS]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "**Planned comparisons** (deployment success, difference in points, Welch "
              "t-test over seeds; bold p < 0.05).", "",
              "| comparison | " + " | ".join(SHORT[m] for m in METHODS) + " |",
              "|---|" + "---:|" * len(METHODS)]
    pairs = [("S8", "T8"), ("S2", "T2")] + [(k, "S8") for k in keys if k not in ("S8",)]
    for a, b in pairs:
        if not (t.has(a) and t.has(b)):
            continue
        cells = [welch(t.get(a, m, "success"), t.get(b, m, "success"))
                 if (a, b) in (("S8", "T8"), ("S2", "T2")) or m == "fm" else ""
                 for m in METHODS]
        lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    lines += ["", "Score penalty against the same network without it, per set:", "",
              "| set | MLP+phase+score − MLP+phase | MLP+score − MLP |", "|---|---:|---:|"]
    for key in keys:
        lines.append(f"| {key} | {welch(t.get(key, 'score_w0.5', 'success'), t.get(key, 'fm', 'success'))}"
                     f" | {welch(t.get(key, 'mlp_score_w0.5', 'success'), t.get(key, 'mlp', 'success'))} |")
    lines.append("")
    findings = out / "START_SPREAD_FINDINGS.md"
    if findings.exists():
        lines[4:4] = [findings.read_text().strip(), ""]
    (out / "START_SPREAD.md").write_text("\n".join(lines))
    print(f"  wrote {out / 'START_SPREAD.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs/skate_deterministic"))
    parser.add_argument("--orig-runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("output/skate_start_spread"))
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-evaluate every policy")
    parser.add_argument("--sets", nargs="+", default=None, choices=list(SETS),
                        help="evaluate only these sets (the rest is read from the CSVs)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)

    diag_path, phase_path = args.out / "data_diag.csv", args.out / "data_diag_phase.csv"
    if args.plot_only and diag_path.exists():
        diag, phase_rows = read_csv(diag_path), read_csv(phase_path)
    else:
        diag, phase_rows = data_diagnostics()
        write_csv(diag_path, diag)
        write_csv(phase_path, phase_rows)
    deploy_path, sweep_path = args.out / "deploy.csv", args.out / "sweep.csv"
    recs = [] if args.force else read_csv(deploy_path)
    sweep = [] if args.force else read_csv(sweep_path)
    if not args.plot_only:
        new, new_sweep = evaluate_all(args, {(r["set"], r["method"], int(r["seed"]))
                                             for r in recs})
        if new:
            recs += new
            sweep += new_sweep
            write_csv(deploy_path, recs)
            write_csv(sweep_path, sweep)
    t = Table(recs)
    fig_profile(phase_rows, args.out)
    if recs:
        fig_deploy(t, args.out)
        fig_sweep(sweep, args.out)
        fig_mix(t, args.out)
        fig_rollouts(args, t, args.out)
        markdown(t, diag, args.out)


if __name__ == "__main__":
    main()
