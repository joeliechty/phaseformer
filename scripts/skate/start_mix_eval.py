"""How much start noise is needed: the MLP arms trained on start-noise mixtures, evaluated from
the single nominal start and from scenario C's held-out starts.

    python scripts/skate/start_mix_eval.py

The training sets go from ``dual_path_fixed`` (0%: one start, the left and right trajectory
repeated) through ``dual_path_mix05``, ``mix10`` and ``mix15`` (p% of the demos from ``dual_path_noisy_start``,
C's start noise with no via jitter; ``scripts/skate/build_start_mixes.py``) to
``dual_path_noisy_start`` itself (100%). Every trained policy (4 arms x 10 seeds per set) is rolled
out, clamped, from

* **the single start** (5.0, 0.9, π/2) -- one deterministic path per seed;
* **C's held-out starts** (``data/test/dual_path.npz``, 1024 of them) -- success and the modes'
  split;
* **the noisy-start held-out set's starts** -- the MMD of the rollouts to those held-out demos
  (phase mean, bandwidth 0.3 as in training). Its starts come from the same distribution as C's,
  and its demos are this task's reference: C's own demos end at scattered goals with free
  headings, so an MMD to them rewards scattered failures over clean arrivals heading up.

Success is reaching within 0.6 m of (5.0, 7.0) with no collision.

Writes ``output/skate_deterministic/start_mix.csv``, ``START_MIX.md`` and ``fig_mix_*``.
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
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cross_start_eval import NOMINAL, evaluate  # noqa: E402
from evaluate_fixed_start import METHODS, SHORT  # noqa: E402
from evaluate_policies import _save, figure_label, method_style  # noqa: E402
from fm_bc.skate.baselines import load_policy  # noqa: E402
from fm_bc.skate.dataset import TrajectoryDataset  # noqa: E402
from fm_bc.skate.evaluation import mmd  # noqa: E402
from fm_bc.skate.plots import COLLISION_COLOR, MODE_COLORS, _world_axes  # noqa: E402
from fm_bc.skate.scenarios import MIX_PERCENTS  # noqa: E402
from skate_planner.viz import PALETTE  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})

N = 1024
# % of noisy-start demos -> training set.
SETS = {0: "dual_path_fixed", **{p: f"dual_path_mix{p:02d}" for p in MIX_PERCENTS},
        100: "dual_path_noisy_start"}
# Evaluated the same way for the combined figure (scripts/skate/plot_p2p_combined.py), outside the
# sweep's own tables and curves: key -> (training set, which runs root).
EXTRA = {"via": ("dual_path_fixed_via", "runs"), "C": ("dual_path", "orig")}
# x position of each fraction on the plots: 0-15 to scale, 100 after a break.
XPOS = {p: (p if p <= 15 else 20) for p in SETS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs/skate_deterministic"))
    parser.add_argument("--orig-runs", type=Path, default=Path("runs"))
    parser.add_argument("--data", type=Path, default=Path("data/skate_deterministic/test"))
    parser.add_argument("--c-test", type=Path, default=Path("data/test/dual_path.npz"))
    parser.add_argument("--out", type=Path, default=Path("output/skate_deterministic"))
    args = parser.parse_args()

    ds = TrajectoryDataset.load(args.data / "dual_path_fixed.npz")  # world, dt, limits
    c_test = TrajectoryDataset.load(args.c_test)
    c_starts = c_test.poses[:, 0]
    ref = TrajectoryDataset.load(args.data / "dual_path_noisy_start.npz")
    ref_demos = torch.as_tensor(ref.poses, dtype=torch.float32)

    rows, single_paths = [], {}
    todo = [(pct, scen, args.runs) for pct, scen in SETS.items()] + [
        (key, scen, args.runs if root == "runs" else args.orig_runs)
        for key, (scen, root) in EXTRA.items()]
    for pct, scen, root in todo:
        for method in METHODS:
            for seed_dir in sorted((root / scen / f"n{N}" / method).glob("seed*")):
                if not (seed_dir / "metrics.json").exists():
                    continue
                model = load_policy(seed_dir / "final.pt")[0].float().eval()
                s = evaluate(model, NOMINAL[None], ds)
                c = evaluate(model, c_starts, ds)
                poses = torch.as_tensor(evaluate(model, ref.poses[:, 0], ds)["poses"],
                                        dtype=torch.float32)
                mmd_phase = [mmd(poses[:, t], ref_demos[:, t]) for t in range(poses.shape[1])]
                rows.append({
                    "pct_noisy": pct, "scenario": scen, "method": method,
                    "seed": int(seed_dir.name[4:]),
                    "single_success": int(s["success"][0]),
                    "single_into_block": int(s["into_block"][0]),
                    "c_success": float(c["success"].mean()),
                    "c_into_block": float(c["into_block"].mean()),
                    "c_collided": float(c["collided"].mean()),
                    "c_left_ok": float(np.mean((c["labels"] == 0) & c["success"])),
                    "c_right_ok": float(np.mean((c["labels"] == 1) & c["success"])),
                    "ref_mmd": float(np.mean(mmd_phase)),
                })
                single_paths.setdefault((pct, method), []).append(
                    (s["poses"][0], bool(s["collided"][0]), int(s["labels"][0]),
                     bool(s["success"][0])))
            print(f"{pct!s:>4} {method}: {len(single_paths.get((pct, method), []))} seeds")

    with (args.out / "start_mix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def get(pct, method, key):
        return np.array([r[key] for r in rows if r["pct_noisy"] == pct and r["method"] == method],
                        dtype=float)

    pcts = [p for p in SETS if any(r["pct_noisy"] == p for r in rows)]

    # ---- success against the fraction of noisy-start demos
    panels = [("single_success", "From the single start", "seeds reaching the goal [%]", 100, False),
              ("c_success", "From scenario C's held-out starts", "success [%]", 100, True),
              ("ref_mmd", "MMD to the noisy-start held-out demos", r"MMD$^2\times10^3$ (phase mean)",
               1e3, True)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    fig.subplots_adjust(wspace=0.28)
    rng = np.random.default_rng(0)
    for ax, (key, title, ylabel, scale, dots) in zip(axes, panels):
        for j, method in enumerate(METHODS):
            st = method_style(method, METHODS)
            xs = np.array([XPOS[p] for p in pcts]) + (j - 1.5) * 0.25
            means = np.array([scale * get(p, method, key).mean() for p in pcts])
            # Two pieces, so no line runs through the axis break before 100%.
            near = np.array([p <= 15 for p in pcts])
            ax.plot(xs[near], means[near], st["ls"], color=st["color"], lw=1.6, marker="o",
                    ms=3.5, label=figure_label(method), zorder=3)
            ax.plot(xs[~near], means[~near], "o", color=st["color"], ms=3.5, zorder=3)
            if dots:
                for x, p in zip(xs, pcts):
                    v = scale * get(p, method, key)
                    ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=4, color=st["color"],
                               alpha=0.35, lw=0, zorder=2)
        ax.set_xticks([XPOS[p] for p in pcts], [str(p) for p in pcts], fontsize=7)
        ax.axvspan(16.3, 18.7, color=PALETTE["surface"], zorder=4)  # the axis break before 100%
        ax.text(17.5, 0.0, "//", transform=ax.get_xaxis_transform(), ha="center", va="center",
                fontsize=8, color=PALETTE["ink_muted"], zorder=5)
        ax.set_xlabel("noisy-start demos in training [%]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if key != "ref_mmd":
            ax.set_ylim(-3, 103)
        else:
            ax.set_yscale("log")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=len(METHODS), fontsize=7.5, frameon=False)
    _save(fig, args.out, "fig_mix_success")

    # ---- the single-start paths, a row per fraction
    show = [p for p in (0, 5, 10, 15, 100) if p in pcts]
    fig, axes = plt.subplots(len(show), len(METHODS),
                             figsize=(2.3 * len(METHODS), 1.55 * len(show)), squeeze=False)
    for i, pct in enumerate(show):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            _world_axes(ax, ds)
            paths = single_paths.get((pct, method), [])
            for poses, collided, label, _ in paths:
                color = COLLISION_COLOR if collided else (MODE_COLORS[label % 2] if label >= 0
                                                          else PALETTE["ink_muted"])
                ax.plot(poses[:, 0], poses[:, 1], "-", lw=1.0, alpha=0.85, color=color)
            ok = sum(succ for *_, succ in paths)
            ax.set_title(f"{SHORT[method]} · {pct}%: {ok}/{len(paths)}", fontsize=7.5, loc="left")
            ax.set_xlabel(""), ax.set_ylabel("")
            ax.set_ylim(3.3, 8)
            ax.tick_params(labelsize=6)
    _save(fig, args.out, "fig_mix_single_paths")

    # ---- markdown
    def fmt(v, scale=100, d=0):
        return f"{scale * v.mean():.{d}f} ± {scale * v.std(ddof=1):.{d}f}" if len(v) > 1 else "—"

    lines = ["# How much start noise is needed (skate, point-to-point dual path)", "",
             "Generated by `scripts/skate/start_mix_eval.py`. Not part of the paper figures.", "",
             "**Superseded** by the start-spread experiment "
             "([START_SPREAD.md](../skate_start_spread/START_SPREAD.md)): spread varied with "
             "everything else fixed, random against sorted sides, evaluated on C's held-out starts "
             "and a lateral line sweep.", "",
             "Training sets: p% of the 1024 demos from the noisy-start set (C's start noise, no via "
             "jitter, goal pose (5.0, 7.0, π/2)), the rest copies of the two fixed trajectories, "
             "half per mode (`scripts/skate/build_start_mixes.py`). 0% is `dual_path_fixed`, 100% "
             "is `dual_path_noisy_start`. Four MLP arms × 10 seeds per set, n = 1024, 20k steps.",
             "",
             "* **single start** -- every policy from (5.0, 0.9, π/2): the number of the 10 seeds "
             "whose one deterministic path reaches within 0.6 m of the goal with no collision.",
             "* **scenario C** -- every policy from C's 1024 held-out starts: success (same "
             "criterion) and how the successes split between the modes.",
             "* **MMD** -- every policy from the noisy-start held-out set's starts (the same start "
             "distribution as C's), MMD×10³ to those held-out demos averaged over phase. Not "
             "against C's demos: they end at scattered goals with free headings, so an MMD to "
             "them rewards scattered failures over clean arrivals heading up.", "",
             "![success against noisy-start fraction](fig_mix_success.png)", "",
             "Every training set side by side -- the one-start sets, these mixtures, and all "
             "starts noisy -- from the single start (bars) and from C's held-out starts (violins "
             "over seeds; `scripts/skate/plot_p2p_combined.py`):", "",
             "![all training sets](fig_p2p_combined.png)", "",
             "| noisy % | arm | single start: seeds reaching | into block | C: success % | "
             "C: reached left/right % | MMD×10³ |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for pct in pcts:
        for method in METHODS:
            ss = get(pct, method, "single_success")
            lines.append(
                f"| {pct} | {figure_label(method)} | {int(ss.sum())}/{len(ss)} | "
                f"{int(get(pct, method, 'single_into_block').sum())}/{len(ss)} | "
                f"{fmt(get(pct, method, 'c_success'))} | "
                f"{100 * get(pct, method, 'c_left_ok').mean():.0f}/"
                f"{100 * get(pct, method, 'c_right_ok').mean():.0f} | "
                f"{fmt(get(pct, method, 'ref_mmd'), 1e3, 1)} |")
    # Score arm vs its unregularized twin, per fraction (Welch, C success over seeds).
    lines += ["", "Score regularizer against the same network without it, success from C's "
              "starts (Welch t-test over seeds; bold p < 0.05):", "",
              "| noisy % | MLP+phase+score − MLP+phase | MLP+score − MLP |", "|---:|---:|---:|"]
    for pct in pcts:
        cells = []
        for a, b in (("score_w0.5", "fm"), ("mlp_score_w0.5", "mlp")):
            x, y = 100 * get(pct, a, "c_success"), 100 * get(pct, b, "c_success")
            # Both arms at ~0 (every seed fails): a t-test of the leftover fractions of a
            # percent would be "significant" and meaningless.
            if (min(len(x), len(y)) < 2 or (np.ptp(x) == 0 and np.ptp(y) == 0)
                    or max(x.mean(), y.mean()) < 1.0):
                cells.append(f"{x.mean() - y.mean():+.1f} (n/a)")
                continue
            p = stats.ttest_ind(x, y, equal_var=False).pvalue
            ps = f"**{p:.2g}**" if p < 0.05 else f"{p:.2g}"
            cells.append(f"{x.mean() - y.mean():+.1f} ({ps})")
        lines.append(f"| {pct} | " + " | ".join(cells) + " |")
    lines += ["", "The single-start path of every seed, a row per mixture; the title counts the "
              "seeds that reach the goal with no collision (red: collided):", "",
              "![single-start paths](fig_mix_single_paths.png)", ""]
    findings = args.out / "START_MIX_FINDINGS.md"
    if findings.exists():
        lines[6:6] = [findings.read_text().strip(), ""]
    (args.out / "START_MIX.md").write_text("\n".join(lines))
    print(f"  wrote {args.out / 'START_MIX.md'}")


if __name__ == "__main__":
    main()
