"""Cross-evaluate the point-to-point policies: every trained policy rolled out from the *same*
starts, whatever its training set's starts were.

    python scripts/skate/cross_start_eval.py

Each set's own evaluation starts from its held-out set's starts -- one pose for ``fixed`` and
``+via``, C's start distribution for ``+start`` and C -- so "trained on noisy starts" and
"evaluated from noisy starts" move together. Here every policy of every set
(``evaluate_fixed_start.SETS`` x ``METHODS`` x seeds) is rolled out, clamped as in training, under
three start conditions shared by all of them:

* **single** -- the nominal start (5.0, 0.9, π/2), once (a deterministic policy drives one path);
* **dist**   -- 1024 starts drawn from C's start distribution (8 cm, 0.05 rad), one fixed draw;
* **offset** -- the nominal start shifted in x by each of ``OFFSETS``, heading unchanged.

Success is reaching within 0.6 m of (5.0, 7.0) -- every set's goal (C's is its mean) -- with no
collision. Writes ``output/skate_deterministic/cross_start.csv`` and ``fig_p2p_cross_*`` and prints
the tables that ``SUMMARY.md`` quotes.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_fixed_start import (  # noqa: E402
    BOX, METHODS, ORIG, SET_COLORS, SET_SHORT, SETS, SHORT, into_block,
)
from evaluate_policies import _save  # noqa: E402
from fm_bc.skate.baselines import load_policy  # noqa: E402
from fm_bc.skate.dataset import TrajectoryDataset  # noqa: E402
from fm_bc.skate.evaluation import collision_breakdown, rollout_policy  # noqa: E402
from fm_bc.skate.scenarios import get_scenario  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})

N = 1024
NOMINAL = np.array([5.0, 0.9, np.pi / 2])
GOAL, REACH = np.array([5.0, 7.0]), 0.6
OFFSETS = np.array([-0.16, -0.08, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08, 0.16])
N_DIST = 1024


@torch.no_grad()
def evaluate(model, starts: np.ndarray, ds: TrajectoryDataset) -> dict[str, np.ndarray]:
    """Per-rollout outcomes from ``starts (M, 3)``."""
    dtype = next(model.parameters()).dtype
    poses, controls, _ = rollout_policy(model, torch.as_tensor(starts, dtype=dtype),
                                        ds.controls.shape[1], ds.dt, ds.limits)
    ob, bd = collision_breakdown(poses.double(), controls.double(), ds.dt, ds.spec.world,
                                 float(ds.config.get("radius", 0.3)))
    poses = poses.numpy()
    collided = np.asarray(ob) | np.asarray(bd)
    reach = np.linalg.norm(poses[:, -1, :2] - GOAL, axis=-1) <= REACH
    labels = ds.spec.label_mode(poses)
    return {"success": reach & ~collided, "collided": collided, "labels": labels,
            "into_block": into_block(poses, BOX) & collided, "poses": poses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs/skate_deterministic"))
    parser.add_argument("--orig-runs", type=Path, default=Path("runs"))
    parser.add_argument("--data", type=Path, default=Path("data/skate_deterministic/test"))
    parser.add_argument("--out", type=Path, default=Path("output/skate_deterministic"))
    args = parser.parse_args()

    # Any point-to-point held-out set carries the world, dt, horizon and limits (all shared).
    ds = TrajectoryDataset.load(args.data / "dual_path_fixed.npz")
    dist = get_scenario(ORIG).start.sample(N_DIST, np.random.default_rng(1234))
    single = NOMINAL[None]
    offsets = NOMINAL[None] + np.outer(OFFSETS, [1.0, 0.0, 0.0])

    rows, per = [], {}
    for scen in SETS:
        root = args.orig_runs if scen == ORIG else args.runs
        for method in METHODS:
            for seed_dir in sorted((root / scen / f"n{N}" / method).glob("seed*")):
                if not (seed_dir / "metrics.json").exists():
                    continue
                model = load_policy(seed_dir / "final.pt")[0].float().eval()
                s, d, o = evaluate(model, single, ds), evaluate(model, dist, ds), \
                    evaluate(model, offsets, ds)
                row = {"scenario": scen, "method": method, "seed": int(seed_dir.name[4:]),
                       "single_success": int(s["success"][0]),
                       "single_into_block": int(s["into_block"][0]),
                       "single_side": int(s["labels"][0]),
                       "dist_success": float(d["success"].mean()),
                       "dist_into_block": float(d["into_block"].mean()),
                       "dist_left": float(np.mean(d["labels"] == 0)),
                       "dist_right": float(np.mean(d["labels"] == 1)),
                       "dist_left_ok": float(np.mean((d["labels"] == 0) & d["success"])),
                       "dist_right_ok": float(np.mean((d["labels"] == 1) & d["success"]))}
                for off, ok in zip(OFFSETS, o["success"]):
                    row[f"offset_{100 * off:+.0f}cm"] = int(ok)
                rows.append(row)
                per.setdefault((scen, method), []).append({"single": s, "dist": d, "offset": o})
            print(f"{scen} {method}: {len(per.get((scen, method), []))} seeds")

    with (args.out / "cross_start.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- tables
    def agg(scen, method, key):
        v = np.array([r[key] for r in rows if r["scenario"] == scen and r["method"] == method],
                     dtype=float)
        return v

    table = ["| trained on | arm | seeds | single start: seeds reaching goal | into block | "
             "start dist.: success % | reached left/right % | own eval. success % |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    own = {}
    for scen in SETS:
        root = args.orig_runs if scen == ORIG else args.runs
        for method in METHODS:
            m = [json.loads(p.read_text())["success_rate"]
                 for p in sorted((root / scen / f"n{N}" / method).glob("seed*/metrics.json"))]
            own[(scen, method)] = 100 * np.mean(m) if m else np.nan
    for scen in SETS:
        for method in METHODS:
            ss = agg(scen, method, "single_success")
            if not len(ss):
                continue
            ds_ = 100 * agg(scen, method, "dist_success")
            lo, ro = 100 * agg(scen, method, "dist_left_ok"), 100 * agg(scen, method, "dist_right_ok")
            table.append(f"| {SET_SHORT[scen]} | {SHORT[method]} | {len(ss)} | {int(ss.sum())}/{len(ss)} | "
                  f"{int(agg(scen, method, 'single_into_block').sum())}/{len(ss)} | "
                  f"{ds_.mean():.0f} ± {ds_.std(ddof=1):.0f} | {lo.mean():.0f}/{ro.mean():.0f} | "
                  f"{own[(scen, method)]:.0f} |")
    print("\n".join(table))
    md = ["## Cross-evaluation: every policy from the same starts", "",
          "Each set's own evaluation starts from its held-out starts, so training-start and "
          "evaluation-start spread move together. Here every trained policy is rolled out "
          "(`scripts/skate/cross_start_eval.py`) from **the single nominal start** "
          "(5.0, 0.9, π/2) -- one deterministic path per seed -- and from **1024 starts drawn "
          "from C's start distribution** (8 cm, 0.05 rad; one draw shared by every policy). "
          "Success is reaching within 0.6 m of (5.0, 7.0) with no collision. **own eval.** is "
          "the set's own evaluation, for reference.", "", *table, "",
          "![cross success](fig_p2p_cross_success.png)", "",
          "From the nominal start shifted in x (heading unchanged), the share of the 10 seeds "
          "that reach the goal; 0 is the exact centre line, where the two modes are "
          "symmetric:", "", "![offsets](fig_p2p_cross_offset.png)", ""]
    (args.out / "CROSS.md").write_text("\n".join(md))
    # ---- figures
    fig, axes = plt.subplots(1, len(METHODS), figsize=(2.3 * len(METHODS), 2.3), sharey=True)
    for ax, method in zip(axes, METHODS):
        for scen in SETS:
            g = per.get((scen, method))
            if not g:
                continue
            ok = 100 * np.mean([x["offset"]["success"] for x in g], axis=0)
            ax.plot(100 * OFFSETS, ok, "o-", ms=3, lw=1.3, color=SET_COLORS[scen],
                    label=SET_SHORT[scen])
        ax.set_title(SHORT[method], fontsize=8.5, loc="left")
        ax.set_xlabel("start x offset [cm]")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xticks([-16, -4, -1, 0, 1, 4, 16], ["-16", "-4", "-1", "0", "1", "4", "16"])
    axes[0].set_ylabel("seeds reaching the goal [%]")
    axes[-1].legend(title="trained on", fontsize=7, title_fontsize=7, frameon=False)
    _save(fig, args.out, "fig_p2p_cross_offset")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5), sharey=True)
    width = 0.2
    for k, scen in enumerate(SETS):
        xs = np.arange(len(METHODS)) + (k - 1.5) * width
        single = [100 * agg(scen, m, "single_success").mean() if len(agg(scen, m, "single_success"))
                  else np.nan for m in METHODS]
        distv = [100 * agg(scen, m, "dist_success").mean() if len(agg(scen, m, "dist_success"))
                 else np.nan for m in METHODS]
        axes[0].bar(xs, single, width, color=SET_COLORS[scen], label=SETS[scen])
        axes[1].bar(xs, distv, width, color=SET_COLORS[scen])
    axes[0].set_title("from the single nominal start (share of seeds)", fontsize=8.5, loc="left")
    axes[1].set_title("from C's start distribution (1024 starts)", fontsize=8.5, loc="left")
    for ax in axes:
        ax.set_xticks(range(len(METHODS)), [SHORT[m] for m in METHODS], fontsize=7)
    axes[0].set_ylabel("success [%]")
    axes[0].set_ylim(0, 100)
    axes[0].legend(title="trained on", fontsize=6.5, title_fontsize=7, frameon=False,
                   loc="upper left")
    _save(fig, args.out, "fig_p2p_cross_success")


if __name__ == "__main__":
    main()
