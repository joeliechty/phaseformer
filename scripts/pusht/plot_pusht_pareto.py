"""Score against inference cost on Push-T: the Pareto frontier on CPU and on a CUDA-graphed GPU.

    python scripts/pusht/plot_pusht_pareto.py

Four models from the scale study (``compare_tf_scale.py``), each with every arm of the paper's
2 x 2 -- phase + score, phase, score, neither -- and the chunk baselines:

* **TF-large** (``_tflarge``) and **TF-med wide** (``_tfmed``), the phase arms on the large MLP
  clock (3 x 724);
* **MLP-med wide** (``_mlpmed``), on the med deep MLP clock (3 x 512);
* **MLP small**, the published MLP, on the paper's clock (2 x 512).

The phase arms run on the model's clock and are timed with it, a clock call and a policy call per
control step; the phase-free arms take no clock. An arm with the score regularizer runs the same
network as the one without, so they share a latency. Scores are the mean over 10 seeds of the
full-episode rollouts (``output/pusht_9010/{tf,mlp}_scale/scores.csv``); latency is the median
milliseconds per control step at batch 1 from ``benchmark_pusht_inference_grid.py``
(``output/pusht_9010/inference_grid.csv``). The baselines' open loop executes 8 steps per
decision, so its cost per step is the decision's over 8; ACT is shown closed loop only, as in the
paper.

The dashed line is the frontier over everything drawn; the vertical lines are the paper's control
budgets (60 Hz visual, 1 kHz tactile). Type is sized as the paper's other figures are
(``plot_pusht.TYPE_RATIOS``), at the width of ``fig_inference_pareto_both``. Writes
``fig_pusht_pareto.{png,pdf}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# (backbone, its clock, model name, clock name, colour, marker of its phase arms: TF diamonds,
# MLP circles; squares are the phase-free arms and triangles are kept for the baselines)
MODELS = [("tflarge", "large", "TF-large", "MLP clock large", "#08306b", "D"),
          ("tfmed", "large", "TF-med wide", "MLP clock large", "#3182bd", "D"),
          ("mlpmed", "d3", "MLP-med wide", "MLP clock med deep", "#e6550d", "o"),
          ("mlp", "512", "MLP small", "MLP clock small", "#fdae6b", "o")]
# (arm, label, uses the phase and so the clock, has the score regularizer)
ARMS = [("score_w2.5_const", "phase + score", True, True), ("fm", "phase", True, False),
        ("mlp_score_w2.5_const", "score", False, True), ("mlp", "neither", False, False)]
BASE_COLOUR = {"dp": "#000000", "act": "#666666", "mip": "#b0b0b0"}
# The paper's type scale (plot_pusht.TYPE_RATIOS / type_pt: points = ratio x 72 x figure width),
# restated as plot_pusht does, at the width fig_inference_pareto_both is drawn
# (plot_inference_combined: TEXT_WIDTH 7.16 x 1.35).
TYPE_RATIOS = {"title": 0.0151, "label": 0.0144, "legend": 0.0126, "tick": 0.0122,
               "note": 0.0115, "data": 0.0112}
WIDTH = 7.16 * 1.35
CONTROL_BUDGETS = ((60.0, "60 Hz visual"), (1000.0, "1 kHz tactile"))


def pt(role: str) -> float:
    return TYPE_RATIOS[role] * 72 * WIDTH


def frontier(points: list[dict], measure: str) -> list[dict]:
    """The points no faster point beats on score, fastest first."""
    out, best = [], -1.0
    for q in sorted(points, key=lambda q: (q["ms"][measure], -q["score"])):
        if q["score"] > best:
            out.append(q)
            best = q["score"]
    return out


def load_points(timings: Path, scores: list[Path], runs: Path) -> list[dict]:
    t = {(r["backbone"], r["clock"]): r for r in csv.DictReader(timings.open())}
    per: dict = {}
    for f in scores:
        for r in csv.DictReader(f.open()):
            per.setdefault((r["arm"], r["backbone"], r["clock"]), []).append(
                float(r["mean_score"]))
    score = {k: st.mean(v) for k, v in per.items()}
    pts = []
    for b, c, name, _, colour, mk in MODELS:
        for arm, label, phase, has_score in ARMS:
            # The phase-free arms ignore the clock: their rollouts are the paper clock's.
            timing = t[(b, c if phase else "none")]
            pts.append(dict(model=name, arm=label, colour=colour, marker=mk if phase else "s",
                            filled=has_score, score=score[(arm, b, c if phase else "512")],
                            ms={m: float(timing[m]) for m in ("cpu", "gpu_graph")}))
    for k, name in (("dp", "DP"), ("mip", "MIP"), ("act", "ACT")):
        for loop, tag, steps in (("open", "chunk", 8), ("closed", "chunk_te0.01", 1)):
            if (k, loop) == ("act", "open"):  # the paper reports ACT closed loop only
                continue
            s = st.mean(json.loads(f.read_text())["mean_score"]
                        for f in sorted((runs / k).glob(f"seed*/rollout_{tag}.json")))
            pts.append(dict(model=name, arm=f"{loop} loop", colour=BASE_COLOUR[k],
                            marker="^" if loop == "open" else "v", filled=True, score=s,
                            base=True, ms={m: float(t[(k, "chunk")][m]) / steps
                                           for m in ("cpu", "gpu_graph")}))
    return pts


def figure(pts: list[dict], out: Path, name: str) -> None:
    plt.rcParams.update({"xtick.labelsize": pt("tick"), "ytick.labelsize": pt("tick")})
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.9))
    for ax, (m, title) in zip(axes, (("cpu", "CPU (Ryzen 7 5700X, 8 threads)"),
                                     ("gpu_graph", "GPU, CUDA graph (RTX 4090)"))):
        for p in pts:
            ax.scatter(p["ms"][m], p["score"], marker=p["marker"],
                       s=34 if p.get("base") else 26, zorder=3, linewidths=1.0,
                       facecolors=p["colour"] if p["filled"] else "white",
                       edgecolors=p["colour"])
        f = frontier(pts, m)
        ax.plot([p["ms"][m] for p in f], [p["score"] for p in f], "k--", lw=0.7,
                drawstyle="steps-post", zorder=2)
        for hz, label in CONTROL_BUDGETS:
            ax.axvline(1000.0 / hz, ls=(0, (4, 3)), lw=0.8, color="#777777", zorder=1)
            ax.text(1000.0 / hz / 1.1, 0.993, f"{label}\n{1000.0 / hz:.3g} ms",
                    fontsize=pt("note"), color="#555555", va="top", ha="right")
        ax.set_xscale("log")
        ax.set_ylim(0.36, 1.0)
        ax.set_xlim(min(p["ms"][m] for p in pts) / 1.6, max(p["ms"][m] for p in pts) * 1.6)
        ax.set_title(title, fontsize=pt("title"))
        ax.set_xlabel("ms per control step (log)", fontsize=pt("label"))
        ax.grid(alpha=0.3, which="both", lw=0.5)
    axes[0].set_ylabel("mean score (10 seeds)", fontsize=pt("label"))

    # Legend: one column per model -- its name and its clock as two header lines, then its four
    # arms -- and a column of baselines under a one-line header. Six one-line rows in every
    # column, so the legend (which fills column by column) needs no padding and the rows line up.
    def header(text):
        return Line2D([], [], ls="", label=text)

    def entry(p):
        return Line2D([], [], ls="", marker=p["marker"], ms=6, mew=1.0, color=p["colour"],
                      mfc=p["colour"] if p["filled"] else "white", label=p["arm"])

    handles, headers = [], []
    for _, _, model, clock, _, _ in MODELS:
        headers += [len(handles), len(handles) + 1]
        handles += [header(model), header(f"({clock})")]
        handles += [entry(p) for p in pts if p["model"] == model and not p.get("base")]
    headers.append(len(handles))
    handles += [header("baselines")]
    handles += [Line2D([], [], ls="", marker=p["marker"], ms=6, color=p["colour"],
                       label=f"{p['model']}, {p['arm']}") for p in pts if p.get("base")]
    assert len(handles) == 30  # 5 columns x 6 rows
    leg = fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0),
                     fontsize=pt("legend"), title_fontsize=pt("legend"), handletextpad=0.3,
                     handlelength=1.2, columnspacing=1.0, borderpad=0.4, labelspacing=0.35)
    for i in headers:
        leg.get_texts()[i].set_fontweight("bold")
    fig.tight_layout(rect=(0, 0.222, 1, 1), w_pad=1.0, pad=0.3)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out / name}.{{png,pdf}}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--timings", type=Path, default=Path("output/pusht_9010/inference_grid.csv"))
    p.add_argument("--scores", type=Path, nargs="+",
                   default=[Path("output/pusht_9010/tf_scale/scores.csv"),
                            Path("output/pusht_9010/mlp_scale/scores.csv")])
    p.add_argument("--runs", type=Path, default=Path("runs/pusht/policy"),
                   help="the baselines' rollouts")
    p.add_argument("--out", type=Path, default=Path("output/pusht_9010/figures"))
    p.add_argument("--name", default="fig_pusht_pareto")
    args = p.parse_args()
    pts = load_points(args.timings, args.scores, args.runs)
    figure(pts, args.out, args.name)
    for m in ("cpu", "gpu_graph"):
        print(f"  {m} frontier: "
              + " -> ".join(f"{p['model']} {p['arm']}" for p in frontier(pts, m)))


if __name__ == "__main__":
    main()
