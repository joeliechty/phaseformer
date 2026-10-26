"""How early does a demonstration's state give away its mode?

    python scripts/skate/mode_separability.py --data data/ --out output/skate/scenarios/ \
        [--baseline data/v1/]

At every phase ``k / N`` this classifies each demonstration's mode from its pose alone -- the
state a phase-free policy sees, embedded exactly as :class:`fm_bc.skate.policy.FlowMLP` does
(``[(x - cx)/hx, (y - cy)/hy, cos theta, sin theta]``) -- by a leave-one-out ``k``-nearest-
neighbour vote among the other demonstrations at the same phase.

Accuracy near 0.5 (chance, for balanced modes) means left- and right-bound demos occupy the
same states: a deterministic state-feedback policy there sees one input with two conflicting
targets and regresses to their mean. Accuracy near 1 means the state already encodes the
mode, so plain regression never has to average anything. The phase at which accuracy first
exceeds ``--threshold`` is the **commit phase**: how far into the task the modes stay mixed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fm_bc.skate.dataset import TrajectoryDataset  # noqa: E402
from fm_bc.skate.scenarios import BOUNDS, SCENARIOS  # noqa: E402
from skate_planner.viz import PALETTE  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
COLORS = {"single_path": "#2a78d6", "dual_goal": "#eb6834", "dual_path": "#1baf7a"}
TITLES = {"single_path": "A · single path", "dual_goal": "B · dual goal",
          "dual_path": "C · dual path"}


def embed(poses: np.ndarray) -> torch.Tensor:
    """The phase-free policy's input features, ``(..., 4)``."""
    x0, y0, x1, y1 = BOUNDS
    p = torch.as_tensor(poses, dtype=torch.double)
    xy = (p[..., :2] - torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2], dtype=torch.double)) / \
        torch.tensor([(x1 - x0) / 2, (y1 - y0) / 2], dtype=torch.double)
    th = p[..., 2:3]
    return torch.cat([xy, torch.cos(th), torch.sin(th)], dim=-1)


def separability(ds: TrajectoryDataset, k: int = 10) -> np.ndarray:
    """Leave-one-out kNN mode accuracy at every phase, ``(N + 1,)``."""
    feats = embed(ds.poses)  # (M, N+1, 4)
    mode = torch.as_tensor(ds.mode)
    n_modes = int(mode.max()) + 1
    acc = []
    for t in range(feats.shape[1]):
        d = torch.cdist(feats[:, t], feats[:, t])
        d.fill_diagonal_(float("inf"))
        nn = d.topk(k, largest=False).indices  # (M, k)
        votes = torch.nn.functional.one_hot(mode[nn], n_modes).sum(dim=1)
        acc.append(float((votes.argmax(dim=1) == mode).double().mean()))
    return np.array(acc)


def commit_phase(acc: np.ndarray, threshold: float) -> float:
    above = np.flatnonzero(acc >= threshold)
    return float(above[0] / (len(acc) - 1)) if len(above) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("output/skate/scenarios"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--label", default="", help="suffix for the figure name")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="another data dir (e.g. a previous scenario layout), drawn dashed")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sources = [(args.data, "", "-", 1.8)]
    if args.baseline is not None:
        sources.append((args.baseline, ", previous layout", "--", 1.1))
    fig, ax = plt.subplots(figsize=(4.2, 2.6), facecolor=PALETTE["surface"])
    print(f"{'scenario':<32}{'commit phase':>13}  accuracy at phase 0 / 0.2 / 0.4 / 0.6")
    for data_dir, tag, ls, lw in sources:
        for name in SCENARIOS:
            path = data_dir / f"{name}.npz"
            if not path.exists():
                continue
            ds = TrajectoryDataset.load(path)
            if ds.mode.max() == 0:
                continue  # one mode: nothing to separate
            acc = separability(ds, args.k)
            phase = np.linspace(0, 1, len(acc))
            c = commit_phase(acc, args.threshold)
            at = [acc[int(round(p * (len(acc) - 1)))] for p in (0, 0.2, 0.4, 0.6)]
            print(f"{name + tag:<32}{c:>13.2f}  " + " / ".join(f"{a:.2f}" for a in at))
            ax.plot(phase, acc, color=COLORS[name], lw=lw, ls=ls,
                    label=f"{TITLES[name]}{tag} (commit {c:.2f})")
    ax.axhline(0.5, color=PALETTE["ink_muted"], lw=0.8, ls=":")
    ax.axhline(args.threshold, color=PALETTE["ink_muted"], lw=0.6, ls="--")
    ax.text(0.99, 0.51, "chance", ha="right", va="bottom", fontsize=7, color=PALETTE["ink_muted"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.02)
    ax.set_facecolor(PALETTE["surface"])
    ax.grid(True, color=PALETTE["grid"], lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("phase")
    ax.set_ylabel(f"mode accuracy from pose ({args.k}-NN)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    name = f"fig_mode_separability{args.label}"
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"{name}.{ext}", dpi=200, facecolor=PALETTE["surface"],
                    bbox_inches="tight")
    print(f"wrote {args.out / name}.{{pdf,png}}")


if __name__ == "__main__":
    main()
