"""Training curves for every model in the two headline violin figures.

    python scripts/combined/plot_training_curves.py

Two figures, one per task, each drawing the same models its violin figure does:

* ``fig_policy_training_curves_n{n}`` (skate nav, ``output/skate/policy/``) -- the arms of
  ``evaluate_policies.VIOLIN_METHODS`` (MLP, MLP+score, MLP+phase, MLP+phase+score and the
  three generative baselines) over the three scenarios. A temporally ensembled re-evaluation
  (``*_te0.01``) is the same trained weights as its original, so it has no curve of its own.
* ``fig_pusht_training_curves`` (Push-T, ``output/pusht_9010/figures/``) -- the arms of
  ``plot_pusht._backbone_groups()``: four arms on the MLP backbone, the same four on the
  transformer, and the three chunk baselines. The violin's open/closed-loop baseline pairs are
  two rollouts of one training run, so each baseline is drawn once.
* ``fig_pusht_training_curves_pareto`` (Push-T) -- the same panels for the four models of the
  Pareto figure (``plot_pusht_pareto.MODELS``: MLP small, MLP-med wide, TF-med wide, TF-large),
  a column each, with the arms named as that figure's legend names them.

**Nothing here is comparable across loss families.** The nav arms regress a body twist, the
Push-T arms a phase velocity, Diffusion Policy denoises, MIP regresses twice and ACT pays
L1 + 10 KL -- four objectives in four units. The curves are therefore drawn on a log axis and
labelled as each model's *own* objective; what a reader should take from a baseline's curve is
its shape and where it stops falling, never its height against an arm's. The quantities that
*are* comparable get their own panels: rollout MMD for nav (the metric every arm is scored on)
and the phase-velocity validation loss shared by all eight Push-T arms.

The two tasks' budgets differ inside a figure as well as between them: the nav arms train for
20k steps and the nav baselines for 100k, on one step axis. Push-T is uniform -- 500 epochs for
every model, DP's low-dim recipe.

Both figures read the run directories the published violins were drawn from; nothing is
retrained or re-rolled, so a curve that disagrees with a violin means one of the two is stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skate"))  # evaluate_policies
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pusht"))  # plot_pusht

import evaluate_policies as ep  # noqa: E402
import plot_pusht as pt  # noqa: E402
import plot_pusht_pareto as pp  # noqa: E402

PALETTE = ep.PALETTE
TEXT_WIDTH = ep.TEXT_WIDTH

# The nav models the n=1024 violin draws, in its order. The `_te0.01` re-evaluations it also
# shows are not trained runs -- they replay their original's weights -- so they are not here.
NAV_METHODS = ["mlp", "mlp_score_w0.5", "fm", "score_w0.5", "dp", "mip", "act"]
# Each nav family's own training objective, for the caption: the arms' regression loss is the
# only one of the four in the figure's own units.
NAV_LOSSES = {"mlp": "velocity MSE", "fm": "velocity MSE", "dp": "denoising MSE",
              "mip": "regression MSE", "act": r"L1 + 10 KL"}
# Push-T: the three chunk baselines, drawn once each (the violin's open/closed pairs are two
# rollouts of one run). Colours are plot_pusht.METHODS' open-loop entries.
PUSHT_BASELINES = ["dp", "mip", "act"]
# Where each Push-T column's legend sits in its validation panel. The top right is free in all
# three: the arms dip early and climb back only to the middle of their range, and the baselines
# only fall.
LEGEND_LOC = ["upper right", "upper right", "upper right"]
# A log axis cannot draw a band that reaches zero; the mean - std edge is clipped to this
# fraction of the mean instead, which keeps the band's upper edge exact and its lower edge
# visibly open rather than silently truncated at some absolute floor.
BAND_FLOOR = 0.02


# --------------------------------------------------------------------------- data
def _log(path: Path):
    """One run's ``log.csv`` as a structured array (``evaluate_policies.load_runs``' reader)."""
    return np.genfromtxt(path, delimiter=",", names=True)


def _seed_logs(method_dir: Path) -> list:
    return [_log(p) for p in sorted(method_dir.glob("seed*/log.csv"))]


def _stack(logs: list, column: str) -> np.ndarray | None:
    """``(seeds, steps)`` of ``column``, or None if no run logs it.

    Runs of one method share a schedule, so their logs share a length; a short one (a run still
    being written) is dropped rather than padded.
    """
    if not logs or column not in logs[0].dtype.names:
        return None
    n = max(len(x[column]) for x in logs)
    rows = [x[column] for x in logs if len(x[column]) == n]
    return np.asarray(rows) if rows else None


def _band(ax, x, series: np.ndarray, style: dict, label: str | None, lw: float = 1.1,
          log: bool = True) -> None:
    """``evaluate_policies._band`` with the lower edge kept positive for a log axis."""
    mu = np.nanmean(series, axis=0)
    ax.plot(x, mu, lw=lw, label=label, **style)
    if series.shape[0] > 1:
        sd = np.nanstd(series, axis=0)
        lo = mu - sd
        if log:
            lo = np.maximum(lo, np.abs(mu) * BAND_FLOOR)
        ax.fill_between(x, lo, mu + sd, color=style["color"], alpha=0.15, lw=0)


def _match_ylim(*axes, log: bool = True, pad: float = 0.06) -> None:
    """Put every axis on the union of what they draw, with a margin -- ``sharey`` would take the
    first one's limits instead and clip the rest."""
    lo = min(min(np.nanmin(line.get_ydata()) for line in ax.get_lines() if len(line.get_ydata()))
             for ax in axes)
    hi = max(max(np.nanmax(line.get_ydata()) for line in ax.get_lines() if len(line.get_ydata()))
             for ax in axes)
    if log:
        span = np.log10(hi / lo)
        lo, hi = lo * 10 ** (-pad * span), hi * 10 ** (pad * span)
    else:
        lo, hi = lo - pad * (hi - lo), hi + pad * (hi - lo)
    for ax in axes:
        ax.set_ylim(lo, hi)


# --------------------------------------------------------------------------- nav
def nav_logs(root: Path, n: int, scenarios, methods) -> dict:
    """``{(scenario, method): [log, ...]}`` for the runs behind the violin."""
    out = {}
    for scenario in scenarios:
        for method in methods:
            logs = _seed_logs(root / scenario / f"n{n}" / method)
            if logs:
                out[(scenario, method)] = logs
    return out


def fig_nav(root: Path, out: Path, n: int = 1024, methods=None, x: str = "progress") -> None:
    """A column per scenario; rollout MMD over training above each model's own training loss.

    The MMD row is the comparison: every model is scored on it, at the same 20 evaluation
    points of its own schedule. The loss row is four objectives on one log axis and is only to
    be read shape-wise -- see the module docstring.

    ``x="progress"`` draws both rows against the fraction of each model's own budget, which is
    what makes one axis legitimate here: the arms train for 20k steps and the baselines for
    100k, but both are evaluated at the same twenty fractions (5 % to 100 %), so the eval points
    line up exactly. ``x="step"`` and ``x="logstep"`` draw the true step instead -- honest about
    the 5x budget difference, at the cost of pressing every arm into the left fifth of the panel
    (linear) or every eval point into its right quarter (log).
    """
    methods = methods or NAV_METHODS
    scenarios = [s for s in ep.SCENARIOS]
    logs = nav_logs(root, n, scenarios, methods)
    scenarios = [s for s in scenarios if any((s, m) in logs for m in methods)]
    if not scenarios:
        print("  no nav runs found")
        return
    drawn = [m for m in methods if any((s, m) in logs for s in scenarios)]
    k = len(scenarios)
    legend_rows = max(1, -(-len(drawn) // 4))
    fig, axes = plt.subplots(2, k, figsize=(TEXT_WIDTH, 3.5 + 0.2 * legend_rows),
                             facecolor=PALETTE["surface"], squeeze=False, sharex=True)
    seeds = sorted({len(v) for v in logs.values()})
    settled: list[float] = []
    for i, scenario in enumerate(scenarios):
        for method in drawn:
            group = logs.get((scenario, method))
            if not group:
                continue
            style = ep.method_style(method, methods)
            label = ep.figure_label(method)
            step = group[0]["step"]
            axis = step / step[-1] if x == "progress" else step
            mmd = _stack(group, "mmd_mean")
            if mmd is not None:
                evald = np.isfinite(mmd[0])
                _band(axes[0, i], axis[evald], mmd[:, evald] * 1e3, style, label)
            # The arms log the regression loss apart from the regularizer; a baseline logs only
            # the objective it is actually trained on.
            loss = _stack(group, "velocity_loss")
            if loss is None or not np.isfinite(loss).any():
                loss = _stack(group, "total_loss")
            if loss is not None:
                _band(axes[1, i], axis, loss, style, None, lw=0.9)
                # The loss row's range is set from the curves *after* the first percent of
                # training: every objective falls by one to three decades in its first few
                # logged steps, and letting those points set the top leaves most of the panel
                # empty. The transients run off the top edge instead.
                tail = np.nanmean(loss, axis=0)[axis >= 0.01 * axis[-1]]
                if tail.size:
                    settled += [np.nanmin(tail), np.nanmax(tail)]
        axes[0, i].set_title(ep.TITLES[scenario], fontsize=9, color=PALETTE["ink"])
        for row, ax in enumerate(axes[:, i]):
            ax.set_yscale("log")
            if x == "logstep":
                ax.set_xscale("log")
            label_x = {"progress": "training progress", "step": "training step",
                       "logstep": "training step"}[x]
            ep._style(ax, label_x if row == 1 else None, size=8)
            if x == "progress":
                ax.set_xlim(0, 1)
                ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            else:
                ax.xaxis.set_major_formatter(
                    lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
            if i:
                ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel(r"rollout MMD$^2\times10^3$", fontsize=8, color=PALETTE["ink"])
    axes[1, 0].set_ylabel("training loss (own objective)", fontsize=8, color=PALETTE["ink"])
    # One y range per row, the union of the three scenarios' (sharey would take A's and clip
    # whichever of B and C runs past it).
    _match_ylim(*axes[0], log=True)
    if settled:
        lo, hi = min(settled), max(settled)
        span = np.log10(hi / lo)
        for ax in axes[1]:
            ax.set_ylim(lo * 10 ** (-0.06 * span), hi * 10 ** (0.12 * span))
    ep._fig_legend(fig, axes, ncol=min(4, len(drawn)), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.055 * legend_rows))
    ep._save(fig, out, f"fig_policy_training_curves_n{n}", tight=False,
             caption=_nav_caption(n, seeds, drawn, x))
    print(f"  nav: {len(drawn)} models x {len(scenarios)} scenarios")


def _nav_caption(n: int, seeds: list[int], drawn: list[str], x: str) -> str:
    seed_text = str(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}--{seeds[-1]}"
    families = ", ".join(f"{ep.figure_label(m)} {NAV_LOSSES[ep.parse_method(m)['baseline'] or m]}"
                         for m in drawn if ep.parse_method(m)["baseline"] is not None)
    return (
        f"Skate navigation, $n = {n}$ demonstrations, mean over {seed_text} seeds (band: "
        r"$\pm 1$ s.d.). \textbf{Top}: squared MMD of the 1024 held-out rollouts against the "
        "held-out demonstrations, at the 20 evaluation points of each run's schedule. Log "
        "scale on both rows. "
        r"\textbf{Bottom}: each model's own training objective -- the arms' "
        "velocity MSE (the regression term alone, without the regularizer), " + families +
        ". The four objectives are in different units and are not comparable by height; only "
        "their shape is. " + (
            "The arms train for 20k steps and the generative baselines for 100k, so both axes "
            "are the fraction of each model's own budget; the twenty evaluation points fall at "
            "the same fractions either way."
            if x == "progress" else
            "The arms train for 20k steps and the generative baselines for 100k, which is why "
            "the arms' curves stop short.")
    )


# --------------------------------------------------------------------------- Push-T
def pusht_entries():
    """``[(method, label, colour, ls), ...]``: the MLP arms, their transformer twins, then the
    chunk baselines -- ``plot_pusht._backbone_groups()`` with the violin's closed-loop
    duplicates dropped, since a pair is two rollouts of one training run."""
    groups = pt._backbone_groups()
    arms = [e for e in groups if e[0] not in pt.BASELINE_TAGS]
    base = {e[0]: e for e in groups if e[0] in pt.BASELINE_TAGS}
    return arms, [base[m] for m in PUSHT_BASELINES if m in base]


def fig_pusht(root: Path, out: Path) -> None:
    """A column per backbone (MLP, transformer, chunk baselines); the training loss above the
    validation loss the checkpoint is selected on, with that checkpoint marked.

    Columns 1 and 2 are the same four arms on two networks and share both rows' axes, so a twin
    pair is read straight across. Column 3 is three objectives that are neither each other's nor
    the arms' (module docstring) and keeps its own.
    """
    arms, baselines = pusht_entries()
    mlp_arms = [e for e in arms if not e[0].endswith(pt.ARCH_SUFFIX)]
    tf_arms = [e for e in arms if e[0].endswith(pt.ARCH_SUFFIX)]
    columns = [("MLP backbone", mlp_arms), (f"{pt.ARCH_LABEL} backbone", tf_arms),
               ("chunk baselines", baselines)]
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 4.1), facecolor=PALETTE["surface"],
                             squeeze=False, sharex=True)
    seeds: set[int] = set()
    for i, (title, entries) in enumerate(columns):
        for method, label, color, ls in entries:
            group = _seed_logs(root / method)
            if not group:
                continue
            seeds.add(len(group))
            style = {"color": color, "ls": ls}
            epoch = group[0]["epoch"]
            # The arms' comparable quantity is the regression term alone: a score arm's
            # train_loss carries lambda * E|(u - u_hat) . score| on top of it.
            train = _stack(group, "train_velocity_loss")
            if train is None:
                train = _stack(group, "train_loss")
            val = _stack(group, "val_loss_ema")
            if train is not None:
                _band(axes[0, i], epoch, train, style, label, lw=1.0)
            if val is not None:
                _band(axes[1, i], epoch, val, style, None, lw=1.0)
                # The evaluated checkpoint: the EMA model at the lowest EMA validation loss.
                mu = np.nanmean(val, axis=0)
                best = int(np.nanargmin(mu))
                axes[1, i].plot(epoch[best], mu[best], "o", ms=3.4, color=color,
                                mec=PALETTE["surface"], mew=0.6, zorder=5)
        # The legend lives in the validation row, the one with a free corner: every column's
        # curves leave its top right. The training row has none -- it fills its top left and
        # bottom right, and the transformer's arms cross the rest on their way down.
        handles, labels = axes[0, i].get_legend_handles_labels()
        leg = axes[1, i].legend(handles, labels, loc=LEGEND_LOC[i], frameon=False, fontsize=6.6,
                                handlelength=1.6, borderpad=0.2, labelspacing=0.3)
        for text in leg.get_texts():
            text.set_color(PALETTE["ink"])
        axes[0, i].set_title(title, fontsize=9, color=PALETTE["ink"])
        for row, ax in enumerate(axes[:, i]):
            # The training losses span three decades (the transformer arms memorize the
            # demonstrations; the MLP arms do not), so that row is log on every column. The
            # arms' validation loss sits in 13-24 and reads better linear; the baselines' own
            # losses, three objectives in one panel, still need the log.
            if row == 0 or i == 2:
                ax.set_yscale("log")
            ep._style(ax, "epoch" if row == 1 else None, size=8)
    # The two backbone columns are matched by hand rather than by sharey: sharing adopts the
    # first column's limits, which cut the transformer's training curves off at the bottom.
    for row in (0, 1):
        _match_ylim(axes[row, 0], axes[row, 1], log=row == 0)
        axes[row, 1].tick_params(labelleft=False)
    # Headroom for the legends: the MLP arms' validation curves climb back to within a point of
    # the panel's top, which is the corner the legend sits in.
    for ax in axes[1, :2]:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    axes[0, 0].set_ylabel("training loss", fontsize=8, color=PALETTE["ink"])
    axes[1, 0].set_ylabel("validation loss (EMA)", fontsize=8, color=PALETTE["ink"])
    fig.tight_layout()
    seed_text = str(min(seeds)) if len(seeds) == 1 else f"{min(seeds)}--{max(seeds)}"
    ep._save(fig, out, "fig_pusht_training_curves", tight=False,
             caption=_pusht_caption(seed_text))
    print(f"  Push-T: {len(mlp_arms) + len(tf_arms) + len(baselines)} models")


def _pusht_caption(seeds: str) -> str:
    return (
        f"Push-T, 500 epochs of Diffusion Policy's low-dimensional recipe, mean over {seeds} "
        r"seeds (band: $\pm 1$ s.d.). \textbf{Top}: training loss. For the eight policy arms "
        "this is the phase-velocity MSE alone, so a score arm is comparable with the arm whose "
        r"network it shares; for the chunk baselines it is their own objective (DP's denoising "
        r"MSE, MIP's regression MSE, ACT's L1 + 10 KL), which is neither the arms' nor each "
        r"other's. \textbf{Bottom}: the EMA model's validation loss, the quantity every arm's "
        "checkpoint is selected on; the dot marks the selected epoch. The training row is log "
        "scale, as is the baselines' validation panel; the arms' validation loss is linear. "
        "The two backbone columns share their axes, so an arm and its transformer twin are read "
        "straight across; the baselines keep their own."
    )


def _pareto_columns(root: Path) -> list:
    """``[(title, run suffix), ...]`` for the Pareto figure's models, smallest first.

    ``plot_pusht_pareto.MODELS`` lists them largest first by backbone key, where the published
    MLP's key ``mlp`` is the one without a run suffix. The title carries the parameter count each
    run recorded for its phase + score arm.
    """
    cols = []
    for backbone, _, name, _, _, _ in reversed(pp.MODELS):
        suffix = "" if backbone == "mlp" else f"_{backbone}"
        cfg = root / f"{pp.ARMS[0][0]}{suffix}" / "seed0" / "config.json"
        n = json.loads(cfg.read_text())["n_params"]
        cols.append((f"{name}\n{n / 1e6:.2f}M params", suffix))
    return cols


def fig_pusht_pareto(root: Path, out: Path) -> None:
    """:func:`fig_pusht` for the Pareto figure's four models: a column per model, then the chunk
    baselines, the training loss above the validation loss the checkpoint is selected on.

    The four model columns plot one quantity, the phase-velocity MSE, and share both rows' axes,
    so a model's arms are read against the next model's straight across. Colour and line style
    name the arm as in the violins; one legend above the panels serves every column.
    """
    width = TEXT_WIDTH
    size = {role: pt.type_pt(width, role) for role in pt.TYPE_RATIOS}
    label = {m: lab for m, lab, _, _ in pp.ARMS}
    arms = [(m, label[m], c, ls) for m, _, c, ls in pt.METHODS if m in label]
    # A baseline's open and closed loop are two rollouts of one run, so its curve takes the bare
    # name, as the Pareto figure's legend gives it.
    baselines = [(m, lab.split(" (")[0], c, ls) for m, lab, c, ls in pusht_entries()[1]]
    columns = [(title, [(f"{m}{suffix}", lab, c, ls) for m, lab, c, ls in arms])
               for title, suffix in _pareto_columns(root)]
    columns.append(("chunk\nbaselines", baselines))
    k = len(columns)
    fig, axes = plt.subplots(2, k, figsize=(width, 3.55), facecolor=PALETTE["surface"],
                             squeeze=False, sharex=True)
    seeds: set[int] = set()
    for i, (title, entries) in enumerate(columns):
        for method, lab, color, ls in entries:
            group = _seed_logs(root / method)
            if not group:
                raise FileNotFoundError(f"no seed*/log.csv under {root / method}")
            seeds.add(len(group))
            style = {"color": color, "ls": ls}
            epoch = group[0]["epoch"]
            train = _stack(group, "train_velocity_loss")
            if train is None:
                train = _stack(group, "train_loss")
            val = _stack(group, "val_loss_ema")
            _band(axes[0, i], epoch, train, style, lab, lw=1.0)
            _band(axes[1, i], epoch, val, style, None, lw=1.0)
            mu = np.nanmean(val, axis=0)
            best = int(np.nanargmin(mu))
            axes[1, i].plot(epoch[best], mu[best], "o", ms=3.0, color=color,
                            mec=PALETTE["surface"], mew=0.5, zorder=5)
        axes[0, i].set_title(title, fontsize=size["title"], color=PALETTE["ink"], pad=3)
        for row, ax in enumerate(axes[:, i]):
            if row == 0 or i == k - 1:
                ax.set_yscale("log")
            ep._style(ax, "epoch" if row == 1 else None, size=size["label"])
            ax.tick_params(labelsize=size["tick"], pad=1.5)
            ax.set_xticks([0, 250, 500])
    models = axes[:, :k - 1]
    for row in (0, 1):
        _match_ylim(*models[row], log=row == 0)
        for ax in models[row, 1:]:
            ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel("training loss", fontsize=size["label"], color=PALETTE["ink"])
    axes[1, 0].set_ylabel("validation loss (EMA)", fontsize=size["label"], color=PALETTE["ink"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    bh, bl = axes[0, k - 1].get_legend_handles_labels()
    leg = fig.legend(handles + bh, labels + bl, loc="upper center", ncol=len(labels + bl),
                     frameon=False, fontsize=size["legend"], handlelength=1.8,
                     columnspacing=1.2, borderpad=0.0, bbox_to_anchor=(0.5, 1.0))
    for text in leg.get_texts():
        text.set_color(PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.955), pad=0.2, w_pad=0.6, h_pad=0.5)
    seed_text = str(min(seeds)) if len(seeds) == 1 else f"{min(seeds)}--{max(seeds)}"
    ep._save(fig, out, "fig_pusht_training_curves_pareto", tight=True, pad=0.02,
             caption=_pusht_pareto_caption(seed_text))
    print(f"  Push-T Pareto: {k - 1} models x {len(arms)} arms, {len(baselines)} baselines")


def _pusht_pareto_caption(seeds: str) -> str:
    return (
        f"Push-T training of the four models in Fig.~\\ref{{fig:pusht_pareto}}, 500 epochs of "
        f"Diffusion Policy's low-dimensional recipe, mean over {seeds} seeds (band: $\\pm 1$ "
        r"s.d.). \textbf{Top}: training loss. For the policy arms this is the phase-velocity MSE "
        "alone, so a score arm is comparable with the arm whose network it shares; for the chunk "
        "baselines it is their own objective (DP's denoising MSE, MIP's regression MSE, ACT's "
        r"L1 + 10 KL), which is neither the arms' nor each other's. \textbf{Bottom}: the EMA "
        "model's validation loss, the quantity every arm's checkpoint is selected on; the dot "
        "marks the selected epoch. The training row is log scale, as is the baselines' "
        "validation panel; the arms' validation loss is linear. The four model columns share "
        "their axes; the baselines keep their own."
    )


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nav-runs", type=Path, default=Path("runs"))
    p.add_argument("--nav-out", type=Path, default=Path("output/skate/policy"))
    p.add_argument("--n", type=int, default=1024, help="nav training-set size (the violin's)")
    p.add_argument("--pusht-runs", type=Path, default=Path("runs/pusht/policy"))
    p.add_argument("--pusht-out", type=Path, default=Path("output/pusht_9010/figures"))
    p.add_argument("--nav-x", choices=("progress", "step", "logstep"), default="progress",
                   help="x axis of the nav figure: the fraction of each model's own budget "
                        "(default -- the arms train 20k steps and the baselines 100k), or the "
                        "true step, linear or log")
    p.add_argument("--only", choices=("nav", "pusht", "pusht_pareto"),
                   help="draw one figure only")
    args = p.parse_args()

    if args.only in (None, "nav"):
        args.nav_out.mkdir(parents=True, exist_ok=True)
        fig_nav(args.nav_runs, args.nav_out, args.n, x=args.nav_x)
    if args.only in (None, "pusht", "pusht_pareto"):
        args.pusht_out.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "pusht"):
        fig_pusht(args.pusht_runs, args.pusht_out)
    if args.only in (None, "pusht_pareto"):
        fig_pusht_pareto(args.pusht_runs, args.pusht_out)


if __name__ == "__main__":
    main()
