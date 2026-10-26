"""Collect the Push-T policy runs into ``output/pusht_9010/SUMMARY.md``.

    python scripts/pusht/summarize_pusht_policy.py

Reads every ``{--runs}/{method}/seed*/metrics.json`` written by
``scripts/pusht/train_pusht_policy.py``: by default the paper's 90/10 runs in
``runs/pusht/policy``; the 70/20/10 runs are ``runs/pusht/policy_702010``, summarized into
``output/pusht/SUMMARY.md``. The score regularizer's hyperparameters are chosen on the
**validation** episodes only -- the configuration with the lowest mean best-checkpoint
validation loss -- and the test episodes are read only for the final comparison of MLP,
MLP+phase and that one configuration. The sweep table therefore shows validation numbers only.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from scipy.stats import ttest_ind, ttest_rel

FIGURE_NAMES = {"mlp": "MLP", "fm": "MLP+phase"}


def short(m: str) -> str:
    """The method name without its run-directory suffix, for prose and column headers."""
    return label(m).split(" (`")[0]


def label(m: str) -> str:
    if m in FIGURE_NAMES:
        return FIGURE_NAMES[m]
    return f"MLP+score (`{m}`)" if m.startswith("mlp_") else f"MLP+phase+score (`{m}`)"


def load(root: Path) -> dict[str, list[dict]]:
    """The MLP arms' runs. The chunk baselines (dp, mip) live beside them but have their own
    metrics -- no per-split velocity errors -- and their own section."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(root.glob("*/seed*/metrics.json")):
        m = json.loads(f.read_text())
        if "val_split" in m.get("best", {}):
            runs[f.parent.parent.name].append(m)
    return dict(runs)


def mean_std(values: list[float]) -> str:
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return f"{st.mean(values):.3f} ± {sd:.3f}"


# The swept arms and their constant-lambda twins, on the MLP backbone: a backbone suffix
# (`_tf`, `_tfbig`) puts a run outside both, which is what score_params can parse.
ANNEALED_SCORE = re.compile(r"score_w[^_]+_cos[^_]+(?:_h[^_]+)?")
CONST_SCORE = re.compile(r"score_w[^_]+_const(?:_h[^_]+)?")


def score_params(name: str) -> dict:
    """``λ``, ``anneal`` (0 = the weight is never annealed) and the kernel bandwidth of a run."""
    m = re.fullmatch(r"score_w([^_]+)_(?:cos([^_]+)|const)(?:_h([^_]+))?", name)
    return {"λ": float(m[1]), "anneal": float(m[2] or 0.0), "h_px": float(m[3] or 32)}


def const_twin(name: str) -> str:
    """The unannealed run of the same lambda and bandwidth: ``score_w2.5_cos0.5`` ->
    ``score_w2.5_const`` (the ``mlp_`` prefix is kept)."""
    prefix, q = ("mlp_" if name.startswith("mlp_") else ""), score_params(name.removeprefix("mlp_"))
    h = "" if q["h_px"] == 32 else f"_h{q['h_px']:g}"
    return f"{prefix}score_w{q['λ']:g}_const{h}"


def split_line(run: dict) -> str:
    """The Data bullet: the split's sizes, and whether it holds out test episodes at all. The 90/10
    split has none -- its held-out set *is* the validation set, and the rollouts are the test."""
    n_train, n_val = len(run["train_episodes"]), len(run["val_episodes"])
    held = run["held_out_episodes"]
    head = ("* **Data** -- Diffusion Policy's Push-T demonstrations (206 human episodes). **Split** "
            "over all episodes (seed 42): ")
    if sorted(held) == sorted(run["val_episodes"]):
        return (head + f"{n_train} train / {n_val} validation, no test episodes: the closed-loop "
                "rollouts are the test, and the offline *test* columns below read the validation "
                "episodes.")
    return head + f"{n_train} train / {n_val} validation / {len(held)} test."


def pick(runs: dict[str, list[dict]], split: str, key: str) -> list[float]:
    return [r["best"][split][key] for r in runs]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs/pusht/policy"))
    p.add_argument("--out", type=Path, default=Path("output/pusht_9010/SUMMARY.md"))
    args = p.parse_args()

    runs = load(args.runs)
    if not runs:
        raise SystemExit(f"no finished runs under {args.runs}")
    # The swept phase arm. The sweep is over the annealed runs -- lambda, the anneal fraction and
    # the bandwidth -- and picks the configuration; the arm reported as MLP+phase+score is then
    # that configuration's constant-lambda twin (`_const`), which is the default schedule. The
    # two are compared head to head in the ablation section below.
    # Only the swept names themselves: `score_w<λ>_cos<frac>[_h<px>]` on the MLP backbone.
    # "not _const" used to stand in for "annealed", which also swept up the backbone twins
    # (`score_w2.5_const_tf`, `..._tfbig`) -- names score_params cannot parse, so they crashed
    # const_twin below rather than merely losing the val-loss pick.
    score = {m: r for m, r in runs.items() if ANNEALED_SCORE.fullmatch(m)}
    val_loss = lambda m: st.mean(r["best"]["val_loss_ema"] for r in runs[m])  # noqa: E731
    best_score = min(score, key=val_loss) if score else None
    annealed_score = best_score
    if best_score and (twin := const_twin(best_score)) in runs:
        best_score = twin
    if best_score is None:  # a tree with the schedule already settled carries no annealed arm
        consts = [m for m in runs if CONST_SCORE.fullmatch(m)]
        best_score = min(consts, key=val_loss) if consts else None
    # Every significance test compares against the full arm, MLP+phase+score (the score
    # configuration with the lowest validation loss); MLP+phase stands in if no score arm ran.
    base_method = best_score if best_score in runs else "fm"
    any_run = next(iter(runs.values()))[0]
    cfg = any_run["train_config"]

    lines = [
        "# Push-T policy evaluation",
        "",
        f"Generated by `scripts/pusht/summarize_pusht_policy.py` from `{args.runs.as_posix()}/`. "
        "Mean ± std over seeds. Design notes and history are in `fm_bc/pusht/STATUS.md`.",
        "",
        "## Setup",
        "",
        split_line(any_run),
        "* **Model** -- inputs `[o, x, ρ]`: the T-block pose `o` (min-max xy, cos θ, sin θ), the "
        "agent position `x` and `2ρ − 1`, with the oracle phase `ρ = t/(L−1)`. Output: the agent's "
        "phase velocity `u = dx/dρ = (L−1)(x_{k+1} − x_k)` in normalized position units. "
        f"{len(cfg['hidden'])}×{cfg['hidden'][0]} {cfg['activation']} MLP.",
        "* **Methods** -- **MLP** (no ρ input), **MLP+phase**, and **MLP+phase+score**: the "
        "phase-velocity MSE plus `λ E|(u − û)·∇log p_ρ(x)|`, the kernel score of the training "
        "episodes' agent positions at the same phase, λ held constant (`_const`) or annealed to 0 "
        "by cosine over the first part of training (`_cos<fraction>`). **MLP+score** is "
        "the no-phase ablation: the MLP's network with MLP+phase+score's regularizer and "
        "hyperparameters (the score is still indexed by the demonstrations' phase, in training "
        "only).",
        f"* **Training** -- Diffusion Policy's `pusht_lowdim` pipeline: {cfg['num_epochs']} epochs, "
        f"batch {cfg['batch_size']}, AdamW {cfg['lr']:g}, cosine with {cfg['lr_warmup_steps']} "
        "warmup steps, EMA. The checkpoint evaluated is the EMA model at the lowest validation "
        "loss.",
        "",
        "## Metrics",
        "",
        "The metrics in this section and the next are **offline**: the policy is evaluated on the "
        "demonstrations' own states. The closed-loop rollouts in gym-pusht, the headline metrics, "
        "follow in their own sections.",
        "",
        "* **u-RMSE** -- RMSE of the predicted phase velocity against the demonstration's, in "
        "normalized units per unit phase (lower is better). **u = 0** (standing still) is given "
        "for scale.",
        "* **next-pos px** -- the same error as the pixel error of the next agent position after "
        "one Euler step `x + û Δρ`.",
        "* **t1 / t2 / t3** -- u-RMSE in the first, middle and last third of phase.",
        f"* **p** -- Welch t-test against {short(base_method)} over seeds, the full arm.",
        "",
    ]

    # The no-phase ablation reuses the hyperparameters selected for MLP+phase+score.
    mlp_score = f"mlp_{best_score}" if best_score and f"mlp_{best_score}" in runs else None
    mlp_annealed = (f"mlp_{annealed_score}"
                    if annealed_score and f"mlp_{annealed_score}" in runs else None)
    finals = ([m for m in ("mlp",) if m in runs] + ([mlp_score] if mlp_score else [])
              + [m for m in ("fm",) if m in runs] + ([best_score] if best_score else []))
    if finals:
        zero_test = any_run["best"]["held_out"]["zero_velocity_rmse"]
        chosen = (f"MLP+phase+score is `{best_score}`: the λ, anneal fraction and bandwidth with "
                  f"the lowest validation loss in the sweep below (`{annealed_score}`), trained "
                  "with λ held constant instead of annealed -- the default schedule, compared "
                  "with the annealed one in the ablation further down. "
                  if best_score and best_score != annealed_score else
                  (f"MLP+phase+score is the configuration with the lowest validation loss in the "
                   f"sweep below (`{best_score}`). " if best_score else ""))
        lines += [
            "## Test set",
            "",
            f"{chosen}u = 0 scores **{zero_test:.3f}** on test.",
            "",
            f"| method | seeds | test u-RMSE | next-pos px | t1 | t2 | t3 | val u-RMSE | best epoch | p vs {short(base_method)} |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        base_test = (pick(runs[base_method], "held_out", "velocity_rmse")
                     if base_method in runs else None)
        for m in finals:
            R = runs[m]
            test = pick(R, "held_out", "velocity_rmse")
            pval = "—"
            if base_test is not None and m != base_method and len(test) > 1 and len(base_test) > 1:
                pval = f"{ttest_ind(test, base_test, equal_var=False).pvalue:.3f}"
            t = lambda k: st.mean(pick(R, "held_out", k))  # noqa: E731
            lines.append(
                f"| {label(m)} | {len(R)} | {mean_std(test)} | {t('rmse_px'):.2f} | "
                f"{t('velocity_rmse_t1'):.2f} | {t('velocity_rmse_t2'):.2f} | "
                f"{t('velocity_rmse_t3'):.2f} | {mean_std(pick(R, 'val_split', 'velocity_rmse'))} | "
                f"{st.mean(r['best']['epoch'] for r in R):.0f} | {pval} |")
        lines.append("")

    rollout_tag = "nominal_inverse_pd"
    rolled = {m: [json.loads(f.read_text()) for f in
                  sorted((args.runs / m).glob(f"seed*/rollout_{rollout_tag}.json"))]
              for m in finals}
    rolled = {m: r for m, r in rolled.items() if r}
    if rolled:
        info = next(iter(rolled.values()))[0]
        lines += [
            "## Closed-loop rollouts (gym-pusht)",
            "",
            f"Diffusion Policy's test protocol: {info['n_envs']} envs (seeds "
            f"{info['seeds'][0]}–{info['seeds'][-1]}), at most {info['max_steps']} steps; **score** "
            "is the episode's max reward `clip(coverage / 0.95, 0, 1)`, **success** is coverage > "
            "0.95. The `best.pt` of every seed is rolled out. **Phase**: nominal clock "
            f"`ρ_t = min(t / (L_med − 1), 1)` with the median training-episode length "
            f"L_med = {info['median_length']:g}, so ρ saturates at step {info['median_length'] - 1:g}; "
            "the displacement `u Δρ` is executed exactly through the inverse of the env's PD "
            "controller (`fm_bc/pusht/sim.py`). A policy that never moves scores **0.139**.",
            "",
            f"| method | seeds | score | success | final coverage | p (score) vs {short(base_method)} |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        base_scores = [x["mean_score"] for x in rolled.get(base_method, [])]
        for m, R in rolled.items():
            sc = [x["mean_score"] for x in R]
            pval = "—"
            if m != base_method and len(sc) > 1 and len(base_scores) > 1:
                pval = f"{ttest_ind(sc, base_scores, equal_var=False).pvalue:.3f}"
            lines.append(f"| {label(m)} | {len(R)} | {mean_std(sc)} | "
                         f"{mean_std([x['success_rate'] for x in R])} | "
                         f"{st.mean(x['mean_final_coverage'] for x in R):.3f} | {pval} |")
        lines += [
            "",
            "This clock saturates at step 122 and the phase arms make little progress after it; "
            "the clock variants below stretch it or replace it with a learned phase estimate.",
            "",
        ]

    stretched = {}
    for m in rolled:
        for tag, length in (("nominal_inverse_pd", None), ("nominal_L246_inverse_pd", "246"),
                            ("nominal_L300_inverse_pd", "300"),
                            ("phase_mlp_inverse_pd", "MLP estimate"),
                            ("phase_mlp_fixedstep_e1e-3_inverse_pd",
                             "MLP estimate, fixed step (ε = 1e-3)"),
                            ("phase_mlp_fixedstep_e1e-4_inverse_pd",
                             "MLP estimate, fixed step (ε = 1e-4)"),
                            ("phase_mlp_start_inverse_pd", "MLP estimate + start pose"),
                            ("phase_mlp_start_fixedstep_e1e-3_inverse_pd",
                             "MLP estimate + start pose, fixed step (ε = 1e-3)"),
                            ("phase_mlp_start_e1o245_inverse_pd",
                             "MLP estimate + start pose (ε = 1/245)"),
                            ("phase_mlp_start_fixedstep_e1o245_inverse_pd",
                             "MLP estimate + start pose, fixed step (ε = 1/245)")):
            R = [json.loads(f.read_text()) for f in
                 sorted((args.runs / m).glob(f"seed*/rollout_{tag}.json"))]
            if R:
                stretched.setdefault(m, []).append((tag, length, R))
    stretched = {m: v for m, v in stretched.items() if len(v) > 1}
    if stretched:
        lines += [
            "### Clock variants (the nominal clock above is the baseline)",
            "",
            "**L = 246 / 300**: `ρ_t = min(t / (L − 1), 1)` with the longest training demo or the "
            "step budget; the step size stays the median demo's, `Δx = u / (L_med − 1)`, so only "
            "how fast ρ advances changes. **MLP estimate**: the supervised phase regressor "
            "(`scripts/pusht/train_pusht_phase.py`, seed-for-seed with the policy) under the "
            "monotone ratchet `ρ_t = max(ρ_{t−1} + ε, MLP(state_t))`, ε = 1/200, and "
            "`Δx = u (ρ_t − ρ_{t−1})`. "
            "**fixed step**: the same estimate conditions the policy but the step stays the median "
            "demo's, `Δx = u / (L_med − 1)` -- so for the phase-free arms it is exactly the "
            "baseline rollout, and their rows repeat it. "
            "**+ start pose**: the estimator also sees the episode's initial block pose, which cuts "
            "its phase RMSE on the demonstrations from 0.025 to 0.007. **ε = 1/245** is one step of the "
            "longest demonstration (246 steps), so the ratchet's floor alone is the L = 246 clock "
            "and the estimate only adds where it runs ahead of it. "
            "**stalled**: the pusher moves < 1 px over the last 50 steps.",
            "",
            "| method | clock | seeds | score | success | final coverage | stalled | p (score) vs L_med |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for m, variants in stretched.items():
            base = [x["mean_score"] for x in variants[0][2]]
            for tag, length, R in variants:
                sc = [x["mean_score"] for x in R]
                stalls = []
                for f in sorted((args.runs / m).glob(f"seed*/rollout_{tag}.npz")):
                    S = np.load(f)["states"][:, :, :2]
                    stalls.append(np.mean(np.abs(S[:, -51:] - S[:, -1:]).max((1, 2)) < 1.0))
                pval = "—" if length is None else f"{ttest_ind(sc, base, equal_var=False).pvalue:.3f}"
                shown = length or f"L_med = {R[0]['median_length']:g}"
                lines.append(
                    f"| {label(m)} | {shown} | {len(R)} | "
                    f"{mean_std(sc)} | {mean_std([x['success_rate'] for x in R])} | "
                    f"{st.mean(x['mean_final_coverage'] for x in R):.3f} | "
                    f"{100 * st.mean(stalls):.0f} % | {pval} |")
        lines.append("")

    ref_file = args.runs / "mmd_reference.json"
    mmds = {m: [json.loads(f.read_text()) for f in
                sorted((args.runs / m).glob("seed*/mmd_demo_clock.json"))] for m in finals}
    mmds = {m: r for m, r in mmds.items() if r}
    if mmds and ref_file.exists():
        ref = json.loads(ref_file.read_text())
        lines += [
            "## MMD to the test demonstrations",
            "",
            f"Every policy is rolled out from the start state of each of the {ref['n_test']} test "
            "demonstrations for that demonstration's length, with its clock `ρ = k / (L − 1)`, and "
            "compared with the demonstrations at matched phase on a 128-point grid: squared MMD "
            "(biased V-statistic), Gaussian kernel of "
            f"{ref['bandwidth_px']:g} px on `[agent_xy, block_xy, 100 cos θ, 100 sin θ]` "
            "(`fm_bc/pusht/evaluation.py`, `scripts/pusht/evaluate_pusht_mmd.py`). Values ×10², "
            f"lower is better; p is Welch over seeds against {short(base_method)}.",
            "",
            "| method | seeds | MMD² mean | p | MMD² at ρ = 1 | p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        def col(R, k):
            return [1e2 * x[k] for x in R]
        for m, R in mmds.items():
            cells = []
            for k in ("mmd_mean", "mmd_final"):
                v = col(R, k)
                pv = "—"
                if m != base_method and base_method in mmds and len(v) > 1:
                    pv = f"{ttest_ind(v, col(mmds[base_method], k), equal_var=False).pvalue:.3f}"
                cells += [mean_std(v), pv]
            lines.append(f"| {label(m)} | {len(R)} | " + " | ".join(cells) + " |")
        lines += [
            f"| *validation demos (floor)* | — | {1e2 * ref['validation_demos']['mmd_mean']:.2f} | — | "
            f"{1e2 * ref['validation_demos']['mmd_final']:.2f} | — |",
            f"| *never-moving pusher* | — | {1e2 * ref['no_motion']['mmd_mean']:.2f} | — | "
            f"{1e2 * ref['no_motion']['mmd_final']:.2f} | — |",
            "",
        ]

    baselines = {}
    for kind, name in (("dp", "Diffusion Policy"), ("mip", "MIP")):
        M = [json.loads(f.read_text())
             for f in sorted((args.runs / kind).glob("seed*/metrics.json"))]
        loops = []
        for pattern, loop in (("rollout_chunk.json", "open"),
                              ("rollout_chunk_te*.json", "closed (temporal ensembling)")):
            R = [json.loads(f.read_text()) for f in sorted((args.runs / kind).glob(f"seed*/{pattern}"))]
            if R:
                loops.append((loop, R))
        if loops:
            baselines[kind] = (name, loops, M)
    if baselines:
        any_b = next(iter(baselines.values()))[1][0][1][0]
        lines += [
            "## Generative chunk baselines (Diffusion Policy, MIP)",
            "",
            f"Same split, epochs, optimizer, EMA and best-EMA-validation-loss selection as the MLP "
            f"arms (`fm_bc/pusht/baseline_training.py`), but the model predicts the environment's "
            f"action: {any_b['obs_steps']} states in, a chunk of {any_b['horizon']} absolute pixel "
            f"targets out, {any_b['action_steps']} executed per re-plan -- so there is no phase "
            "clock and no inverse PD. Rolled out on the same protocol "
            "(`scripts/pusht/rollout_pusht_baselines.py`).",
            "",
            "**open loop** executes `action_steps` of each chunk, the policy as trained; "
            "**closed loop** re-plans every step and temporally ensembles ACT-style "
            "(`exp(-0.01 i)`), the two modes the skate baselines are reported in.",
            "",
            "| method | loop | seeds | score | success | final coverage | params | best epoch |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for kind, (name, loops, M) in baselines.items():
            for loop, R in loops:
                lines.append(
                    f"| {name} | {loop} | {len(R)} | {mean_std([x['mean_score'] for x in R])} | "
                    f"{mean_std([x['success_rate'] for x in R])} | "
                    f"{st.mean(x['mean_final_coverage'] for x in R):.3f} | "
                    f"{M[0]['n_params'] / 1e6:.1f} M | "
                    f"{st.mean(m['best']['epoch'] for m in M):.0f} |")
        # The reference line needs the nominal-clock rollouts; a tree rolled out only on the
        # eval clock still has baselines to report, so it drops the line rather than crashing.
        if rolled:
            best_rollout = max(rolled.items(),
                               key=lambda kv: st.mean(x["mean_score"] for x in kv[1]))
            lines += [
                "",
                f"For reference, the best MLP arm on the nominal clock is "
                f"{label(best_rollout[0])} at "
                f"{mean_std([x['mean_score'] for x in best_rollout[1]])}.",
                "",
            ]

    # Checkpoint selection: the lowest-validation-loss model against the final, overfit one.
    # Both are rolled out on the same protocol; `last.pt`'s rollouts carry the `_last` tag.
    conditions = [("phase_mlp_start_fixedstep_e1o245_inverse_pd", "eval clock", list(rolled)),
                  ("nominal_inverse_pd", "nominal clock", list(rolled)),
                  ("chunk", "open loop", list(baselines)),
                  ("chunk_te0.01", "closed loop", list(baselines))]
    name_of = lambda m: baselines[m][0] if m in baselines else label(m)  # noqa: E731
    ckpt_rows, ckpt_epochs = [], {}
    for tag, condition, methods in conditions:
        for m in methods:
            pair = []
            for suffix in ("", "_last"):
                pair.append([json.loads(f.read_text()) for f in
                             sorted((args.runs / m).glob(f"seed*/rollout_{tag}{suffix}.json"))])
            A, B = pair
            if not A or len(A) != len(B):
                continue
            a, b = [x["mean_score"] for x in A], [x["mean_score"] for x in B]
            pval = f"{ttest_rel(b, a).pvalue:.3f}" if len(a) > 1 else "—"
            ckpt_rows.append(
                f"| {name_of(m)} | {condition} | {len(A)} | {mean_std(a)} | {mean_std(b)} | "
                f"{st.mean(b) - st.mean(a):+.3f} | {pval} | "
                f"{100 * st.mean(x['success_rate'] for x in A):.1f} % → "
                f"{100 * st.mean(x['success_rate'] for x in B):.1f} % | "
                f"{st.mean(x['mean_final_coverage'] for x in A):.3f} → "
                f"{st.mean(x['mean_final_coverage'] for x in B):.3f} |")
            M = [json.loads(f.read_text())
                 for f in sorted((args.runs / m).glob("seed*/metrics.json"))]
            if M and "epoch" in M[0].get("best", {}):
                # The validation loss each checkpoint was picked by (velocity MSE for the MLP
                # arms, the denoiser's own objective for the chunk baselines).
                final = lambda r: (r["last"]["val_loss_ema"] if "val_loss_ema" in r["last"]  # noqa: E731
                                   else r["last"]["val_split"]["velocity_mse"])
                ckpt_epochs[name_of(m)] = (st.mean(x["best"]["epoch"] for x in M),
                                           M[0]["train_config"]["num_epochs"],
                                           st.mean(x["best"]["val_loss_ema"] for x in M),
                                           st.mean(final(x) for x in M))
    if ckpt_rows:
        chosen_at = ", ".join(f"{k} at epoch {e:.0f}/{n} (val loss {a:.3g} → {b:.3g} by the end)"
                              for k, (e, n, a, b) in ckpt_epochs.items())
        lines += [
            "## Checkpoint selection: best validation loss vs the final model",
            "",
            "Every arm above is the EMA model at the **lowest EMA validation loss** (`best.pt`). "
            "That is the standard choice for a regression policy, but a generative chunk policy is "
            "usually trained to the end and the final weights are kept -- Diffusion Policy's own "
            "recipe never early-stops -- and the validation loss of a denoiser is a poor proxy for "
            "rollout score. The same rollouts are therefore repeated with each run's `last.pt` "
            "(`--checkpoint last.pt`, which tags its outputs `_last`). p is a **paired** t-test "
            "over seeds, `last.pt` against `best.pt`; the same network, the same 50 envs, the same "
            f"clock. Validation picks: {chosen_at}.",
            "",
            "| method | condition | seeds | score (best.pt) | score (last.pt) | Δ | p | success | "
            "final coverage |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ] + ckpt_rows + [""]

    # Schedule ablation: lambda held at score_weight for the whole run (`_const`) against the
    # annealed arm of the same lambda and bandwidth, at both checkpoints.
    annealed_arms = [m for m in (mlp_annealed, annealed_score)
                     if m and m in runs and const_twin(m) in runs]
    clocks = [("phase_mlp_start_fixedstep_e1o245_inverse_pd", "eval clock"),
              ("nominal_inverse_pd", "nominal clock")]
    ablation, val_losses = [], []
    scores_at = lambda name, tag, sfx: [  # noqa: E731
        json.loads(f.read_text())["mean_score"]
        for f in sorted((args.runs / name).glob(f"seed*/rollout_{tag}{sfx}.json"))]
    for annealed in annealed_arms:
        m = const_twin(annealed)
        for tag, clock in clocks:
            anneal_frac = score_params(annealed.removeprefix("mlp_"))["anneal"]
            for name, schedule in ((annealed, f"cosine → 0 over {anneal_frac:g}"),
                                   (m, "**constant**")):
                a, b = (scores_at(name, tag, ""), scores_at(name, tag, "_last"))
                if not a or len(a) != len(b):
                    continue
                cells = []
                for values, other in ((a, scores_at(annealed, tag, "")),
                                      (b, scores_at(annealed, tag, "_last"))):
                    # Paired over seeds against the annealed arm of the same lambda.
                    pval = ("—" if name == annealed or len(values) != len(other) or len(a) < 2
                            else f" (p {ttest_rel(values, other).pvalue:.2f})")
                    cells.append(mean_std(values) + ("" if pval == "—" else pval))
                S = lambda sfx: [json.loads(f.read_text())["success_rate"] for f in  # noqa: E731
                                 sorted((args.runs / name).glob(f"seed*/rollout_{tag}{sfx}.json"))]
                ablation.append(
                    f"| {short(name)} | {schedule} | {clock} | {len(a)} | {cells[0]} | "
                    f"{cells[1]} | {st.mean(b) - st.mean(a):+.3f} | "
                    f"{100 * st.mean(S('')):.1f} % → {100 * st.mean(S('_last')):.1f} % |")
        for name in (annealed, m):
            R = runs[name]
            val_losses.append(f"{short(name)} {'constant' if name == m else 'annealed'} "
                              f"{st.mean(r['best']['val_loss_ema'] for r in R):.2f} → "
                              f"{st.mean(r['last']['val_split']['velocity_mse'] for r in R):.2f}")
    if ablation:
        lam = score_params(annealed_arms[0].removeprefix("mlp_"))["λ"]
        lines += [
            "### The regularizer without its schedule (constant λ)",
            "",
            f"The score arms anneal λ to 0, so the weights they end on are trained by the plain "
            f"velocity MSE and the regularizer only shapes the path there. `_const` holds "
            f"λ = {lam:g} for all 500 epochs instead (`--score-anneal-fracs 0`), so it is still "
            "acting on `last.pt`. Same λ, bandwidth, seeds, split and protocol as the annealed arm "
            "above; both checkpoints rolled out.",
            "",
            "p is paired over seeds against the annealed arm on the same clock and checkpoint.",
            "",
            "| method | λ schedule | clock | seeds | score (best.pt) | score (last.pt) | Δ | "
            "success |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ] + ablation + [
            "",
            f"Validation loss (velocity MSE), best → last: {'; '.join(dict.fromkeys(val_losses))}.",
            "",
        ]

    FIGURE_TAG = "phase_mlp_start_fixedstep_e1o245_inverse_pd"
    FIGURE_TAG_LABEL = ("phase-MLP clock with the start pose, ε = 1/245 and the fixed step")
    stalls, escapes = {}, {}
    for m in rolled:
        s_m, e_m = [], []
        for f in sorted((args.runs / m).glob(f"seed*/rollout_{FIGURE_TAG}.npz")):
            S = np.load(f)["states"]
            s_m.append(np.mean(np.abs(S[:, -51:, :2] - S[:, -1:, :2]).max((1, 2)) < 1.0))
            e_m.append(np.mean(((S[:, :, 2:4] < 0) | (S[:, :, 2:4] > 512)).any((1, 2))))
        if s_m:
            stalls[m], escapes[m] = 100 * st.mean(s_m), 100 * st.mean(e_m)
    stall_line = ", ".join(f"{label(m)} {v:.0f} %" for m, v in stalls.items()) or "n/a"
    escape_line = (", ".join(f"{label(m)} {v:.1f} %" for m, v in escapes.items() if v > 0)
                   or "no rollout of any method")

    fig_dir = args.out.parent / "figures"
    figures = [
        ("fig_pusht_demos", "Example training episodes: pusher path by phase, block at ρ = 0, ¼, ½, ¾, 1"),
        ("fig_pusht_distribution", "All training pusher and block paths, and episode lengths"),
        ("fig_pusht_phase_evolution", "Per-phase density of the state on the shared phase grid"),
        ("fig_pusht_velocity_score", "Demo phase velocity and the kernel score at five phases"),
        ("fig_pusht_violin", "Episode scores and per-seed rollout metrics per method"),
        ("fig_pusht_inference", "Inference cost per decision and per control step, and the score it buys"),
        ("fig_pusht_coverage", "Rollout coverage against step, with the nominal clock's saturation"),
        ("fig_pusht_rollout_grid", "The same test envs rolled out by every method (seed 0)"),
        ("fig_pusht_filmstrip", "Simulator frames of one test env"),
        ("fig_pusht_filmstrip_ranked_mlp", "MLP: best, median and worst of 250 episodes"),
        ("fig_pusht_filmstrip_ranked_mlp_score_w2.5_const", "MLP+score: best, median and worst of 250 episodes"),
        ("fig_pusht_filmstrip_ranked_fm", "MLP+phase: best, median and worst of 250 episodes"),
        ("fig_pusht_filmstrip_ranked_score_w2.5_const", "MLP+phase+score: best, median and worst of 250 episodes"),
        ("fig_pusht_final_poses", "Final block poses of all 50 test envs (seed 0), coloured by coverage"),
    ]
    present = [(f, cap) for f, cap in figures if (fig_dir / f"{f}.png").exists()]
    if present:
        lines += [
            "## Figures",
            "",
            "Regenerate with `python scripts/pusht/plot_pusht.py`. Drawn in the simulator's screen "
            "frame (y down), so they match rendered frames.",
            "",
            f"They show the **{FIGURE_TAG_LABEL}** rollouts "
            f"(`rollout_{FIGURE_TAG}.*`), the evaluation condition; the tables above keep the "
            "nominal clock as the baseline. Pass `--rollout-tag` to plot another clock.",
            "",
            "In rollouts, **stalling is a failure of every method**: the pusher does not move over "
            f"the last 50 steps in {stall_line}. The block leaves the arena (the kinematic pusher "
            f"can drag it through a wall) in {escape_line}.",
            "",
        ]
        for f, cap in present:
            lines += [f"**{cap}**", "", f"![{cap}](figures/{f}.png)", ""]

    if score:
        zero_val = any_run["best"]["val_split"]["zero_velocity_rmse"]
        lines += [
            "## Score hyperparameter sweep (validation only)",
            "",
            f"Sorted by mean best-checkpoint validation loss (velocity MSE). u = 0 scores "
            f"{zero_val:.3f} u-RMSE on validation. **λ** is the initial weight, **anneal** the "
            "fraction of training over which it decays to 0, **h** the kernel bandwidth.",
            "",
            "| λ | anneal | h (px) | seeds | val loss | val u-RMSE | best epoch |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        ref = [m for m in ("fm", "mlp") if m in runs]
        for m in sorted(score, key=val_loss):
            q, R = score_params(m), runs[m]
            bold = "**" if m == annealed_score else ""
            lines.append(
                f"| {bold}{q['λ']:g}{bold} | {q['anneal']:g} | {q['h_px']:g} | {len(R)} | "
                f"{mean_std([r['best']['val_loss_ema'] for r in R])} | "
                f"{mean_std(pick(R, 'val_split', 'velocity_rmse'))} | "
                f"{st.mean(r['best']['epoch'] for r in R):.0f} |")
        for m in ref:
            R = runs[m]
            lines.append(
                f"| {FIGURE_NAMES[m]} | — | — | {len(R)} | "
                f"{mean_std([r['best']['val_loss_ema'] for r in R])} | "
                f"{mean_std(pick(R, 'val_split', 'velocity_rmse'))} | "
                f"{st.mean(r['best']['epoch'] for r in R):.0f} |")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}  ({len(runs)} methods, best score config {best_score})")


if __name__ == "__main__":
    main()
