# Skate navigation (`fm_bc/skate/`)

`fm_bc.skate` uses the planner as a demonstration generator for the paper's navigation experiments:
collision-free demo *distributions* for the three Fig. 6 obstacle scenarios, their SE(2)
divergence and score, and a phase-conditioned MLP policy trained with flow matching (**FM**), with
or without the conditional divergence matching regularizer (**FM+CDM**, `L_FM + λ L_CDM`). A
third baseline, **MLP (no phase)** (`--no-phase`), is the same network and loss without the phase
input: a plain state-feedback behaviour-cloning policy.
Rollouts saturate the policy's commands to the planner's `SkateLimits` (v ∈ [0, 2] m/s,
|ω| ≤ 2 rad/s) before executing them, as a real robot's actuators would; `--no-clamp` executes
raw commands instead.

```bash
# demos: 1024 per scenario for training, an independent 1024 (seed 1) held out for evaluation,
# with their kNN (k=16) divergence and kernel score -- the shipped datasets, rebuilt and checked
python scripts/skate/reproduce_datasets.py --out data/reproduced    # or --check data/
python scripts/skate/mode_separability.py  --data data/        # how long the pose hides the mode (B, C)

# policies: scenario x training demos x CDM weight x seed (resumable; skips finished runs)
python scripts/skate/run_experiments.py --cdm-weights 0 0.05 --n-train 1024 512 256 128 64 --seeds 0 1 2 3 4
python scripts/skate/run_experiments.py --no-phase --cdm-weights 0 --n-train 1024 512 256 128 64 --seeds 0 1 2 3 4
# CDM ablation: each regularizer term alone and both together, λ annealed 0.05 → 0 over half the steps
python scripts/skate/run_experiments.py --cdm-weights 0.05 --cdm-terms both div score --cdm-schedule cosine \
    --cdm-anneal-frac 0.5 --n-train 1024 512 256 128 64 --seeds 0 1 2 3 4
# action-chunk baselines at n = 1024: Diffusion Policy, MIP and ACT
python scripts/skate/train_baselines.py --methods dp mip act --n-train 1024 --seeds 0 1 2 3 4
# ACT's own batch size is 8; this is the data-matched control at Diffusion Policy's 256
python scripts/skate/train_baselines.py --methods act --batch-size 256 --n-train 1024 \
    --seeds 0 1 2 3 4 --out runs_actb256
python scripts/skate/benchmark_inference.py --runs runs/ --out output/skate/policy/   # inference cost
python scripts/skate/evaluate_policies.py --runs runs/ --test data/test/ --out output/skate/policy/
```

The shipped demonstrations come from two generations of the planner. `single_path` was planned
before commit `2eaf39d` (against a distance-transform grid SDF, from seeds that snap to the goal
heading), `dual_goal` and `dual_path` by today's planner, and all six files' divergence and score
with a form of the SE(2) log that differs from today's in the last bits.
`scripts/skate/reproduce_datasets.py` switches exactly those back and rebuilds every array bit for
bit, checking it against `scripts/skate/nav_datasets.sha256.json`; `--check data/` verifies a copy
without rebuilding. `generate_datasets.py` followed by `compute_divergence.py` makes *new* datasets
with today's planner — for `single_path` a different distribution (98 % of problems accepted,
against 34 % for the shipped one).

The baselines (`fm_bc/skate/baselines.py`) are the standard BC policies, run in Diffusion Policy's
action-chunk convention: from the last two poses (no phase) each predicts 16 controls and executes
8 before re-planning, so a *stochastic* one can split between modes within one start rather than
only across starts. **Diffusion Policy** is a 100-step DDPM (`squaredcos_cap_v2`, ε prediction) on
its conditional 1D UNet; **MIP** ("Much Ado About Noising") is two regression steps on its residual
MLP, the second from the first's output with noise injected at t\* = 0.9 in training only;
**MeanFlow** learns the average velocity `u(z, r, t)` through the MeanFlow identity (a forward-mode
JVP) and samples in one step, on the Diffusion Policy UNet. Each keeps its paper's optimizer, batch
size and EMA (`fm_bc/skate/baseline_training.py`); the networks are ported from the reference
repos into `fm_bc/nets/`. None has a velocity field of its own, so they report the rollouts'
divergence but not a model divergence.

**ACT** (Zhao et al. 2023) is the odd one out and is here as the contrast: a CVAE transformer
trained with L1 + 10·KL whose latent is fixed to the prior mean `z = 0` at inference, so its chunk
is a **deterministic** function of the observation window. It cannot split between modes at all —
its diversity metrics are structurally zero and its `mode_tv` reflects only across-start variation.
That is the expected result, not a bug. The port (`fm_bc/nets/act.py`) follows
[LeRobot](https://github.com/huggingface/lerobot)'s `ACTPolicy`, which is what the community
benchmarks against, at its defaults: 512-wide, 8 heads, 3200 feed-forward, 4 encoder and 4
VAE-encoder layers, latent 32, post-norm, AdamW 1e-5 with weight decay 1e-4 and no EMA. Two points
worth knowing:

* **One decoder layer is faithful, not a shortcut.** The released ACT builds seven and then reads
  the *first* layer's output, so the other six never receive gradient; LeRobot ships
  `n_decoder_layers = 1` for exactly this reason. ACT is therefore ~40 M parameters against
  Diffusion Policy's 65 M — report the counts rather than claiming a matched comparison.
* **ACT is reported closed loop.** Training writes the open-loop rollout, matching LeRobot's
  `temporal_ensemble_coeff = None`; the figures and tables show ACT only with the ACT paper's own
  temporal ensembling (`act_te0.01`, from `scripts/skate/evaluate_temporal_ensemble.py --methods act
  --m 0.01`), because executing 8 steps per decision would make its per-step cost look 8× lower
  than under the loop it is evaluated in.
* ACT's published batch size is 8 against Diffusion Policy's 256, so at a shared 100k-step budget
  it sees 32× fewer samples. The reported ACT runs use its own recipe (batch 8); the command block
  above also trains a batch-matched control at 256.

The reported baselines are **DP**, **MIP** and **ACT** at n = 1024 (3 scenarios × 10 seeds, 100k
steps each; DP and MIP open and closed loop, ACT closed loop). **MeanFlow** is implemented and tested but not in the
tables: its one-step chunks stay ~2× noisier
than the demonstrations, which drives the robot into obstacles (≈ 7 % success at half budget), so
it was parked rather than tuned. Multi-step sampling of the same flow map, or putting training
mass on the `(r = 0, t = 1)` pair the one-step sampler queries, are the things to try first.
`scripts/skate/benchmark_inference.py` times every policy — per decision, per executed control step
and per rollout — into the SUMMARY's inference table.

The ablation splits the CDM regularizer into its two terms, `a = ∇·u − ∇·v` (the demos' kNN
divergence minus the model's) and `b = (u − v)·∇log p_ρ` (the twist error against the kernel
score): `--cdm-terms div` trains `E|a|`, `score` trains `E|b|` and `both` (the default) the full
CDM `E|a + b|`. `--cdm-schedule cosine` anneals λ to 0 over the first `--cdm-anneal-frac` of
training, after which the loss is pure FM.

`evaluate_policies.py` writes the results to `output/skate/policy/SUMMARY.md` (tables with the
figures inline), with `table_policy.tex` / `table_policy_summary.tex` for the paper. The analysis
behind the kNN divergence estimator is `scripts/skate/paper_divergence_figures.py` →
`output/skate/paper/`.

