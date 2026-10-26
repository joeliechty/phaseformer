# Push-T extension — state of play

Where the Push-T work stands, what is verified, and what the next person (or session) should do
first. Written to be picked up cold.

**Branch:** `push_t`. **Commits so far:** `d1f6bbb` (Phase 1 data infrastructure), `ac7c57a`
(task-progress CVAE).

The CVAE's full design rationale — the objective term by term, the ranked failure modes and the
metric definitions — is in a separate design note, not included here. This
file is the state, not the design.

---

## 1. Why this work exists

`fm_bc` studies flow-matching BC with a conditional divergence-matching (CDM) regularizer. Until
now it had one task: SE(2) skate-robot navigation, with demonstrations *generated* by the Theseus
planner at a fixed horizon `N = 50`. Because every demo is the same length, the phase grid `k/N`
is free, and `divergence_knn` / `score_kernel` — which take neighbours **at the same phase** —
operate directly on a rectangular `(M, N+1, 3)` array.

Push-T (Chi et al. 2023) is the second SE(2) task. None of the Lie-group math changes. Three
things do, and they drive every design decision below:

1. **Episodes vary in length** (206 episodes, 25 650 steps, 30–246 each at 10 Hz). The shared
   phase grid must be constructed.
2. **Actions are absolute pixel targets** in a 512×512 frame, applied as a PD setpoint — not
   body twists. The skate assumption that *actions are the velocity field* does not carry over.
3. **The pusher has no orientation.** It is a circle, so SE(2) enters only through the T-block
   pose, which the policy observes but never predicts deltas for.

**Environment.** Real runs happen on a GPU machine. Development also has to work on a laptop with
no `fm_bc` env, where Theseus cannot be installed at all (theseus-ai 0.2.3 ships cp38–cp310
wheels only). Everything in the Push-T subtree is therefore torch-only and Theseus-free, which is
enforced by a test rather than left as a convention.

---

## 2. What is DONE and verified

### Phase 1 — data pipeline (commit `d1f6bbb`)

| File | Role |
|---|---|
| `scripts/pusht/download_pusht.sh` | Anonymous fetch → size-verify → unzip. `--dry-run`, `--force`, `--help`, idempotent. |
| `fm_bc/__init__.py` | **Modified**: PEP 562 lazy `__getattr__`/`__dir__`. Public API unchanged. |
| `fm_bc/windows.py` | `window_indices` (moved verbatim from `chunks.py`) + `episode_window_indices`. |
| `fm_bc/skate/chunks.py` | **Modified**: two lines, re-imports `window_indices`. All call sites unchanged. |
| `fm_bc/pusht/constants.py` | Workspace, goal pose, control rate, `ROT_SCALE_PX_PER_RAD`. |
| `fm_bc/pusht/data.py` | `PushTEpisodes`: variable-length episodes in one flat buffer. |
| `fm_bc/pusht/convert.py` | `from_zarr` — the only zarr dependency, imported lazily. |
| `fm_bc/pusht/splits.py` | DP's episode-level split; `TRAIN`/`VAL`/`UNUSED`. |
| `fm_bc/pusht/normalize.py` | `(scale, offset)` normalizers: `box`/`minmax`/`gaussian`. |
| `fm_bc/pusht/grid.py` | Uniform phase resampling, geodesic in SE(2); per-episode `grid_dt`. |
| `fm_bc/pusht/prepared.py` | `PushTPrepared`: splits + normalizers + grid. |
| `fm_bc/pusht/chunks.py` | `OBS_DIM = 6`, `embed_state`, episode-aware `make_windows`. |
| `scripts/pusht/{convert,prepare,inspect}_pusht.py` | The three CLIs. |
| `tests/pusht/test_pusht_{data,windows}.py` | 44 tests. |

`.gitignore` needed **no change** — the existing `data/` rule covers `data/pusht/**` (verified
with `git check-ignore` on both the npz and a file inside the zarr store), while
`scripts/pusht/download_pusht.sh` stays tracked.

**Verified, not merely written:**

- End-to-end on a synthetic zarr in the exact published layout with realistic lengths (31–246):
  convert → prepare → inspect all pass, exit 0.
- **The decisive check**: the phase grid feeds `divergence_knn` and `score_kernel` directly, both
  `metric="se2"` (block pose, D=3) and `metric="xy"` (agent, D=2), finite throughout, with the
  per-episode `dt` division applied across an 8.17× time-scale spread.
- Tested against **zarr 3.4.0**, so the "zarr 2.x/3.x common API" claim is exercised on the newer
  API rather than assumed.
- The no-Theseus property is a test (`sys.meta_path` hook blocking `theseus`/`skate_planner`).
- Skate path unaffected: `window_indices` re-exports from `fm_bc.skate.chunks` with identical
  behaviour; the exact import line in `tests/skate/test_baselines.py` resolves.

### Phase 2a — task-progress CVAE (commit `ac7c57a`)

`fm_bc/pusht/progress.py` (~23 KB), `fm_bc/pusht/progress_training.py` (~25 KB),
`scripts/pusht/train_progress.py`, `tests/pusht/test_pusht_progress.py`.

Latent is `z = [z_phase (scalar), z_style (Z-dim)]`, with **KL on `z_style` only** — so `z_phase`
is the rate-free channel the optimizer routes information through, and a monotonicity hinge
decides *which* information. Ground-truth `PushTEpisodes.phase` never enters the graph; it is
named `phase_holdout` in `progress_rows` so it cannot be passed in by accident. **Only the
encoder ships** (`ProgressCVAE.phase`); the decoder is a discarded training scaffold that is
load-bearing precisely because reconstruction is the only term grounding `z_phase` in real state
structure.

Key API: `ProgressCVAE`, `progress_rows`, `sample_pairs`, `monotonicity_hinge`,
`load_progress_model`; `ProgressConfig`, `train_progress`, `evaluate_progress`,
`phase_baselines`, `spearman`, `encode_rows`, `progress_run_dir`.

**Current test status: 73 passed**, ~17 s:

```bash
pytest tests/pusht/test_pusht_{progress,data,windows}.py
```

Run on a laptop with **torch only — no scipy, no Theseus**. That is why `spearman` is implemented
in `progress_training.py` with tie-aware average ranks in pure torch rather than taken from
`scipy.stats`; a scipy import would break the subtree property that
`test_pusht_imports_without_theseus` enforces, and it now covers both progress modules.

`pytest tests/` as a whole reports **9 collection errors** — `test_baselines`, `test_costs`,
`test_divergence`, `test_planner`, `test_policy`, `test_scenarios`, `test_se2`, `test_skate_model`,
`test_world` — all `ModuleNotFoundError: No module named 'theseus'`. **Pre-existing**, confirmed
identical at `HEAD` with `git stash`: the Theseus half of the repo on a machine that cannot install
it, not a regression.

#### Two measurements that shaped the tests

**The `_toy()` fixture cannot test generalization, so the behavioural test does not pretend to.**
Its agent walks `50 + 3*t` in the **global** flat index, so the last episode lies entirely outside
the coordinate range of the others. Measured: per-episode Spearman **+1.000 on all four training
episodes** and **−0.817 on episode 4**, pinned at exactly those values across every budget from 700
to 4000 steps. That is extrapolation, not undertraining, so
`test_training_recovers_monotone_progress_on_the_episodes_it_saw` scores the training episodes and
says so in its docstring. Generalization can only be measured on the real dataset.

**The OOD score's separation strengthens with training**, because it depends on the KL having
actually shaped the posterior. Using that same extrapolated episode as a genuine distribution
shift: at 3000 steps all three seeds separate by ≥9.7× (best 23×), while at ≤2000 steps one seed
does not separate at all. `_trained()` defaults to `steps=3000` for this reason — if that test ever
looks flaky, the cause is training budget, not the assertion.

---

## 3. The four decisions that shape everything

These were taken deliberately; do not silently reverse them.

**Source: the Diffusion Policy zarr only.** It is the *only* source carrying the T-block SE(2)
pose. LeRobot's `lerobot/pusht` has `observation.state` of shape `[2]` — agent position only,
block pose dropped upstream — so it cannot support the SE(2) framing. Not scripted on purpose;
there is a comment at the bottom of the download script saying so.

**Two artifacts, because there are two consumers.** `pusht_episodes.npz` is canonical: episodes
concatenated flat with per-episode bounds, feeding the action-chunk baselines.
`pusht_prepared.npz` is derived and regenerable: splits, normalizer stats fitted on training
episodes only, and the rectangular phase grid the divergence estimators need. Flat+bounds rather
than padded-rectangular because padding wastes ~half the buffer and forces a validity mask every
consumer must remember to honour.

**Uniform phase resampling, geodesic in SE(2).** The block pose interpolates with `se2_log` /
`se2_exp_retract`. The alternative (scatter to nearest bin, gap-fill) preserves original samples
but loads the interpolation artifact onto *short* episodes specifically — a 60-step episode
contributes ~75 % fabricated points to every neighbour query, a 246-step one none — and episode
length correlates with task difficulty, which is the signal the divergence target carries.

**Store absolute actions plus both derived velocity forms.** `vel_action` (commanded,
`action − agent_pos`) and `agent_delta` (achieved, forward difference, **edge-held** at each
episode's last step so it never reads as a full stop).

---

## 4. Three constants that silently corrupt results if changed

**`grid_dt` is per-episode.** After gridding, one phase step is `(L_e − 1)·dt/N` seconds, varying
~4× across episodes. `divergence_knn` takes a scalar `dt` and is exactly linear in `1/dt`, so a
global value scales a half-length episode's divergence by exactly ½. Always:

```python
div = divergence_knn(prep.grid.block_poses(), dt=1.0, k=16, metric="se2",
                     rot_scale=ROT_SCALE_PX_PER_RAD) / prep.grid.grid_dt[:, None]   # 1/s
```

**`ROT_SCALE_PX_PER_RAD = 100`.** In pixel units translations are O(100) and rotations O(π), so
the skate default of `1.0` would make the SE(2) metric ~99.9 % translation. 100 px/rad is the
block's characteristic radius.

**`kappa` in the CVAE is a supervision dial, not a hyperparameter.** With `z_phase ∈ [0,1]` and
per-episode margin `m = kappa/(L−1)`, the hinge telescopes to `p_{L−1} − p_0 ≥ kappa`. At
`kappa = 1` the only feasible solution is `p_t = t/(L−1)` — *exactly* ground-truth phase. Never
run `kappa = 1` and call it unsupervised. Default is 0.5.

---

## 5. Where to pick up

### Done on the GPU machine (2026-09-16)

- **Real data through the pipeline.** Download → convert → prepare (`--n-grid 128`) → inspect all
  pass. 206 episodes / 25 650 steps, 90/4/112 split as expected. Two estimates below were off:
  shortest episode is **49** steps (not 30), and `grid_dt` spread is **5.10×** (0.0378–0.1929 s),
  not ≈4×. Archive sha256 pinned in `scripts/pusht/download_pusht.sh`. `zarr` had to be installed
  into the `fm_bc` env (`pip install "zarr>=2.16"` → 2.18.3).
- **`pytest tests -q` with Theseus present: 186 passed.** Run with `PYTHONNOUSERSITE=1` — a `dash`
  pytest plugin in `~/.local` otherwise breaks collection (unrelated to this repo).
- **Baselines on real data:** `rho_geom = +0.975`, `rho_linear = +0.685`. The hand-coded
  distance-to-goal null is almost a perfect per-episode progress measure.
- **Default CVAE, 3 seeds** (`Z=8, κ=0.5, To=2`, 20k steps, ~100 s each on a 4090;
  `runs/progress/pusht/progress/z8_k0.5/seed{0,1,2}`), scored on the 112 unused episodes:

  | seed | ρ/ep mean | median | p10 | no-cond | phase_abl | cal_gap | roughness | ρ_t3 |
  |---|---|---|---|---|---|---|---|---|
  | 0 | 0.952 | 0.975 | 0.897 | 0.966 | 370 | 0.068 | 0.314 | 0.670 |
  | 1 | 0.955 | 0.971 | 0.910 | 0.969 | 380 | 0.064 | 0.323 | 0.638 |
  | 2 | 0.960 | 0.980 | 0.920 | 0.971 | 352 | 0.068 | 0.316 | 0.693 |

  Gate 1 (phase ablation) passes by a wide margin; calibration gap passes. **Gate 2 fails**:
  mean 0.956 is *below* `rho_geom` 0.975 (medians tie). Roughness sits just over 0.3, and
  `spearman_no_cond` is consistently *above* `with_cond`. Per §5's reading order, the CVAE is not
  yet shown to learn anything beyond distance-to-goal; decide whether that is a problem (or
  whether `rho_geom` is simply the right progress signal for Push-T) before running the ablation
  table.

### Current Push-T results → `output/pusht/SUMMARY.md`

Regenerate with `python scripts/pusht/summarize_pusht_policy.py`. Split is now **70/20/10 over all
206 episodes** (`data/pusht/pusht_prepared_702010.npz`, 144/41/21, seed 42); DP's 4 validation
episodes were too few and too easy to select checkpoints. Score hyperparameters were swept on
validation (λ ∈ {0.5, 2.5, 10} × anneal {0.5, 1} × h {16, 32, 64} px, 3 seeds); the winner
`score_w2.5_cos0.5` (λ·E|b| ≈ the MSE at the start) was rerun to 5 seeds. Test u-RMSE: MLP
6.098 ± 0.077, MLP+phase 6.019 ± 0.020, MLP+phase+score 5.846 ± 0.054 (p = 0.001 vs MLP+phase);
u = 0 scores 6.910. Offline only — no simulator, so no MMD or success yet.

**ρ is normalized time `k/(L−1)`, an oracle** — not geometric. It needs the episode length, so a
closed-loop rollout needs a phase estimate. A geometric phase (block-to-goal progress) would make
`dx/dρ` ill-posed as is: the block is still on ~54 % of steps while the agent moves (Δρ = 0).

**Closed-loop rollouts** now run in gym-pusht (`fm_bc/pusht/sim.py`,
`scripts/pusht/rollout_pusht_policy.py`; install extra `pusht-sim`, pymunk < 7). Placement is exact
once the angle is set before the position (gym-pusht's legacy reset misplaces the block ~60 px); the
policy's displacement goes through an exact inverse of the env's PD loop. With the nominal clock
(L_med = 123) on DP's 50-seed protocol: score MLP 0.487, MLP+phase 0.469, +score 0.598 (p = 0.001
vs MLP+phase); success 1 % / 1 % / 6 %; stand-still 0.139. Phase-conditioned policies stall once
ρ saturates at step 122 — fix the clock next.

**MLP+score ablation** (score regularizer, no phase input; `mlp_score_w…`) on both tasks. Push-T,
closed loop: 0.594 ≈ MLP+phase+score 0.598 (p = 0.90) and > MLP 0.487 (p = 0.03); offline it does
not beat MLP (p = 0.47). Skate (`output/skate/policy/SUMMARY.md`, all 3 scenarios × 5 n × 5 seeds):
indistinguishable from MLP in every cell, and worse than MLP+phase+score on MMD in 8/15 cells —
on skate the score term only helps together with phase.

The sections below record how we got here; their DP-split numbers are superseded.

### Data audit and first policy arms (2026-09-16)

**Data audit, all against the raw zarr:** state/action/keypoint/episode_ends bit-identical;
split identical to DP's own `sampler.py` (fetched and executed); block heading consistent with the
keypoints (a flipped sign would be off by 236 px); grid endpoints exact; no window crosses an
episode. Known and harmless: 47 frames, all in the first 1–4 steps of an episode, where keypoints
disagree with the block pose by ≤ 2 px (keypoints are not a policy input); `action[t]` leads the
agent by ~2 steps (PD lag, not an off-by-one); `prep.normalizer("state", "box")` scales theta by
256 — never use it on the heading.

**Policy arms** (`fm_bc/pusht/policy.py`, `policy_training.py`,
`scripts/pusht/train_pusht_policy.py`). Current definition, after two corrections from the user (do
not revert either):

- **Output is the agent's phase velocity** `u = dx/dρ = (L−1)(x_{k+1} − x_k)` in normalized
  position units, so `x_{k+1} = x_k + u·Δρ`. Not the dataset's absolute `action` column — that
  first version was wrong for the paper's math.
- **Inputs are `[o, x, ρ]` at the current step**: block `[xy, cos θ, sin θ]`, agent xy, `2ρ−1`
  (oracle ρ = t/(L−1)). No observation history.
- **Training follows DP's `pusht_lowdim` workspace**, not the skate recipe: epochs over batches of
  256, AdamW 1e-4 (0.95, 0.999, wd 1e-6), cosine with 500 warmup steps, EMA (power 0.75), min-max
  normalizer (fitted on training episodes; DP fits on all). Val loss on DP's 4 validation
  episodes every epoch; `best.pt` by EMA val loss, EMA checkpoints every 50 epochs, `last.pt`.
  The 112 unused episodes are the held-out test set (DP uses env rollouts instead).
- Score term: agent-xy kernel score at the row's phase (32 px), λ 0.05 → 0 over the first half.

**Results, 500 epochs × 5 seeds** (`runs/pusht/policy_dp_split/`, ~35 s/run; held-out
phase-velocity RMSE, u = 0 scores 6.745):

| arm | val-selected `best.pt` | `last.pt` | best epoch on held-out (not selectable) |
|---|---|---|---|
| MLP | 6.416 (epoch ~10) | 7.133 | 6.00 (epoch ~100) |
| MLP+phase | 6.416 (epoch ~10) | 6.632 | 5.94 (epoch ~140) |
| MLP+phase+score | 6.416 (epoch ~10) | 6.629 | 5.94 (epoch ~140) |

Three problems, in order of importance:

1. **`[o, x, ρ]` barely determines the phase velocity.** Model-free nulls on the same inputs:
   kNN explains R² ≈ 0.16 held-out (0.19 for per-step displacement, so the phase normalization is
   not the cause). The human demos move the pusher very differently from the same configuration.
   Everything beyond that is memorization (train RMSE → 2.7).
2. **The 4 validation episodes cannot select a model.** Their u = 0 RMSE is 4.95 against 6.75 on
   the held-out set — slow, easy episodes — so val loss rises as soon as the model predicts
   non-trivial velocities, and selects epoch ~10, still inside LR warmup and identical for all
   three arms. DP never selected on val loss (top-k is by rollout success).
3. **The score term is effectively off** at λ = 0.05: λ·E|b| is ~2 % of the MSE (it was ~3× the
   MSE for the absolute-action target), because the MSE scales as (L−1)² and b as (L−1). λ needs
   rescaling (~2.5 for an equal share) before MLP+phase+score says anything.

(Superseded runs kept for reference: `runs/pusht/policy_abs_action`, `policy_velocity_to2`,
`policy_oxrho_skate_recipe`. With a 2-step history the phase velocity was far more predictable,
held-out best ≈ 2.5 vs u = 0 6.25, because the previous displacement leaks into the input.)

### Formerly blocking: the data had never been downloaded

`data/pusht/` does not exist. Every artifact so far was validated against a **synthetic** zarr in
the published layout — the real demonstrations have never been through this pipeline:

```bash
scripts/pusht/download_pusht.sh                       # ~30 MB; prints the archive sha256
python scripts/pusht/convert_pusht.py                 # → data/pusht/pusht_episodes.npz
python scripts/pusht/prepare_pusht.py --n-grid 128    # → data/pusht/pusht_prepared.npz
python scripts/pusht/inspect_pusht.py --data data/pusht/pusht_prepared.npz   # must exit 0
```

Then expect: 206 episodes / 25 650 steps, 90 train / 4 val / 112 unused, `grid_dt` spread ≈ 4×.
Paste the printed sha256 into `PUSHT_ZIP_SHA256`'s default in the download script to pin it.

### Also outstanding

- **`pytest tests -q` has never run with Theseus present.** The only changes to pre-existing code
  are the two-line `chunks.py` re-import and the lazy `fm_bc/__init__.py`; both were verified
  against stubbed modules, but the real suite is the check that matters. Run it in the `fm_bc`
  env.
- **The CVAE ablation table is unrun**: `λ = 0`; `κ ∈ {0.1, 0.3, 0.5, 0.8}` with **`κ = 1.0`
  labelled as the supervised ceiling, never a data point** (§4); `To ∈ {2, 4, 8}`;
  `Z ∈ {2, 4, 8, 16}`; fixed vs log-uniform gaps; 3 seeds each.
- No figures and no `SUMMARY.md` for the progress model — deliberately out of scope for the first
  cut.

### Reading a CVAE run's results, in order

**Run `scripts/pusht/train_progress.py --baselines-only` before anything else.** It reports two
label-free nulls the CVAE must clearly beat before any of its numbers mean anything: `rho_geom`
(hand-coded `−d_SE2(block, goal)` — the goal pose is constant across every episode, so this is
already a progress measure) and `rho_linear` (a linear probe on the same window). On the synthetic
fixture `rho_geom` alone scores **0.88**. If the CVAE does not beat these, it has learned nothing,
and that is worth knowing before spending GPU time.

Then read the run in this order. The first item is a gate and voids everything after it;
`scripts/pusht/train_progress.py::_report` prints it as an explicit pass/fail and says plainly when
nothing below it is meaningful.

1. **`phase_ablation` > 5 %.** Reconstruction is the only term grounding `z_phase` in task
   structure — the hinge alone is satisfied by *any* monotone-along-time statistic. If the decoder
   was not using `z_phase`, nothing shaped the encoder and every rank correlation is meaningless
   however good it looks.
2. **`spearman_per_episode_mean` clearly above `max(rho_geom, rho_linear)`.**
3. **`calibration_gap` < 0.15.** Per-episode rank correlation is invariant to *any* monotone
   reparameterization, so it cannot tell you whether `z_phase = 0.6` means the same thing in two
   episodes — exactly what reading the scalar out at inference assumes. If the gap is large, set
   `kappa_hi = 2.0` for the two-sided band.
4. **`roughness` < 0.3**, `ood_p95` usable as a threshold, `spearman_no_cond` not far below
   `spearman_with_cond`, `active_units` ≥ 2, `spearman_t3` not collapsing in the tail.

Headline held-out set is the **112 UNUSED episodes**, not the 4 validation ones (which are reported
separately as `val_split`). `PushTPrepared.unused_ids` exposes them, and evaluation **falls back to
`val_ids` when `unused_ids` is empty** — which is what happens on the toy fixture and on any split
where `max_train_episodes` covers every episode.

### Still deferred (ingredients are stored)

- **Observation vector**: the 6-dim `[agent_xy_n, block_xy_n, cos θ, sin θ]` currently in
  `pusht/chunks.py`, vs DP's benchmark-comparable 20-dim `[keypoint(18), agent_xy(2)]`. Both are
  derivable from the stored artifact; `keypoint` is retained for exactly this reason.
- **Action head**: absolute position vs delta/velocity.
- **Divergence metric**: `xy` on the agent (D=2) vs `se2` on the block (D=3).
- **`log(r_k + 1e-8)` in `fm_bc/divergence.py`.** This fabricates large divergence spikes where
  demos share identical states. It will bite harder on Push-T than on the skate task, because a
  stalled pusher produces genuinely identical consecutive states. Not yet addressed.
- **Evaluation needs a simulator.** `fm_bc/skate/evaluation.py` is skate-specific throughout
  (`spec.world`, `spec.label_mode`, `SkateLimits`). Push-T rollouts need `gym-pusht` or the DP
  env, and nothing of that exists yet.
- Goal conditioning is a meta constant (`[256, 256, π/4]`), not a per-episode field.

---

## 6. Facts worth not rediscovering

- **No frame of the released demonstrations reaches success.** Coverage maxes at 0.9489 against a
  0.95 threshold — human operators stop just short. A `success` column derived from the demos is
  identically false and is not a usable label.
- **DP's "Push-T, 200 demos" actually trains on 90 episodes.** `seed 42`, `val_ratio 0.02`,
  `max_train_episodes 90`. The split logic in `splits.py` was checked verbatim against
  `diffusion_policy/common/sampler.py`, including the subtlety that `downsample_mask` draws
  indices *into the train-index array*, and that each function makes its own `default_rng(seed)`.
- **zarr 3.0+ requires Python ≥ 3.11**, but Theseus pins the repo to 3.10. The reader uses only
  the API common to 2.x and 3.x (`zarr.open(store, mode="r")`, `root[key][:]`), so the pin is
  `zarr>=2.16` with no upper bound and the version resolves per-environment.
- **The `(-π,π]` seam is not a bug.** Gridded `block_theta` legitimately shows a raw ~2π step
  wherever a trajectory crosses the boundary. Any check on heading continuity must use the
  **wrapped** difference — the meaningful failure is a wrapped step near π, meaning the geodesic
  went the long way. This caught me twice: once in a test and once as a genuine bug in
  `inspect_pusht.py`.
- **Action chunks start at the oldest observed step** `k − To + 1`, not at `k`. The first `To−1`
  entries are the past.
- `fm_bc/se2.py`, `fm_bc/divergence.py` and `fm_bc/nets/` are Theseus-free and shared.
  `fm_bc/skate/policy.py`, `baselines.py`, `training.py`, `baseline_training.py`,
  `evaluation.py` all reach `skate_planner` → `theseus` and must not be imported from the Push-T
  subtree. The progress modules restate `_ACTIVATIONS`, `lr_factor`, the power EMA and
  `_write_log` for this reason (and `fm_bc.skate.training._write_log` could not be reused anyway —
  it writes against a module-global `_LOG_FIELDS` with `extrasaction="ignore"` and would silently
  drop columns).
  **If you add a logged column to the CVAE, add it to the local `_LOG_FIELDS`** or it vanishes
  silently.
- **`ood = ‖μ‖²/Z` is not ≈1 in distribution.** Its absolute scale depends on how hard the KL bit
  during training; on the toy fixture it runs ~0.007–0.2. Calibrate downstream thresholds against
  the `ood_p95` written to `metrics.json`, never against a fixed constant. It is deliberately not
  reconstruction error — that would mean shipping the decoder, and only the encoder is meant to.
- **`w_theta` is a buffer computed from the normalizer's scale**, not a hardcoded constant, so the
  heading weighting adapts if the normalizer mode changes. Under `box` it is
  `(ROT_SCALE_PX_PER_RAD / 256.000256)² ≈ 0.1526`, which is what the test pins. Unweighted, the two
  `cos/sin` dims would take ~74 % of the reconstruction gradient purely as an artifact of the
  embedding — their variance is ~0.5 against ~0.09 for box-normalized positions.
- **The CVAE decoder emits a 2-vector projected onto the unit circle, never an angle.** The chord
  term `2(1 − cos dθ)` is a von Mises NLL up to constants, so it is a proper likelihood with no
  Bessel function; emitting `θ` directly would reintroduce exactly the wrap discontinuity the
  `cos/sin` embedding exists to avoid.
- **`sample_pairs` resolves partners through a `row_of_step` scatter, never `idx + 1`.** Rows are
  one per flat step in buffer order, so an increment would pair one episode's last window with the
  next episode's first — and under `episode_ids` filtering the rows are a subsequence where `r + 1`
  is not even the next step. Invalid partners carry a `0` sentinel and are masked out of the
  hinge's mean, which `test_invalid_pairs_do_not_contribute_to_the_monotonicity_term` guards.

## 7. Deviations from the original plan

- Added `fm_bc/pusht/prepared.py`, which the plan implied would live in `grid.py`. It holds
  splits and normalizers as well as the grid, so that would have been a muddy home.
- The skate task was **not** migrated to `fm_bc/skate/`. Symmetric, but a ~15-file rename on a
  50+-commit branch for no Phase-1 benefit. The asymmetry is deliberate.
- On-disk format is `.npz` + a JSON `meta` string (matching `TrajectoryDataset.save`), with torch
  tensors in memory and conversion at the I/O boundary only. `.pt` was the alternative; the repo
  reserves that for checkpoints.

## Stretched clock (2026-09-17)

`NominalClock(median_length, phase_length)` separates how fast ρ advances (`phase_length`) from the
step size (still `1 / (L_med − 1)`); `scripts/pusht/rollout_pusht_policy.py --phase-length L` writes
`rollout_nominal_L{L}_inverse_pd.*`. With L = 246 (longest demo) stalling drops from 71–72 % to
38 % (MLP+phase) / 21 % (+score) and final coverage rises slightly (0.419 → 0.458 for +score), but
score (max coverage over the episode) is unchanged (0.598 → 0.603, p = 0.85): the slower clock
makes the early push weaker (coverage at step 122 0.40 → 0.23). L = 300 is no better. Table in
`output/pusht/SUMMARY.md`.

## MMD to the test demos (2026-09-17)

`fm_bc/pusht/evaluation.py` + `scripts/pusht/evaluate_pusht_mmd.py`: the skate metric for Push-T.
Roll out from each of the 21 test demos' start states for its length with its own clock, squared MMD
at matched phase (128-point grid, 100 px Gaussian kernel on
`[agent_xy, block_xy, 100 cos θ, 100 sin θ]`). All four methods land at 9–11 ×10⁻² (validation-demo
floor 6.1, never-moving pusher 21.8), with no significant difference against MLP+phase -- MMD does
not separate what the rollout score does. `place_state` now resets to the requested state before
placing it (a random reset left contacts that shifted the block by up to 1.4 px in ~15 % of
placements; the DP-protocol rollouts never used it). `fig_pusht_violin` shows the per-episode score
distribution with percentile lines and the MMD panels.

## Learned phase clock (2026-09-17)

`fm_bc/pusht/phase_mlp.py` + `scripts/pusht/train_pusht_phase.py`: a 3-layer MLP (512, 512, sigmoid)
on `[agent_xy, block_xy, cos θ, sin θ]`, MSE against `ρ = k / (L − 1)`, fit on all 206 demos, one
per policy seed. Phase RMSE 0.025 (≈ 3 steps). `MonotonePhase` wraps it as a rollout clock,
`ρ_t = max(ρ_{t−1} + ε, MLP(state_t))`, with optional EMA smoothing and a rate cap;
`scripts/pusht/rollout_pusht_policy.py --clock phase_mlp` writes `rollout_phase_mlp_inverse_pd.*`.
Nominal-clock results are untouched and stay the baseline.

**It does not beat the nominal clock.** With ε = 1/200 the score arms lose (MLP+phase+score
0.598 → 0.521, p = 0.005; MLP+score 0.594 → 0.537); MLP+phase is unchanged (0.469 → 0.468). Why:
the estimator is accurate on demo states but its estimate rises slowly on rollout states (it beats
the ratchet on only 5 % of steps), so the ε floor sets the rate and the clock degenerates to a
1/200 clock. Smaller ε (1e-3, 1e-4) is worse still (score 0.28 / 0.15 on 20 envs): with `Δx = u Δρ`,
a phase that only advances when the block moves cannot get the pusher started -- a deadlock the
open-loop clock cannot have. EMA smoothing and rate caps change nothing.

**Decoupled (`--phase-fixed-step`, full protocol).** The estimate conditions the policy, the step
stays `1 / (L_med − 1)`. The phase-free arms then reproduce the baseline rollout exactly (a
consistency check). The phase arms stay below it: MLP+phase+score 0.542 ± 0.042 (ε = 1e-3, p = 0.05
vs 0.598) and 0.513 (ε = 1e-4); MLP+phase 0.457 / 0.433 vs 0.469. Stalling drops sharply
(72 % → 21 %, 71 % → 12 %) and final coverage holds, so the estimate does keep the robot working --
it just does not reach a higher best coverage than the median-length clock.

**Start-pose estimator (`--use-start`, `runs/pusht/phase_start_702010`).** Feeding the episode's
initial block pose alongside the current state cuts the estimator's phase RMSE from 0.025 to
**0.007** (~0.9 steps) and raises its correlation with geometric progress along a rollout from 0.63
to 0.81; the estimate at a rollout's first step falls from 0.10 to 0.065. It does **not** move the
rollout score: with the estimate's own step 0.568 ± 0.058 (vs 0.521 for the plain estimator, 0.598
baseline), with the fixed step 0.536 ± 0.045 (vs 0.542). Both estimators still beat the ratchet on
only 5 % of steps and reach ρ = 0.69 at step 122, so the clock rate is set by ε either way: on
rollout states the estimate rises more slowly than the demonstrations' phase, however accurate it is
on demonstrations.

**ε = 1/245 (one step of the longest demonstration).** The floor of the ratchet is then the L = 246
clock itself, and the estimate rides above it (79 % of steps, mean gap 0.10 with the fixed step).
This is the best learned clock so far: with the fixed step MLP+phase reaches 0.490 ± 0.042 (baseline
0.469, and equal to the plain L = 246 clock) and MLP+phase+score 0.589 ± 0.044 (baseline 0.598,
L = 246 0.603); final coverage 0.338 / 0.416 against 0.312 / 0.419. With the estimate's own step it
is worse again (0.461 / 0.548) but stalls least (36 % / 30 %). So the gain over ε = 1e-3 comes from
the floor, not from the estimate: a learned phase has still not beaten an open-loop clock here.

**Evaluation condition for the figures.** `scripts/pusht/plot_pusht.py` now defaults to
`rollout_phase_mlp_start_fixedstep_e1o245_inverse_pd` (start-pose estimator, ε = 1/245, fixed step)
-- `--rollout-tag` switches clocks, and the tables in `output/pusht/SUMMARY.md` keep every clock
with the nominal one as the baseline. The figures mark step 245, where the ratchet's floor reaches
ρ = 1, in place of the old step 122, and the filmstrip frames follow that clock.

## 90/10 split (2026-09-17)

`data/pusht/pusht_prepared_9010.npz` (`--split ratio --fractions 0.9 0.1 0.0`, seed 42): 185 train /
21 val / no test -- the rollouts are the test. Every arm retrained into `runs_9010/` (21 methods ×
5 seeds, `scripts/pusht/train_pusht_policy.py --prepared ... --out runs_9010`), phase estimators
into `runs_9010/pusht/phase_start`; `output/pusht_9010/SUMMARY.md`. The 70/20/10 runs are untouched.
Since 2026-09-24 the 90/10 runs (the CHPC 10-seed set) live in `runs/pusht/policy/` and
`runs/pusht/phase_start/`, the 70/20/10 ones in `runs/pusht/policy_702010/` and
`runs/pusht/phase_start_702010/`; the commands below use the new paths.
λ = 2.5, anneal 0.5, h = 32 px is still the best score configuration on the new split.

**Rollouts (eval clock: start-pose estimator, ε = 1/245, fixed step).** The phase arms gain and the
phase-free arms lose: MLP+phase 0.490 → **0.589** (p = 0.008), MLP+phase+score 0.589 → **0.619**,
MLP 0.487 → 0.459 (p = 0.07), MLP+score 0.594 → 0.523. Under the nominal clock the same pattern is
weaker (MLP+phase 0.469 → 0.501, MLP+phase+score 0.598 → 0.595). With 90/10 and the eval clock the
ordering is finally the one the paper argues for: MLP 0.459 < MLP+score 0.523 < MLP+phase 0.589 <
MLP+phase+score 0.619.

**Throughput.** These 4×512 MLPs at batch 256 are launch-bound: one process leaves the 4090 at ~5 %
utilization and 61 s/run. Five parallel processes give 2.7 runs/min, ten 4.0 runs/min (the sweep's
score arms are heavier, ~314 s/run with fifteen). Shard by seed with `setsid nohup`; the whole
105-run retrain took ~45 min instead of ~1 h 45 min.

## ACT chunk baseline (2026-09-18) -- implemented; since trained, 10 seeds

`fm_bc/nets/act.py` (`ACT`) + `PushTACTPolicy` in `fm_bc/pusht/baselines.py`, kind `pusht_act`,
registered as `act`. A CVAE transformer trained with L1 + 10*KL whose latent is fixed to the prior
mean `z = 0` at inference, so **it is deterministic**: MMD and any diversity metric will read
~zero for this arm by construction, not by failure. Ported at LeRobot's defaults, including
`n_decoder_layers = 1` -- the released ACT builds seven and reads the first layer's output, so the
rest never train, and LeRobot matches that deliberately. ~40 M parameters against DP's 65 M.

Three deviations, all recorded in the run's `config.json`:

* **Batch 256, not ACT's 8.** This loop is epoch-based, so batch 8 would cost 32x the optimizer
  steps for the same data. The nav side keeps ACT's 8 (step-budgeted) and offers `--batch-size 256`
  as the data-matched control.
* **Chunk 16 / 8 executed**, not LeRobot's 100 / 100 -- to stay comparable to DP and MIP here.
* **Edge-padded windows enter the CVAE encoder.** ACT masks padded actions out of it;
  `fm_bc.pusht.chunks` edge-pads and the other chunk baselines train on those targets, so ACT does
  too.

Trained with `--lr 1e-5 --no-ema` (ACT's recipe). `use_ema=False` makes `best.pt` the raw model at
its lowest validation loss; the `selection` field in `metrics.json` says which rule was used.
Reported **open loop** (`--loop open`), matching LeRobot's `temporal_ensemble_coeff = None`.

Verified on CPU: shapes, determinism, the fixed-generator validation loss, the checkpoint round
trip, the KL term's effect, and a real 4-epoch `train_pusht_baseline` run on synthetic episodes.
Not yet verified: anything needing the real dataset, a GPU, or `gym-pusht`.

## Generative chunk baselines: DP and MIP (2026-09-17)

`fm_bc/pusht/baselines.py` (`PushTDiffusionPolicy`, `PushTMIPPolicy`) + `baseline_training.py` +
`scripts/pusht/train_pusht_baselines.py` / `rollout_pusht_baselines.py`. Same 90/10 split, 500
epochs, optimizer, EMA and best-EMA-validation-loss selection as the MLP arms; the model predicts
the env's action instead (2 states in, 16 absolute pixel targets out, 8 executed per re-plan), so no
clock and no inverse PD. Both loops of the skate comparison: open (chunk as trained) and closed
(ACT temporal ensembling, m = 0.01). `rollout_chunk_policy` runs the 50 envs in lockstep so the
sampler is batched -- closed-loop DP is 300 DDPM chains per episode and is only affordable batched.

| method | loop | score | success | final coverage |
|---|---|---:|---:|---:|
| Diffusion Policy (3 seeds) | open | **0.959 ± 0.021** | 42.7 % | 0.837 |
| Diffusion Policy | closed | 0.819 ± 0.012 | 32.0 % | 0.634 |
| MIP (5 seeds) | open | 0.410 ± 0.046 | 2.4 % | 0.215 |
| MIP | closed | 0.411 ± 0.030 | 0.4 % | 0.185 |
| MLP+phase+score (5 seeds, eval clock) | — | 0.619 ± 0.061 | 6.0 % | 0.456 |

DP is far ahead of everything: action chunks plus a multimodal sampler beat single-step regression,
which is the averaging failure the MLP rollouts stall on. MIP's two-step sampler does not carry it.
Both baselines select very early checkpoints (DP epoch 44-54 of 500, MIP 119-155) and overfit hard
afterwards (DP val loss 0.027 -> 0.10), so 500 epochs is far more than they need here.

**Figures with the baselines.** `plot_pusht.py` now also reads the chunk baselines' rollouts
(`BASELINE_TAGS`: `rollout_chunk` open loop, `rollout_chunk_te0.01` closed) and shows all eight arms
in `fig_pusht_violin` and `fig_pusht_coverage`; the per-env grids and filmstrips stay readable with
`GRID_METHODS` (the four MLP arms and DP open loop). `scripts/pusht/benchmark_pusht_inference.py`
writes `inference.csv` (batch-1 latency on GPU and CPU; a phase arm is timed *with* its phase
estimator) and `fig_pusht_inference` follows `evaluate_policies.fig_inference` exactly -- same bar
rows (fill = GPU, hatch = CPU), value labels with the control rate, latency-budget lines (now 10 Hz
Push-T as well as 60 Hz and 1 kHz), and a score-vs-cost panel with a Pareto frontier per device,
circles for the policy as trained and squares for closed loop.

| method | ms/step (GPU) | ms/step (CPU) | score |
|---|---:|---:|---:|
| MLP / MLP+score | 0.21 / 0.20 | 0.13 | 0.459 / 0.523 |
| MLP+phase (+ estimator) | 0.49 | 0.28 | 0.589 |
| MLP+phase+score (+ estimator) | 0.48 | 0.28 | 0.619 |
| MIP open / closed | 0.25 / 1.98 | 0.88 / 7.01 | 0.410 / 0.411 |
| DP open / closed | 57.5 / 460 | 495 / 3958 | 0.959 / 0.819 |

The frontier is MLP (cheapest) -> MLP+score -> MLP+phase+score -> DP open loop: DP buys its score
with ~120x the per-step cost of the phase arms on a GPU, and cannot run at 10 Hz on this CPU
(495 ms per control step open loop).

## Checkpoint selection: `best.pt` vs `last.pt` (2026-09-17)

Every arm is selected as the EMA model at the lowest EMA validation loss, but Diffusion Policy's own
recipe never early-stops, and a denoiser's validation loss is a poor proxy for rollout score. So
every rollout condition was repeated with each run's final weights: `--checkpoint last.pt` on
`rollout_pusht_policy.py` / `rollout_pusht_baselines.py`, which appends `_last` to the tag so both
evaluations coexist next to the checkpoint. `summarize_pusht_policy.py` pairs them ("Checkpoint
selection" in `output/pusht_9010/SUMMARY.md`, paired t-test over seeds).

| method | condition | score `best.pt` | score `last.pt` | Δ | p | success |
|---|---|---:|---:|---:|---:|---|
| MLP | eval clock | 0.459 | 0.408 | −0.052 | 0.015 | 2.0 → 7.6 % |
| MLP+score | eval clock | 0.523 | 0.412 | −0.111 | 0.023 | 7.6 → 7.6 % |
| MLP+phase | eval clock | 0.589 | 0.603 | +0.014 | 0.565 | 4.4 → 10.4 % |
| MLP+phase+score | eval clock | 0.619 | 0.582 | −0.036 | 0.367 | 6.0 → 8.4 % |
| DP | open | 0.959 | 0.966 | +0.007 | 0.672 | 42.7 → **61.3 %** |
| DP | closed | 0.819 | **0.955** | +0.135 | 0.009 | 32.0 → **60.7 %** |
| MIP | open | 0.410 | 0.459 | +0.049 | 0.347 | 2.4 → 4.0 % |
| MIP | closed | 0.411 | 0.452 | +0.041 | 0.422 | 0.4 → 0.4 % |

**The two families want opposite checkpoints.** The chunk baselines gain from training to the end:
DP's validation loss rises 0.027 → 0.12 after its epoch-47 minimum, yet closed-loop score goes
0.819 → 0.955 and success doubles in both loops -- early stopping was throwing away most of DP's
closed-loop performance. MIP gains a little in both loops. The single-step regression arms lose:
MLP −0.052 and MLP+score −0.111 (both significant), while the two phase arms are flat, so
`best.pt` stays the right choice for them and the headline comparison is unchanged.

**Success rises for every arm even where score falls.** Score is `max clip(coverage / 0.95, 0, 1)`
and success is coverage > 0.95, so overfit weights that reproduce the demonstrations more sharply
finish a few more episodes outright while doing worse on average (MLP 2.0 → 7.6 % success at
−0.052 score). The gap is largest for DP open loop, which was already scoring 0.96: almost all of
`last.pt`'s gain there shows up in success, not score.

Figures and the headline tables stay on `best.pt`.
`plot_pusht.py --rollout-tag phase_mlp_start_fixedstep_e1o245_inverse_pd_last` regenerates the whole
set on the final checkpoints (a `_last` tag switches the chunk baselines to their `_last` rollouts
too).

## The score regularizer without its schedule (constant λ, 2026-09-17)

The score arms anneal λ to 0 over the first half of training, so the weights they *end* on are
trained by the plain velocity MSE -- the regularizer only shapes the path there, and `last.pt` of an
annealed arm has no regularizer acting on it at all. `score_anneal_frac = 0` now means "never
anneal" (`score_weight_at` returns λ at every step) and names the run `score_w<λ>_const` /
`mlp_score_w<λ>_const`. Same λ = 2.5, h = 32 px, 90/10 split, 500 epochs and seeds 0-4.

| method | λ schedule | clock | score `best.pt` | score `last.pt` | Δ |
|---|---|---|---:|---:|---:|
| MLP+score | annealed | either | 0.523 ± 0.048 | 0.412 ± 0.029 | −0.111 |
| MLP+score | constant | either | 0.546 ± 0.033 | 0.466 ± 0.046 | −0.080 |
| MLP+phase+score | annealed | eval | 0.619 ± 0.061 | 0.582 ± 0.066 | −0.036 |
| MLP+phase+score | constant | eval | 0.630 ± 0.022 | 0.593 ± 0.033 | −0.038 |
| MLP+phase+score | annealed | nominal | 0.595 ± 0.064 | 0.532 ± 0.026 | −0.063 |
| MLP+phase+score | constant | nominal | 0.572 ± 0.036 | 0.556 ± 0.041 | −0.016 |

**No significant difference in score either way** (paired over seeds, p = 0.17-0.74 on all eight
comparisons). Constant λ is ahead in seven of eight cells and behind only for MLP+phase+score at
`best.pt` on the nominal clock, but with five seeds none of it separates.

**What does change is the spread and the overfitting.** Constant λ roughly halves the seed std of
MLP+phase+score (0.061 → 0.022 at `best.pt`, 0.066 → 0.033 at `last.pt`), and its validation loss
degrades far less over the second half of training: 13.64 → 14.29 against the annealed arm's
13.41 → 15.64 (MLP+score 13.95 → 16.69 against 13.85 → 18.57). Holding the regularizer on acts as
a regularizer in the ordinary sense. It does not close the `best.pt`/`last.pt` gap, though: the
final model is still worse in every arm, so unlike DP (which gains from training to the end), the
single-step regression arms want early stopping whether or not λ is annealed.

Offline cost of holding λ on is negligible: best-epoch validation loss 13.64 vs 13.41
(MLP+phase+score) and 13.95 vs 13.85 (MLP+score).

**Constant λ is now the default arm** (2026-09-17). `plot_pusht.py` draws both score arms from the
`_const` runs and `--score-arms annealed` switches back; `summarize_pusht_policy.py` still sweeps
λ, the anneal fraction and the bandwidth over the *annealed* runs -- that is what selects the
configuration -- and then reports its constant-λ twin as MLP+phase+score, with the head-to-head in
the ablation above. Every figure in `output/pusht_9010/figures/` and every table in its
`SUMMARY.md` is the constant-λ arm; the ranked filmstrips are named `..._score_w2.5_const`.
The inference figure resolves a schedule it has no measurement for to its twin's row -- the two
schedules are the same network, so the latency is the same and nothing was re-benchmarked.

## Transformer backbone: the phase-former arms (2026-09-18)

The same four arms on an AdaLN transformer instead of the MLP
(`PushTTransformer` in `fm_bc/pusht/policy.py`, `--arch transformer`, run names suffixed `_tf`).
The MLP concatenates `2 rho - 1` to its input vector; the transformer tokenizes the observation
per entity per step, reads it with cross-attention, and injects `rho` multiplicatively at every
normalization instead. **Everything else is held fixed**: the 90/10 split, λ = 2.5 held constant,
bandwidth 32 px, 500 epochs, batch 256, lr 1e-4, EMA, `obs_steps = 1`, seeds 0-4. Parameters are
matched to within 10 % (792,578 MLP vs 874,498 transformer), so this isolates the backbone rather
than the budget.

**The transformer is ahead in all four conditions**, eval clock at `best.pt`, paired over seeds:

| condition | run name | MLP | transformer | p |
|---|---|---|---|---|
| no phase, no score | `mlp_tf` | 0.459 ± 0.019 | 0.582 ± 0.089 | 0.034 |
| no phase, score | `mlp_score_w2.5_const_tf` | 0.546 ± 0.033 | 0.607 ± 0.057 | 0.058 |
| phase, no score | `fm_tf` | 0.589 ± 0.048 | 0.647 ± 0.023 | 0.055 |
| phase, score | `score_w2.5_const_tf` | 0.630 ± 0.022 | **0.683 ± 0.035** | 0.005 |

The arm ordering is unchanged -- score helps, phase helps, both is best -- so the backbone lifts
the whole family rather than reshuffling it, and the best arm is still MLP+phase+score's
counterpart at 0.683. `best.pt` still beats `last.pt` (0.683 vs 0.658), so the early-stopping
finding above survives the backbone change. On the nominal clock the phase arms give up most of
the gain's significance (0.572 → 0.601, p = 0.29 at `best.pt`) but keep the direction.

**Validation loss ranks these arms wrongly -- do not select on it.** Offline, the transformer's
gain looks like it lives entirely in the phase-free arms (13.97 → 12.03 and 13.95 → 12.30) and
vanishes when the phase is added (13.54 → 13.57, 13.64 → 13.60). The rollouts say the opposite:
the phase arms are exactly where the transformer's advantage is largest and most significant.
This is the same score/validation divergence as the `best.pt`/`last.pt` section above, and it is
now sharp enough to be a rule: **on this task, offline velocity MSE does not order policies.**

**The phase arms converge in ~15 epochs.** Best epoch is 13 (`fm_tf`) and 16
(`score_w2.5_const_tf`) against 150 and 221 for their MLP twins; the phase-free transformer arms
take 52 and 78 against 111 and 152. Driving `rho` through adaptive layer norm at every block is a
far stronger conditioning path than concatenating it to the input, and the model exploits it
almost immediately, after which validation drifts while the rolled-out policy stays good. 500
epochs is ~30x more than the phase arms need; budget accordingly before running a sweep.

Reproduce (about 30 min wall for all 20 runs, four arms in parallel on one 4090, ~6 min per run):

```
python scripts/pusht/train_pusht_policy.py --arch transformer --methods mlp --seeds 0 1 2 3 4 \
    --prepared data/pusht/pusht_prepared_9010.npz --out runs --device cuda
# ... and --methods fm / --methods mlp_score / --methods score, the latter two adding
#     --score-weights 2.5 --score-anneal-fracs 0
python scripts/pusht/rollout_pusht_policy.py --runs runs/pusht/policy \
    --prepared data/pusht/pusht_prepared_9010.npz \
    --methods mlp_tf fm_tf mlp_score_w2.5_const_tf score_w2.5_const_tf \
    --clock phase_mlp --phase-runs runs/pusht/phase_start \
    --phase-eps 0.004081632653061225 --phase-fixed-step \
    --tag-suffix _start_fixedstep_e1o245 --checkpoint best.pt --workers 10
```

**Not yet in the figures or `SUMMARY.md`.** `summarize_pusht_policy.py` discovers arms by globbing
run directories, so the `_tf` arms enter its tables on the next run -- and its `best_rollout` picks
the top-scoring arm across everything rolled out, which is now `score_w2.5_const_tf`. Decide how
the transformer arms should be presented before regenerating, or that selection silently promotes
them to the headline method.

**Figures** (2026-09-18). `plot_pusht.py --arch transformer` draws every rollout figure from the
`_tf` runs and writes it as `fig_pusht_*_tf`, so the two backbones' sets coexist and neither
overwrites the other; "TF" takes "MLP"'s place in the labels, naming the backbone exactly as "MLP"
does for the MLP arms. The baselines are unchanged by the flag, so their ranked filmstrips are not
redrawn under a `_tf` name. `fig_pusht_violin_backbones` is the head-to-head: each arm's MLP and
transformer side by side, the pair sharing the arm's colour with the transformer hatched -- eight
policy violins plus DP and MIP would otherwise need more categorical hues than stay separable.

**The transformer costs about 9x the MLP per decision**, and it had to be measured: unlike the two
λ schedules, this is a different network, so the inference figure's twin fallback does not apply.
`benchmark_pusht_inference.py` now takes `--methods`, `--baselines` and `--append` (merge into the
existing `inference.csv`, replacing only the rows it re-measures, so adding an arm does not perturb
published numbers), and reads `use_phase` off the loaded model rather than a hardcoded list of
phase-arm names. On this machine, per decision at batch 1: 0.21 → 1.89 ms on the 4090 and
0.13 → 1.17 ms on the CPU without the phase estimator, 0.49 → 2.29 ms and 0.28 → 1.45 ms with it.
That is still ~40x inside Push-T's own 10 Hz budget and three orders of magnitude cheaper than
DP's 460 ms, so the backbone's cost does not change any deployment conclusion.

**Both checkpoints, both backbones** (2026-09-18). `--rollout-tag <tag>_last` now also tags the
file names, so the overfit set lands beside the best-validation one instead of overwriting it:
the two tags are the backbone (`_tf`) then the checkpoint (`_last`), giving
`fig_pusht_violin`, `_tf`, `_last` and `_tf_last`. Four sets of every rollout figure, plus
`fig_pusht_violin_backbones` and `fig_pusht_violin_backbones_last` for the head-to-head.

The overfit figures restate the split from the checkpoint section above, now on both backbones.
The chunk baselines want the final model -- DP's median episode score is 1.00 at `last.pt` with
97 % of episodes ≥ 0.9 and success 61 %, against 0.99 / 94 % / 43 % at `best.pt`. The phase-free
regression arms want early stopping and lose badly: MLP 0.459 → 0.408 and the transformer
0.582 → 0.459, so **the transformer gives up more of its advantage to overfitting than the MLP
does** (its phase-free gain is +0.12 at `best.pt` but only +0.05 at `last.pt`, and its score arm's
edge disappears entirely, 0.466 vs 0.462, p = 0.92). The phase arms are the exception in both
backbones -- roughly flat to slightly better by median at `last.pt` -- which is why the
`best.pt`/`last.pt` gap never looked like a single story across the arms.

## Final position and orientation error replace final coverage in the violins (2026-09-18)

The violins' per-episode panels are now **score**, **final position error [px]** and **final
orientation error [deg]** to the goal (`goal_error`), not score and final coverage. Coverage is an
area overlap, so it saturates near the goal and is nearly blind to orientation. The two components
are reported apart rather than blended into one SE(2) distance: blending needs a px-per-radian
exchange rate (`ROT_SCALE_PX_PER_RAD` = 100) that decides the answer, and they are not
interchangeable to the task -- the T can sit on the goal at the wrong angle. Both axes are
logarithmic (position spans 0.2-371 px, orientation 0-177 deg, and everything separating the arms
is at the bottom of that range), and the share above each violin is within **3.5 px** / **2.5 deg**:
the success set's edges, since every episode with coverage > 0.95 ends inside both. The dotted
reference is the block left at its start pose, the counterpart of the never-moving pusher's 0.139.

**The split matters, because the two components disagree.** The phase arms' deficit is entirely
positional -- their orientation is as good as or better than the phase-free arms':

| arm (transformer) | final pos [px] | final rot [deg] | within 3.5 px | within 2.5 deg |
|---|---|---|---|---|
| TF | 22 | 7.2 | 16 % | 25 % |
| TF+score | 18 | 4.5 | 22 % | 40 % |
| TF+phase | 42 | 6.7 | 4 % | 24 % |
| TF+phase+score | 43 | 4.5 | 6 % | 32 % |
| DP (open) | 4.2 | 1.6 | 47 % | 67 % |

**And the arms that end worst get closest.** Componentwise best over the episode against the final
value (median over 250 episodes):

| arm (transformer) | closest pos | final pos | drift | closest rot | final rot |
|---|---|---|---|---|---|
| TF | 20.4 | 21.7 | +1.3 | 3.1 | 7.2 |
| TF+score | 14.4 | 18.0 | +3.7 | 1.0 | 4.5 |
| TF+phase | **13.9** | 42.1 | **+28.3** | **0.2** | 6.7 |
| TF+phase+score | **12.2** | 43.3 | **+31.1** | **0.1** | 4.5 |
| DP (open) | 2.0 | 4.2 | +2.2 | 0.1 | 1.6 |

The phase arms pass within 12-14 px and a fraction of a degree of the goal -- closer than any other
arm, DP excepted -- and then push the block back out by ~30 px. The phase-free arms stop where they
got (drift 1-4 px), which is the same fact as their much higher stalled-episode count (35 against
20). So **the phase input earns its keep on the approach and loses it after the clock saturates**:
past rho = 1 the policy extrapolates outside its training domain and keeps commanding motion.
A terminal behaviour -- hold position once the clock reaches 1, or taper the commanded velocity as
rho -> 1 -- is the obvious thing to try, and it is worth roughly 30 px of final position error.

This is also why the rollout **score** ranks the phase arms first while these panels rank them
last: score is `max` coverage **over** the episode, so it credits the closest approach and ignores
the drift. `fig_final_poses`' own annotations agree with the new panels, not with score -- its mean
final coverage is 0.38 for `fm_tf` against 0.50 for `mlp_tf`.

## Score-weight sweep on the transformer's phase+score arm (2026-09-18)

λ ∈ {0.5, 2.5, 5, 10, 25} for `score_w<λ>_const_tf`, 5 seeds each, everything else held at the
arm's settings (90/10 split, constant schedule, bandwidth 32 px, 500 epochs). 25 runs, ~35 min
wall with four λ in parallel on one 4090.

**λ is a clean, monotonic brake on the memorization** documented above -- final training loss
spans 180x across the sweep, and the overfitting rebound falls to a third:

| λ | final train | min val | best epoch | val rebound |
|---|---|---|---|---|
| 0 (no score) | 0.008 | 13.57 | 13 | +3.25 |
| 0.5 | 0.038 | **13.36** | 13 | +2.60 |
| 2.5 | 0.181 | 13.60 | 16 | +2.29 |
| 5 | 0.370 | 13.75 | 17 | +2.18 |
| 10 | 0.701 | 14.06 | 21 | +1.61 |
| 25 | 1.463 | 14.75 | 24 | **+0.94** |

**None of it reaches the rollouts.** Eval clock at `best.pt`, mean score over 5 seeds:

| λ | 0.5 | **2.5** | 5 | 10 | 25 |
|---|---|---|---|---|---|
| score | 0.668 ± 0.013 | **0.683 ± 0.035** | 0.644 ± 0.046 | 0.630 ± 0.027 | 0.639 ± 0.045 |

Paired against λ = 2.5, only λ = 10 separates (−0.053, p = 0.003); 0.5 (p = 0.45), 5 (p = 0.17) and
25 (p = 0.21) do not. **The existing default λ = 2.5 is already the best value in the sweep** -- the
score is flat below it and declines above. It is top or joint-top on score in three of the four
clock × checkpoint conditions.

Two caveats worth keeping. λ = 0.5 scores within noise of 2.5 at **a third of the seed spread**
(± 0.013 against ± 0.035), so it is the steadier choice if variance matters more than the mean.
And final position error cannot discriminate here at all: its per-seed spread is ± 13 to ± 35 px
against between-λ differences of 3-15 px, so every pairing is insignificant (p = 0.32-0.78). Ranking
λ on position error would need far more than 5 seeds.

**The useful negative result is that suppressing the memorization does not improve the policy.**
λ = 25 leaves 180x more training loss than λ = 0 and overfits a third as much, and rolls out no
better -- slightly worse. Together with the earlier observation that the most completely memorizing
arm (`fm_tf`, train loss 0.008) is among the best policies, memorization is not what limits these
arms, and regularizing it harder is not the lever. The ~30 px of post-saturation drift is.

**The sweep judged on terminal error instead of score** (2026-09-18). Final coverage, final
position error and final orientation error, eval clock at `best.pt`, 5 seeds x 50 envs per λ:

| λ | final coverage | cov > 0.95 | median pos | ≤ 3.5 px | median rot | ≤ 2.5° |
|---|---|---|---|---|---|---|
| 0 (none) | 0.332 ± 0.067 | 2 % | 42.1 | 4 % | 6.7 | 24 % |
| 0.5 | 0.344 ± 0.058 | 4 % | 39.0 | 4 % | 4.6 | 32 % |
| 2.5 | 0.336 ± 0.105 | 5 % | 43.3 | 6 % | 4.5 | 32 % |
| 5 | 0.329 ± 0.080 | 4 % | 44.2 | 5 % | **3.5** | **40 %** |
| 10 | 0.306 ± 0.075 | 2 % | 49.1 | 2 % | 5.0 | 30 % |
| 25 | 0.327 ± 0.093 | 2 % | 42.7 | 2 % | 5.6 | 30 % |

**Nothing separates.** Paired against λ = 2.5 over seeds, all 30 tests (5 arms x 3 metrics x 2
checkpoints) are null but one at p = 0.04, which is what 30 tests return by chance. The seed spread
swamps the effect: final coverage varies ± 0.06-0.11 across seeds against a 0.04 range between λ.
These metrics therefore neither confirm nor overturn λ = 2.5; they simply cannot rank λ at 5 seeds.
λ = 5 is nominally the best on orientation (3.5°, 40 % within 2.5°) and on both coverage and
position at `last.pt`, but never significantly.

**What they do reveal is the checkpoint, and it reverses the earlier conclusion.** Pooled over all
six arms (30 seed-pairs), `best.pt` -> `last.pt`:

* **final coverage +0.073, p = 0.0002**
* **mean score −0.001, p = 0.95**

Training to the end leaves the score untouched and moves the block substantially closer to the
goal -- every one of the six arms improves, and median position error falls with it (λ = 2.5:
43.3 -> 35.9 px; λ = 5: 44.2 -> 30.7 px). The checkpoint section above concluded from *score* that
the single-step regression arms want early stopping; on where the block actually ends up, the
transformer's phase+score arms want the **final** model. This is the same score-versus-terminal
split as the drift finding: score is a max over the episode and cannot see the end of it.

**The violins' error panels are now per seed** (2026-09-18). Every `fig_pusht_violin*` figure
now reports the two terminal errors *averaged within a seed*, drawn like the mean score beside
them: five dots, a mean bar, a linear axis and the paired † test, with the never-moving reference
as the mean start-pose error (116.3 px, 100°). The per-episode version they replace was a
three-decade log cloud of 250 dots per arm -- unreadable at `fig_pusht_violin_backbones`' twelve
violins, and not much better at eight -- and its median-and-share annotation bought less than the
averaging does: the mean per seed is what makes the position/orientation split legible. Score per
episode stays pooled, the one panel where the per-episode spread is the point. `fig_violin` no
longer has a per-episode error path at all (it was `seed_errors`, briefly); `ea34b89` is the last
commit that draws one.

Reading it per seed sharpens the drift finding. The two transformer phase arms are the **best two
of the eight on orientation and the worst two on position**:

| arm | mean final position [px] | mean final orientation [°] |
| --- | --- | --- |
| MLP / TF | 65.8 / 55.3 | 42.4 / 36.8 |
| MLP+score / TF+score | 56.4 / **50.0** | 33.4 / 29.9 |
| MLP+phase / TF+phase | 66.7 / *84.4* | 33.4 / 23.7 |
| MLP+phase+score / TF+phase+score | 56.5 / *68.4* | 31.1 / **11.4** |

TF+phase+score gets the block to within 11° of the goal orientation -- twice as square as any arm
but DP -- and then leaves it 68 px away, worse than the plain TF it beats on every score panel.
Position is where the drift is spent, and no arm here recovers it; DP open loop, at 9.9 px, is the
only thing on the figure that arrives. MIP's 100-109 px is barely inside the never-move reference
(116.3 px) while its orientation is not (35-39° against 100°): it turns the block roughly in place.

**ADD joins the violins as a third terminal error** (2026-09-18). The average-distance metric from
6D pose estimation, on SE(2): carry a fixed set of body points through both the final pose and the
goal pose and average the Euclidean distances,
`E_ADD = (1/N) Σ ||(R_gt x_i + t_gt) − (R_pred x_i + t_pred)||`. It is the single continuous number
the position/orientation split was the decomposition of, and it needs no px-per-radian exchange
rate: the block's own geometry sets one, so an angular error costs more out at the tip of the stem
than at the crossbar. `add_error` averages over the T filled on a 4 px grid (357 points) rather
than its 8 outline vertices, which would overweight the corners. `tests/pusht/test_pusht_metrics.py`
pins it against the closed forms -- exactly the displacement for a pure translation, exactly
`mean|x_i| · 2 sin(Δθ/2)` (mean radius 52.1 px) for a pure rotation.

**ADD sides with position, and that settles the phase arms' split verdict.** Mean per seed, eval
clock, `best.pt`:

| arm | position [px] | orientation [°] | ADD [px] |
| --- | --- | --- | --- |
| MLP / TF | 65.8 / 55.3 | 42.4 / 36.8 | 61.0 / **51.2** |
| MLP+score / TF+score | 56.4 / 50.0 | 33.4 / 29.9 | 51.5 / **47.4** |
| MLP+phase / TF+phase | 66.7 / 84.4 | 33.4 / 23.7 | 63.8 / *84.9* |
| MLP+phase+score / TF+phase+score | 56.5 / 68.4 | 31.1 / **11.4** | 52.3 / 67.7 |
| DP open / closed | 9.9 / 32.5 | 5.8 / 14.2 | 10.2 / 31.3 |
| never-moving pusher | 116.3 | 100.0 | 113.1 |

ADD tracks position within a few px for every arm and reverses none of its orderings. TF+phase+score
ends 25° squarer than plain TF and still 16 px worse in ADD, because at a mean radius of 52 px a
degree is worth only 0.91 px and the position deficit is 13 px: **the geometry's own exchange rate
does not buy the orientation win.** Two details worth keeping: ADD sits *below* the position error
for almost every arm (61.0 vs 65.8 for MLP), so the block's rotation partly cancels its translation
error rather than adding to it; and the never-moving reference is 113.1 px, so MIP's 96-106 px is
the only policy on the figure that ADD cannot separate from not moving at all.

**Two narrower backbone figures** (2026-09-18). `fig_pusht_violin_backbones` keeps all seven
panels; beside it `fig_violin` now takes `keys`, and `fig_violin_backbones` draws two cuts of the
same twelve arms with it:

* `fig_pusht_violin_backbones_score` -- mean score per seed, stalled episodes per seed.
* `fig_pusht_violin_backbones_goal` -- mean final position error, mean final orientation error,
  success rate per seed.

Panels lay out in one row up to four and then a second, the grid taking as many columns as both
rows divide, and the caption is now built from the panels actually drawn and wrapped to the
figure's width (the figures save on a tight bounding box, so an unwrapped caption line would
widen the file past its own axes). Both cuts test against **TF+phase+score**, and the four phase
arms' x labels are bold in every violin figure.

The two cuts save to **one width** so they stack in a document: `_violin_width` is passed the
wider cut's three columns, so the two-panel figure stretches its panels instead of shrinking the
page. Two things had to change for the widths to come out equal to the pixel. `_save` grew
`fixed_width`, which trims only top and bottom -- a tight box trims to the content, whose left
edge is the widest y tick label of the first panel (`140` is wider than `1.1`), so the two cuts
saved 7 px apart however equal their figure widths were. And `wspace` now scales with the column
count, making the gutter a fixed share of the figure: as a fraction of a column it had grown with
the stretched panels, spending the extra width on the gap rather than on the violins.

The two cuts are drawn at `font_scale = 1.2` -- every type size in `fig_violin` goes through one
multiplier, and the header and footer grow with it, so nothing collides at a larger size. They are
the figures meant to be read at less than a full page width; the seven-panel figures stay at 1.0.

Every violin panel also carries a **y axis label** now. The title says what the panel is and what
it is averaged over ("mean final position error per seed [px]"); the label says what the axis
carries ("position error [px]"), so a panel cropped out of the figure is not unitless.

**A dagger bug went with it.** `base` defaulted to `SCORE_ARMS[SCORE_ARM][1]`, an MLP arm, so
`--arch transformer` figures tested against an arm they do not draw and silently showed no
significance marks at all. It now defaults to that arm *of the plotted backbone*
(`+ ARCH_TAG`), and `fig_pusht_violin_tf*` carry daggers for the first time.

**Success is the env's coverage, not IoU**, and the panel now says so: `coverage =
area(goal ∩ block) / area(goal) > 0.95`. The denominator is the goal's area alone, not the union.
Block and goal are congruent, so `union = 2A − I` and the 0.95 threshold is an IoU of
`0.95 / (2 − 0.95) = 0.905`; anything quoted as IoU from these runs is that number, not 0.95.

**What the two cuts say, against TF+phase+score (Welch over 5 seeds).** On score it is the best
policy arm -- ahead of MLP (p = 0.000), MLP+score (0.000), MLP+phase (0.009), MLP+phase+score
(0.026) and TF+score (0.040), with only plain TF (0.062) and TF+phase (0.099) not separating --
and it stalls least, 16.0 of 50 against 29-35 for every MLP arm (p ≤ 0.002). On where the block
ends up the picture inverts: it is the best arm on orientation by a wide margin (11.4°, p ≤ 0.027
against every other policy arm) and **indistinguishable from all seven of them on position**
(68.4 px, every p ≥ 0.18) -- the only position daggers belong to the four baselines. Success rate
separates nothing among the policies (every p ≥ 0.09): at 2-13 % over 5 seeds there is no power
there, which is the argument for reading the terminal errors instead.

**`fig_pusht_inference_backbones`** (2026-09-18). The inference figure with both backbones' arms,
the same pairing as the violin figure -- each arm followed by its transformer twin, sharing the
arm's colour, with the baselines after them. `fig_violin_backbones` and it now share
`_backbone_pairs()`, and `fig_inference` takes `methods` and a `name` the way `fig_violin` does.

The hatch is already spent here: it separates the two *devices*, not the backbones. So the
backbone is carried by the row labels and by the pair sitting adjacent in one colour, which is
enough -- the ten rows read down as MLP, TF, MLP+score, TF+score, ... and the comparison is the
difference between neighbours. Read that way, **the transformer costs 9x the MLP per decision on
the GPU and the phase estimator is no longer what dominates**: 0.21 -> 1.89 ms plain, 0.49 -> 2.29
with phase, so the estimator adds 0.28 ms to an MLP (more than doubling it) and 0.40 ms to a
transformer (a fifth). Both backbones stay inside Push-T's 10 Hz budget by more than two orders of
magnitude; the choice between them is not a compute argument.

The score-vs-cost panel's label column had to change to hold ten arms: it now stacks evenly from
the best-scoring arm with the spacing closed up to fit, rather than following each label toward
its own score, which let a tight cluster walk the stack off the bottom of the axes (visible on
`fig_pusht_inference_last`, where `MLP` sat on the axis floor).

**A score arm shares its twin's bar row** (2026-09-18). The score term is a training-time loss,
not a second head: an arm and its score variant run the identical forward pass, and the timers
separate them only by rounding (0.21 vs 0.20 ms, 1.89 vs 1.88, 0.49 vs 0.48). So
`fig_pusht_inference_backbones` now folds each pair into one bar row at their mean cost, labelled
with both names (`MLP+phase / +score`) and sliced horizontally into their two colours -- blue over
pink on the phase rows, a single orange where the pair already shares a colour. Ten rows become
six and the figure is 28 % shorter, at no cost in information: what the four dropped rows showed
was the timer's last digit. `fig_inference` takes a `merge` mapping (`_score_merge()`) and the
score-vs-cost panel still carries a point per arm -- the cost is shared, the score is not.

On the shorter axes the two DP labels in that panel collided, so far-right labels closer than 4
score points are now pushed apart and joined back to their point by a hairline (it shows on
`fig_pusht_inference_last` too, which is redrawn; the `best.pt` sets are unchanged).

**Leaders meet the label at its left edge.** `annotate`'s own arrow leaves the text box at
whichever corner the target lies past, so a label stacked above or below its point was joined at a
corner and read as pointing nowhere in particular -- worst in the score-vs-cost label column,
where every leader travels up-left. `_leader()` draws the hairline as its own textless annotation
ending just left of the label, at its mid height, and the label is placed separately.

## The 67M phase-former: `--arch transformer_big` (2026-09-19)

A third backbone, `PushTBigTransformer` in `fm_bc/pusht/policy.py`, at **66,877,698 parameters**
against the `_tf` arms' 907,522 (874,498 without the phase input). It is a subclass rather than
options on `PushTTransformer`, and a separate `ARCHS` entry rather than a size flag, so the
published backbone's defaults, module tree, `state_dict` and run names are untouched by
construction. Run names take `_tfbig`: `mlp_tfbig`, `fm_tfbig`, `score_w2.5_const_tfbig`,
`mlp_score_w2.5_const_tfbig`. **Not yet trained.**

Three things differ from `_tf`, and each follows from cross-attention being the mechanism the
backbone is scaled around.

**The block is tokenized as points on it.** `_tf` emits two context tokens per observed step --
the block's `[xy_n, cos, sin]` and the agent's `[xy_n]` -- so at `obs_steps = 1` cross-attention
is a two-way softmax, and widening the network buys value and output projections rather than a
richer attention pattern. `transformer_big` adds the T's eight vertices and its two rectangle
centroids as their own tokens, `M = 2 -> 12`. Those points are a fixed rigid function of the same
`(block_x, block_y, theta)` (`TEE_POINTS` in `constants.py`, `tee_points()` in `chunks.py`), so
**the observation carries exactly the information it did before** -- what changes is that the
query can attend to the stem tip or a crossbar corner separately instead of to one summary of the
pose. It costs 9,984 parameters (0.015 %) and no measurable time: batch-1 latency is flat in `M`
(17.16 ms at `M = 2`, 17.36 at 10, 16.36 at 40, CPU single-thread).

These are deliberately **not** the zarr's nine `keypoint`s. Those are furthest-point-sampled from
a rendered mask with a random perturbation, so they are an artifact of one sampling run and cannot
be recovered from the geometry; the polygon can, and it is the same one `plot_pusht.py` already
drew the figures with (now one constant, imported by both).

**There is no self-attention.** With `horizon = 1` there is a single query token, and a softmax
over one key is the constant 1 however the query is projected -- so `qkv`'s q and k rows receive
*exactly* zero gradient (`test_self_attention_has_no_gradient_at_one_query_token` pins it, and
pins that they are non-zero at `horizon = 3`), and the surviving path `proj(W_v(adaLN(x)))` is a
linear gated residual the block's own MLP subsumes. At this shape `num_layers = 8` without it is
the same parameter and FLOP budget as `num_layers = 6` with it (`8 * 12 d^2 = 6 * 16 d^2`), so the
~7M dead weights become two more live blocks. `use_self_attn` is a new `PhaseTransformer` option
defaulting to `True`; the skate `FlowTransformer` shares that file and is unaffected (verified
bit-identical `state_dict` and forward at the default).

**`phase_dim` does not track the width.** Each adaLN site is `Linear(phase_dim, 3 d)`, so
`phase_dim = hidden_dim = 768` would spend **37.2 %** of the model transforming a scalar, leaving
cross-attention 15.2 %. Held at 256 the modulation is 16.5 % and cross-attention 20.8 %.

So the shape is `hidden_dim 768, num_layers 8, num_heads 12, phase_dim 256`, and it is the class's
own default -- `--arch transformer_big` needs no other flag. For that to work the five transformer
size flags on `scripts/pusht/train_pusht_policy.py` now default to `None` and only the ones actually
passed reach `net_kwargs`; left at their old literals they would have silently shrunk the big
backbone back to 128 x 2. One visible consequence: a default `--arch transformer` run now records
`train_config.net_kwargs = {}` instead of the five explicit keys. The model, the run name and
every number are unchanged, and nothing reads that field.

**Expect it to overfit hard, and do not select it on validation loss.** The `_tf` phase arms
already peak at epoch 13-16 of 500 on ~18k training rows; 67M is ~75x the parameters on the same
data. The section above -- *Validation loss ranks these arms wrongly* -- applies with more force,
so `--checkpoint-every` wants to be small enough to bracket a peak that will arrive earlier than
epoch 50, and the checkpoint to report is whichever rolls out best, as Diffusion Policy's own
recipe does (`topk` on `test_mean_score`, never on validation loss).

At 268 MB per checkpoint, `--checkpoint-every 10` over 500 epochs is ~13 GB per seed.

**Known issue, pre-existing.** `scripts/pusht/summarize_pusht_policy.py`'s score-name regex is an
anchored `fullmatch` with no trailing-backbone group, so it already returns `None` for
`score_w2.5_const_tf` and `const_twin`/`score_params` then raise `TypeError`. It crashes today on
any runs tree containing the published transformer score runs; the `_tfbig` runs land in the same
tree. Fix it before summarizing either backbone.

## The phase-former at twice the parameters: `--arch transformer_medium` (2026-10-02)

Does the published phase-former improve with scale alone? `PushTMediumTransformer` (`_tfmed`) is
`PushTTransformer` widened, `hidden_dim = phase_dim` 128 -> 180, same two blocks, four heads and
tokenization: **1,791,002 parameters against 907,522 (1.97x)**. Its rollout clock was doubled the
same way, two hidden layers of 724 instead of 512 (533,589 vs 268,801 parameters,
`runs/pusht/phase_start_w724/`, otherwise trained exactly as `phase_start`). Everything else is the
`_tf` protocol: 4 arms x 10 seeds, 500 epochs, `best.pt`, the paper's rollout clock settings.
Report and figure: `output/pusht_9010/tf_scale/` (`scripts/pusht/compare_tf_scale.py`).

**No reliable gain.** Mean score per seed, Welch p against the published TF on the paper's clock:

| arm | TF 0.9M | TF-med 1.8M | p | TF-med + 724 clock | p |
|---|---|---|---|---|---|
| TF | 0.574 | 0.556 | 0.52 | — | |
| TF+score | 0.595 | 0.591 | 0.88 | — | |
| TF+phase | 0.654 | 0.644 | 0.56 | 0.640 | 0.49 |
| TF+phase+score | 0.672 | **0.701** | 0.053 | 0.696 | 0.19 |

The one hint is the full arm, TF+phase+score, which gains 0.029 on the paper's clock with a
tighter spread (sd 0.023 vs 0.037) -- borderline at p = 0.053, and not significant once it is one
of the eight comparisons made here. The other three arms move by -0.018 to +0.004, inside the seed
noise. Success rates are unchanged (0.03-0.12 either way).

**The bigger clock does nothing measurable either.** It fits the demos a little better (rho RMSE
0.0075 vs 0.0083, both under one grid step of 0.0041) but moves the rollouts by +0.022 (TF+phase,
p = 0.17), +0.004 (TF+phase+score) and -0.004/-0.005 on TF-med -- the policy is not limited by the
clock's accuracy. The phase-free arms were not re-rolled with it: at a fixed step they ignore the
clock, so their rollouts would be identical.

**Width does not change when training peaks.** Best epochs are 11 and 13 for TF-med's phase arms
against 13 and 15 for TF, and 38 / 66 against 49 / 75 for the phase-free ones -- so doubling the
width neither speeds the overfit up much nor gives the validation-selected model anything new.

Against the 67M backbone, which also re-tokenizes the T and drops self-attention, the contrast is
informative: `_tfbig` lifts the phase-free arms (0.574 -> 0.649, p = 0.010) and *hurts* both phase
arms badly (0.672 -> 0.514, p < 0.001), while doubling the published shape leaves all four where
they were. So the published 0.9M size is not the bottleneck for the phase arms, and the big
model's phase-arm loss comes from its design or its size far outside the data, not from scale
per se. The `_tf` and `_tfbig` seeds trained on CHPC and `_tfmed` on the 4090; GPU training is not
deterministic, so that difference is distributional only.

## Depth, not width: medium-deep and large phase-formers and clocks (2026-10-02)

The medium section above widened the phase-former and its clock. This one adds the other two
resizings of each: **medium-deep** spends the same doubling on depth (`transformer_medium_deep`,
`_tfdeep`, four blocks at width 128, 1,730,818 parameters; clock `phase_start_d3`, three hidden
layers of 512, 531,457) and **large** takes both changes at once (`transformer_large`, `_tflarge`,
width 180 and four blocks, 3,416,762; clock `phase_start_w724_d3`, three layers of 724,
1,058,489). Same protocol, 4 arms x 10 seeds each, and every phase-arm checkpoint of all four
backbones rolled out on all four clocks. Report: `output/pusht_9010/tf_scale/`.

**The best TF+phase+score is TF-large with the large clock: 0.721 ± 0.039, against 0.672 ± 0.037
for the published configuration (+0.049, Welch p = 0.010).** Its best seed scores 0.811. Of the 16
backbone x clock combinations the published one ranks 15th; the top six (0.702-0.721) all have
depth somewhere -- a 4-block backbone, a 3-layer clock, or both. The lead of the top one over
ranks 2-3 (medium-deep with the large or deep clock, 0.715 and 0.712, p = 0.018 and 0.016) is
within seed noise, so the claim is the top cluster, not the single winner. **Success rate does not
move**: 0.042 for the winner and the published configuration alike -- the gain is in how close the
block gets (score is max coverage), not in finishing more episodes.

**Depth in the clock is the cleanest effect.** The deep clock fits rho to RMSE 0.0046 and the
large one 0.0045, against 0.0083 for the paper's and 0.0075 for the wide one, and the rollouts
follow the fit: pooled over every backbone and both phase arms, the deep clock adds +0.020 to the
score (9 of 10 clock seeds positive, p < 0.001 averaged per clock seed, since all eight checkpoints
of a seed share its clock), the large clock +0.019 (10/10, p < 0.001), the wide clock +0.001
(5/10, p = 0.80). Width barely improves the fit and does nothing; depth halves the error and
gives a consistent +0.02. The clock is not the bottleneck for success either (no change).

**Depth in the backbone helps most where phase is absent.** On the paper's clock, the phase-free
TF arm goes 0.574 -> 0.638 (deep, p = 0.030) -> 0.656 (large, p = 0.012), and TF+score 0.595 ->
0.622 -> 0.645 (p = 0.083), while width alone is flat (0.556, 0.591). The phase arms gain less
(TF+phase 0.654 -> 0.670 -> 0.684, p = 0.13; TF+phase+score 0.672 -> 0.671 -> 0.696, p = 0.17), so
the phase arms' lead over the phase-free ones narrows from ~0.10 at 0.9M to ~0.04 at 3.4M: some of
what the phase input buys the small model, depth buys too. TF+phase+score stays the best arm at
every size.

**Every size peaks at the same epoch** (11-13 for the phase arms, 29-40 for the phase-free ones
at 4 blocks), so the larger models are not overfitting faster.

Caveats. Many tests were run here; individual p values near 0.05 should be read as direction,
not proof. The pooled clock test and the deep/large backbone gain on the phase-free arm are the
two effects that hold up. One odd cell: medium-deep TF+phase+score's success rate on the paper's
clock is 0.014 (vs 0.042), but the same checkpoints reach 0.036-0.062 on every other clock, so it
reads as a clock-checkpoint interaction in a 1-in-50-episode metric rather than a property of the
backbone.

## Transformer phase clocks: robot and object as separate tokens (2026-10-03)

`PhaseClockTransformer` (`fm_bc/pusht/phase_mlp.py`) reads the start-pose clock's inputs as four
tokens -- a readout token, the robot (agent xy), the object now and the object at the episode's
start (block xy, cos, sin; one shared projection, told apart by slot embeddings) -- through a pre-LN
self-attention encoder, with `PhaseMLP`'s sigmoid head and call signature. `load_phase_mlp`
dispatches on the checkpoint's kind, so rollouts and plots take it unchanged. One twin per MLP
clock, within 3% of its parameters: base d=104 x 2 layers (263,849), wide 148 x 2 (531,765), deep
104 x 4 (526,137), large 148 x 4 (1,061,309), in `runs/pusht/phase_start_xf{,_wide,_deep,_large}`.
Recipe from a seed-0 probe: lr 3e-4, 500 warmup steps, cosine to 0 (the MLP clocks' constant 1e-3
fit 0.0051 / 0.0047 at base / large against 0.0037 / 0.0023). 10 seeds each; every phase-arm
checkpoint of all four backbones rolled out on all four (320 rollouts).

**The best TF+phase+score is unchanged: TF-large with the MLP large clock, 0.721 ± 0.039** (vs the
published 0.672, p = 0.010). The best transformer-clock combination is TF-med deep with the
transformer large clock at 0.719 ± 0.041 (rank 2 of 32), statistically a tie. The top ten of the
32 combinations mix both clock families; the published configuration ranks 31st.

**The transformer clocks fit the demonstrations better, and far more consistently**: rho RMSE
0.0037 / 0.0031 / 0.0026 / 0.0023 at base / wide / deep / large, against the MLP twins' 0.0083 /
0.0075 / 0.0046 / 0.0045, with a seed-to-seed sd of 0.00005 against 0.0005-0.0024. **Part of that
is the recipe, not the architecture.** A seed-0 control refitting the four MLP clocks with a
decayed learning rate (lr 1e-3, warmup, cosine) reaches 0.0062 / 0.0053 / 0.0027 / 0.0025: at deep
and large sizes a decayed MLP nearly matches the transformer, and only at the two-layer sizes does
the transformer keep a clear (~40%) edge. The MLP clocks' constant learning rate is what leaves
them noisy across seeds. (Control in the session scratchpad only, not in `runs/`.)

**Rollouts do not follow the fit.** Pooled over all 80 phase-arm checkpoints, against the paper's
clock (per-clock-seed means, one-sample t over 10 seeds):

| clock | fit RMSE | Δ score | p | vs its MLP twin | p |
|---|---|---|---|---|---|
| transformer base | 0.0037 | +0.001 | 0.63 | (twin = paper's clock) | |
| transformer wide | 0.0031 | +0.015 | < 0.001 | **+0.014** | 0.008 |
| transformer deep | 0.0026 | +0.013 | 0.015 | -0.008 | 0.25 |
| transformer large | 0.0023 | +0.007 | 0.074 | **-0.012** (1/10 seeds > 0) | 0.006 |
| MLP deep | 0.0046 | +0.020 | < 0.001 | | |
| MLP large | 0.0045 | +0.019 | < 0.001 | | |

The best-fitting clock in the study, transformer large, is *worse* in closed loop than the MLP
large clock it out-fits by 2x. Fit is measured in-sample on the demonstrations, while rollout
states drift off them; past ~0.0045 a tighter fit to the demos stops paying and can cost. That
reading is plausible but untested -- a held-out (leave-episodes-out) fit is the metric that would
test it, and the candidate predictor of rollout value. **Success rate is where the transformer
clocks show something new**: wide +0.014 (9/10, p = 0.002) and deep +0.009 (8/10, p = 0.003)
against the paper's clock -- the first success gains anywhere in this study, though under one
episode in 50.

Open: whether the MLP clocks retrained with the decayed recipe (fitting like the transformers)
also lose their rollout edge, which would confirm that in-sample fit is the wrong target.

## The MLP at more parameters: wide, deep and large (2026-10-05)

The MLP ladder, built to stay parameter-matched to the transformer ladder as the published pair
is: **MLP-med wide** (`mlp_medium`, `_mlpmed`, 4 x 768, 1,779,458 vs TF-med wide's 1,791,002),
**MLP-med deep** (`mlp_medium_deep`, `_mlpdeep`, 8 x 512, 1,843,714 vs TF-med deep's 1,730,818)
and **MLP-large** (`mlp_large`, `_mlplarge`, 7 x 768, 3,551,234 vs TF-large's 3,416,762). Same
protocol, 4 arms x 10 seeds each; every arm on the paper's clock and the phase arms of all four
MLP sizes (the published one included) on all seven resized clocks. Report:
`output/pusht_9010/mlp_scale/` (`compare_tf_scale.py --family mlp`).

**The best MLP+phase+score is MLP-med wide with the transformer wide clock, 0.675 ± 0.054**,
against 0.637 ± 0.037 for the published MLP (+0.038, p = 0.085). None of the 32 MLP combinations
clears p < 0.05 against the published configuration; the top ten span 0.663-0.675.

**The transformer stays ahead at every size.** +phase+score arm, Welch over seeds:

| size | MLP, paper clock | TF, paper clock | p | MLP, its best clock | TF, its best clock | p |
|---|---|---|---|---|---|---|
| base | 0.637 | 0.672 | 0.051 | 0.666 (transformer large) | 0.702 (MLP deep) | 0.051 |
| medium wide | 0.654 | 0.701 | 0.017 | 0.675 (transformer wide) | 0.705 (transformer deep) | 0.15 |
| medium deep | 0.658 | 0.671 | 0.63 | 0.667 (transformer large) | 0.719 (transformer large) | 0.049 |
| large | 0.650 | 0.696 | 0.007 | 0.662 (transformer base) | 0.721 (MLP large) | 0.001 |

The best TF+phase+score (0.721) is 0.046 above the best MLP+phase+score (0.675). The MLPs do
succeed somewhat more often at their best clocks (0.046-0.090 vs 0.038-0.054), the one metric where
they lead.

**Scale helps the MLP where phase is absent and nowhere else.** On the paper's clock the plain MLP
goes 0.458 -> 0.483 (wide) / 0.584 (deep, p < 0.001) / 0.616 (large, p < 0.001), and MLP+score
0.519 -> 0.537 / 0.591 (p = 0.014) / 0.579 (p = 0.030) -- depth again, not width, as in the
transformer study. The phase arms do not move: MLP+phase 0.581 -> 0.596 / 0.586 / 0.561 and
MLP+phase+score 0.637 -> 0.654 / 0.658 / 0.650, all p > 0.3. So the phase input's advantage
shrinks with size, and at MLP-large it reverses for phase alone (MLP+phase 0.561 against the plain
MLP's 0.616). The score regularizer keeps the full arm on top at every size.

**Bigger clocks barely help the MLPs.** Pooled over all 80 MLP phase-arm checkpoints, the deep MLP
clock adds +0.007 (p = 0.083) and every other clock -0.001 to +0.007, against +0.020 (p < 0.001)
for the deep clock on the transformers. The clock's accuracy pays off on the transformer, which
reads the phase at every block, more than on the MLP, which concatenates it once.

**Larger MLPs peak earlier** -- best epoch 222 -> 149 / 93 / 60 for MLP+phase+score -- the same
early-stopping pattern as the transformers, starting from a much later epoch.
