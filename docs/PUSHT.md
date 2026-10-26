# Push-T (`fm_bc/pusht/`)

A second SE(2) task, from human demonstrations rather than planner output: push a T-shaped block
onto a goal pose with a circular pusher ([Diffusion Policy][dp], Chi et al. 2023). None of the
Lie-group machinery changes. What does change is that **episodes vary in length** (206 episodes,
25 650 steps, 30–246 each at 10 Hz), that **actions are absolute pixel targets** in a 512×512
frame rather than body twists, and that the **pusher has no orientation** — it is a circle, so
SE(2) enters only through the block's pose, which the policy observes but never predicts deltas
for.

[dp]: https://diffusion-policy.cs.columbia.edu/

```bash
scripts/pusht/download_pusht.sh                       # ~30 MB → data/pusht/ (git-ignored)
python scripts/pusht/convert_pusht.py                 # zarr → data/pusht/pusht_episodes.npz
python scripts/pusht/prepare_pusht.py --n-grid 128    # splits, normalizers, phase grid
python scripts/pusht/inspect_pusht.py --data data/pusht/pusht_prepared.npz   # audit; non-zero on failure
```

The download script hardcodes no paths: the destination is `$1`, else `$PUSHT_DATA_DIR`, else
`<repo>/data/pusht`, resolved from the script's own location. It verifies the archive size, is
idempotent, and takes `--dry-run` / `--force`.

Two artifacts, because there are two consumers. **`pusht_episodes.npz`** is the canonical
conversion: episodes concatenated into one flat buffer with per-episode bounds, feeding the
action-chunk baselines through `fm_bc/pusht/chunks.py`. **`pusht_prepared.npz`** is derived and
regenerable: split labels, normalization statistics fitted on training episodes only, and the
rectangular phase grid that `divergence_knn` / `score_kernel` need — they take neighbours *at
the same phase*, which the fixed-horizon skate task gets for free and Push-T does not.

Gridding resamples each episode uniformly in phase, interpolating the block pose **along the
SE(2) geodesic** with `se2_log` / `se2_exp_retract`. The alternative — scatter each real step to
its nearest bin and interpolate the gaps — preserves original samples but loads the
interpolation artifact onto short episodes specifically (a 60-step episode contributes ~75 %
fabricated points to every neighbour query, a 246-step one none), and episode length correlates
with task difficulty, which is the signal the divergence target carries. Uniform resampling
fabricates more points but spreads the artifact evenly, with exact endpoints and monotone phase
by construction.

One field deserves singling out. After gridding, a phase step is `grid_dt[e] = (L_e - 1)·dt/N`
seconds, which varies ~4× across episodes. `divergence_knn` takes a scalar `dt` and is linear in
`1/dt`, so a global value would scale a half-length episode's divergence by exactly ½. Divide
per episode instead:

```python
div = divergence_knn(prep.grid.block_poses(), dt=1.0, k=16, metric="se2",
                     rot_scale=ROT_SCALE_PX_PER_RAD) / prep.grid.grid_dt[:, None]   # 1/s
```

`rot_scale` matters as much. In pixel units translations are O(100) and rotations O(π), so the
skate default of 1.0 would make the metric ~99.9 % translation; `ROT_SCALE_PX_PER_RAD = 100`
(the block's characteristic radius) balances them.

Splits reproduce Diffusion Policy's convention (`seed 42`, `val_ratio 0.02`,
`max_train_episodes 90` → 4 validation, 90 training), with a third label for the 112 episodes
that convention leaves unused, so an `--n-train` sweep can reclaim them without touching the
held-out set. Note that **no frame of the released demonstrations reaches the 0.95 coverage
success threshold** (max 0.9489): a `success` column derived from the demos is identically
false, and is not a usable label.

Only `scripts/pusht/convert_pusht.py` needs zarr (`pip install -e '.[pusht]'`), and it imports it
lazily; everything downstream is torch alone, so the data pipeline and its tests run without
Theseus — `pytest tests/pusht/test_pusht_{data,windows,progress}.py`.

[`fm_bc/pusht/STATUS.md`](../fm_bc/pusht/STATUS.md) records where this work stands: what is verified
against real data versus synthetic fixtures, the constants that silently corrupt results if
changed (`grid_dt`, `rot_scale`, the CVAE's `kappa`), and what to run first.

## Task progress (`fm_bc/pusht/progress.py`)

A lightweight CVAE over state windows whose latent splits into a scalar **phase** and a Gaussian
**style** code, with the KL applied *only* to the style half. That asymmetry is the design: the
phase channel is rate-free, so the optimizer routes information through it, and a monotonicity
constraint decides which information. Progress is never supervised — `ep.phase` is held out as the
yardstick. Only the encoder ships; `phase()` is a deterministic forward pass with no sampling and
no decoder.

```bash
python scripts/pusht/train_progress.py --baselines-only      # run this FIRST
python scripts/pusht/train_progress.py --seeds 0 1 2 --out runs
```

Three things are worth knowing before touching the hyperparameters.

**The margin is a supervision dial.** With `z_phase` bounded to `[0,1]` and a per-episode margin
`m = κ/(L−1)`, the constraint `p_{t+1} − p_t ≥ m` telescopes to `p_{L−1} − p_0 ≥ κ`. At `κ = 1`
the required rise is the whole range, so the unique feasible solution is `p_t = t/(L−1)` — exactly
the ground-truth phase. A margin at `κ = 1` *is* phase supervision wearing a hinge, so every number
is indexed by κ (default 0.5) and `κ = 1` belongs in an ablation table as the labelled ceiling,
never as a run. A length-*independent* margin cannot work at all: feasibility for the 246-step
episodes caps it at `1/245`, at which point the 30-step episodes are asked to rise by 12 % of the
range.

**`phase_ablation` is a gate, not a diagnostic.** Reconstruction is the only term grounding
`z_phase` in task structure — the hinge alone is satisfied by *any* monotone-along-time statistic.
If shuffling `z_phase` barely changes the decoder's error, nothing shaped the encoder and a rank
correlation of 0.95 means nothing. Check it before reading any other number. For the same reason
the discarded decoder is load-bearing, and the two null baselines (`rho_geom`, a hand-coded
`−d_SE2(block, goal)`; `rho_linear`, a least-squares probe) are printed on every run: the goal pose
is a constant, so a three-line feature is already a progress measure, and a CVAE that does not
beat it has shown nothing.

**Scored on the 112 `UNUSED` episodes, not the 4 validation ones** — never trained on either way,
but four episodes cannot put a standard error on a per-episode correlation. `spearman_pooled` and
`calibration_gap` sit alongside the per-episode mean because per-episode rank correlation is
invariant to *any* monotone reparameterization, so it cannot tell you whether `z_phase = 0.6` means
the same thing in two episodes — which is exactly what reading the scalar out assumes. The encoder
also returns `ood = ‖μ‖²/Z`, since `z_phase = sigmoid(u)` is bounded and will otherwise answer
confidently on rollout states it has never seen.

