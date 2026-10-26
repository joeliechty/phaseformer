# Phaseformer, Appendix B — Lean 4 formalization

A Lean 4 + Mathlib formalization of every proof in **Appendix B (Proofs, B.1–B.6)** of the
Phaseformer theory appendix (Draft 4). Statement numbers follow the rendered draft:
Eq. (7)–(10), (13), (14), (20), (21); Lemma 1, Propositions 1–3, Lemma 2, Theorems 4–5.

* No `sorry`, no `admit`, no new `axiom`. `Audit.lean` prints the axioms behind ~100 main
  theorems. Each one should show only Lean's standard `propext`, `Classical.choice` and
  `Quot.sound`.
* There are four places where the appendix itself cites an external theorem that Mathlib
  doesn't have. In each, the cited fact is either an explicit hypothesis of the Lean
  theorem or left as a stated citation (see the table). Nothing is assumed silently.

## Build and check

One command does everything below and checks the axiom audit automatically:

```bash
./verify.sh
```

Or step by step:

```bash
cd lean
lake exe cache get     # downloads prebuilt Mathlib (a few GB); skip it and the build takes hours
lake build             # compiles all seven files
lake env lean Audit.lean   # prints the axioms each main theorem depends on
```

`lean-toolchain` pins **Lean v4.34.1** and `lakefile.toml` pins **Mathlib v4.34.1**. If you
have `elan`, it fetches that toolchain automatically. A Lean installed without `elan` has to
be exactly v4.34.1, because Mathlib only builds with the toolchain it pins.

## Files

| File | Appendix section |
|---|---|
| `PhaseformerProofs/Gronwall.lean` | B.4 "Grönwall" paragraph (Eq. 13, closed loop, Euler, unicycle); rollout uniqueness used throughout |
| `PhaseformerProofs/B1_Setting.lean` | B.1: Eq. (7)–(10), Eq. (8), Lemma 1 |
| `PhaseformerProofs/B2_Marginals.lean` | B.2: Proposition 1 and the closed-form Gaussian example |
| `PhaseformerProofs/B3_PhaseNecessary.lean` | B.3: Proposition 2 (a) overshoot, (b) stall |
| `PhaseformerProofs/B4_InputVariation.lean` | B.4: Lemma 2, Proposition 3 |
| `PhaseformerProofs/B5_Ideal.lean` | B.5: Theorem 4 |
| `PhaseformerProofs/B6_Score.lean` | B.6: Theorem 5, worked examples, stationarity, realizability |

## Coverage

✅ fully proved  🟡 proved, with a cited or modelling step taken as an explicit hypothesis  📚 cited in the appendix, not formalized

### B.1 Proofs for the Setting and Lemma 1
| Step | Status | Lean |
|---|---|---|
| Eq. (7) decomposition: fit error + irreducible + cost of not seeing `Z` | ✅ general `L²(μ;E)`, conditional expectation = Mathlib `condExpL2` | `eq7_L2`, core `pythagoras_nested` |
| `Z' = h(Z)` gives `σ(Z') ⊆ σ(Z)`; `‖f‖² = ∫‖f‖²` | ✅ | `comap_comp_le`, `L2_norm_sq_eq_integral` |
| Special cases Eq. (9)–(10); the minimizer over policies of `Z'` is `m'` | ✅ | `eq9_L2`, `eq7_minimizer` |
| Eq. (8) (Bayes: `v̄_free` mixes the phase-wise `v̄`) | ✅ finite-dataset form | `fiberMean_tower` |
| Lemma 1(a) BC loss = CFM loss for every coupling and policy | ✅ | `lemma1a` |
| Lemma 1(b) FM inference = closed-loop rollout when unique | ✅ | `lemma1b` |
| Lemma 1(c) straight paths have ρ-independent velocity; `sin(πρ)` isn't straight | ✅ | `lemma1c_*` |

### B.2 Proposition 1 and the closed-form example
| Step | Status | Lean |
|---|---|---|
| Weak continuity equation `d/dρ E h(x_ρ) = E[∇h·ẋ] = E[∇h·v̄]` ("average the velocity first") | ✅ finite weighted dataset of differentiable paths | `average_velocity_first`, `prop1_weak_continuity` |
| Uniqueness of continuity-equation solutions for Lipschitz `v̄` (Ambrosio–Gigli–Savaré §8.1) | 📚 | — |
| Phase-free identity `∇·(μ v̄_free) = p₀ − p₁` (weak) | ✅ | `prop1_phase_free_identity` |
| 1-D identity `μ v̄_free = F_S − F_G` (weak) | ✅ | `phase_free_identity_1d` |
| Regularity bounds | ✅ | `regularity_vbar_zero`, `regularity_path_norm` |
| Example: `L(ρ) ≤ 1`, rollout `σ_ρ x(0)`, its uniqueness, return to start, `v̄ = 0` at ½, no crossing, `σ_ρ x(0) ∼ N(0, σ_ρ²)` | ✅ | `gauss_*` |
| Example: `Cov(x₁−x₀, x_ρ) = 2ρ−1`, `Var x_ρ = σ_ρ²`, residual uncorrelated | ✅ | `gauss_covariance` |
| Jointly Gaussian ⇒ `E[x₁−x₀ | x_ρ]` is linear | 📚 standard | — |

### B.3 Proposition 2
| Step | Status | Lean |
|---|---|---|
| (a) `T` increasing and separates the blocks; groups disjoint at each phase; `C = 0` | ✅ (no-conflict proved directly, without Brenier) | `T_strictMonoOn`, `T_block_images`, `groups_disjoint_at_phase`, `prop2a_no_conflict` |
| (a) occupation density = mass/speed; `μ₁ = ¼`, `μ₂ = ⅛` | ✅ | `occupation_per_demo`, `group_occupation`, `occupation_values` |
| (a) `v̄_free = (2μ₁+4μ₂)/(μ₁+μ₂)` is 2, 4, or strictly between; `8/3`; `A → ⅓` | ✅ | `vfree_convex`, `vfree_overlap`, `aliasing_limit` |
| (a) travel time `∫ dz/v = 1`, `Φ = G⁻¹(G+1)` strictly increasing, rollout stays in `I` | ✅ | `travel_time`, `rollout_strictMono`, `rollout_stays_in` |
| (a) overshoot `Φ(a) > T(a)`, undershoot `Φ(b) < T(b)` | 🟡 the `v̄_free` profile from the rows above is the hypothesis `VfreeProfile`; rollouts of the continuous field are assumed to exist, as in the paper | `prop2a_overshoot`, `prop2a_undershoot` |
| (a) `Φ_#P_S ≠ P_G` (CDFs differ at `Φ(b)`) | 🟡 same hypotheses | `cdf_lt_of_undershoot`, `prop2a_not_PG` |
| (a) the δ→0 limit: endpoints 2.5 and 4.5 | ✅ arithmetic | `prop2a_limit` |
| (b) `C = 0`, `v̄ = π cos πρ`, demos are its rollouts; `v̄_free = 0` (flux cancels); stall; disjoint supports; `A > 0` | ✅ | `prop2b_*` |

### B.4 Input variation
| Step | Status | Lean |
|---|---|---|
| Eq. (13) Grönwall with time-varying `L(ρ)` | ✅ for continuous `L` (the paper needs only integrable `L`; every use here has continuous or bounded `L`) | `gronwall_rollouts`, `rollouts_eq_of_eq` |
| Closed-loop constant `L_x + L_u L`; Euler product; unicycle `L_u = 1`, `L_x = |v|` | ✅ | `closedLoop_lipschitz`, `euler_lipschitz`, `unicycle_Lu`, `unicycle_Lx` |
| Lemma 2(a) branch average; single-branch target | ✅ | `lemma2a_branch_average`, `fiberMean_const` |
| Lemma 2(b) `L(ρ) ≥ 1/(ρ−ρ*)`, not integrable | ✅ | `lemma2b_lipschitz_lower`, `lemma2b_not_integrable` |
| Lemma 2(c) one state, one branch | ✅ (uniqueness from a Lipschitz closed loop on the joint robot+world state) | `lemma2c_one_branch` |
| Prop. 3(a) undecided starts, Eq. (14) | ✅ | `prop3_separation`, `prop3a_undecided`, `prop3a_prob` |
| Prop. 3(b) undecided stretch of length ≥ `g/K` on every segment | ✅ | `prop3b_segment`, `prop3b_interval_1d` |
| Prop. 3(b) Gaussian: `ℓ > 4σ` impossible; `Pr(undecided) ≥ φ(2)ℓ/σ`; `φ(2) ∈ (0.0539, 0.0541)` | ✅ | `prop3_gauss_too_wide`, `prop3_gauss_prob`, `phi2_bounds` |
| Remark after the proof (lines in `ℝᵈ`) | not formalized (it's a remark, not part of a proof) | — |

### B.5 Theorem 4
| Step | Status | Lean |
|---|---|---|
| (i) Eq. (20) Jensen; equality iff velocity a.e. constant; the path is then straight | ✅ | `jensen_identity`, `jensen_effort`, `jensen_eq_iff`, `straight_of_const_velocity` |
| (i) every minimizer attains both bounds; positive-weight demos are straight | ✅ | `minimizer_attains_bounds`, `thm4i_straight` |
| (i) Brenier: `γ*` unique, supported on the graph of `∇f` | 📚 Mathlib has no optimal transport | — |
| (ii) `L_BC = L^{γ*}_CFM` | ✅ | `thm4ii` |
| (iii) `∇f` monotone for convex `f`; Eq. (21); no crossing; `L(ρ) = max{1/ρ, 1/(1−ρ)}` | ✅ (also for the multivalued `∂f`) | `gradient_convex_monotone`, `eq21`, `no_crossing`, `velocity_lipschitz`, `inv_min_eq_max` |
| (iii) every `x` is `(1−ρ)a + ρt`, `t ∈ ∂f(a)` (Rockafellar–Wets 12.60 / Minty) | 🟡 existence is the field of the structure `IdealField`; uniqueness is proved | `IdealField` |
| (iii) `v̄` fits every demo, is `L(ρ)`-Lipschitz on `ℝᵈ`; each demo is the unique rollout on `[ρ₀,ρ₁] ⊂ (0,1)` | ✅ given the above | `idealField_on_demo`, `idealField_lipschitz`, `demo_isRollout`, `demo_unique_rollout` |

### B.6 Theorem 5 and the score penalty
| Step | Status | Lean |
|---|---|---|
| Pointwise reduction (minimize input by input) | ✅ finite dataset, both directions | `total_eq_sum_fibers`, `min_of_pointwise_min`, `pointwise_min_of_min` |
| Split of `J` into orthogonal part + `g(a)` | ✅ any probability law of `u` with finite second moment | `sqLoss_eq`, `penalty_eq`, `J_split` |
| `s = 0` ⇒ `π_λ = v̄`; (i) orthogonal component; `a_λ` minimizes `g` | ✅ | `thm5_s_zero`, `thm5_i`, `thm5_along` |
| (ii) `|a_λ − b̄| ≤ λ‖s‖/2`; between `b̄` and every median; `→ m` as `λ‖s‖ → ∞` | ✅ | `thm5_ii_bound`, `thm5_ii_between`, `thm5_ii`, `thm5_ii_limit` |
| (iii) `a_λ = b̄` iff `b̄` is a median; constant or symmetric `b` ⇒ `π_λ = v̄`; unique minimizer | ✅ | `thm5_iii_iff`, `median_of_const`, `median_of_symmetric`, `thm5_iii`, `g_unique_min` |
| `E|a−b|` convex, 1-Lipschitz, continuous | ✅ | `absDev_convex`, `absDev_lipschitz`, `absDev_continuous` |
| "Size and direction" stationarity formula | ✅ finite law, no atom at `a` | `stationarity` |
| Examples: symmetric tie (`a_λ = 0`), ¾/¼ law with `λ‖s‖ = 2` (`a_λ = ½`), bias (`a_λ = max{1−λ‖s‖/6, 0}`) | ✅ | `ex_symmetric_tie`, `ex_shift_against`, `ex_bias` |
| Realizability: score of `N(1, v)` at 0 is `1/v` | ✅ | `realizability_score` |

## How the Lean differs from the paper

* **Conditional means given a state.** Eq. (7) is proved in full `L²` generality. Statements
  that condition on "the demonstrations at state `x` and phase `ρ`" (Eq. 8, Prop. 1's weak
  form, Lemma 2(a), the pointwise reduction of Thm 5) use a finite weighted dataset, i.e. the
  empirical distribution. That is also what training minimizes.
* **Rollouts** follow Mathlib's ODE conventions: continuous on `[ρ₀, ρ₁]`, with a right
  derivative on `[ρ₀, ρ₁)`.
* **Theorem 5(ii)** is proved by an elementary perturbation argument (compare `a` with
  `a + t(b̄ − a)` and let `t → 0`) instead of subgradients. The conclusions are the same.
  Medians are defined as minimizers of `a ↦ E|a − b|`.
* **Squared distances in Grönwall.** We differentiate `‖z − z'‖²`, which is smooth, instead
  of `‖z − z'‖`, which isn't differentiable where the rollouts meet. The bound is the same.

## Findings

Every proved step goes through as the appendix argues it; formalizing it turned up no errors.
Two small precisions came out of it:

* **Prop. 2(a), the endpoint starts.** The starts `−δ` and `1 + δ` sit on the boundary of
  `I = (−δ, 5+δ)` (for `1 + δ`, `b + 4 = 5 + δ`). The paper's ranges `(−δ, δ]` and
  `[1−δ, 1+δ)` already exclude them, and in `Φ_#P_S ≠ P_G` they form a null set, which the
  Lean proof removes explicitly.
* **Grönwall** is formalized for continuous `L(ρ)`. The appendix uses it with integrable
  `L`, but every application in the appendix has `L` continuous or bounded on the interval.
