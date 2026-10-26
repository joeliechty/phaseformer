import Mathlib
import PhaseformerProofs.Gronwall

/-!
# App. B.1 — Proofs for the Setting and Lemma 1

* **Eq. (7)** (`eq7_L2`).  The paper's proof is the orthogonality of conditional
  expectation.  In `L²(μ; E)` conditional expectation onto a sub-σ-algebra is Mathlib's
  `condExpL2`, the orthogonal projection onto the closed subspace `lpMeas` of
  `m`-measurable functions.  With `m = σ(Z)`, `m' = σ(Z') ⊆ σ(Z)` (because `Z' = h(Z)`,
  `comap_comp_le`), `Y = ẋ_ρ`, `m(Z) = condExpL2 m Y`, `m'(Z') = condExpL2 m' Y`, and any
  square-integrable policy `pol'` of `Z'` (an element of `lpMeas m'`, by Doob–Dynkin),
  `‖pol' - Y‖² = ‖pol' - m'‖² + ‖Y - m‖² + ‖m - m'‖²`, where `‖f‖² = ∫ ‖f‖² dμ`
  (`L2_norm_sq_eq_integral`).  The three terms are the fit error, the irreducible part,
  and the cost of not seeing `Z`.  The pure Hilbert-space core is `pythagoras_nested`.
* **Special cases, Eqs. (9)–(10)**: `eq9_L2` (`Z' = Z`), and the minimizer statement
  `eq7_minimizer` ("the minimizer over all policies of `Z'` is `m'`").
* **Eq. (8) / Bayes** (finite form, `fiberMean_tower`): the coarse conditional mean is the
  visitation-weighted average of the fine conditional means.
* **Lemma 1**: (a) `lemma1a` — BC loss on straight demonstrations equals the CFM loss for
  every policy; (b) `lemma1b` — FM inference and the closed-loop rollout coincide whenever
  the solution is unique; (c) `lemma1c_straight_velocity`, `lemma1c_sin_not_straight`.
-/

open MeasureTheory Set
open scoped RealInnerProductSpace

namespace Phaseformer

section Hilbert

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Pythagoras for three pairwise-orthogonal vectors. -/
lemma norm_add_add_sq_of_orth {a b c : E} (hab : ⟪a, b⟫ = 0) (hac : ⟪a, c⟫ = 0)
    (hbc : ⟪b, c⟫ = 0) : ‖a + b + c‖ ^ 2 = ‖a‖ ^ 2 + ‖b‖ ^ 2 + ‖c‖ ^ 2 := by
  rw [norm_add_sq_real, norm_add_sq_real, inner_add_left, hab, hac, hbc]
  ring

/-- **Eq. (7), Hilbert-space core.**  Let `K' ≤ K` be subspaces with orthogonal projections
`P' = K'.starProjection` and `P = K.starProjection` (`P y = m`, `P' y = m'`).  For every
`p' ∈ K'`, `‖p' - y‖² = ‖p' - m'‖² + ‖y - m‖² + ‖m - m'‖²`.  The cross terms vanish exactly
as in the paper: `y - m ⊥ K ⊇ K'`, and `m - m' = m - P' m ⊥ K'` by the tower property
`P' ∘ P = P'`. -/
theorem pythagoras_nested {K K' : Submodule ℝ E} [K.HasOrthogonalProjection]
    [K'.HasOrthogonalProjection] (hle : K' ≤ K) (y p' : E) (hp' : p' ∈ K') :
    ‖p' - y‖ ^ 2 =
      ‖p' - K'.starProjection y‖ ^ 2 + ‖y - K.starProjection y‖ ^ 2 +
        ‖K.starProjection y - K'.starProjection y‖ ^ 2 := by
  set m := K.starProjection y
  set m' := K'.starProjection y
  have hm'K : m' ∈ K := hle (K'.starProjection_apply_mem y)
  have hmK : m ∈ K := K.starProjection_apply_mem y
  have hperp : y - m ∈ Kᗮ := K.sub_starProjection_mem_orthogonal y
  -- tower property: `P' m = m'`
  have hPP : K'.starProjection m = m' := by
    have := congrArg (fun f => f y) (Submodule.starProjection_comp_starProjection_of_le hle)
    simpa using this
  have hperp' : m - m' ∈ K'ᗮ := by
    have := K'.sub_starProjection_mem_orthogonal m
    rwa [hPP] at this
  have a_mem : p' - m' ∈ K' := K'.sub_mem hp' (K'.starProjection_apply_mem y)
  have h1 : ⟪p' - m', m' - m⟫ = 0 := by
    rw [show m' - m = -(m - m') by abel, inner_neg_right, real_inner_comm,
      (Submodule.mem_orthogonal' _ _).1 hperp' _ a_mem, neg_zero]
  have h2 : ⟪p' - m', m - y⟫ = 0 := by
    rw [show m - y = -(y - m) by abel, inner_neg_right, real_inner_comm,
      (Submodule.mem_orthogonal' _ _).1 hperp _ (hle a_mem), neg_zero]
  have h3 : ⟪m' - m, m - y⟫ = 0 := by
    rw [show m - y = -(y - m) by abel, inner_neg_right, real_inner_comm,
      (Submodule.mem_orthogonal' _ _).1 hperp _ (K.sub_mem hm'K hmK), neg_zero]
  have hsplit : p' - y = (p' - m') + (m' - m) + (m - y) := by abel
  rw [hsplit, norm_add_add_sq_of_orth h1 h2 h3, ← norm_neg (m - y), neg_sub,
    ← norm_neg (m' - m), neg_sub]
  ring

end Hilbert

section L2

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
variable {α : Type*} {m m' m0 : MeasurableSpace α} {μ : Measure α}

omit [CompleteSpace E] in
/-- A coarser σ-algebra has fewer `L²` functions measurable with respect to it. -/
lemma lpMeas_mono (h : m' ≤ m) : lpMeas E ℝ m' 2 μ ≤ lpMeas E ℝ m 2 μ := by
  intro f hf
  rw [mem_lpMeas_iff_aestronglyMeasurable] at hf ⊢
  exact hf.mono h

omit [CompleteSpace E] in
/-- The squared `L²` norm is the expected squared norm, so `‖pol - Y‖²` is the BC loss. -/
lemma L2_norm_sq_eq_integral (f : α →₂[μ] E) : ‖f‖ ^ 2 = ∫ a, ‖f a‖ ^ 2 ∂μ := by
  rw [← real_inner_self_eq_norm_sq, L2.inner_def]
  simp

/-- If `Z' = h ∘ Z` with `h` measurable then `σ(Z') ≤ σ(Z)`. -/
lemma comap_comp_le {β γ : Type*} [mβ : MeasurableSpace β] [mγ : MeasurableSpace γ]
    (Z : α → β) (h : β → γ) (hh : Measurable h) :
    mγ.comap (h ∘ Z) ≤ mβ.comap Z := by
  rw [← MeasurableSpace.comap_comp]
  exact MeasurableSpace.comap_mono hh.comap_le

/-- **Eq. (7).**  `m ≤ m0` is `σ(Z)`, `m' ≤ m` is `σ(Z')`, `Y ∈ L²` is the demonstrated
velocity and `pol'` is any square-integrable `σ(Z')`-measurable policy. -/
theorem eq7_L2 (hm : m ≤ m0) (hm' : m' ≤ m) (Y pol' : α →₂[μ] E)
    (hpol' : pol' ∈ lpMeas E ℝ m' 2 μ) :
    ‖pol' - Y‖ ^ 2 =
      ‖pol' - (condExpL2 E ℝ (hm'.trans hm) Y : α →₂[μ] E)‖ ^ 2      -- fit error
      + ‖Y - (condExpL2 E ℝ hm Y : α →₂[μ] E)‖ ^ 2                  -- irreducible given Z
      + ‖(condExpL2 E ℝ hm Y : α →₂[μ] E)
          - (condExpL2 E ℝ (hm'.trans hm) Y : α →₂[μ] E)‖ ^ 2 := by -- cost of not seeing Z
  have : Fact (m ≤ m0) := ⟨hm⟩
  have : Fact (m' ≤ m0) := ⟨hm'.trans hm⟩
  exact pythagoras_nested (lpMeas_mono hm') Y pol' hpol'

/-- **Eq. (9)** (`Z' = Z`): `L_BC(π) = ‖π - v̄‖² + C` with `C = ‖Y - v̄‖²`. -/
theorem eq9_L2 (hm : m ≤ m0) (Y pol : α →₂[μ] E) (hpol : pol ∈ lpMeas E ℝ m 2 μ) :
    ‖pol - Y‖ ^ 2 = ‖pol - (condExpL2 E ℝ hm Y : α →₂[μ] E)‖ ^ 2
      + ‖Y - (condExpL2 E ℝ hm Y : α →₂[μ] E)‖ ^ 2 := by
  have h := eq7_L2 hm le_rfl Y pol hpol
  simpa using h

/-- **Minimizer.**  Among all policies of `Z'`, the loss is minimized by `m' = E[Y | Z']`,
with minimum `C + A`, and only by it. -/
theorem eq7_minimizer (hm : m ≤ m0) (hm' : m' ≤ m) (Y pol' : α →₂[μ] E)
    (hpol' : pol' ∈ lpMeas E ℝ m' 2 μ) :
    ‖(condExpL2 E ℝ (hm'.trans hm) Y : α →₂[μ] E) - Y‖ ^ 2 ≤ ‖pol' - Y‖ ^ 2 ∧
    (‖pol' - Y‖ ^ 2 = ‖(condExpL2 E ℝ (hm'.trans hm) Y : α →₂[μ] E) - Y‖ ^ 2 ↔
      pol' = (condExpL2 E ℝ (hm'.trans hm) Y : α →₂[μ] E)) := by
  have hmem : (condExpL2 E ℝ (hm'.trans hm) Y : α →₂[μ] E) ∈ lpMeas E ℝ m' 2 μ :=
    (condExpL2 E ℝ (hm'.trans hm) Y).2
  have h1 := eq7_L2 hm hm' Y pol' hpol'
  have h2 := eq7_L2 hm hm' Y _ hmem
  rw [sub_self, norm_zero] at h2
  refine ⟨?_, ?_⟩
  · rw [h1, h2]; nlinarith [sq_nonneg ‖pol' - (condExpL2 E ℝ (hm'.trans hm) Y : α →₂[μ] E)‖]
  · constructor
    · intro h
      rw [h1, h2] at h
      have : ‖pol' - (condExpL2 E ℝ (hm'.trans hm) Y : α →₂[μ] E)‖ ^ 2 = 0 := by linarith
      exact sub_eq_zero.1 (norm_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 this))
    · intro h; rw [h]

end L2

/-! ### Eq. (8) in finite form: the coarse average mixes the fine averages -/

section Finite

variable {Ω β γ V : Type*} [Fintype Ω] [DecidableEq β] [DecidableEq γ]
  [AddCommGroup V] [Module ℝ V]

/-- Total weight of the demonstrations (samples) whose input `Z` equals `z`. -/
def fiberMass (w : Ω → ℝ) (Z : Ω → β) (z : β) : ℝ :=
  ∑ ω ∈ Finset.univ.filter (fun ω => Z ω = z), w ω

/-- Weighted sum of `Y` over the fiber `Z = z`. -/
def fiberSum (w : Ω → ℝ) (Y : Ω → V) (Z : Ω → β) (z : β) : V :=
  ∑ ω ∈ Finset.univ.filter (fun ω => Z ω = z), w ω • Y ω

/-- Conditional mean `E[Y | Z = z]` for the weighted empirical distribution. -/
noncomputable def fiberMean (w : Ω → ℝ) (Y : Ω → V) (Z : Ω → β) (z : β) : V :=
  (fiberMass w Z z)⁻¹ • fiberSum w Y Z z

lemma fiberMass_smul_fiberMean {w : Ω → ℝ} (hw : ∀ ω, 0 ≤ w ω) (Y : Ω → V) (Z : Ω → β)
    (z : β) : fiberMass w Z z • fiberMean w Y Z z = fiberSum w Y Z z := by
  unfold fiberMean
  by_cases h : fiberMass w Z z = 0
  · have hz : ∀ ω ∈ Finset.univ.filter (fun ω => Z ω = z), w ω = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg (fun ω _ => hw ω)).1 h
    have : fiberSum w Y Z z = 0 := Finset.sum_eq_zero fun ω hω => by rw [hz ω hω, zero_smul]
    rw [this, h]; simp
  · rw [smul_smul, mul_inv_cancel₀ h, one_smul]

/-- **Eq. (8), finite form (Bayes / tower).**  For `Z' = h ∘ Z` (e.g. `h(x, ρ) = x`),
`E[Y | Z' = z'] = Σ_{z : h z = z'} Pr(Z = z) E[Y | Z = z] / Pr(Z' = z')`: the phase-free
target mixes the phase-conditioned targets `v̄(x, ρ)` with weights given by how often each
phase visits `x`. -/
theorem fiberMean_tower {w : Ω → ℝ} (hw : ∀ ω, 0 ≤ w ω) (Y : Ω → V) (Z : Ω → β)
    (h : β → γ) (z' : γ) :
    fiberMass w (h ∘ Z) z' • fiberMean w Y (h ∘ Z) z' =
      ∑ z ∈ (Finset.univ.image Z).filter (fun z => h z = z'),
        fiberMass w Z z • fiberMean w Y Z z := by
  rw [fiberMass_smul_fiberMean hw]
  simp_rw [fiberMass_smul_fiberMean hw]
  unfold fiberSum
  rw [← Finset.sum_fiberwise_of_maps_to (s := Finset.univ.filter (fun ω => (h ∘ Z) ω = z'))
    (t := (Finset.univ.image Z).filter (fun z => h z = z')) (g := Z)]
  · refine Finset.sum_congr rfl fun z hz => ?_
    refine Finset.sum_congr ?_ fun _ _ => rfl
    ext ω
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Function.comp]
    constructor
    · exact fun h' => h'.2
    · intro h'; refine ⟨?_, h'⟩
      rw [h']; exact (Finset.mem_filter.1 hz).2
  · intro ω hω
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Function.comp,
      Finset.mem_image] at hω ⊢
    exact ⟨⟨ω, rfl⟩, hω⟩

end Finite

/-! ### Lemma 1: flow matching is phase-conditioned BC for any coupling -/

section Lemma1

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The straight path between paired endpoints, `x_ρ = (1-ρ) x₀ + ρ x₁`. -/
def straightPath (p : E × E) (ρ : ℝ) : E := (1 - ρ) • p.1 + ρ • p.2

lemma hasDerivAt_straightPath (p : E × E) (ρ : ℝ) :
    HasDerivAt (straightPath p) (p.2 - p.1) ρ := by
  have h := (((hasDerivAt_id ρ).const_sub 1).smul_const p.1).add
    ((hasDerivAt_id ρ).smul_const p.2)
  have e : straightPath p = fun ρ => (1 - id ρ) • p.1 + id ρ • p.2 := rfl
  rw [e]
  convert h using 1
  simp only [neg_smul, one_smul]; abel

/-- Squared-loss BC objective, Eq. (6): average over demonstrations `ω ∼ μ` and phases
`ρ ∼ U[0,1]` of `‖π(x_ρ, ρ) - ẋ_ρ‖²`. -/
noncomputable def bcLoss {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (X : Ω → ℝ → E)
    (pol : E → ℝ → E) : ℝ :=
  ∫ ω, (∫ ρ in (0 : ℝ)..1, ‖pol (X ω ρ) ρ - deriv (X ω) ρ‖ ^ 2) ∂μ

/-- Conditional flow-matching objective with coupling `γ`, Eq. (12). -/
noncomputable def cfmLoss [MeasurableSpace E] (γ : Measure (E × E)) (pol : E → ℝ → E) : ℝ :=
  ∫ p, (∫ ρ in (0 : ℝ)..1, ‖pol ((1 - ρ) • p.1 + ρ • p.2) ρ - (p.2 - p.1)‖ ^ 2) ∂γ

/-- **Lemma 1(a).**  For any coupling `γ` and any policy, the BC loss on the straight
demonstrations `x_ρ = (1-ρ)x₀ + ρx₁`, `(x₀,x₁) ∼ γ`, equals the CFM loss: substituting
`x_ρ` and `ẋ_ρ = x₁ - x₀` into Eq. (6) gives Eq. (12) term by term.  (Both are then
minimized by `v̄` with minimum `C_γ`, by `eq9_L2`.) -/
theorem lemma1a [MeasurableSpace E] (γ : Measure (E × E)) (pol : E → ℝ → E) :
    bcLoss γ straightPath pol = cfmLoss γ pol := by
  unfold bcLoss cfmLoss
  congr 1; funext p
  congr 1; funext ρ
  rw [(hasDerivAt_straightPath p ρ).deriv]
  rfl

/-- **Lemma 1(b).**  Flow-matching inference integrates `dx/dρ = π(x, ρ)` from `x₀`, which
is the closed-loop rollout of the phase-conditioned policy; whenever the solution is unique
(here: `π` Lipschitz in the state with continuous `L`) the two coincide. -/
theorem lemma1b {pol : E → ℝ → E} {L : ℝ → ℝ} (hL : Continuous L)
    (hLip : LipschitzInState pol L 0 1) {zFM zBC : ℝ → E}
    (hFM : IsRollout pol zFM 0 1) (hBC : IsRollout pol zBC 0 1) (h0 : zFM 0 = zBC 0) :
    EqOn zFM zBC (Icc 0 1) :=
  rollouts_eq_of_eq hL hLip hFM hBC h0

/-- **Lemma 1(c), first half.**  For straight demonstrations the velocity `x₁ - x₀` does
not depend on `ρ`. -/
theorem lemma1c_straight_velocity (p : E × E) (ρ ρ' : ℝ) :
    deriv (straightPath p) ρ = deriv (straightPath p) ρ' := by
  rw [(hasDerivAt_straightPath p ρ).deriv, (hasDerivAt_straightPath p ρ').deriv]

/-- **Lemma 1(c), second half.**  The single demonstration `x_ρ = sin(πρ)` is BC data that no
straight-line dataset produces: no straight path agrees with it on `[0,1]`, and its velocity
`π cos(πρ)` changes with `ρ` (it is `π` at `ρ = 0` and `-π` at `ρ = 1`). -/
theorem lemma1c_sin_not_straight :
    (¬ ∃ p : ℝ × ℝ, ∀ ρ ∈ Icc (0 : ℝ) 1, Real.sin (Real.pi * ρ) = straightPath p ρ) ∧
    deriv (fun ρ => Real.sin (Real.pi * ρ)) 0 = Real.pi ∧
    deriv (fun ρ => Real.sin (Real.pi * ρ)) 1 = -Real.pi := by
  have hd : ∀ ρ, HasDerivAt (fun ρ => Real.sin (Real.pi * ρ))
      (Real.cos (Real.pi * ρ) * Real.pi) ρ := fun ρ => by
    simpa using ((hasDerivAt_id ρ).const_mul Real.pi).sin
  refine ⟨?_, ?_, ?_⟩
  · rintro ⟨p, hp⟩
    have h0 := hp 0 (by norm_num)
    have h1 := hp 1 (by norm_num)
    have hh := hp (1 / 2) (by norm_num)
    simp only [straightPath, smul_eq_mul] at h0 h1 hh
    rw [show Real.pi * (1 / 2) = Real.pi / 2 by ring, Real.sin_pi_div_two] at hh
    simp at h0 h1
    rw [← h0, ← h1] at hh
    norm_num at hh
  · rw [(hd 0).deriv]; simp
  · rw [(hd 1).deriv]; simp

end Lemma1

end Phaseformer
