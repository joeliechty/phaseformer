import Mathlib
import PhaseformerProofs.Gronwall
import PhaseformerProofs.B1_Setting

/-!
# App. B.2 — Proof of Proposition 1, and a Closed-Form Example

Demonstrations are a finite weighted family of differentiable paths `X ω`, `ω : Ω`, with
velocities `V ω ρ` (the empirical distribution of a dataset).  `v̄(x, ρ)` is the
conditional mean of the velocity over the demonstrations at state `x` and phase `ρ`
(`fiberMean`, B.1).

* `average_velocity_first` — "the second step averages the velocity first, which is
  allowed because `∇h(x_ρ)` is fixed once `x_ρ` is".
* `prop1_weak_continuity` — the weak form of the continuity equation
  `d/dρ E h(x_ρ) = E[∇h(x_ρ) · v̄(x_ρ, ρ)]`, for every differentiable test function `h`.
* `prop1_phase_free_identity` — integrating over `ρ ∈ [0,1]` gives the phase-free identity
  `∇·(μ v̄_free) = p₀ - p₁` in weak form:
  `∫₀¹ E[∇h(x_ρ) · v̄(x_ρ, ρ)] dρ = E h(x₁) - E h(x₀)`.
* `phase_free_identity_1d` — in one dimension, `μ v̄_free = F_S - F_G` in weak form:
  `∫₀¹ E[φ(x_ρ) ẋ_ρ] dρ = ∫ φ(t) (F_S(t) - F_G(t)) dt`.
* Regularity: `regularity_vbar_zero`, `regularity_path_norm`.
* Closed-form example (Fig. 6): `gauss_sigmaSq_pos`, `gauss_lipschitz_le_one`,
  `gauss_rollout`, `gauss_rollout_unique`, `gauss_rollout_returns`, `gauss_field_half`,
  `gauss_no_crossing`, `gauss_pushforward`, `gauss_covariance`.

**Not formalized (cited):** the uniqueness of solutions of the continuity equation for a
Lipschitz field (Ambrosio–Gigli–Savaré 2008, §8.1), which turns the weak equation into
"rolling out `v̄` reproduces `p_ρ`"; and the Gaussian conditional-mean formula
`E[x₁ - x₀ | x_ρ] = Cov/Var · x_ρ` for jointly Gaussian variables (we prove the covariance
algebra, `gauss_covariance`, from which that formula reads off).
-/

open MeasureTheory Set
open scoped RealInnerProductSpace

namespace Phaseformer

section WeakCE

variable {Ω E : Type*} [Fintype Ω] [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [DecidableEq E]

omit [InnerProductSpace ℝ E] [DecidableEq E] in
/-- **Averaging the velocity first.**  For any state-dependent linear functional `A(x)`
(here `A(x) = ∇h(x)`), `E[A(x)(ẋ)] = E[A(x)(E[ẋ | x])]`. -/
theorem average_velocity_first [NormedSpace ℝ E] {β : Type*} [DecidableEq β]
    {w : Ω → ℝ} (hw : ∀ ω, 0 ≤ w ω) (A : β → E →L[ℝ] ℝ) (Y : Ω → E) (Z : Ω → β) :
    ∑ ω, w ω * A (Z ω) (Y ω) = ∑ ω, w ω * A (Z ω) (fiberMean w Y Z (Z ω)) := by
  have key : ∀ F : Ω → ℝ, ∑ ω, F ω =
      ∑ z ∈ Finset.univ.image Z, ∑ ω ∈ Finset.univ.filter (fun ω => Z ω = z), F ω :=
    fun F => (Finset.sum_fiberwise_of_maps_to (s := Finset.univ) (g := Z)
      (fun ω _ => Finset.mem_image_of_mem Z (Finset.mem_univ ω)) F).symm
  rw [key, key]
  refine Finset.sum_congr rfl fun z _ => ?_
  have e1 : ∑ ω ∈ Finset.univ.filter (fun ω => Z ω = z), w ω * A (Z ω) (Y ω) =
      A z (fiberSum w Y Z z) := by
    unfold fiberSum
    rw [map_sum]
    refine Finset.sum_congr rfl fun ω hω => ?_
    rw [(Finset.mem_filter.1 hω).2, map_smul, smul_eq_mul]
  have e2 : ∑ ω ∈ Finset.univ.filter (fun ω => Z ω = z),
      w ω * A (Z ω) (fiberMean w Y Z (Z ω)) = A z (fiberSum w Y Z z) := by
    rw [← fiberMass_smul_fiberMean hw, map_smul, smul_eq_mul]
    unfold fiberMass
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun ω hω => ?_
    rw [(Finset.mem_filter.1 hω).2]
  rw [e1, e2]

/-- `v̄(x, ρ)`: average velocity of the demonstrations at state `x` and phase `ρ`. -/
noncomputable def vbar (w : Ω → ℝ) (X V : Ω → ℝ → E) (ρ : ℝ) (x : E) : E :=
  fiberMean w (fun ω => V ω ρ) (fun ω => X ω ρ) x

/-- **Proposition 1, weak continuity equation.**  For every differentiable test function
`h`, `d/dρ E h(x_ρ) = E[∇h(x_ρ) · ẋ_ρ] = E[∇h(x_ρ) · v̄(x_ρ, ρ)]`. -/
theorem prop1_weak_continuity {w : Ω → ℝ} (hw : ∀ ω, 0 ≤ w ω) {X V : Ω → ℝ → E}
    (hX : ∀ ω ρ, HasDerivAt (X ω) (V ω ρ) ρ) {h : E → ℝ} {h' : E → E →L[ℝ] ℝ}
    (hh : ∀ x, HasFDerivAt h (h' x) x) (ρ : ℝ) :
    HasDerivAt (fun ρ => ∑ ω, w ω * h (X ω ρ))
      (∑ ω, w ω * h' (X ω ρ) (vbar w X V ρ (X ω ρ))) ρ := by
  have hsum : HasDerivAt (fun ρ => ∑ ω, w ω * h (X ω ρ))
      (∑ ω, w ω * h' (X ω ρ) (V ω ρ)) ρ := by
    have := HasDerivAt.fun_sum (u := Finset.univ) (A := fun ω ρ => w ω * h (X ω ρ))
      (A' := fun ω => w ω * h' (X ω ρ) (V ω ρ)) (x := ρ)
      (fun ω _ => ((hh (X ω ρ)).comp_hasDerivAt ρ (hX ω ρ)).const_mul (w ω))
    simpa using this
  exact hsum.congr_deriv (average_velocity_first hw (fun x => h' x) (fun ω => V ω ρ)
    (fun ω => X ω ρ))

/-- **Proposition 1, phase-free identity.**  Integrating the weak continuity equation over
`ρ ∈ [0,1]` gives `∇·(μ v̄_free) = p₀ - p₁` in weak form (`∫₀¹ p_ρ v̄ dρ = μ v̄_free` is
Eq. (8)):  `∫₀¹ E[∇h(x_ρ) · v̄(x_ρ, ρ)] dρ = E h(x₁) - E h(x₀)`. -/
theorem prop1_phase_free_identity {w : Ω → ℝ} (hw : ∀ ω, 0 ≤ w ω) {X V : Ω → ℝ → E}
    (hX : ∀ ω ρ, HasDerivAt (X ω) (V ω ρ) ρ) (hV : ∀ ω, Continuous (V ω))
    {h : E → ℝ} {h' : E → E →L[ℝ] ℝ} (hh : ∀ x, HasFDerivAt h (h' x) x)
    (hh' : Continuous h') :
    ∫ ρ in (0 : ℝ)..1, ∑ ω, w ω * h' (X ω ρ) (vbar w X V ρ (X ω ρ)) =
      ∑ ω, w ω * h (X ω 1) - ∑ ω, w ω * h (X ω 0) := by
  have hcongr : (fun ρ => ∑ ω, w ω * h' (X ω ρ) (vbar w X V ρ (X ω ρ))) =
      fun ρ => ∑ ω, w ω * h' (X ω ρ) (V ω ρ) := by
    funext ρ
    exact (average_velocity_first hw (fun x => h' x) (fun ω => V ω ρ)
      (fun ω => X ω ρ)).symm
  rw [hcongr]
  have hXc : ∀ ω, Continuous (X ω) := fun ω =>
    continuous_iff_continuousAt.2 fun ρ => (hX ω ρ).continuousAt
  have hcont : Continuous fun ρ => ∑ ω, w ω * h' (X ω ρ) (V ω ρ) := by
    refine continuous_finsetSum _ fun ω _ => continuous_const.mul ?_
    exact (hh'.comp (hXc ω)).clm_apply (hV ω)
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun ρ => ∑ ω, w ω * h (X ω ρ))
    (fun ρ _ => ?_) (hcont.intervalIntegrable 0 1)]
  have := HasDerivAt.fun_sum (u := Finset.univ) (A := fun ω ρ => w ω * h (X ω ρ))
    (A' := fun ω => w ω * h' (X ω ρ) (V ω ρ)) (x := ρ)
    (fun ω _ => ((hh (X ω ρ)).comp_hasDerivAt ρ (hX ω ρ)).const_mul (w ω))
  simpa using this

end WeakCE

section OneDim

variable {Ω : Type*} [Fintype Ω]

/-- Empirical cumulative distribution function `F(t) = Σ w_ω 1[x_ω ≤ t]`. -/
noncomputable def ecdf (w : Ω → ℝ) (x : Ω → ℝ) (t : ℝ) : ℝ :=
  ∑ ω, w ω * (Ici (x ω)).indicator (fun _ => (1 : ℝ)) t

/-- **The phase-free identity in one dimension.**  `μ v̄_free = F_S - F_G` in weak form:
for every continuous integrable `φ`,
`∫₀¹ E[φ(x_ρ) ẋ_ρ] dρ = ∫ φ(t) (F_S(t) - F_G(t)) dt`.
(The left side is `∫ φ · μ v̄_free`, by Eq. (8).) -/
theorem phase_free_identity_1d (w : Ω → ℝ) {X V : Ω → ℝ → ℝ}
    (hX : ∀ ω ρ, HasDerivAt (X ω) (V ω ρ) ρ) (hV : ∀ ω, Continuous (V ω))
    {φ : ℝ → ℝ} (hφ : Continuous φ) (hφi : Integrable φ) :
    ∫ ρ in (0 : ℝ)..1, ∑ ω, w ω * (φ (X ω ρ) * V ω ρ) =
      ∫ t, φ t * (ecdf w (fun ω => X ω 0) t - ecdf w (fun ω => X ω 1) t) := by
  have hXc : ∀ ω, Continuous (X ω) := fun ω =>
    continuous_iff_continuousAt.2 fun ρ => (hX ω ρ).continuousAt
  -- left side: each pass contributes `∫_{x₀}^{x₁} φ`
  have hL : ∫ ρ in (0 : ℝ)..1, ∑ ω, w ω * (φ (X ω ρ) * V ω ρ) =
      ∑ ω, w ω * ∫ t in X ω 0..X ω 1, φ t := by
    rw [intervalIntegral.integral_finsetSum]
    · refine Finset.sum_congr rfl fun ω _ => ?_
      rw [intervalIntegral.integral_const_mul]
      congr 1
      exact intervalIntegral.integral_comp_mul_deriv (fun ρ _ => hX ω ρ)
        (hV ω).continuousOn hφ
    · intro ω _
      exact (continuous_const.mul ((hφ.comp (hXc ω)).mul (hV ω))).intervalIntegrable _ _
  -- right side: `∫ φ · 1[x₀ ≤ t] - ∫ φ · 1[x₁ ≤ t] = ∫_{x₀}^{x₁} φ`
  have hind : ∀ a : ℝ, Integrable (fun t => φ t * (Ici a).indicator (fun _ => (1 : ℝ)) t) := by
    intro a
    have : (fun t => φ t * (Ici a).indicator (fun _ => (1 : ℝ)) t) = (Ici a).indicator φ := by
      funext t; by_cases ht : t ∈ Ici a <;> simp [ht]
    rw [this]; exact hφi.indicator measurableSet_Ici
  have hR : ∫ t, φ t * (ecdf w (fun ω => X ω 0) t - ecdf w (fun ω => X ω 1) t) =
      ∑ ω, w ω * ∫ t in X ω 0..X ω 1, φ t := by
    have expand : (fun t => φ t * (ecdf w (fun ω => X ω 0) t - ecdf w (fun ω => X ω 1) t)) =
        fun t => ∑ ω, w ω * (φ t * (Ici (X ω 0)).indicator (fun _ => (1 : ℝ)) t -
          φ t * (Ici (X ω 1)).indicator (fun _ => (1 : ℝ)) t) := by
      funext t; unfold ecdf
      rw [mul_sub, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun ω _ => by ring
    rw [expand, integral_finsetSum]
    · refine Finset.sum_congr rfl fun ω _ => ?_
      rw [integral_const_mul, integral_sub (hind _) (hind _)]
      congr 1
      have e : ∀ a : ℝ, ∫ t, φ t * (Ici a).indicator (fun _ => (1 : ℝ)) t = ∫ t in Ici a, φ t := by
        intro a
        rw [← integral_indicator measurableSet_Ici]
        congr 1; funext t; by_cases ht : t ∈ Ici a <;> simp [ht]
      rw [e, e]
      exact intervalIntegral.integral_Ici_sub_Ici' hφi.integrableOn hφi.integrableOn
    · intro ω _
      exact ((hind _).sub (hind _)).const_mul _
  rw [hL, hR]

end OneDim

section Regularity

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

omit [InnerProductSpace ℝ E] [CompleteSpace E] in
/-- **Regularity, first bound.**  If `v̄(·, ρ)` is `L(ρ)`-Lipschitz then
`‖v̄(0, ρ)‖ ≤ ‖v̄(x, ρ)‖ + L(ρ) ‖x‖` for every `x`. -/
theorem regularity_vbar_zero {v : E → E} {L : ℝ} (hv : ∀ x y, ‖v x - v y‖ ≤ L * ‖x - y‖)
    (x : E) : ‖v 0‖ ≤ ‖v x‖ + L * ‖x‖ := by
  have := hv 0 x
  rw [zero_sub, norm_neg] at this
  calc ‖v 0‖ = ‖v x + (v 0 - v x)‖ := by congr 1; abel
    _ ≤ ‖v x‖ + ‖v 0 - v x‖ := norm_add_le _ _
    _ ≤ ‖v x‖ + L * ‖x‖ := by linarith

omit [InnerProductSpace ℝ E] in
/-- **Regularity, second bound.**  Along a path with integrable velocity,
`‖x_ρ‖ ≤ ‖x_{ρ₀}‖ + ∫₀¹ ‖ẋ‖` for `0 ≤ ρ₀ ≤ ρ ≤ 1`; averaging over demonstrations gives
`E‖x_ρ‖ ≤ E‖x_{ρ₀}‖ + E ∫₀¹ ‖ẋ_ρ‖ dρ`. -/
theorem regularity_path_norm [NormedSpace ℝ E] {x v : ℝ → E}
    (hx : ∀ ρ, HasDerivAt x (v ρ) ρ) (hv : Continuous v) {ρ₀ ρ : ℝ}
    (h0 : 0 ≤ ρ₀) (h01 : ρ₀ ≤ ρ) (h1 : ρ ≤ 1) :
    ‖x ρ‖ ≤ ‖x ρ₀‖ + ∫ s in (0 : ℝ)..1, ‖v s‖ := by
  have hftc : ∫ s in ρ₀..ρ, v s = x ρ - x ρ₀ :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt (fun s _ => hx s) (hv.intervalIntegrable _ _)
  have hn : ‖x ρ - x ρ₀‖ ≤ ∫ s in ρ₀..ρ, ‖v s‖ := by
    rw [← hftc]; exact intervalIntegral.norm_integral_le_integral_norm h01
  have hmono : ∫ s in ρ₀..ρ, ‖v s‖ ≤ ∫ s in (0 : ℝ)..1, ‖v s‖ :=
    intervalIntegral.integral_mono_interval h0 h01 h1
      (Filter.Eventually.of_forall fun _ => norm_nonneg _) (hv.norm.intervalIntegrable _ _)
  calc ‖x ρ‖ = ‖x ρ₀ + (x ρ - x ρ₀)‖ := by congr 1; abel
    _ ≤ ‖x ρ₀‖ + ‖x ρ - x ρ₀‖ := norm_add_le _ _
    _ ≤ _ := by linarith

end Regularity

/-! ### The closed-form example (standard CFM with independent Gaussian endpoints) -/

section Gaussian

open Real ProbabilityTheory

/-- `σ_ρ² = (1-ρ)² + ρ²`. -/
def sigmaSq (ρ : ℝ) : ℝ := (1 - ρ) ^ 2 + ρ ^ 2

/-- `σ_ρ`. -/
noncomputable def sigma (ρ : ℝ) : ℝ := sqrt (sigmaSq ρ)

theorem gauss_sigmaSq_pos (ρ : ℝ) : 0 < sigmaSq ρ := by
  unfold sigmaSq; nlinarith [sq_nonneg (2 * ρ - 1)]

theorem sigma_pos (ρ : ℝ) : 0 < sigma ρ := sqrt_pos.2 (gauss_sigmaSq_pos ρ)

theorem sigma_sq (ρ : ℝ) : sigma ρ ^ 2 = sigmaSq ρ := sq_sqrt (gauss_sigmaSq_pos ρ).le

/-- The MSE optimum of standard CFM between independent `N(0,1)` endpoints:
`v̄(x, ρ) = (2ρ - 1)/σ_ρ² · x`. -/
noncomputable def vbarGauss (x ρ : ℝ) : ℝ := (2 * ρ - 1) / sigmaSq ρ * x

/-- `v̄` is `L(ρ)`-Lipschitz with `L(ρ) = |2ρ - 1|/σ_ρ² ≤ 1`. -/
theorem gauss_lipschitz_le_one (ρ : ℝ) : |2 * ρ - 1| / sigmaSq ρ ≤ 1 := by
  rw [div_le_one (gauss_sigmaSq_pos ρ)]
  unfold sigmaSq
  cases abs_cases (2 * ρ - 1) with
  | inl h => rw [h.1]; nlinarith [sq_nonneg (2 * ρ - 1 - 1)]
  | inr h => rw [h.1]; nlinarith [sq_nonneg (2 * ρ - 1 + 1)]

theorem gauss_lipschitzInState : LipschitzInState vbarGauss (fun _ => 1) 0 1 := by
  intro ρ _ x y
  simp only [vbarGauss, Real.norm_eq_abs, one_mul]
  rw [← mul_sub, abs_mul, abs_div, abs_of_pos (gauss_sigmaSq_pos ρ)]
  exact mul_le_of_le_one_left (abs_nonneg _) (gauss_lipschitz_le_one ρ)

/-- `dσ_ρ/dρ = (2ρ - 1)/σ_ρ`, so `x(ρ) = σ_ρ x(0)` solves `dx/dρ = v̄(x, ρ)`. -/
theorem gauss_rollout (x0 ρ : ℝ) :
    HasDerivAt (fun ρ => sigma ρ * x0) (vbarGauss (sigma ρ * x0) ρ) ρ := by
  have hS : HasDerivAt sigmaSq (2 * (2 * ρ - 1)) ρ := by
    have := (((hasDerivAt_id' ρ).const_sub 1).pow 2).add ((hasDerivAt_id' ρ).pow 2)
    exact this.congr_deriv (by push_cast; ring)
  have hσ : HasDerivAt sigma (2 * (2 * ρ - 1) / (2 * sqrt (sigmaSq ρ))) ρ :=
    hS.sqrt (gauss_sigmaSq_pos ρ).ne'
  convert hσ.mul_const x0 using 1
  have hs := sigma_pos ρ
  have e : sigmaSq ρ = sigma ρ ^ 2 := (sigma_sq ρ).symm
  show (2 * ρ - 1) / sigmaSq ρ * (sigma ρ * x0) = 2 * (2 * ρ - 1) / (2 * sigma ρ) * x0
  rw [e]; field_simp

theorem gauss_isRollout (x0 : ℝ) : IsRollout vbarGauss (fun ρ => sigma ρ * x0) 0 1 :=
  ⟨fun ρ _ => (gauss_rollout x0 ρ).continuousAt.continuousWithinAt,
    fun ρ _ => (gauss_rollout x0 ρ).hasDerivWithinAt⟩

/-- The rollout of `v̄` from `x(0)` is unique, so it is `σ_ρ x(0)`. -/
theorem gauss_rollout_unique {z : ℝ → ℝ} (hz : IsRollout vbarGauss z 0 1) :
    EqOn z (fun ρ => sigma ρ * z 0) (Icc 0 1) := by
  refine rollouts_eq_of_eq continuous_const gauss_lipschitzInState hz (gauss_isRollout (z 0)) ?_
  simp [sigma, sigmaSq]

/-- Each rollout ends where it started: `σ₁ = σ₀ = 1`. -/
theorem gauss_rollout_returns (x0 : ℝ) : sigma 1 * x0 = x0 ∧ sigma 0 * x0 = x0 := by
  simp [sigma, sigmaSq]

/-- At `ρ = ½`, `v̄ = 0` everywhere and every rollout momentarily stops. -/
theorem gauss_field_half (x : ℝ) : vbarGauss x (1 / 2) = 0 := by
  simp [vbarGauss]

/-- The rollouts never cross: starts in increasing order stay in increasing order. -/
theorem gauss_no_crossing {x0 y0 : ℝ} (h : x0 < y0) (ρ : ℝ) : sigma ρ * x0 < sigma ρ * y0 :=
  mul_lt_mul_of_pos_left h (sigma_pos ρ)

/-- The rollout matches `p_ρ` at every phase: `σ_ρ x(0) ∼ N(0, σ_ρ²)` when
`x(0) ∼ N(0, 1)`. -/
theorem gauss_pushforward (ρ : ℝ) :
    (gaussianReal 0 1).map (fun x => sigma ρ * x) = gaussianReal 0 (sigmaSq ρ).toNNReal := by
  rw [gaussianReal_map_const_mul]
  congr 1
  · simp
  · ext; simp [sigma_sq, (gauss_sigmaSq_pos ρ).le]

/-- **Covariance algebra.**  For `x₀, x₁` with unit second moments and `E[x₀x₁] = 0`
(e.g. independent `N(0,1)`), `x_ρ = (1-ρ)x₀ + ρx₁` satisfies
`E[(x₁ - x₀) x_ρ] = 2ρ - 1` and `E[x_ρ²] = σ_ρ²`, so the residual
`(x₁ - x₀) - (2ρ-1)/σ_ρ² · x_ρ` is uncorrelated with `x_ρ`.  For jointly Gaussian
variables (cited, not formalized) this makes `E[x₁ - x₀ | x_ρ] = (2ρ-1)/σ_ρ² · x_ρ`. -/
theorem gauss_covariance {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) {x0 x1 : Ω → ℝ}
    (h00 : Integrable (fun ω => x0 ω * x0 ω) P) (h11 : Integrable (fun ω => x1 ω * x1 ω) P)
    (h01 : Integrable (fun ω => x0 ω * x1 ω) P)
    (m00 : ∫ ω, x0 ω * x0 ω ∂P = 1) (m11 : ∫ ω, x1 ω * x1 ω ∂P = 1)
    (m01 : ∫ ω, x0 ω * x1 ω ∂P = 0) (ρ : ℝ) :
    ∫ ω, (x1 ω - x0 ω) * ((1 - ρ) * x0 ω + ρ * x1 ω) ∂P = 2 * ρ - 1 ∧
    ∫ ω, ((1 - ρ) * x0 ω + ρ * x1 ω) ^ 2 ∂P = sigmaSq ρ ∧
    ∫ ω, ((x1 ω - x0 ω) - (2 * ρ - 1) / sigmaSq ρ * ((1 - ρ) * x0 ω + ρ * x1 ω)) *
      ((1 - ρ) * x0 ω + ρ * x1 ω) ∂P = 0 := by
  -- every integrand is a combination `a·x₀² + b·x₁² + c·x₀x₁`
  have comb : ∀ a b c : ℝ, ∫ ω, (a * (x0 ω * x0 ω) + b * (x1 ω * x1 ω) + c * (x0 ω * x1 ω)) ∂P
      = a * 1 + b * 1 + c * 0 := by
    intro a b c
    rw [integral_add, integral_add, integral_const_mul, integral_const_mul,
      integral_const_mul, m00, m11, m01]
    all_goals first
      | exact h00.const_mul a | exact h11.const_mul b | exact h01.const_mul c
      | exact (h00.const_mul a).add (h11.const_mul b)
  have e1 : ∫ ω, (x1 ω - x0 ω) * ((1 - ρ) * x0 ω + ρ * x1 ω) ∂P = 2 * ρ - 1 := by
    have := comb (-(1 - ρ)) ρ (1 - 2 * ρ)
    rw [show (fun ω => (x1 ω - x0 ω) * ((1 - ρ) * x0 ω + ρ * x1 ω)) =
      fun ω => -(1 - ρ) * (x0 ω * x0 ω) + ρ * (x1 ω * x1 ω) + (1 - 2 * ρ) * (x0 ω * x1 ω) by
        funext ω; ring, this]
    ring
  have e2 : ∫ ω, ((1 - ρ) * x0 ω + ρ * x1 ω) ^ 2 ∂P = sigmaSq ρ := by
    have := comb ((1 - ρ) ^ 2) (ρ ^ 2) (2 * ρ * (1 - ρ))
    rw [show (fun ω => ((1 - ρ) * x0 ω + ρ * x1 ω) ^ 2) =
      fun ω => (1 - ρ) ^ 2 * (x0 ω * x0 ω) + ρ ^ 2 * (x1 ω * x1 ω) +
        2 * ρ * (1 - ρ) * (x0 ω * x1 ω) by funext ω; ring, this]
    unfold sigmaSq; ring
  refine ⟨e1, e2, ?_⟩
  set k := (2 * ρ - 1) / sigmaSq ρ
  have := comb (-(1 - ρ) - k * (1 - ρ) ^ 2) (ρ - k * ρ ^ 2)
    (1 - 2 * ρ - k * (2 * ρ * (1 - ρ)))
  rw [show (fun ω => ((x1 ω - x0 ω) - k * ((1 - ρ) * x0 ω + ρ * x1 ω)) *
      ((1 - ρ) * x0 ω + ρ * x1 ω)) =
    fun ω => (-(1 - ρ) - k * (1 - ρ) ^ 2) * (x0 ω * x0 ω) + (ρ - k * ρ ^ 2) * (x1 ω * x1 ω) +
      (1 - 2 * ρ - k * (2 * ρ * (1 - ρ))) * (x0 ω * x1 ω) by funext ω; ring, this]
  have hs := (gauss_sigmaSq_pos ρ).ne'
  simp only [k]
  unfold sigmaSq at hs ⊢
  field_simp
  ring

end Gaussian

end Phaseformer
