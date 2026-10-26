import Mathlib
import PhaseformerProofs.Gronwall
import PhaseformerProofs.B1_Setting

/-!
# App. B.5 — Proof of Theorem 4 (ideal case: BC on minimum-effort demonstrations is OT-CFM)

## (i) Demonstrations are OT-paired straight lines
* `jensen_effort` — Eq. (20): `½‖x₁ - x₀‖² ≤ ∫₀¹ ½‖ẋ_ρ‖² dρ`, via the exact identity
  `∫₀¹ ‖ẋ - (x₁ - x₀)‖² = ∫₀¹ ‖ẋ‖² - ‖x₁ - x₀‖²`.
* `jensen_eq_iff` — equality iff the velocity is (a.e.) constant, and then the path is the
  straight line `(1-ρ)x₀ + ρx₁` (`straight_of_const_velocity`).
* `minimizer_attains_bounds` — the order argument: `𝓔 ≥ ½E‖x₁-x₀‖² ≥ ½W₂²`, both bounds
  attained by straight lines between `γ*`-paired endpoints, so every minimizer attains both.
* `thm4i_straight` — for a (weighted, finite) family of demonstrations whose effort equals
  its endpoint cost, every demonstration of positive weight is a straight line.

  **Cited, not formalized:** Brenier's theorem (`γ*` is unique and supported on the graph
  of `T = ∇f`, `f` convex) — Mathlib has no optimal-transport theory yet.

## (ii) Same objective
`thm4ii` — `L_BC = L^{γ*}_CFM` for every policy (substitution; Lemma 1(a) with `γ = γ*`).

## (iii) No label conflict
* `gradient_convex_monotone` — "because `f` is convex, `T = ∇f` is monotone".
* `eq21` — Eq. (21): `‖(1-ρ)Δ₀ + ρΔ₁‖² ≥ (1-ρ)²‖Δ₀‖² + ρ²‖Δ₁‖²` when `⟨Δ₀, Δ₁⟩ ≥ 0`.
* `no_crossing` — for `ρ < 1`, equal states force equal starts (and for `0 < ρ < 1` equal
  endpoints too, so this holds for the subdifferential `∂f` as well): demonstrations never
  cross, `C = 0`.
* `velocity_lipschitz` — `‖Δ₁ - Δ₀‖ ≤ ‖x_ρ - x'_ρ‖ / min(ρ, 1-ρ)`, i.e.
  `L(ρ) = max{1/ρ, 1/(1-ρ)}`.
* `IdealField`, `idealField_lipschitz`, `idealField_on_demo`, `demo_isRollout`,
  `demo_unique_rollout` — given that every `x` can be written as `(1-ρ)a + ρt` with
  `(a, t)` in the monotone graph (**cited, not formalized:** Minty's theorem /
  Rockafellar–Wets Prop. 12.60, via strong convexity of `(1-ρ)/2‖·‖² + ρf`), the field
  `v̄(x, ρ) = t - a` is well defined on all of `ℝᵈ`, `L(ρ)`-Lipschitz, fits every
  demonstration exactly, and each demonstration is its unique rollout on any
  `[ρ₀, ρ₁] ⊂ (0, 1)`.
-/

open MeasureTheory Set
open scoped RealInnerProductSpace

namespace Phaseformer

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ### (i) Jensen and straightness -/

section Jensen

variable [CompleteSpace E]

/-- The exact identity behind Eq. (20): with `c = x₁ - x₀ = ∫₀¹ ẋ`,
`∫₀¹ ‖ẋ - c‖² = ∫₀¹ ‖ẋ‖² - ‖c‖²`. -/
theorem jensen_identity {x v : ℝ → E} (hx : ∀ ρ ∈ uIcc (0 : ℝ) 1, HasDerivAt x (v ρ) ρ)
    (hv : IntervalIntegrable v volume 0 1)
    (hv2 : IntervalIntegrable (fun ρ => ‖v ρ‖ ^ 2) volume 0 1) :
    IntervalIntegrable (fun ρ => ‖v ρ - (x 1 - x 0)‖ ^ 2) volume 0 1 ∧
    ∫ ρ in (0 : ℝ)..1, ‖v ρ - (x 1 - x 0)‖ ^ 2 =
      (∫ ρ in (0 : ℝ)..1, ‖v ρ‖ ^ 2) - ‖x 1 - x 0‖ ^ 2 := by
  set c := x 1 - x 0
  have hftc : ∫ ρ in (0 : ℝ)..1, v ρ = c := intervalIntegral.integral_eq_sub_of_hasDerivAt hx hv
  have hexp : (fun ρ => ‖v ρ - c‖ ^ 2) =
      fun ρ => ‖v ρ‖ ^ 2 - 2 * ⟪c, v ρ⟫ + ‖c‖ ^ 2 := by
    funext ρ; rw [norm_sub_sq_real, real_inner_comm]
  have hin : IntervalIntegrable (fun ρ => ⟪c, v ρ⟫) volume 0 1 := by
    rw [intervalIntegrable_iff_integrableOn_Ioc_of_le zero_le_one] at hv ⊢
    exact hv.const_inner c
  have hint : ∫ ρ in (0 : ℝ)..1, ⟪c, v ρ⟫ = ⟪c, c⟫ := by
    rw [intervalIntegral.integral_of_le zero_le_one, integral_inner, ← hftc,
      intervalIntegral.integral_of_le zero_le_one]
    rw [intervalIntegrable_iff_integrableOn_Ioc_of_le zero_le_one] at hv
    exact hv
  refine ⟨?_, ?_⟩
  · rw [hexp]
    exact ((hv2.sub (hin.const_mul 2)).add intervalIntegrable_const)
  · rw [hexp, intervalIntegral.integral_add (hv2.sub (hin.const_mul 2)) intervalIntegrable_const,
      intervalIntegral.integral_sub hv2 (hin.const_mul 2), intervalIntegral.integral_const_mul,
      hint, real_inner_self_eq_norm_sq, intervalIntegral.integral_const]
    simp; ring

/-- **Eq. (20) (Jensen).**  `½‖x₁ - x₀‖² ≤ ∫₀¹ ½‖ẋ_ρ‖² dρ`. -/
theorem jensen_effort {x v : ℝ → E} (hx : ∀ ρ ∈ uIcc (0 : ℝ) 1, HasDerivAt x (v ρ) ρ)
    (hv : IntervalIntegrable v volume 0 1)
    (hv2 : IntervalIntegrable (fun ρ => ‖v ρ‖ ^ 2) volume 0 1) :
    1 / 2 * ‖x 1 - x 0‖ ^ 2 ≤ ∫ ρ in (0 : ℝ)..1, 1 / 2 * ‖v ρ‖ ^ 2 := by
  obtain ⟨-, h⟩ := jensen_identity hx hv hv2
  have hnn : 0 ≤ ∫ ρ in (0 : ℝ)..1, ‖v ρ - (x 1 - x 0)‖ ^ 2 :=
    intervalIntegral.integral_nonneg zero_le_one fun _ _ => sq_nonneg _
  rw [intervalIntegral.integral_const_mul]
  linarith

/-- **Equality in Eq. (20)** holds iff the velocity is a.e. constant (`= x₁ - x₀`) on `[0,1]`. -/
theorem jensen_eq_iff {x v : ℝ → E} (hx : ∀ ρ ∈ uIcc (0 : ℝ) 1, HasDerivAt x (v ρ) ρ)
    (hv : IntervalIntegrable v volume 0 1)
    (hv2 : IntervalIntegrable (fun ρ => ‖v ρ‖ ^ 2) volume 0 1) :
    (1 / 2 * ‖x 1 - x 0‖ ^ 2 = ∫ ρ in (0 : ℝ)..1, 1 / 2 * ‖v ρ‖ ^ 2) ↔
      ∀ᵐ ρ ∂(volume.restrict (Ioc (0 : ℝ) 1)), v ρ = x 1 - x 0 := by
  obtain ⟨hi, h⟩ := jensen_identity hx hv hv2
  rw [intervalIntegral.integral_const_mul]
  have key : (∫ ρ in (0 : ℝ)..1, ‖v ρ - (x 1 - x 0)‖ ^ 2 = 0) ↔
      ∀ᵐ ρ ∂(volume.restrict (Ioc (0 : ℝ) 1)), v ρ = x 1 - x 0 := by
    rw [intervalIntegral.integral_of_le zero_le_one,
      integral_eq_zero_iff_of_nonneg (fun _ => sq_nonneg _)
        ((intervalIntegrable_iff_integrableOn_Ioc_of_le zero_le_one).1 hi)]
    constructor
    · intro hz; filter_upwards [hz] with ρ hρ
      simpa [sub_eq_zero] using hρ
    · intro hz; filter_upwards [hz] with ρ hρ
      simp [hρ]
  rw [← key]
  constructor <;> intro h' <;> linarith

/-- A path whose velocity is a.e. the constant `x₁ - x₀` on `[0,1]` is the straight line
`x_ρ = (1-ρ)x₀ + ρx₁`. -/
theorem straight_of_const_velocity {x v : ℝ → E}
    (hx : ∀ ρ ∈ uIcc (0 : ℝ) 1, HasDerivAt x (v ρ) ρ) (hv : IntervalIntegrable v volume 0 1)
    (hc : ∀ᵐ ρ ∂(volume.restrict (Ioc (0 : ℝ) 1)), v ρ = x 1 - x 0) :
    ∀ ρ ∈ Icc (0 : ℝ) 1, x ρ = (1 - ρ) • x 0 + ρ • x 1 := by
  intro ρ hρ
  have hsub : uIcc (0 : ℝ) ρ ⊆ uIcc 0 1 := by
    rw [uIcc_of_le hρ.1, uIcc_of_le zero_le_one]; exact Icc_subset_Icc le_rfl hρ.2
  have hftc : ∫ s in (0 : ℝ)..ρ, v s = x ρ - x 0 :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt (fun s hs => hx s (hsub hs))
      (hv.mono_set hsub)
  have hconst : ∫ s in (0 : ℝ)..ρ, v s = ∫ s in (0 : ℝ)..ρ, (x 1 - x 0) := by
    apply intervalIntegral.integral_congr_ae
    rw [ae_restrict_iff' measurableSet_Ioc] at hc
    filter_upwards [hc] with s hs hs'
    rw [uIoc_of_le hρ.1] at hs'
    exact hs ⟨hs'.1, hs'.2.trans hρ.2⟩
  rw [hconst, intervalIntegral.integral_const, sub_zero] at hftc
  have e : x ρ = x 0 + ρ • (x 1 - x 0) := by rw [hftc]; abel
  rw [e, sub_smul, one_smul, smul_sub]
  abel

/-- **The order argument of (i).**  If `cost ≤ effort` for every candidate (Jensen),
`W ≤ cost` for every admissible candidate (`W = ½W₂²`), and some admissible candidate has
effort `W` (straight lines between `γ*`-paired endpoints), then every minimizer of the
effort has effort `W` and endpoint cost `W`: its endpoints are `W₂`-optimally paired and
Jensen holds with equality. -/
theorem minimizer_attains_bounds {ι : Type*} (adm : ι → Prop) (eff cost : ι → ℝ) (W : ℝ)
    (hJ : ∀ i, cost i ≤ eff i) (hW : ∀ i, adm i → W ≤ cost i)
    (hstar : ∃ i, adm i ∧ eff i = W) {i₀ : ι} (hadm : adm i₀)
    (hmin : ∀ i, adm i → eff i₀ ≤ eff i) : eff i₀ = W ∧ cost i₀ = W := by
  obtain ⟨i, hi, hie⟩ := hstar
  have h1 := hmin i hi
  have h2 := hW i₀ hadm
  have h3 := hJ i₀
  constructor <;> linarith

/-- **Theorem 4(i), straightness.**  For a finite weighted family of demonstrations, if the
total effort equals the total endpoint cost (as it does for a minimizer, by
`minimizer_attains_bounds`), every demonstration of positive weight is a straight line. -/
theorem thm4i_straight {Ω : Type*} [Fintype Ω] {w : Ω → ℝ} (hw : ∀ ω, 0 ≤ w ω)
    {x v : Ω → ℝ → E} (hx : ∀ ω, ∀ ρ ∈ uIcc (0 : ℝ) 1, HasDerivAt (x ω) (v ω ρ) ρ)
    (hv : ∀ ω, IntervalIntegrable (v ω) volume 0 1)
    (hv2 : ∀ ω, IntervalIntegrable (fun ρ => ‖v ω ρ‖ ^ 2) volume 0 1)
    (heq : ∑ ω, w ω * (∫ ρ in (0 : ℝ)..1, 1 / 2 * ‖v ω ρ‖ ^ 2) =
      ∑ ω, w ω * (1 / 2 * ‖x ω 1 - x ω 0‖ ^ 2)) :
    ∀ ω, 0 < w ω → ∀ ρ ∈ Icc (0 : ℝ) 1, x ω ρ = (1 - ρ) • x ω 0 + ρ • x ω 1 := by
  intro ω hωpos
  have hgap : ∀ ω, 0 ≤ w ω * ((∫ ρ in (0 : ℝ)..1, 1 / 2 * ‖v ω ρ‖ ^ 2) -
      1 / 2 * ‖x ω 1 - x ω 0‖ ^ 2) := fun ω =>
    mul_nonneg (hw ω) (sub_nonneg.2 (jensen_effort (hx ω) (hv ω) (hv2 ω)))
  have hsum : ∑ ω, w ω * ((∫ ρ in (0 : ℝ)..1, 1 / 2 * ‖v ω ρ‖ ^ 2) -
      1 / 2 * ‖x ω 1 - x ω 0‖ ^ 2) = 0 := by
    simp only [mul_sub, Finset.sum_sub_distrib, heq, sub_self]
  have hz := (Finset.sum_eq_zero_iff_of_nonneg fun ω _ => hgap ω).1 hsum ω (Finset.mem_univ _)
  have hz' : (∫ ρ in (0 : ℝ)..1, 1 / 2 * ‖v ω ρ‖ ^ 2) - 1 / 2 * ‖x ω 1 - x ω 0‖ ^ 2 = 0 := by
    rcases mul_eq_zero.1 hz with h | h
    · exact absurd h hωpos.ne'
    · exact h
  have hc := (jensen_eq_iff (hx ω) (hv ω) (hv2 ω)).1 (by linarith)
  exact straight_of_const_velocity (hx ω) (hv ω) hc

end Jensen

/-! ### (ii) Same objective -/

/-- **Theorem 4(ii).**  With straight demonstrations between `γ*`-paired endpoints,
`L_BC(π) = L^{γ*}_CFM(π)` for every phase-conditioned policy (Lemma 1(a) with `γ = γ*`;
Lemma 1 extends the identity to every coupling). -/
theorem thm4ii [MeasurableSpace E] (γstar : Measure (E × E)) (pol : E → ℝ → E) :
    bcLoss γstar straightPath pol = cfmLoss γstar pol :=
  lemma1a γstar pol

/-! ### (iii) No label conflict -/

section Monotone

/-- **The gradient of a convex function is monotone** (`T = ∇f` is the Brenier map). -/
theorem gradient_convex_monotone [CompleteSpace E] {f : E → ℝ} {T : E → E} (hf : ConvexOn ℝ univ f)
    (hT : ∀ x, HasGradientAt f (T x) x) (a b : E) : 0 ≤ ⟪a - b, T a - T b⟫ := by
  set g : ℝ →ᵃ[ℝ] E := AffineMap.lineMap b a
  have hg : ∀ t, g t = b + t • (a - b) := fun t => by
    simp [g, AffineMap.lineMap_apply, vsub_eq_sub, vadd_eq_add, add_comm]
  have hφc : ConvexOn ℝ univ (f ∘ g) := by simpa using hf.comp_affineMap g
  have hgd : ∀ t : ℝ, HasDerivAt (fun t : ℝ => b + t • (a - b)) (a - b) t := fun t => by
    simpa using ((hasDerivAt_id t).smul_const (a - b)).const_add b
  have hφd : ∀ t, HasDerivAt (f ∘ g) ⟪T (g t), a - b⟫ t := by
    intro t
    have h1 := ((hasGradientAt_iff_hasFDerivAt.1 (hT (g t))).comp_hasDerivAt t
      (by simpa only [← hg] using hgd t))
    rw [hg] at h1 ⊢
    simpa [InnerProductSpace.toDual_apply_apply] using h1
  have hmono := hφc.monotoneOn_deriv fun t _ => (hφd t).differentiableAt
  have h01 := hmono (mem_univ 0) (mem_univ 1) zero_le_one
  rw [(hφd 0).deriv, (hφd 1).deriv, hg, hg] at h01
  simp only [zero_smul, add_zero, one_smul, add_sub_cancel] at h01
  rw [inner_sub_right, real_inner_comm (T a) (a - b), real_inner_comm (T b) (a - b)]
  linarith

/-- **Eq. (21).**  If `⟨Δ₀, Δ₁⟩ ≥ 0` and `ρ ∈ [0,1]`,
`‖(1-ρ)Δ₀ + ρΔ₁‖² ≥ (1-ρ)²‖Δ₀‖² + ρ²‖Δ₁‖²`. -/
theorem eq21 {Δ₀ Δ₁ : E} (h : 0 ≤ ⟪Δ₀, Δ₁⟫) {ρ : ℝ} (hρ : ρ ∈ Icc (0 : ℝ) 1) :
    (1 - ρ) ^ 2 * ‖Δ₀‖ ^ 2 + ρ ^ 2 * ‖Δ₁‖ ^ 2 ≤ ‖(1 - ρ) • Δ₀ + ρ • Δ₁‖ ^ 2 := by
  rw [norm_add_sq_real, norm_smul, norm_smul, inner_smul_left, inner_smul_right,
    Real.norm_of_nonneg (by linarith [hρ.2] : (0 : ℝ) ≤ 1 - ρ), Real.norm_of_nonneg hρ.1]
  simp only [RCLike.conj_to_real]
  have : 0 ≤ 2 * ((1 - ρ) * (ρ * ⟪Δ₀, Δ₁⟫)) :=
    mul_nonneg zero_le_two (mul_nonneg (by linarith [hρ.2]) (mul_nonneg hρ.1 h))
  nlinarith

/-- A set of start–end pairs is **monotone** (e.g. the graph of `∇f`, or of `∂f`). -/
def MonotoneGraph (R : Set (E × E)) : Prop :=
  ∀ p ∈ R, ∀ q ∈ R, 0 ≤ ⟪p.1 - q.1, p.2 - q.2⟫

/-- The phase-`ρ` state of the straight demonstration from `p.1` to `p.2`. -/
def stateAt (p : E × E) (ρ : ℝ) : E := (1 - ρ) • p.1 + ρ • p.2

lemma stateAt_sub (p q : E × E) (ρ : ℝ) :
    stateAt p ρ - stateAt q ρ = (1 - ρ) • (p.1 - q.1) + ρ • (p.2 - q.2) := by
  simp only [stateAt, smul_sub]; abel

/-- **No crossing.**  For a monotone graph and `ρ < 1`, two demonstrations in the same state
at phase `ρ` have the same start; for `0 < ρ < 1` they also have the same end.  So the
phase-`ρ` state determines the demonstration, hence its velocity: `C = 0`. -/
theorem no_crossing {R : Set (E × E)} (hR : MonotoneGraph R) {p q : E × E} (hp : p ∈ R)
    (hq : q ∈ R) {ρ : ℝ} (hρ : ρ ∈ Ico (0 : ℝ) 1) (h : stateAt p ρ = stateAt q ρ) :
    p.1 = q.1 ∧ (0 < ρ → p.2 = q.2) := by
  have h21 := eq21 (hR p hp q hq) (Ico_subset_Icc_self hρ)
  rw [← stateAt_sub, h, sub_self, norm_zero] at h21
  have h1ρ : 0 < 1 - ρ := by linarith [hρ.2]
  have hA : (1 - ρ) ^ 2 * ‖p.1 - q.1‖ ^ 2 = 0 := by
    nlinarith [mul_nonneg (sq_nonneg (1 - ρ)) (sq_nonneg ‖p.1 - q.1‖),
      mul_nonneg (sq_nonneg ρ) (sq_nonneg ‖p.2 - q.2‖)]
  refine ⟨?_, fun hρ0 => ?_⟩
  · have := (mul_eq_zero.1 hA).resolve_left (by positivity)
    exact sub_eq_zero.1 (norm_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 this))
  · have hB : ρ ^ 2 * ‖p.2 - q.2‖ ^ 2 = 0 := by
      nlinarith [mul_nonneg (sq_nonneg (1 - ρ)) (sq_nonneg ‖p.1 - q.1‖),
        mul_nonneg (sq_nonneg ρ) (sq_nonneg ‖p.2 - q.2‖)]
    have := (mul_eq_zero.1 hB).resolve_left (by positivity)
    exact sub_eq_zero.1 (norm_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 this))

/-- **The Lipschitz bound.**  For `ρ ∈ (0, 1)`, the velocities `Δ = t - a` of two
demonstrations satisfy `‖Δ - Δ'‖ ≤ ‖x_ρ - x'_ρ‖ / min(ρ, 1-ρ)`, i.e.
`L(ρ) = max{1/ρ, 1/(1-ρ)}`. -/
theorem velocity_lipschitz {R : Set (E × E)} (hR : MonotoneGraph R) {p q : E × E}
    (hp : p ∈ R) (hq : q ∈ R) {ρ : ℝ} (hρ : ρ ∈ Ioo (0 : ℝ) 1) :
    ‖(p.2 - p.1) - (q.2 - q.1)‖ ≤ ‖stateAt p ρ - stateAt q ρ‖ / min ρ (1 - ρ) := by
  have hmin : 0 < min ρ (1 - ρ) := lt_min hρ.1 (by linarith [hρ.2])
  rw [le_div_iff₀ hmin]
  set Δ₀ := p.1 - q.1
  set Δ₁ := p.2 - q.2
  have hmono : 0 ≤ ⟪Δ₀, Δ₁⟫ := hR p hp q hq
  have h21 := eq21 hmono (Ioo_subset_Icc_self hρ)
  rw [← stateAt_sub] at h21
  have hd : (p.2 - p.1) - (q.2 - q.1) = Δ₁ - Δ₀ := by simp only [Δ₀, Δ₁]; abel
  rw [hd]
  -- `‖Δ₁ - Δ₀‖² ≤ ‖Δ₀‖² + ‖Δ₁‖²`
  have hsq : ‖Δ₁ - Δ₀‖ ^ 2 ≤ ‖Δ₀‖ ^ 2 + ‖Δ₁‖ ^ 2 := by
    rw [norm_sub_sq_real, real_inner_comm]; linarith
  -- `min² (‖Δ₀‖² + ‖Δ₁‖²) ≤ (1-ρ)²‖Δ₀‖² + ρ²‖Δ₁‖²`
  have hm1 : min ρ (1 - ρ) ^ 2 ≤ (1 - ρ) ^ 2 :=
    pow_le_pow_left₀ hmin.le (min_le_right _ _) 2
  have hm2 : min ρ (1 - ρ) ^ 2 ≤ ρ ^ 2 := pow_le_pow_left₀ hmin.le (min_le_left _ _) 2
  have key : (‖Δ₁ - Δ₀‖ * min ρ (1 - ρ)) ^ 2 ≤ ‖stateAt p ρ - stateAt q ρ‖ ^ 2 := by
    calc (‖Δ₁ - Δ₀‖ * min ρ (1 - ρ)) ^ 2 = min ρ (1 - ρ) ^ 2 * ‖Δ₁ - Δ₀‖ ^ 2 := by ring
      _ ≤ min ρ (1 - ρ) ^ 2 * (‖Δ₀‖ ^ 2 + ‖Δ₁‖ ^ 2) :=
          mul_le_mul_of_nonneg_left hsq (sq_nonneg _)
      _ ≤ (1 - ρ) ^ 2 * ‖Δ₀‖ ^ 2 + ρ ^ 2 * ‖Δ₁‖ ^ 2 := by
          nlinarith [sq_nonneg ‖Δ₀‖, sq_nonneg ‖Δ₁‖]
      _ ≤ _ := h21
  exact (pow_le_pow_iff_left₀ (mul_nonneg (norm_nonneg _) hmin.le) (norm_nonneg _)
    two_ne_zero).1 key

/-- `1/min(ρ, 1-ρ) = max{1/ρ, 1/(1-ρ)}` on `(0, 1)`. -/
theorem inv_min_eq_max {ρ : ℝ} (hρ : ρ ∈ Ioo (0 : ℝ) 1) :
    1 / min ρ (1 - ρ) = max (1 / ρ) (1 / (1 - ρ)) := by
  have h1 : 0 < 1 - ρ := by linarith [hρ.2]
  rcases le_total ρ (1 - ρ) with h | h
  · rw [min_eq_left h, max_eq_left (one_div_le_one_div_of_le hρ.1 h)]
  · rw [min_eq_right h, max_eq_right (one_div_le_one_div_of_le h1 h)]

/-- The ideal field on all of `ℝᵈ`.  `Rep ρ x` is a representation `x = (1-ρ)a + ρt` with
`(a, t) ∈ R`; its existence for every `x` is the cited Minty / Rockafellar–Wets step.
`v̄(x, ρ) = t - a`. -/
structure IdealField (R : Set (E × E)) where
  rep : ℝ → E → E × E
  rep_mem : ∀ ρ ∈ Ioo (0 : ℝ) 1, ∀ x, rep ρ x ∈ R
  rep_state : ∀ ρ ∈ Ioo (0 : ℝ) 1, ∀ x, stateAt (rep ρ x) ρ = x

/-- `v̄(x, ρ) = t - a` for the representation `x = (1-ρ)a + ρt`. -/
def IdealField.vbar {R : Set (E × E)} (F : IdealField R) (x : E) (ρ : ℝ) : E :=
  (F.rep ρ x).2 - (F.rep ρ x).1

/-- The representation is unique (no crossing), so `v̄` fits every demonstration exactly:
`v̄(x_ρ, ρ) = x₁ - x₀`. -/
theorem idealField_on_demo {R : Set (E × E)} (hR : MonotoneGraph R) (F : IdealField R)
    {p : E × E} (hp : p ∈ R) {ρ : ℝ} (hρ : ρ ∈ Ioo (0 : ℝ) 1) :
    F.vbar (stateAt p ρ) ρ = p.2 - p.1 := by
  have h := no_crossing hR (F.rep_mem ρ hρ _) hp (Ioo_subset_Ico_self hρ)
    (F.rep_state ρ hρ (stateAt p ρ))
  unfold IdealField.vbar
  rw [h.1, h.2 hρ.1]

/-- **`v̄` is `L(ρ)`-Lipschitz on all of `ℝᵈ`**, `L(ρ) = 1/min(ρ, 1-ρ)`. -/
theorem idealField_lipschitz {R : Set (E × E)} (hR : MonotoneGraph R) (F : IdealField R)
    {ρ : ℝ} (hρ : ρ ∈ Ioo (0 : ℝ) 1) (x y : E) :
    ‖F.vbar x ρ - F.vbar y ρ‖ ≤ (1 / min ρ (1 - ρ)) * ‖x - y‖ := by
  have h := velocity_lipschitz hR (F.rep_mem ρ hρ x) (F.rep_mem ρ hρ y) hρ
  rw [F.rep_state ρ hρ x, F.rep_state ρ hρ y] at h
  unfold IdealField.vbar
  rw [one_div_mul_eq_div]; exact h

/-- Each demonstration is a rollout of `v̄` on `[ρ₀, ρ₁] ⊂ (0, 1)`. -/
theorem demo_isRollout {R : Set (E × E)} (hR : MonotoneGraph R) (F : IdealField R)
    {p : E × E} (hp : p ∈ R) {ρ₀ ρ₁ : ℝ} (h0 : 0 < ρ₀) (h1 : ρ₁ < 1) :
    IsRollout F.vbar (stateAt p) ρ₀ ρ₁ := by
  have hd : ∀ ρ, HasDerivAt (stateAt p) (p.2 - p.1) ρ := hasDerivAt_straightPath p
  refine ⟨fun ρ _ => (hd ρ).continuousAt.continuousWithinAt, fun ρ hρ => ?_⟩
  rw [idealField_on_demo hR F hp ⟨h0.trans_le hρ.1, hρ.2.trans h1⟩]
  exact (hd ρ).hasDerivWithinAt

/-- **Theorem 4(iii), uniqueness.**  On `[ρ₀, ρ₁] ⊂ (0, 1)`, `L(ρ)` is bounded, so by
Grönwall the demonstration is the only rollout of `v̄` from its phase-`ρ₀` state. -/
theorem demo_unique_rollout {R : Set (E × E)} (hR : MonotoneGraph R) (F : IdealField R)
    {p : E × E} (hp : p ∈ R) {ρ₀ ρ₁ : ℝ} (h0 : 0 < ρ₀) (h1 : ρ₁ < 1) {z : ℝ → E}
    (hz : IsRollout F.vbar z ρ₀ ρ₁) (hz0 : z ρ₀ = stateAt p ρ₀) :
    EqOn z (stateAt p) (Icc ρ₀ ρ₁) := by
  set Lc := max (1 / ρ₀) (1 / (1 - ρ₁))
  have hLip : LipschitzInState F.vbar (fun _ => Lc) ρ₀ ρ₁ := by
    intro ρ hρ x y
    have hρ' : ρ ∈ Ioo (0 : ℝ) 1 := ⟨h0.trans_le hρ.1, hρ.2.trans_lt h1⟩
    refine (idealField_lipschitz hR F hρ' x y).trans (mul_le_mul_of_nonneg_right ?_ (norm_nonneg _))
    rw [inv_min_eq_max hρ']
    have hpos1 : 0 < 1 - ρ₁ := by linarith
    exact max_le_max (one_div_le_one_div_of_le h0 hρ.1)
      (one_div_le_one_div_of_le hpos1 (by linarith [hρ.2]))
  exact rollouts_eq_of_eq continuous_const hLip hz (demo_isRollout hR F hp h0 h1) hz0

end Monotone

end Phaseformer
