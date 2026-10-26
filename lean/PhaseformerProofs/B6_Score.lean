import Mathlib
import PhaseformerProofs.B1_Setting

/-!
# App. B.6 — Proof of Theorem 5, and Properties of the Score Penalty

**Pointwise reduction** (`total_eq_sum_fibers`, `min_of_pointwise_min`,
`pointwise_min_of_min`): both losses average over the demonstrations at each policy input,
and `s` is a function of that input, so the minimizer over all phase-conditioned policies is
obtained input by input (finite dataset form).

**The pointwise problem.**  At an input with demonstrated velocities `u ∼ ν` (any
probability law on `E` with finite second moment) and score `s`, the policy value minimizes
`J(v) = E‖v - u‖² + λ E|(u - v)·s|`.

* `J_split` — with `ŝ = s/‖s‖`, `a = v·ŝ`, `b = u·ŝ`, `b̄ = E b`:
  `J(v) = ‖P⊥(v - ū)‖² + [(a - b̄)² + λ‖s‖ E|a - b|] + E‖u - ū‖²`.
* `thm5_s_zero` — where `s = 0`, `π_λ = v̄`.
* `thm5_i` — the component of `π_λ` orthogonal to `s` equals that of `v̄`.
* `thm5_along` — `a_λ = π_λ·ŝ` minimizes `g(a) = (a - b̄)² + λ‖s‖ E|a - b|`.
* `thm5_ii_bound` — `|a_λ - b̄| ≤ λ‖s‖/2`.
* `thm5_ii_between` — `a_λ` lies between `b̄` and every median `m` of `b`.
* `thm5_ii_limit` — `a_λ → m` (the median closest to `b̄`) as `λ‖s‖ → ∞`.
* `thm5_iii_iff` — `a_λ = b̄` exactly when `b̄` is a median of `b`;
  `median_of_const`, `median_of_symmetric` — this holds when `b` is constant or symmetric
  about `b̄`; `thm5_iii` — then `π_λ = v̄`.
* `g_unique_min` — `g` has at most one minimizer (strict convexity).

Medians are defined as minimizers of `a ↦ E|a - b|`.  The 1-D analysis uses only that this
function is convex and 1-Lipschitz (`absDev_convex`, `absDev_lipschitz`).

**Worked examples** (`ex_symmetric_tie`, `ex_shift_against`, `ex_bias`), the stationarity
condition (`stationarity`) and the realizability score computation (`realizability_score`).
-/

open MeasureTheory Set Filter Topology
open scoped RealInnerProductSpace

namespace Phaseformer

/-! ### Pointwise reduction (finite dataset) -/

section Pointwise

variable {Ω β V : Type*} [Fintype Ω] [DecidableEq β]

/-- The objective restricted to the demonstrations at input `z`. -/
def fiberObj (w : Ω → ℝ) (Z : Ω → β) (ℓ : V → Ω → ℝ) (z : β) (v : V) : ℝ :=
  ∑ ω ∈ Finset.univ.filter (fun ω => Z ω = z), w ω * ℓ v ω

lemma total_eq_sum_fibers (w : Ω → ℝ) (Z : Ω → β) (ℓ : V → Ω → ℝ) (pol : β → V) :
    ∑ ω, w ω * ℓ (pol (Z ω)) ω = ∑ z ∈ Finset.univ.image Z, fiberObj w Z ℓ z (pol z) := by
  rw [← Finset.sum_fiberwise_of_maps_to (s := Finset.univ) (g := Z)
    (fun ω _ => Finset.mem_image_of_mem Z (Finset.mem_univ ω))]
  refine Finset.sum_congr rfl fun z _ => Finset.sum_congr rfl fun ω hω => ?_
  rw [(Finset.mem_filter.1 hω).2]

/-- If `pol z` minimizes the objective at every input `z`, `pol` minimizes the total loss. -/
theorem min_of_pointwise_min (w : Ω → ℝ) (Z : Ω → β) (ℓ : V → Ω → ℝ) (pol : β → V)
    (h : ∀ z v, fiberObj w Z ℓ z (pol z) ≤ fiberObj w Z ℓ z v) (pol' : β → V) :
    ∑ ω, w ω * ℓ (pol (Z ω)) ω ≤ ∑ ω, w ω * ℓ (pol' (Z ω)) ω := by
  rw [total_eq_sum_fibers, total_eq_sum_fibers]
  exact Finset.sum_le_sum fun z _ => h z (pol' z)

/-- Conversely, a minimizer of the total loss minimizes the objective at every input that
some demonstration has. -/
theorem pointwise_min_of_min [DecidableEq V] (w : Ω → ℝ) (Z : Ω → β) (ℓ : V → Ω → ℝ)
    (pol : β → V) (h : ∀ pol' : β → V, ∑ ω, w ω * ℓ (pol (Z ω)) ω ≤ ∑ ω, w ω * ℓ (pol' (Z ω)) ω)
    {z : β} (hz : z ∈ Finset.univ.image Z) (v : V) :
    fiberObj w Z ℓ z (pol z) ≤ fiberObj w Z ℓ z v := by
  have h' := h (Function.update pol z v)
  rw [total_eq_sum_fibers, total_eq_sum_fibers, ← Finset.add_sum_erase _ _ hz,
    ← Finset.add_sum_erase _ _ hz] at h'
  have : ∑ x ∈ (Finset.univ.image Z).erase z, fiberObj w Z ℓ x (Function.update pol z v x) =
      ∑ x ∈ (Finset.univ.image Z).erase z, fiberObj w Z ℓ x (pol x) :=
    Finset.sum_congr rfl fun x hx => by rw [Function.update_of_ne (Finset.ne_of_mem_erase hx)]
  rw [this, Function.update_self] at h'
  linarith

end Pointwise

/-! ### The one-dimensional problem along the score -/

section OneDim

/-- `g(a) = (a - b̄)² + c · h(a)` with `c = λ‖s‖` and `h(a) = E|a - b|`. -/
def gObj (bbar c : ℝ) (h : ℝ → ℝ) (a : ℝ) : ℝ := (a - bbar) ^ 2 + c * h a

variable {bbar c : ℝ} {h : ℝ → ℝ}

/-- **`|a_λ - b̄| ≤ λ‖s‖/2`.**  Proof without subgradients: compare with `a_t = a + t(b̄ - a)`,
use that `h` is 1-Lipschitz, and let `t → 0`. -/
theorem thm5_ii_bound (hc : 0 ≤ c) (hlip : ∀ x y, |h x - h y| ≤ |x - y|) {a : ℝ}
    (ha : ∀ x, gObj bbar c h a ≤ gObj bbar c h x) : |a - bbar| ≤ c / 2 := by
  set D := |a - bbar|
  by_contra hcon
  push Not at hcon
  have hD : 0 < D := by linarith
  set t := min 1 ((2 * D - c) / (2 * D))
  have ht0 : 0 < t := lt_min one_pos (div_pos (by linarith) (by linarith))
  have ht1 : t ≤ 1 := min_le_left _ _
  have htε : t ≤ (2 * D - c) / (2 * D) := min_le_right _ _
  have hcmp := ha (a + t * (bbar - a))
  unfold gObj at hcmp
  have hq : (a + t * (bbar - a) - bbar) ^ 2 = (1 - t) ^ 2 * D ^ 2 := by
    rw [sq_abs]; ring
  have hl : h (a + t * (bbar - a)) ≤ h a + t * D := by
    have := hlip (a + t * (bbar - a)) a
    have e : |a + t * (bbar - a) - a| = t * D := by
      rw [show a + t * (bbar - a) - a = t * (bbar - a) by ring, abs_mul, abs_of_pos ht0,
        abs_sub_comm]
    rw [e] at this
    linarith [le_abs_self (h (a + t * (bbar - a)) - h a)]
  rw [hq, show (a - bbar) ^ 2 = D ^ 2 from (sq_abs _).symm] at hcmp
  have h1 : D ^ 2 ≤ (1 - t) ^ 2 * D ^ 2 + c * t * D := by nlinarith
  -- `(2 - t) D ≤ c`
  have h2 : (2 - t) * D ≤ c := by
    have : t * D * ((2 - t) * D - c) ≤ 0 := by nlinarith
    have hpos : 0 < t * D := mul_pos ht0 hD
    nlinarith
  have h3 : t * (2 * D) ≤ 2 * D - c := by rwa [le_div_iff₀ (by linarith)] at htε
  nlinarith

/-- **`a_λ` lies between `b̄` and every median.**  If `m` minimizes `h` (a median), then
`min(b̄, m) ≤ a_λ ≤ max(b̄, m)`: beyond both, moving toward them lowers the quadratic term
strictly and does not raise `h`. -/
theorem thm5_ii_between (hc : 0 ≤ c) (hconv : ConvexOn ℝ univ h) {m : ℝ}
    (hm : ∀ x, h m ≤ h x) {a : ℝ} (ha : ∀ x, gObj bbar c h a ≤ gObj bbar c h x) :
    min bbar m ≤ a ∧ a ≤ max bbar m := by
  -- on a segment from the minimizer `m`, `h` is at most its value at the far end
  have hseg : ∀ x y, (m ≤ x ∧ x ≤ y ∨ y ≤ x ∧ x ≤ m) → h x ≤ h y := by
    intro x y hxy
    rcases eq_or_ne y m with hym | hym
    · have : x = m := by rcases hxy with h' | h' <;> linarith [h'.1, h'.2]
      rw [this, hym]
    · set s := (x - m) / (y - m)
      have hs0 : 0 ≤ s := by
        rcases hxy with h' | h'
        · exact div_nonneg (by linarith [h'.1]) (by linarith [h'.1, h'.2, lt_of_le_of_ne (h'.1.trans h'.2) (Ne.symm hym)])
        · exact div_nonneg_of_nonpos (by linarith [h'.2]) (by linarith [lt_of_le_of_ne (h'.1.trans h'.2) hym])
      have hs1 : s ≤ 1 := by
        rcases hxy with h' | h'
        · exact div_le_one_of_le₀ (by linarith [h'.2]) (by linarith [h'.1, h'.2])
        · rw [div_le_one_of_neg (by linarith [lt_of_le_of_ne (h'.1.trans h'.2) hym])]; linarith [h'.1]
      have hx : x = (1 - s) • m + s • y := by
        simp only [smul_eq_mul, s]; field_simp [sub_ne_zero.2 hym]; ring
      have key : h ((1 - s) • m + s • y) ≤ (1 - s) • h m + s • h y :=
        hconv.2 (mem_univ m) (mem_univ y) (by linarith) hs0 (by ring)
      rw [← hx] at key
      simp only [smul_eq_mul] at key
      nlinarith [hm y]
  constructor
  · by_contra hlt
    push Not at hlt
    set p := min bbar m
    have hq : (p - bbar) ^ 2 < (a - bbar) ^ 2 := by
      have : a < p := hlt
      have : p ≤ bbar := min_le_left _ _
      nlinarith
    have hh : h p ≤ h a := hseg p a (Or.inr ⟨hlt.le, min_le_right _ _⟩)
    have := ha p
    unfold gObj at this
    nlinarith [mul_le_mul_of_nonneg_left hh hc]
  · by_contra hlt
    push Not at hlt
    set p := max bbar m
    have hq : (p - bbar) ^ 2 < (a - bbar) ^ 2 := by
      have : p < a := hlt
      have : bbar ≤ p := le_max_left _ _
      nlinarith
    have hh : h p ≤ h a := hseg p a (Or.inl ⟨le_max_right _ _, hlt.le⟩)
    have := ha p
    unfold gObj at this
    nlinarith [mul_le_mul_of_nonneg_left hh hc]

/-- **`g` has at most one minimizer** (the quadratic term is strictly convex). -/
theorem g_unique_min (hc : 0 ≤ c) (hconv : ConvexOn ℝ univ h) {a₁ a₂ : ℝ}
    (h₁ : ∀ x, gObj bbar c h a₁ ≤ gObj bbar c h x) (h₂ : ∀ x, gObj bbar c h a₂ ≤ gObj bbar c h x) :
    a₁ = a₂ := by
  have hmid := hconv.2 (mem_univ a₁) (mem_univ a₂) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num)
  simp only [smul_eq_mul] at hmid
  have e := h₁ (1 / 2 * a₁ + 1 / 2 * a₂)
  have e2 := h₂ a₁
  unfold gObj at e e2
  have := mul_le_mul_of_nonneg_left hmid hc
  nlinarith [sq_nonneg (a₁ - a₂)]

/-- **`a_λ = b̄` iff `b̄` is a median** (for `λ‖s‖ > 0`). -/
theorem thm5_iii_iff (hc : 0 < c) (hconv : ConvexOn ℝ univ h) {a : ℝ}
    (ha : ∀ x, gObj bbar c h a ≤ gObj bbar c h x) : a = bbar ↔ ∀ x, h bbar ≤ h x := by
  constructor
  · rintro rfl x
    by_contra hlt
    push Not at hlt
    -- move from `b̄` toward `x` by a small `t`
    have hxne : x ≠ a := fun hx => by rw [hx] at hlt; exact lt_irrefl _ hlt
    have hd : 0 < (x - a) ^ 2 := by positivity
    set δh := h a - h x
    have hδ : 0 < δh := by simp only [δh]; linarith
    set t := min 1 (c * δh / (2 * (x - a) ^ 2))
    have ht0 : 0 < t := lt_min one_pos (by positivity)
    have ht1 : t ≤ 1 := min_le_left _ _
    have htε : t ≤ c * δh / (2 * (x - a) ^ 2) := min_le_right _ _
    have hcv := hconv.2 (mem_univ a) (mem_univ x) (by linarith : 0 ≤ 1 - t) ht0.le (by ring)
    simp only [smul_eq_mul] at hcv
    have e := ha ((1 - t) * a + t * x)
    unfold gObj at e
    have hq : ((1 - t) * a + t * x - a) ^ 2 = t ^ 2 * (x - a) ^ 2 := by ring
    rw [hq, sub_self, zero_pow two_ne_zero, zero_add] at e
    have h3 : t * (2 * (x - a) ^ 2) ≤ c * δh := by rwa [le_div_iff₀ (by positivity)] at htε
    have := mul_le_mul_of_nonneg_left hcv hc.le
    nlinarith [mul_pos ht0 hδ, mul_pos hc (mul_pos ht0 hδ)]
  · intro hmed
    refine g_unique_min hc.le hconv ha fun x => ?_
    unfold gObj
    nlinarith [hmed x, sq_nonneg (x - bbar), mul_le_mul_of_nonneg_left (hmed x) hc.le]

/-- **`a_λ → m` as `λ‖s‖ → ∞`**, where `m` is the median closest to `b̄`.  Here `a c` is a
minimizer of `g` for each `c > 0`; `h` is continuous and convex. -/
theorem thm5_ii_limit (hconv : ConvexOn ℝ univ h) (hcont : Continuous h) {m : ℝ}
    (hm : ∀ x, h m ≤ h x) (hclosest : ∀ m', (∀ x, h m' ≤ h x) → |m - bbar| ≤ |m' - bbar|)
    (a : ℝ → ℝ) (ha : ∀ c, 0 < c → ∀ x, gObj bbar c h (a c) ≤ gObj bbar c h x) :
    Tendsto a atTop (𝓝 m) := by
  set J := Icc (min bbar m) (max bbar m)
  have haJ : ∀ c, 0 < c → a c ∈ J := fun c hc => thm5_ii_between hc.le hconv hm (ha c hc)
  -- the only median in `J` is `m`
  have honly : ∀ y ∈ J, (∀ x, h y ≤ h x) → y = m := by
    intro y hy hy'
    have hy : y ∈ Icc (min bbar m) (max bbar m) := hy
    have := hclosest y hy'
    rcases le_total bbar m with hbm | hbm
    · rw [min_eq_left hbm, max_eq_right hbm] at hy
      rw [abs_of_nonneg (by linarith), abs_of_nonneg (by linarith [hy.1])] at this
      linarith [hy.2]
    · rw [min_eq_right hbm, max_eq_left hbm] at hy
      rw [abs_of_nonpos (by linarith), abs_of_nonpos (by linarith [hy.2])] at this
      linarith [hy.1]
  rw [Metric.tendsto_atTop]
  intro ε hε
  set Kε := J ∩ {y | ε ≤ |y - m|}
  have hKc : IsCompact Kε :=
    isCompact_Icc.inter_right (isClosed_le continuous_const (continuous_id.sub continuous_const).abs)
  rcases Kε.eq_empty_or_nonempty with hK | hK
  · refine ⟨1, fun c hc => ?_⟩
    by_contra hge
    push Not at hge
    have : a c ∈ Kε := ⟨haJ c (by linarith), by
      show ε ≤ |a c - m|; rw [← Real.dist_eq]; exact hge⟩
    rw [hK] at this; exact this
  · obtain ⟨y₀, hy₀, hmin⟩ := hKc.exists_isMinOn hK hcont.continuousOn
    have hη : h m < h y₀ := by
      rcases lt_or_eq_of_le (hm y₀) with h' | h'
      · exact h'
      · exfalso
        have hy0m : y₀ = m := honly y₀ hy₀.1 (fun x => by rw [← h']; exact hm x)
        have h2 : ε ≤ |y₀ - m| := hy₀.2
        rw [hy0m, sub_self, abs_zero] at h2; linarith
    set η := h y₀ - h m
    have hηpos : 0 < η := by simp only [η]; linarith
    set N := max 1 ((m - bbar) ^ 2 / η + 1)
    refine ⟨N, fun c hc => ?_⟩
    have hc1 : 1 ≤ c := (le_max_left _ _).trans hc
    have hc0 : 0 < c := by linarith
    have hcN : (m - bbar) ^ 2 / η + 1 ≤ c := (le_max_right _ _).trans hc
    by_contra hge
    push Not at hge
    have hK' : a c ∈ Kε := ⟨haJ c hc0, by
      show ε ≤ |a c - m|; rw [← Real.dist_eq]; exact hge⟩
    have h1 : h y₀ ≤ h (a c) := hmin hK'
    have e := ha c hc0 m
    unfold gObj at e
    -- `c (h(a_c) - h(m)) ≤ (m - b̄)²`, but `h(a_c) - h(m) ≥ η` and `c > (m - b̄)²/η`
    have h2 : c * η ≤ (m - bbar) ^ 2 := by
      nlinarith [sq_nonneg (a c - bbar), mul_le_mul_of_nonneg_left h1 hc0.le]
    have h3 : (m - bbar) ^ 2 < c * η := by
      have := (div_lt_iff₀ hηpos).1 (by linarith : (m - bbar) ^ 2 / η < c)
      linarith
    linarith

end OneDim

/-! ### Mean absolute deviation `h(a) = E|a - b|` -/

section AbsDev

variable {α : Type*} [MeasurableSpace α] (ν : Measure α) [IsProbabilityMeasure ν]

/-- `h(a) = E|a - b|` for a random variable `b`. -/
noncomputable def absDev (b : α → ℝ) (a : ℝ) : ℝ := ∫ x, |a - b x| ∂ν

variable {ν}

lemma integrable_absDev {b : α → ℝ} (hb : Integrable b ν) (a : ℝ) :
    Integrable (fun x => |a - b x|) ν :=
  ((integrable_const a).sub hb).abs

/-- `E|a - b|` is convex in `a`. -/
theorem absDev_convex {b : α → ℝ} (hb : Integrable b ν) : ConvexOn ℝ univ (absDev ν b) := by
  refine ⟨convex_univ, fun a₁ _ a₂ _ t₁ t₂ ht₁ ht₂ ht => ?_⟩
  simp only [smul_eq_mul, absDev]
  rw [← integral_const_mul, ← integral_const_mul, ← integral_add
    ((integrable_absDev hb a₁).const_mul t₁) ((integrable_absDev hb a₂).const_mul t₂)]
  refine integral_mono (integrable_absDev hb _)
    (((integrable_absDev hb a₁).const_mul t₁).add ((integrable_absDev hb a₂).const_mul t₂))
    fun x => ?_
  calc |t₁ * a₁ + t₂ * a₂ - b x| = |t₁ * (a₁ - b x) + t₂ * (a₂ - b x)| := by
        congr 1; linear_combination (b x) * ht
    _ ≤ |t₁ * (a₁ - b x)| + |t₂ * (a₂ - b x)| := abs_add_le _ _
    _ = t₁ * |a₁ - b x| + t₂ * |a₂ - b x| := by
        rw [abs_mul, abs_mul, abs_of_nonneg ht₁, abs_of_nonneg ht₂]

/-- `E|a - b|` is 1-Lipschitz in `a`. -/
theorem absDev_lipschitz {b : α → ℝ} (hb : Integrable b ν) (a a' : ℝ) :
    |absDev ν b a - absDev ν b a'| ≤ |a - a'| := by
  unfold absDev
  rw [← integral_sub (integrable_absDev hb a) (integrable_absDev hb a')]
  calc |∫ x, (|a - b x| - |a' - b x|) ∂ν| ≤ ∫ x, |(|a - b x| - |a' - b x|)| ∂ν :=
        abs_integral_le_integral_abs
    _ ≤ ∫ _x, |a - a'| ∂ν := by
        refine integral_mono ((integrable_absDev hb a).sub (integrable_absDev hb a')).abs
          (integrable_const _) fun x => ?_
        have := abs_abs_sub_abs_le_abs_sub (a - b x) (a' - b x)
        simpa using this
    _ = |a - a'| := by simp

theorem absDev_continuous {b : α → ℝ} (hb : Integrable b ν) : Continuous (absDev ν b) :=
  (LipschitzWith.of_dist_le_mul (K := 1) fun x y => by
    rw [NNReal.coe_one, one_mul, Real.dist_eq, Real.dist_eq]
    exact absDev_lipschitz hb x y).continuous

omit [IsProbabilityMeasure ν] in
/-- If `b` is a.s. constant `= b̄`, then `b̄` is a median. -/
theorem median_of_const {b : α → ℝ} {bbar : ℝ} (hb : ∀ᵐ x ∂ν, b x = bbar) (a : ℝ) :
    absDev ν b bbar ≤ absDev ν b a := by
  have : absDev ν b bbar = 0 := by
    unfold absDev
    rw [integral_eq_zero_of_ae]
    filter_upwards [hb] with x hx
    simp [hx]
  rw [this]
  exact integral_nonneg fun _ => abs_nonneg _

/-- If the law of `b` is symmetric about `b̄` (`b` and `2b̄ - b` have the same law), then `b̄`
is a median. -/
theorem median_of_symmetric {b : α → ℝ} (hb : Integrable b ν) (hbm : Measurable b) {bbar : ℝ}
    (hsym : ν.map b = ν.map (fun x => 2 * bbar - b x)) (a : ℝ) :
    absDev ν b bbar ≤ absDev ν b a := by
  have hrefl : absDev ν b a = absDev ν b (2 * bbar - a) := by
    unfold absDev
    have e1 : ∫ x, |a - b x| ∂ν = ∫ y, |a - y| ∂(ν.map b) :=
      (integral_map hbm.aemeasurable
        (continuous_const.sub continuous_id).abs.aestronglyMeasurable).symm
    have e2 : ∫ y, |a - y| ∂(ν.map (fun x => 2 * bbar - b x)) =
        ∫ x, |a - (2 * bbar - b x)| ∂ν :=
      integral_map (hbm.const_sub _).aemeasurable
        (continuous_const.sub continuous_id).abs.aestronglyMeasurable
    rw [e1, hsym, e2]
    congr 1; funext x
    rw [show a - (2 * bbar - b x) = -(2 * bbar - a - b x) by ring, abs_neg]
  have hconv := absDev_convex hb
  have := hconv.2 (mem_univ a) (mem_univ (2 * bbar - a)) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num)
  simp only [smul_eq_mul] at this
  rw [show 1 / 2 * a + 1 / 2 * (2 * bbar - a) = bbar by ring] at this
  linarith

end AbsDev

/-! ### The vector problem and Theorem 5 -/

section Vector

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  [MeasurableSpace E]
variable (ν : Measure E) [IsProbabilityMeasure ν]

/-- Mean demonstrated velocity `ū = v̄(x, ρ)` at the input. -/
noncomputable def ubar : E := ∫ u, u ∂ν

/-- Squared loss at the input, `E‖v - u‖²`. -/
noncomputable def sqLoss (v : E) : ℝ := ∫ u, ‖v - u‖ ^ 2 ∂ν

/-- Score penalty at the input, `E|(u - v)·s|`. -/
noncomputable def penalty (s v : E) : ℝ := ∫ u, |⟪u - v, s⟫| ∂ν

/-- The pointwise objective `J(v) = E‖v - u‖² + λ E|(u - v)·s|`. -/
noncomputable def Jobj (lam : ℝ) (s v : E) : ℝ := sqLoss ν v + lam * penalty ν s v

variable {ν}

omit [InnerProductSpace ℝ E] [CompleteSpace E] in
lemma integrable_id_of_memLp (h2 : MemLp id 2 ν) : Integrable (fun u : E => u) ν :=
  h2.integrable one_le_two

/-- `E‖v - u‖² = ‖v - ū‖² + E‖ū - u‖²`. -/
theorem sqLoss_eq (h2 : MemLp id 2 ν) (v : E) :
    sqLoss ν v = ‖v - ubar ν‖ ^ 2 + sqLoss ν (ubar ν) := by
  have hi := integrable_id_of_memLp h2
  have hsq : ∀ w : E, Integrable (fun u => ‖w - u‖ ^ 2) ν := by
    intro w
    have : MemLp (fun u : E => w - u) 2 ν := (memLp_const w).sub h2
    exact (memLp_two_iff_integrable_sq_norm this.aestronglyMeasurable).1 this
  have hexp : (fun u => ‖v - u‖ ^ 2) = fun u =>
      ‖v - ubar ν‖ ^ 2 + 2 * ⟪v - ubar ν, ubar ν - u⟫ + ‖ubar ν - u‖ ^ 2 := by
    funext u
    rw [show v - u = (v - ubar ν) + (ubar ν - u) by abel, norm_add_sq_real]
  have hin : Integrable (fun u => ⟪v - ubar ν, ubar ν - u⟫) ν :=
    ((integrable_const _).sub hi).const_inner _
  have hf : Integrable (fun u => ubar ν - u) ν := (integrable_const _).sub hi
  have hzero : ∫ u, ⟪v - ubar ν, ubar ν - u⟫ ∂ν = 0 := by
    rw [integral_inner hf, integral_sub (integrable_const _) hi]
    simp [ubar]
  have i1 : Integrable (fun u => ‖v - ubar ν‖ ^ 2 + 2 * ⟪v - ubar ν, ubar ν - u⟫) ν :=
    (integrable_const _).add (hin.const_mul 2)
  unfold sqLoss
  rw [hexp, integral_add i1 (hsq _),
    integral_add (integrable_const _) (hin.const_mul 2), integral_const_mul, hzero]
  simp

omit [CompleteSpace E] [MeasurableSpace E] in
/-- Pythagoras along a unit vector `e`: `‖w‖² = (w·e)² + ‖w - (w·e)e‖²`. -/
lemma norm_sq_split {e : E} (he : ‖e‖ = 1) (w : E) :
    ‖w‖ ^ 2 = ⟪w, e⟫ ^ 2 + ‖w - ⟪w, e⟫ • e‖ ^ 2 := by
  rw [norm_sub_sq_real, norm_smul, he, inner_smul_right, Real.norm_eq_abs, mul_one, sq_abs]
  ring

omit [CompleteSpace E] [IsProbabilityMeasure ν] in
/-- The penalty through the unit score `ŝ = s/‖s‖`: `|(u - v)·s| = ‖s‖ |v·ŝ - u·ŝ|`. -/
lemma penalty_eq {s : E} (hs : s ≠ 0) (v : E) :
    penalty ν s v = ‖s‖ * absDev ν (fun u => ⟪u, ‖s‖⁻¹ • s⟫) ⟪v, ‖s‖⁻¹ • s⟫ := by
  have hn : ‖s‖ ≠ 0 := norm_ne_zero_iff.2 hs
  have key : ∀ u, |⟪u - v, s⟫| = ‖s‖ * |⟪v, ‖s‖⁻¹ • s⟫ - ⟪u, ‖s‖⁻¹ • s⟫| := by
    intro u
    rw [inner_smul_right, inner_smul_right, ← mul_sub, abs_mul,
      abs_of_pos (inv_pos.2 (norm_pos_iff.2 hs)), ← mul_assoc, mul_inv_cancel₀ hn, one_mul]
    simp only [inner_sub_left]
    rw [abs_sub_comm]
  unfold penalty absDev
  rw [← integral_const_mul]
  congr 1; funext u
  exact key u

/-- **The split of `J`.**  With `ŝ = s/‖s‖`, `a = v·ŝ`, `b̄ = ū·ŝ`, `P⊥ w = w - (w·ŝ)ŝ`:
`J(v) = ‖P⊥(v - ū)‖² + g(a) + E‖ū - u‖²`, `g(a) = (a - b̄)² + λ‖s‖ E|a - b|`. -/
theorem J_split (h2 : MemLp id 2 ν) {s : E} (hs : s ≠ 0) (lam : ℝ) (v : E) :
    Jobj ν lam s v =
      ‖(v - ubar ν) - ⟪v - ubar ν, ‖s‖⁻¹ • s⟫ • (‖s‖⁻¹ • s)‖ ^ 2 +
      gObj ⟪ubar ν, ‖s‖⁻¹ • s⟫ (lam * ‖s‖) (absDev ν fun u => ⟪u, ‖s‖⁻¹ • s⟫)
        ⟪v, ‖s‖⁻¹ • s⟫ + sqLoss ν (ubar ν) := by
  have he : ‖‖s‖⁻¹ • s‖ = 1 := by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.2 hs)]
  unfold Jobj gObj
  rw [sqLoss_eq h2, penalty_eq hs, norm_sq_split he (v - ubar ν), inner_sub_left]
  ring

/-- **Where `s = 0`, `π_λ = v̄`**: `J` is the squared loss alone, uniquely minimized by `ū`. -/
theorem thm5_s_zero (h2 : MemLp id 2 ν) (lam : ℝ) {v : E}
    (hv : ∀ w, Jobj ν lam 0 v ≤ Jobj ν lam 0 w) : v = ubar ν := by
  have h := hv (ubar ν)
  unfold Jobj penalty at h
  simp only [inner_zero_right, abs_zero, integral_zero, mul_zero, add_zero] at h
  rw [sqLoss_eq h2 v] at h
  have : ‖v - ubar ν‖ ^ 2 ≤ 0 := by nlinarith
  exact sub_eq_zero.1 (norm_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1
    (le_antisymm this (sq_nonneg _))))

/-- **Theorem 5(i).**  The component of `π_λ` orthogonal to `s` equals that of `v̄`. -/
theorem thm5_i (h2 : MemLp id 2 ν) {s : E} (hs : s ≠ 0) (lam : ℝ) {v : E}
    (hv : ∀ w, Jobj ν lam s v ≤ Jobj ν lam s w) :
    v - ⟪v, ‖s‖⁻¹ • s⟫ • (‖s‖⁻¹ • s) = ubar ν - ⟪ubar ν, ‖s‖⁻¹ • s⟫ • (‖s‖⁻¹ • s) := by
  set e := ‖s‖⁻¹ • s
  have he : ‖e‖ = 1 := by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.2 hs)]
  set P := (v - ubar ν) - ⟪v - ubar ν, e⟫ • e
  -- `w = v - P` has the same component along `e` and no orthogonal error
  have hPe : ⟪P, e⟫ = 0 := by
    simp only [P, inner_sub_left, inner_smul_left, real_inner_self_eq_norm_sq, he,
      RCLike.conj_to_real]; ring
  have hcmp := hv (v - P)
  rw [J_split h2 hs, J_split h2 hs] at hcmp
  have h1 : ⟪v - P, e⟫ = ⟪v, e⟫ := by rw [inner_sub_left, hPe, sub_zero]
  have h2' : (v - P - ubar ν) - ⟪v - P - ubar ν, e⟫ • e = 0 := by
    have : v - P - ubar ν = ⟪v - ubar ν, e⟫ • e := by simp only [P]; abel
    rw [this, inner_smul_left, real_inner_self_eq_norm_sq, he]; simp
  rw [h1, h2', norm_zero] at hcmp
  have hP : ‖P‖ ^ 2 = 0 := by nlinarith [sq_nonneg ‖P‖]
  have hP0 : P = 0 := norm_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 hP)
  have : v - ubar ν = ⟪v - ubar ν, e⟫ • e := sub_eq_zero.1 hP0
  rw [inner_sub_left, sub_smul] at this
  rw [sub_eq_sub_iff_sub_eq_sub, this]

/-- **Along the score.**  `a_λ = π_λ·ŝ` minimizes `g(a) = (a - b̄)² + λ‖s‖ E|a - b|`. -/
theorem thm5_along (h2 : MemLp id 2 ν) {s : E} (hs : s ≠ 0) (lam : ℝ) {v : E}
    (hv : ∀ w, Jobj ν lam s v ≤ Jobj ν lam s w) (a : ℝ) :
    gObj ⟪ubar ν, ‖s‖⁻¹ • s⟫ (lam * ‖s‖) (absDev ν fun u => ⟪u, ‖s‖⁻¹ • s⟫) ⟪v, ‖s‖⁻¹ • s⟫ ≤
      gObj ⟪ubar ν, ‖s‖⁻¹ • s⟫ (lam * ‖s‖) (absDev ν fun u => ⟪u, ‖s‖⁻¹ • s⟫) a := by
  set e := ‖s‖⁻¹ • s
  have he : ‖e‖ = 1 := by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.2 hs)]
  have hee : ⟪e, e⟫ = 1 := by rw [real_inner_self_eq_norm_sq, he]; norm_num
  have hi := thm5_i h2 hs lam hv
  have hcmp := hv (v + (a - ⟪v, e⟫) • e)
  rw [J_split h2 hs, J_split h2 hs] at hcmp
  have hP : ∀ w : E, (w - ubar ν) - ⟪w - ubar ν, e⟫ • e =
      (w - ⟪w, e⟫ • e) - (ubar ν - ⟪ubar ν, e⟫ • e) := by
    intro w; rw [inner_sub_left, sub_smul]; abel
  have hshift : (v + (a - ⟪v, e⟫) • e) - ⟪v + (a - ⟪v, e⟫) • e, e⟫ • e = v - ⟪v, e⟫ • e := by
    rw [inner_add_left, inner_smul_left, hee]; simp only [RCLike.conj_to_real, mul_one]
    rw [add_smul, sub_smul]; abel
  have ha : ⟪v + (a - ⟪v, e⟫) • e, e⟫ = a := by
    rw [inner_add_left, inner_smul_left, hee]; simp
  rw [hP, hP, hi, sub_self, hshift, hi, sub_self, ha] at hcmp
  linarith

/-- **Theorem 5(ii).**  `|a_λ - b̄| ≤ λ‖s‖/2`, and `a_λ` lies between `b̄` and every median of
`b = u·ŝ`. -/
theorem thm5_ii (h2 : MemLp id 2 ν) {s : E} (hs : s ≠ 0) {lam : ℝ} (hlam : 0 ≤ lam) {v : E}
    (hv : ∀ w, Jobj ν lam s v ≤ Jobj ν lam s w) :
    |⟪v, ‖s‖⁻¹ • s⟫ - ⟪ubar ν, ‖s‖⁻¹ • s⟫| ≤ lam * ‖s‖ / 2 ∧
    ∀ m, (∀ x, absDev ν (fun u => ⟪u, ‖s‖⁻¹ • s⟫) m ≤ absDev ν (fun u => ⟪u, ‖s‖⁻¹ • s⟫) x) →
      min ⟪ubar ν, ‖s‖⁻¹ • s⟫ m ≤ ⟪v, ‖s‖⁻¹ • s⟫ ∧ ⟪v, ‖s‖⁻¹ • s⟫ ≤ max ⟪ubar ν, ‖s‖⁻¹ • s⟫ m := by
  have hb : Integrable (fun u => ⟪u, ‖s‖⁻¹ • s⟫) ν :=
    (integrable_id_of_memLp h2).inner_const _
  have hc : 0 ≤ lam * ‖s‖ := mul_nonneg hlam (norm_nonneg _)
  exact ⟨thm5_ii_bound hc (absDev_lipschitz hb) (thm5_along h2 hs lam hv),
    fun m hm => thm5_ii_between hc (absDev_convex hb) hm (thm5_along h2 hs lam hv)⟩

/-- **Theorem 5(iii).**  If `b̄` is a median of `b = u·ŝ` (e.g. `b` constant — the
demonstrations agree along `s`, in particular where they do not cross — or `b` symmetric
about `b̄`: `median_of_const`, `median_of_symmetric`), then `π_λ = v̄`. -/
theorem thm5_iii (h2 : MemLp id 2 ν) {s : E} (hs : s ≠ 0) {lam : ℝ} (hlam : 0 < lam) {v : E}
    (hv : ∀ w, Jobj ν lam s v ≤ Jobj ν lam s w)
    (hmed : ∀ x, absDev ν (fun u => ⟪u, ‖s‖⁻¹ • s⟫) ⟪ubar ν, ‖s‖⁻¹ • s⟫ ≤
      absDev ν (fun u => ⟪u, ‖s‖⁻¹ • s⟫) x) : v = ubar ν := by
  have hb : Integrable (fun u => ⟪u, ‖s‖⁻¹ • s⟫) ν :=
    (integrable_id_of_memLp h2).inner_const _
  have hc : 0 < lam * ‖s‖ := mul_pos hlam (norm_pos_iff.2 hs)
  have ha := (thm5_iii_iff hc (absDev_convex hb) (thm5_along h2 hs lam hv)).2 hmed
  have hi := thm5_i h2 hs lam hv
  rw [ha] at hi
  exact sub_left_injective hi

end Vector

/-! ### Worked examples and the stationarity condition -/

section Examples

/-- `g` for a two-point law: value `b₁` with probability `p`, `b₂` with `1 - p`. -/
def g2 (p b₁ b₂ c a : ℝ) : ℝ :=
  p * (a - b₁) ^ 2 + (1 - p) * (a - b₂) ^ 2 + c * (p * |a - b₁| + (1 - p) * |a - b₂|)

/-- **Symmetric tie.**  Velocities `±1` along the score, equally likely: `a_λ = 0` for every
`λ`, so an exact symmetric tie is not broken. -/
theorem ex_symmetric_tie {c : ℝ} (hc : 0 ≤ c) (a : ℝ) :
    g2 (1 / 2) 1 (-1) c 0 ≤ g2 (1 / 2) 1 (-1) c a := by
  unfold g2
  norm_num
  have h1 : 1 ≤ 1 / 2 * |a - 1| + 1 / 2 * |a + 1| := by
    cases abs_cases (a - 1) <;> cases abs_cases (a + 1) <;> linarith
  nlinarith [sq_nonneg a, mul_le_mul_of_nonneg_left h1 hc]

/-- **Shift against `s`.**  Velocity `0` with probability `¾` and `4` with `¼`, `λ‖s‖ = 2`:
mean `1`, median `0`, and `a_λ = ½`. -/
theorem ex_shift_against (a : ℝ) :
    g2 (3 / 4) 0 4 2 (1 / 2) ≤ g2 (3 / 4) 0 4 2 a ∧
    (3 / 4 : ℝ) * 0 + 1 / 4 * 4 = 1 := by
  refine ⟨?_, by norm_num⟩
  unfold g2
  norm_num
  rcases le_or_gt 0 a with h0 | h0 <;> rcases le_or_gt a 4 with h4 | h4
  · rw [abs_of_nonneg h0, abs_of_nonpos (by linarith)]; nlinarith [sq_nonneg (a - 1 / 2)]
  · rw [abs_of_nonneg h0, abs_of_pos (by linarith)]; nlinarith [sq_nonneg (a - 1 / 2)]
  · rw [abs_of_neg h0, abs_of_nonpos (by linarith)]; nlinarith [sq_nonneg (a - 1 / 2)]
  · linarith

/-- **Bias.**  A third of the demonstrations move along `ŝ` at speed `3`, the rest do not:
`b̄ = 1`, `m = 0`, and `a_λ = max{1 - λ‖s‖/6, 0}`; for `λ‖s‖ ≥ 6` the regularized policy
ignores the moving third. -/
theorem ex_bias {c : ℝ} (hc : 0 ≤ c) (a : ℝ) :
    g2 (1 / 3) 3 0 c (max (1 - c / 6) 0) ≤ g2 (1 / 3) 3 0 c a := by
  unfold g2
  rcases le_total (1 - c / 6) 0 with hm | hm
  · rw [max_eq_right hm]
    norm_num
    rcases le_or_gt 0 a with h0 | h0 <;> rcases le_or_gt a 3 with h3 | h3
    · rw [abs_of_nonneg h0, abs_of_nonpos (by linarith : a - 3 ≤ 0)]; nlinarith
    · rw [abs_of_nonneg h0, abs_of_pos (by linarith : 0 < a - 3)]; nlinarith
    · rw [abs_of_neg h0, abs_of_nonpos (by linarith : a - 3 ≤ 0)]; nlinarith
    · linarith
  · rw [max_eq_left hm]
    have hm0 : 0 ≤ 1 - c / 6 := hm
    have hm3 : 1 - c / 6 - 3 ≤ 0 := by linarith
    rw [abs_of_nonpos hm3, show 1 - c / 6 - 0 = 1 - c / 6 by ring, abs_of_nonneg hm0]
    rcases le_or_gt 0 a with h0 | h0 <;> rcases le_or_gt a 3 with h3 | h3
    · rw [abs_of_nonpos (by linarith : a - 3 ≤ 0), sub_zero, abs_of_nonneg h0]
      nlinarith [sq_nonneg (a - (1 - c / 6))]
    · rw [abs_of_pos (by linarith : 0 < a - 3), sub_zero, abs_of_nonneg h0]
      nlinarith [sq_nonneg (a - (1 - c / 6))]
    · rw [abs_of_nonpos (by linarith : a - 3 ≤ 0), sub_zero, abs_of_neg h0]
      nlinarith [sq_nonneg (a - (1 - c / 6))]
    · linarith

/-- **Stationarity (size and direction of the shift).**  For a finite law of `b` with no
atom at `a`, if `a` minimizes `g` then
`a = b̄ + (λ‖s‖/2) (Pr(b > a) - Pr(b < a))`: the prediction moves toward the side with more
demonstrations. -/
theorem stationarity {ι : Type*} [Fintype ι] {w b : ι → ℝ} (hw1 : ∑ i, w i = 1) {c a : ℝ}
    (hatom : ∀ i, b i ≠ a)
    (hmin : ∀ x, (∑ i, w i * (x - b i) ^ 2) + c * ∑ i, w i * |x - b i| ≥
      (∑ i, w i * (a - b i) ^ 2) + c * ∑ i, w i * |a - b i|) :
    a = (∑ i, w i * b i) + c / 2 *
      ((∑ i ∈ Finset.univ.filter (fun i => a < b i), w i) -
        (∑ i ∈ Finset.univ.filter (fun i => b i < a), w i)) := by
  set G : ℝ → ℝ := fun x => (∑ i, w i * (x - b i) ^ 2) + c * ∑ i, w i * |x - b i|
  have hd : HasDerivAt G ((∑ i, w i * (2 * (a - b i))) +
      c * ∑ i, w i * (SignType.sign (a - b i) : ℝ)) a := by
    have h1 : HasDerivAt (fun x => ∑ i, w i * (x - b i) ^ 2) (∑ i, w i * (2 * (a - b i))) a := by
      have := HasDerivAt.fun_sum (u := Finset.univ) (A := fun i x => w i * (x - b i) ^ 2)
        (A' := fun i => w i * (2 * (a - b i))) (x := a)
        (fun i _ => by
          have := (((hasDerivAt_id' a).sub_const (b i)).pow 2).const_mul (w i)
          exact this.congr_deriv (by push_cast; ring))
      simpa using this
    have h2 : HasDerivAt (fun x => ∑ i, w i * |x - b i|)
        (∑ i, w i * (SignType.sign (a - b i) : ℝ)) a := by
      have := HasDerivAt.fun_sum (u := Finset.univ) (A := fun i x => w i * |x - b i|)
        (A' := fun i => w i * (SignType.sign (a - b i) : ℝ)) (x := a)
        (fun i _ => by
          have := (hasDerivAt_abs (sub_ne_zero.2 (hatom i).symm)).comp a
            ((hasDerivAt_id' a).sub_const (b i))
          simpa using this.const_mul (w i))
      simpa using this
    exact h1.add (h2.const_mul c)
  have hzero := IsLocalMin.hasDerivAt_eq_zero (Filter.Eventually.of_forall hmin) hd
  -- evaluate the sums
  have hsq : ∑ i, w i * (2 * (a - b i)) = 2 * (a - ∑ i, w i * b i) := by
    rw [show (fun i => w i * (2 * (a - b i))) = fun i => 2 * a * w i - 2 * (w i * b i) by
      funext i; ring, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hw1]; ring
  have hsg : ∑ i, w i * (SignType.sign (a - b i) : ℝ) =
      (∑ i ∈ Finset.univ.filter (fun i => b i < a), w i) -
        (∑ i ∈ Finset.univ.filter (fun i => a < b i), w i) := by
    rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun i => b i < a)]
    have e1 : ∑ i ∈ Finset.univ.filter (fun i => b i < a),
        w i * (SignType.sign (a - b i) : ℝ) = ∑ i ∈ Finset.univ.filter (fun i => b i < a), w i :=
      Finset.sum_congr rfl fun i hi => by
        rw [sign_pos (sub_pos.2 (Finset.mem_filter.1 hi).2)]; simp
    have e2 : ∑ i ∈ Finset.univ.filter (fun i => ¬ b i < a),
        w i * (SignType.sign (a - b i) : ℝ) =
          -∑ i ∈ Finset.univ.filter (fun i => a < b i), w i := by
      rw [← Finset.sum_neg_distrib]
      apply Finset.sum_congr
      · ext i; simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_lt]
        exact ⟨fun h => lt_of_le_of_ne h (hatom i).symm, fun h => h.le⟩
      · intro i hi
        rw [sign_neg (sub_neg.2 (Finset.mem_filter.1 hi).2)]; simp
    rw [e1, e2]; ring
  rw [hsq, hsg] at hzero
  linarith

/-- **Realizability.**  With `x_{ρ*} ∼ N(1, v)` the score `∂ₓ log p` at `x = 0` is `1/v`:
`1` for `v = 1`, and `1/(1 + h²)` for a Gaussian kernel estimate with bandwidth `h`
(`v = 1 + h²`). -/
theorem realizability_score {v : ℝ} (hv : 0 < v) :
    HasDerivAt (fun x => Real.log (ProbabilityTheory.gaussianPDFReal 1 v.toNNReal x)) (1 / v) 0 := by
  have hv' : (v.toNNReal : ℝ) = v := Real.coe_toNNReal _ hv.le
  have e : (fun x => Real.log (ProbabilityTheory.gaussianPDFReal 1 v.toNNReal x)) =
      fun x => Real.log (Real.sqrt (2 * Real.pi * v))⁻¹ + (-(x - 1) ^ 2 / (2 * v)) := by
    funext x
    rw [ProbabilityTheory.gaussianPDFReal_def, hv', Real.log_mul (by positivity) (by positivity),
      Real.log_exp]
  rw [e]
  have := (((hasDerivAt_id' (0 : ℝ)).sub_const 1).pow 2).neg.div_const (2 * v) |>.const_add
    (Real.log (Real.sqrt (2 * Real.pi * v))⁻¹)
  refine this.congr_deriv ?_
  field_simp
  ring

end Examples

end Phaseformer
