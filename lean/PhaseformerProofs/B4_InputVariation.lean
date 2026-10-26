import Mathlib
import PhaseformerProofs.Gronwall
import PhaseformerProofs.B1_Setting

/-!
# App. B.4 — Proofs for Input Variation (Lemma 2 and Proposition 3)

The Grönwall paragraph of B.4 (Eq. (13), closed-loop and Euler constants, the unicycle) is
in `PhaseformerProofs/Gronwall.lean`.

## Lemma 2 (a split after a shared input must come from the input)
* (a) `lemma2a_branch_average` — at a shared input the regression target is the weighted
  branch average `(w₁u₁ + w₂u₂)/(w₁ + w₂)`; `fiberMean_const` — if all demonstrations at an
  input take one branch, the target is that branch.
* (b) `lemma2b_lipschitz_lower` — following both branches forces `L(ρ) ≥ 1/(ρ - ρ*)`;
  `lemma2b_not_integrable` — then `L` is not integrable at `ρ*`.
* (c) `lemma2c_one_branch` — with a deterministic (unique-solution) closed loop, rollouts in
  the same state at `ρ*` continue identically, so they follow at most one branch.

## Proposition 3 (bridge bound)
* (a) `prop3_separation`, `prop3a_undecided`, `prop3a_prob` (Eq. (14)).
* (b) `prop3b_segment` — every segment from `S_A` to `S_B` contains an open stretch of
  undecided starts of length at least `g/K`.
* 1-D Gaussian starts: `prop3_gauss_too_wide` (`ℓ > 4σ` ⇒ the starts in `[m-2σ, m+2σ]`
  cannot reach both `A` and `B`), `prop3_gauss_prob` (`Pr(undecided) ≥ φ(2) ℓ/σ`), and
  `phi2_bounds` (`φ(2) = e⁻²/√(2π) ∈ (0.0539, 0.0541)`, i.e. `≈ 0.054`).
-/

open MeasureTheory Set Metric
open scoped RealInnerProductSpace ENNReal

namespace Phaseformer

/-! ### Lemma 2 -/

section Lemma2

variable {Ω β V : Type*} [Fintype Ω] [DecidableEq β] [AddCommGroup V] [Module ℝ V]

/-- **Lemma 2(a).**  If the demonstrations at input `z` leave along two branches with
velocities `u₁` (those with `br ω`) and `u₂` (the rest), of total weights `W₁, W₂`, the
regression target `v̄(z)` is `(W₁ u₁ + W₂ u₂)/(W₁ + W₂)`. -/
theorem lemma2a_branch_average (w : Ω → ℝ) (Z : Ω → β) (z : β) (br : Ω → Prop)
    [DecidablePred br] (u₁ u₂ : V) :
    fiberMean w (fun ω => if br ω then u₁ else u₂) Z z =
      (fiberMass w Z z)⁻¹ •
        ((∑ ω ∈ (Finset.univ.filter fun ω => Z ω = z).filter br, w ω) • u₁ +
         (∑ ω ∈ (Finset.univ.filter fun ω => Z ω = z).filter (fun ω => ¬ br ω), w ω) • u₂) ∧
    fiberMass w Z z =
      (∑ ω ∈ (Finset.univ.filter fun ω => Z ω = z).filter br, w ω) +
      (∑ ω ∈ (Finset.univ.filter fun ω => Z ω = z).filter (fun ω => ¬ br ω), w ω) := by
  constructor
  · unfold fiberMean fiberSum
    congr 1
    rw [← Finset.sum_filter_add_sum_filter_not _ br, Finset.sum_smul, Finset.sum_smul]
    congr 1
    · exact Finset.sum_congr rfl fun ω hω => by simp [(Finset.mem_filter.1 hω).2]
    · exact Finset.sum_congr rfl fun ω hω => by simp [(Finset.mem_filter.1 hω).2]
  · unfold fiberMass
    rw [Finset.sum_filter_add_sum_filter_not]

/-- If every demonstration at the input takes the same branch `u`, the target is `u`
("the MSE optimum steers different inputs to different branches only where the
demonstrations at those inputs take different branches"). -/
theorem fiberMean_const (w : Ω → ℝ) (Z : Ω → β) (z : β) (u : V) (Y : Ω → V)
    (hY : ∀ ω, Z ω = z → Y ω = u) (hm : fiberMass w Z z ≠ 0) :
    fiberMean w Y Z z = u := by
  unfold fiberMean fiberSum
  rw [Finset.sum_congr rfl fun ω hω => by rw [hY ω (Finset.mem_filter.1 hω).2],
    ← Finset.sum_smul]
  change (fiberMass w Z z)⁻¹ • (fiberMass w Z z • u) = u
  rw [smul_smul, inv_mul_cancel₀ hm, one_smul]

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **Lemma 2(b), the bound.**  If a policy whose rollouts follow both branches
`x* + (ρ - ρ*) uₖ` (so `π(x* + (ρ-ρ*)uₖ, ρ) = uₖ`) is `L(ρ)`-Lipschitz at phase `ρ > ρ*`,
then `L(ρ) ≥ 1/(ρ - ρ*)`: the branches are `(ρ-ρ*)‖u₁-u₂‖` apart and their velocities differ
by `‖u₁ - u₂‖`. -/
theorem lemma2b_lipschitz_lower {pol : E → ℝ → E} {L : ℝ → ℝ} {xs u₁ u₂ : E} {ρs ρ : ℝ}
    (hu : u₁ ≠ u₂) (hρ : ρs < ρ)
    (hLip : ∀ x y, ‖pol x ρ - pol y ρ‖ ≤ L ρ * ‖x - y‖)
    (h₁ : pol (xs + (ρ - ρs) • u₁) ρ = u₁) (h₂ : pol (xs + (ρ - ρs) • u₂) ρ = u₂) :
    1 / (ρ - ρs) ≤ L ρ := by
  have h := hLip (xs + (ρ - ρs) • u₁) (xs + (ρ - ρs) • u₂)
  rw [h₁, h₂, show xs + (ρ - ρs) • u₁ - (xs + (ρ - ρs) • u₂) = (ρ - ρs) • (u₁ - u₂) by
    rw [smul_sub]; abel, norm_smul, Real.norm_of_nonneg (by linarith)] at h
  have hn : 0 < ‖u₁ - u₂‖ := norm_pos_iff.2 (sub_ne_zero.2 hu)
  rw [div_le_iff₀ (by linarith)]
  nlinarith

/-- **Lemma 2(b), non-integrability.**  `L(ρ) ≥ 1/(ρ - ρ*)` on `(ρ*, 1]` is not integrable at
`ρ*`, because `∫_{ρ*}^1 dρ/(ρ - ρ*) = ∞`. -/
theorem lemma2b_not_integrable {L : ℝ → ℝ} {ρs : ℝ} (hρs : ρs < 1)
    (hL : ∀ ρ ∈ Ioc ρs 1, 1 / (ρ - ρs) ≤ L ρ) : ¬ IntervalIntegrable L volume ρs 1 := by
  intro hint
  have : IntervalIntegrable (fun ρ => (ρ - ρs)⁻¹) volume ρs 1 := by
    refine hint.mono_fun ?_ ?_
    · exact ((measurable_id.sub_const ρs).inv).aestronglyMeasurable
    · rw [uIoc_of_le hρs.le]
      refine (ae_restrict_mem measurableSet_Ioc).mono fun ρ hρ => ?_
      have h1 : 0 < ρ - ρs := sub_pos.2 hρ.1
      have h2 := hL ρ hρ
      show ‖(ρ - ρs)⁻¹‖ ≤ ‖L ρ‖
      rw [Real.norm_of_nonneg (inv_nonneg.2 h1.le), ← one_div, Real.norm_eq_abs]
      exact h2.trans (le_abs_self _)
  rw [intervalIntegrable_sub_inv_iff] at this
  rcases this with h | h
  · linarith
  · exact h (by rw [uIcc_of_le hρs.le]; exact ⟨le_rfl, hρs.le⟩)

/-- **Lemma 2(c).**  A deterministic policy in a deterministic environment makes the closed
loop a deterministic system: with unique solutions (here, a Lipschitz field on the joint
state space `E`, which may include the world state), rollouts that are in the same state at
`ρ*` continue identically, so they cannot follow two distinct branches. -/
theorem lemma2c_one_branch {pol : E → ℝ → E} {L : ℝ → ℝ} {ρs ρ₁ : ℝ}
    (hL : Continuous L) (hLip : LipschitzInState pol L ρs ρ₁) {z z' : ℝ → E}
    (hz : IsRollout pol z ρs ρ₁) (hz' : IsRollout pol z' ρs ρ₁) (h0 : z ρs = z' ρs)
    {xs u₁ u₂ : E} (hu : u₁ ≠ u₂) (hρ : ρs < ρ₁)
    (hb₁ : ∀ ρ ∈ Icc ρs ρ₁, z ρ = xs + (ρ - ρs) • u₁) :
    ¬ ∀ ρ ∈ Icc ρs ρ₁, z' ρ = xs + (ρ - ρs) • u₂ := by
  intro hb₂
  have heq := rollouts_eq_of_eq hL hLip hz hz' h0 (right_mem_Icc.2 hρ.le)
  rw [hb₁ ρ₁ (right_mem_Icc.2 hρ.le), hb₂ ρ₁ (right_mem_Icc.2 hρ.le), add_right_inj] at heq
  exact hu (smul_right_injective E (sub_ne_zero.2 hρ.ne') heq)

end Lemma2

/-! ### Proposition 3 -/

section Prop3

variable {E : Type*} [NormedAddCommGroup E]
variable {Φ : E → E} {K g : ℝ} {A B : Set E}

/-- Starts sent into `A` and into `B` are at least `g/K` apart. -/
theorem prop3_separation (hK : 0 < K) (hΦ : ∀ x y, ‖Φ x - Φ y‖ ≤ K * ‖x - y‖)
    (hAB : ∀ a ∈ A, ∀ b ∈ B, g ≤ ‖a - b‖) {x y : E} (hx : x ∈ Φ ⁻¹' A) (hy : y ∈ Φ ⁻¹' B) :
    g / K ≤ ‖x - y‖ := by
  rw [div_le_iff₀ hK]
  have := (hAB _ hx _ hy).trans (hΦ x y)
  linarith

/-- **Proposition 3(a).**  Every start whose distance to `S_A` is positive but less than
`g/K` is undecided: its rollout ends in neither `A` nor `B`. -/
theorem prop3a_undecided (hK : 0 < K) (hΦ : ∀ x y, ‖Φ x - Φ y‖ ≤ K * ‖x - y‖)
    (hAB : ∀ a ∈ A, ∀ b ∈ B, g ≤ ‖a - b‖) {x : E} (h0 : 0 < infDist x (Φ ⁻¹' A))
    (h1 : infDist x (Φ ⁻¹' A) < g / K) : x ∉ Φ ⁻¹' A ∧ x ∉ Φ ⁻¹' B := by
  refine ⟨fun hx => by rw [infDist_zero_of_mem hx] at h0; exact lt_irrefl _ h0, fun hx => ?_⟩
  have hne : (Φ ⁻¹' A).Nonempty := by
    by_contra hne
    rw [not_nonempty_iff_eq_empty] at hne
    rw [hne, infDist_empty] at h0
    exact lt_irrefl _ h0
  have : g / K ≤ infDist x (Φ ⁻¹' A) := by
    rw [le_infDist hne]
    intro y hy
    rw [dist_eq_norm, ← norm_neg, neg_sub]
    exact prop3_separation hK hΦ hAB hy hx
  linarith

/-- **Eq. (14).**  For starts drawn from any distribution,
`Pr(undecided) ≥ Pr(0 < dist(x_{ρ₀}, S_A) < g/K)`. -/
theorem prop3a_prob [MeasurableSpace E] (P : Measure E) (hK : 0 < K)
    (hΦ : ∀ x y, ‖Φ x - Φ y‖ ≤ K * ‖x - y‖) (hAB : ∀ a ∈ A, ∀ b ∈ B, g ≤ ‖a - b‖) :
    P {x | 0 < infDist x (Φ ⁻¹' A) ∧ infDist x (Φ ⁻¹' A) < g / K} ≤
      P {x | x ∉ Φ ⁻¹' A ∧ x ∉ Φ ⁻¹' B} :=
  measure_mono fun _ hx => prop3a_undecided hK hΦ hAB hx.1 hx.2

variable [NormedSpace ℝ E]

/-- The segment from `a` to `b`, `t ↦ a + t (b - a)`. -/
def seg (a b : E) (t : ℝ) : E := a + t • (b - a)

lemma continuous_seg (a b : E) : Continuous (seg a b) := by
  unfold seg; fun_prop

/-- **Proposition 3(b).**  Every segment from a point of `S_A` to a point of `S_B` contains an
open stretch `{seg a b t : t₁ < t < t₂}` of undecided starts whose endpoints are at least
`g/K` apart.  (`Φ` continuous and `A`, `B` closed make `S_A`, `S_B` closed.) -/
theorem prop3b_segment (hK : 0 < K) (hg : 0 < g) (hΦ : ∀ x y, ‖Φ x - Φ y‖ ≤ K * ‖x - y‖)
    (hΦc : Continuous Φ) (hA : IsClosed A) (hB : IsClosed B)
    (hAB : ∀ a ∈ A, ∀ b ∈ B, g ≤ ‖a - b‖) {a b : E} (ha : a ∈ Φ ⁻¹' A) (hb : b ∈ Φ ⁻¹' B) :
    ∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ < t₂ ∧ t₂ ≤ 1 ∧
      seg a b t₁ ∈ Φ ⁻¹' A ∧ seg a b t₂ ∈ Φ ⁻¹' B ∧
      g / K ≤ ‖seg a b t₂ - seg a b t₁‖ ∧
      ∀ t ∈ Ioo t₁ t₂, seg a b t ∉ Φ ⁻¹' A ∧ seg a b t ∉ Φ ⁻¹' B := by
  have hSA : IsClosed (Φ ⁻¹' A) := hA.preimage hΦc
  have hSB : IsClosed (Φ ⁻¹' B) := hB.preimage hΦc
  set TA := {t | t ∈ Icc (0 : ℝ) 1 ∧ seg a b t ∈ Φ ⁻¹' A}
  have hTAc : IsClosed TA := isClosed_Icc.inter (hSA.preimage (continuous_seg a b))
  have hTAne : TA.Nonempty := ⟨0, ⟨le_rfl, zero_le_one⟩, by simpa [seg] using ha⟩
  have hTAb : BddAbove TA := ⟨1, fun t ht => ht.1.2⟩
  set t₁ := sSup TA
  have ht₁ : t₁ ∈ TA := hTAc.csSup_mem hTAne hTAb
  set TB := {t | t ∈ Icc t₁ 1 ∧ seg a b t ∈ Φ ⁻¹' B}
  have hTBc : IsClosed TB := isClosed_Icc.inter (hSB.preimage (continuous_seg a b))
  have hTBne : TB.Nonempty := ⟨1, ⟨ht₁.1.2, le_rfl⟩, by simpa [seg] using hb⟩
  have hTBb : BddBelow TB := ⟨t₁, fun t ht => ht.1.1⟩
  set t₂ := sInf TB
  have ht₂ : t₂ ∈ TB := hTBc.csInf_mem hTBne hTBb
  have hsep : g / K ≤ ‖seg a b t₂ - seg a b t₁‖ := by
    rw [← norm_neg, neg_sub]; exact prop3_separation hK hΦ hAB ht₁.2 ht₂.2
  have hlt : t₁ < t₂ := by
    rcases eq_or_lt_of_le ht₂.1.1 with h | h
    · exfalso; rw [← h, sub_self, norm_zero] at hsep; exact absurd hsep (not_le.2 (by positivity))
    · exact h
  refine ⟨t₁, t₂, ht₁.1.1, hlt, ht₂.1.2, ht₁.2, ht₂.2, hsep, fun t ht => ⟨?_, ?_⟩⟩
  · intro hA'
    have : t ∈ TA := ⟨⟨ht₁.1.1.trans ht.1.le, ht.2.le.trans ht₂.1.2⟩, hA'⟩
    exact absurd (le_csSup hTAb this) (not_le.2 ht.1)
  · intro hB'
    have : t ∈ TB := ⟨⟨ht.1.le, ht.2.le.trans ht₂.1.2⟩, hB'⟩
    exact absurd (csInf_le hTBb this) (not_le.2 ht.2)

/-! #### One-dimensional Gaussian starts -/

open ProbabilityTheory Real

/-- In one dimension the undecided stretch of (b) is an open interval `(p, q)` with
`q - p ≥ g/K`, lying between the two starts. -/
theorem prop3b_interval_1d {Φ : ℝ → ℝ} (hK : 0 < K) (hg : 0 < g)
    (hΦ : ∀ x y, ‖Φ x - Φ y‖ ≤ K * ‖x - y‖) (hΦc : Continuous Φ)
    {A B : Set ℝ} (hA : IsClosed A) (hB : IsClosed B) (hAB : ∀ a ∈ A, ∀ b ∈ B, g ≤ ‖a - b‖)
    {a b : ℝ} (ha : a ∈ Φ ⁻¹' A) (hb : b ∈ Φ ⁻¹' B) :
    ∃ p q : ℝ, g / K ≤ q - p ∧ min a b ≤ p ∧ q ≤ max a b ∧
      ∀ y ∈ Ioo p q, y ∉ Φ ⁻¹' A ∧ y ∉ Φ ⁻¹' B := by
  obtain ⟨t₁, t₂, h0, h12, h1, -, -, hsep, hund⟩ := prop3b_segment hK hg hΦ hΦc hA hB hAB ha hb
  simp only [seg, smul_eq_mul, Real.norm_eq_abs] at hsep hund
  have hd : seg a b t₂ - seg a b t₁ = (t₂ - t₁) * (b - a) := by simp [seg]; ring
  have hab : a ≠ b := by
    intro h; subst h
    have : a + t₂ * (a - a) - (a + t₁ * (a - a)) = 0 := by ring
    rw [this, abs_zero] at hsep; exact absurd hsep (not_le.2 (by positivity))
  rcases lt_or_gt_of_ne hab with hlt | hgt
  · refine ⟨a + t₁ * (b - a), a + t₂ * (b - a), ?_, ?_, ?_, ?_⟩
    · have : |a + t₂ * (b - a) - (a + t₁ * (b - a))| = a + t₂ * (b - a) - (a + t₁ * (b - a)) :=
        abs_of_nonneg (by nlinarith)
      linarith
    · rw [min_eq_left hlt.le]; nlinarith
    · rw [max_eq_right hlt.le]; nlinarith
    · intro y hy
      have hba : 0 < b - a := sub_pos.2 hlt
      have ht : (y - a) / (b - a) ∈ Ioo t₁ t₂ :=
        ⟨by rw [lt_div_iff₀ hba]; linarith [hy.1], by rw [div_lt_iff₀ hba]; linarith [hy.2]⟩
      have h := hund _ ht
      have e : a + (y - a) / (b - a) * (b - a) = y := by field_simp; ring
      rwa [e] at h
  · refine ⟨a + t₂ * (b - a), a + t₁ * (b - a), ?_, ?_, ?_, ?_⟩
    · have : |a + t₂ * (b - a) - (a + t₁ * (b - a))| = a + t₁ * (b - a) - (a + t₂ * (b - a)) := by
        rw [abs_of_nonpos (by nlinarith)]; ring
      linarith
    · rw [min_eq_right hgt.le]; nlinarith
    · rw [max_eq_left hgt.le]; nlinarith
    · intro y hy
      have hab' : b - a < 0 := sub_neg.2 hgt
      have ht : (y - a) / (b - a) ∈ Ioo t₁ t₂ :=
        ⟨by rw [lt_div_iff_of_neg hab']; linarith [hy.2],
          by rw [div_lt_iff_of_neg hab']; linarith [hy.1]⟩
      have h := hund _ ht
      have e : a + (y - a) / (b - a) * (b - a) = y := by field_simp [hab'.ne]; ring
      rwa [e] at h

/-- **Too wide a shell.**  With `ℓ = g/K > 4σ`, starts in `W = [m - 2σ, m + 2σ]` cannot reach
both `A` and `B`. -/
theorem prop3_gauss_too_wide {Φ : ℝ → ℝ} (hK : 0 < K) (hg : 0 < g)
    (hΦ : ∀ x y, ‖Φ x - Φ y‖ ≤ K * ‖x - y‖) (hΦc : Continuous Φ)
    {A B : Set ℝ} (hA : IsClosed A) (hB : IsClosed B) (hAB : ∀ a ∈ A, ∀ b ∈ B, g ≤ ‖a - b‖)
    {m σ : ℝ} (hℓ : 4 * σ < g / K) :
    ¬ ∃ a ∈ Icc (m - 2 * σ) (m + 2 * σ), ∃ b ∈ Icc (m - 2 * σ) (m + 2 * σ),
      a ∈ Φ ⁻¹' A ∧ b ∈ Φ ⁻¹' B := by
  rintro ⟨a, ha, b, hb, haA, hbB⟩
  obtain ⟨p, q, hlen, hp, hq, -⟩ := prop3b_interval_1d hK hg hΦ hΦc hA hB hAB haA hbB
  have h1 : m - 2 * σ ≤ min a b := le_min ha.1 hb.1
  have h2 : max a b ≤ m + 2 * σ := max_le ha.2 hb.2
  linarith

/-- The standard normal density at 2, `φ(2) = e⁻²/√(2π)`. -/
noncomputable def phi2 : ℝ := exp (-2) / sqrt (2 * π)

/-- `φ(2) ≈ 0.054`. -/
theorem phi2_bounds : 0.0539 < phi2 ∧ phi2 < 0.0541 := by
  have he1 := Real.exp_one_gt_d9
  have he2 := Real.exp_one_lt_d9
  have hp1 := Real.pi_gt_d2
  have hp2 := Real.pi_lt_d2
  have hexp : exp (-2) = 1 / exp 1 ^ 2 := by
    rw [show (-2 : ℝ) = -(1 * 2) by norm_num, exp_neg, ← exp_nat_mul]; norm_num
  have hs1 : sqrt (2 * π) < 2.51 := by
    rw [sqrt_lt' (by norm_num)]; nlinarith
  have hs2 : 2.505 < sqrt (2 * π) := by
    rw [lt_sqrt (by norm_num)]; nlinarith
  have hspos : 0 < sqrt (2 * π) := by linarith
  unfold phi2
  rw [hexp]
  have hepos : 0 < exp 1 := exp_pos 1
  have hE1 : 7.389 < exp 1 ^ 2 := by nlinarith
  have hE2 : exp 1 ^ 2 < 7.3891 := by nlinarith
  constructor
  · rw [lt_div_iff₀ hspos, lt_div_iff₀ (by positivity)]
    nlinarith
  · rw [div_lt_iff₀ hspos, div_lt_iff₀ (by positivity)]
    nlinarith

/-- On `W = [m - 2σ, m + 2σ]` the density of `N(m, σ²)` is at least `φ(2)/σ`. -/
theorem gauss_density_lower {m σ x : ℝ} (hσ : 0 < σ) (hx : x ∈ Icc (m - 2 * σ) (m + 2 * σ)) :
    phi2 / σ ≤ gaussianPDFReal m (σ ^ 2).toNNReal x := by
  rw [gaussianPDFReal_def]
  simp only [Real.coe_toNNReal _ (sq_nonneg σ)]
  have hsq : sqrt (2 * π * σ ^ 2) = σ * sqrt (2 * π) := by
    rw [mul_comm (2 * π), sqrt_mul (sq_nonneg _), sqrt_sq hσ.le]
  rw [hsq, phi2]
  have hspos : 0 < sqrt (2 * π) := sqrt_pos.2 (by positivity)
  have hexp : exp (-2) ≤ exp (-(x - m) ^ 2 / (2 * σ ^ 2)) := by
    apply exp_le_exp.2
    rw [le_div_iff₀ (by positivity)]
    have : (x - m) ^ 2 ≤ (2 * σ) ^ 2 :=
      sq_le_sq' (by linarith [hx.1]) (by linarith [hx.2])
    nlinarith
  calc exp (-2) / sqrt (2 * π) / σ = exp (-2) * (σ * sqrt (2 * π))⁻¹ := by
        field_simp
    _ ≤ exp (-(x - m) ^ 2 / (2 * σ ^ 2)) * (σ * sqrt (2 * π))⁻¹ :=
        mul_le_mul_of_nonneg_right hexp (by positivity)
    _ = (σ * sqrt (2 * π))⁻¹ * exp (-(x - m) ^ 2 / (2 * σ ^ 2)) := mul_comm _ _

/-- **Proposition 3(b), Gaussian bound.**  If starts `x_{ρ₀} ∼ N(m, σ²)` in
`W = [m - 2σ, m + 2σ]` reach both `A` and `B`, then `Pr(undecided) ≥ φ(2) ℓ/σ`,
`ℓ = g/K`. -/
theorem prop3_gauss_prob {Φ : ℝ → ℝ} (hK : 0 < K) (hg : 0 < g)
    (hΦ : ∀ x y, ‖Φ x - Φ y‖ ≤ K * ‖x - y‖) (hΦc : Continuous Φ)
    {A B : Set ℝ} (hA : IsClosed A) (hB : IsClosed B) (hAB : ∀ a ∈ A, ∀ b ∈ B, g ≤ ‖a - b‖)
    {m σ : ℝ} (hσ : 0 < σ) {a b : ℝ} (ha : a ∈ Icc (m - 2 * σ) (m + 2 * σ))
    (hb : b ∈ Icc (m - 2 * σ) (m + 2 * σ)) (haA : a ∈ Φ ⁻¹' A) (hbB : b ∈ Φ ⁻¹' B) :
    ENNReal.ofReal (phi2 * (g / K) / σ) ≤
      gaussianReal m (σ ^ 2).toNNReal {x | x ∉ Φ ⁻¹' A ∧ x ∉ Φ ⁻¹' B} := by
  obtain ⟨p, q, hlen, hp, hq, hund⟩ := prop3b_interval_1d hK hg hΦ hΦc hA hB hAB haA hbB
  have hW : Ioo p q ⊆ Icc (m - 2 * σ) (m + 2 * σ) := fun y hy =>
    ⟨(le_min ha.1 hb.1).trans (hp.trans hy.1.le), (hy.2.le.trans hq).trans (max_le ha.2 hb.2)⟩
  have hv : (σ ^ 2).toNNReal ≠ 0 := by
    rw [Ne, Real.toNNReal_eq_zero, not_le]; positivity
  calc ENNReal.ofReal (phi2 * (g / K) / σ)
      ≤ ENNReal.ofReal (phi2 / σ) * volume (Ioo p q) := by
        rw [Real.volume_Ioo, ← ENNReal.ofReal_mul (div_nonneg (phi2_bounds.1.le.trans' (by norm_num)) hσ.le)]
        apply ENNReal.ofReal_le_ofReal
        rw [mul_div_right_comm]
        exact mul_le_mul_of_nonneg_left hlen (div_nonneg (by linarith [phi2_bounds.1]) hσ.le)
    _ = ∫⁻ _ in Ioo p q, ENNReal.ofReal (phi2 / σ) := (setLIntegral_const _ _).symm
    _ ≤ ∫⁻ x in Ioo p q, gaussianPDF m (σ ^ 2).toNNReal x := by
        apply setLIntegral_mono (measurable_gaussianPDF _ _)
        intro x hx
        rw [gaussianPDF_def]
        exact ENNReal.ofReal_le_ofReal (gauss_density_lower hσ (hW hx))
    _ = gaussianReal m (σ ^ 2).toNNReal (Ioo p q) := (gaussianReal_apply m hv _).symm
    _ ≤ _ := measure_mono fun y hy => hund y hy

end Prop3

end Phaseformer
