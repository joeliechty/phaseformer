import Mathlib
import PhaseformerProofs.Gronwall
import PhaseformerProofs.B2_Marginals

/-!
# App. B.3 — Proof of Proposition 2 (phase can be necessary)

## (a) Overshoot
`P_S = ½ U[-δ, δ] + ½ U[1-δ, 1+δ]`, `0 < δ < ½`, `T(a) = a + 2` on the first block and
`a + 4` on the second, straight demonstrations `x_ρ = x₀ + ρ (T x₀ - x₀)`.

* `T_strictMonoOn`, `T_block_images` — `T` is increasing and separates the blocks.
* `groups_disjoint_at_phase`, `prop2a_no_conflict` — at each state and phase only one
  demonstration is present, so `C = 0` and `v̄ = 2` or `4` on the swept regions.
* `occupation_per_demo`, `group_occupation` — a pass at speed `c` spends time `1/c` per unit
  length; the occupation densities on the full-presence region are `μ₁ = ½·½ = ¼`,
  `μ₂ = ½·¼ = ⅛` (`occupation_values`).
* `vfree_convex` — `v̄_free = (2μ₁ + 4μ₂)/(μ₁ + μ₂)` is `2`, `4`, or strictly between.
  `vfree_overlap`: `8/3` on the overlap; `aliasing_limit`: `A → ¼(2-8/3)² + ⅛(4-8/3)² = ⅓`.
* Travel time for 1-D autonomous rollouts (`travel_time`): `∫_{x(0)}^{x(1)} dz / v(z) = 1`;
  `rollout_stays_in` ("it stays in `I` because its speed is at most 4");
  `rollout_strictMono` (`Φ` is strictly increasing); `overshoot_of_lt`, `undershoot_of_gt`.
* `prop2a_overshoot`, `prop2a_undershoot` — every first-group rollout of `v̄_free`
  overshoots `T(a)`, every second-group rollout undershoots `T(b)`.
* `cdf_lt_of_undershoot`, `prop2a_not_PG` — `Φ_# P_S ≠ P_G`: the distribution functions
  differ at `Φ(b)`.
* `prop2a_limit` — in the limit `δ → 0` the rollouts from `0` and `1` end at `2.5` and `4.5`.

The profile of `v̄_free` (`= 2`, `∈ (2,4)`, `= 4` on the three regions, continuous on `I`) is
taken as a hypothesis in `prop2a_*`; it is what `vfree_convex` gives from the occupation
densities.  Existence of rollouts of `v̄_free` (a continuous field) is assumed, as in the
paper.

## (b) Stall
`P_S = U[-δ, δ]`, `x_ρ = x₀ + sin(πρ)`.  `prop2b_velocity`, `prop2b_no_conflict` (`C = 0`,
`v̄ = π cos(πρ)`, 0-Lipschitz), `prop2b_demo_is_rollout`, `prop2b_flux_zero`
(`v̄_free = 0`: every pass out is cancelled by the pass back), `prop2b_stall`
(rollouts of the zero field never move), `prop2b_supports_disjoint` (`p_{1/2}` and `P_S` have
disjoint supports), `prop2b_shift_pos`, `prop2b_aliasing` (`A > 0`).
-/

open MeasureTheory Set Real

namespace Phaseformer

/-! ### Travel time for one-dimensional autonomous rollouts -/

section Travel

/-- `x` is a rollout of the autonomous 1-D field `v` on `[0, 1]`. -/
def IsRollout1D (v : ℝ → ℝ) (x : ℝ → ℝ) : Prop :=
  ContinuousOn x (Icc 0 1) ∧ ∀ ρ ∈ Ioo (0 : ℝ) 1, HasDerivAt x (v (x ρ)) ρ

variable {v : ℝ → ℝ} {p q : ℝ}

lemma inv_continuousOn (hv : ContinuousOn v (Ioo p q)) (hpos : ∀ y ∈ Ioo p q, 0 < v y) :
    ContinuousOn (fun y => 1 / v y) (Ioo p q) :=
  continuousOn_const.div hv fun y hy => (hpos y hy).ne'

lemma inv_intervalIntegrable (hv : ContinuousOn v (Ioo p q)) (hpos : ∀ y ∈ Ioo p q, 0 < v y)
    {a b : ℝ} (ha : a ∈ Ioo p q) (hb : b ∈ Ioo p q) :
    IntervalIntegrable (fun y => 1 / v y) volume a b :=
  ((inv_continuousOn hv hpos).mono (ordConnected_Ioo.uIcc_subset ha hb)).intervalIntegrable

/-- **Travel time.**  If `x` is a rollout of `v` staying in an interval `I` where `v` is
continuous and positive, then `∫_{x(0)}^{x(1)} dz / v(z) = 1`, i.e. with
`G(y) = ∫_0^y dz / v(z)` the rollout map is `Φ(x) = G⁻¹(G(x) + 1)`. -/
theorem travel_time (hv : ContinuousOn v (Ioo p q)) (hpos : ∀ y ∈ Ioo p q, 0 < v y)
    {x : ℝ → ℝ} (hx : IsRollout1D v x) (hin : ∀ ρ ∈ Icc (0 : ℝ) 1, x ρ ∈ Ioo p q) :
    ∫ y in x 0..x 1, 1 / v y = 1 := by
  have h0 := hin 0 (by norm_num)
  set G : ℝ → ℝ := fun y => ∫ t in x 0..y, 1 / v t
  have hG : ∀ y ∈ Ioo p q, HasDerivAt G (1 / v y) y := fun y hy =>
    intervalIntegral.integral_hasDerivAt_right (inv_intervalIntegrable hv hpos h0 hy)
      ((inv_continuousOn hv hpos).stronglyMeasurableAtFilter isOpen_Ioo y hy)
      ((inv_continuousOn hv hpos).continuousAt (isOpen_Ioo.mem_nhds hy))
  have hGc : ContinuousOn G (Ioo p q) := fun y hy => (hG y hy).continuousAt.continuousWithinAt
  have hcomp : ContinuousOn (G ∘ x) (Icc 0 1) := hGc.comp hx.1 (fun ρ hρ => hin ρ hρ)
  have hd : ∀ ρ ∈ Ioo (0 : ℝ) 1, HasDerivAt (G ∘ x) 1 ρ := by
    intro ρ hρ
    have hxin := hin ρ (Ioo_subset_Icc_self hρ)
    have := (hG (x ρ) hxin).comp ρ (hx.2 ρ hρ)
    convert this using 1
    field_simp [(hpos _ hxin).ne']
  have := intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le zero_le_one hcomp hd
    intervalIntegrable_const
  simp only [intervalIntegral.integral_const, sub_zero, smul_eq_mul, mul_one,
    Function.comp, G, intervalIntegral.integral_same] at this
  linarith

/-- If the travel time from `x(0)` to `T` is less than 1, the rollout passes `T`. -/
theorem overshoot_of_lt (hv : ContinuousOn v (Ioo p q)) (hpos : ∀ y ∈ Ioo p q, 0 < v y)
    {x : ℝ → ℝ} (hx : IsRollout1D v x) (hin : ∀ ρ ∈ Icc (0 : ℝ) 1, x ρ ∈ Ioo p q)
    {T : ℝ} (hT : T ∈ Ioo p q) (h : ∫ y in x 0..T, 1 / v y < 1) : T < x 1 := by
  by_contra hle
  push Not at hle
  have h0 := hin 0 (by norm_num)
  have h1 := hin 1 (by norm_num)
  have hsplit := intervalIntegral.integral_add_adjacent_intervals
    (inv_intervalIntegrable hv hpos h0 h1) (inv_intervalIntegrable hv hpos h1 hT)
  have hnn : 0 ≤ ∫ y in x 1..T, 1 / v y := intervalIntegral.integral_nonneg hle fun u hu =>
    (one_div_pos.2 (hpos u (ordConnected_Ioo.out h1 hT hu))).le
  rw [travel_time hv hpos hx hin] at hsplit
  linarith

/-- If the travel time from `x(0)` to `T` exceeds 1, the rollout stops short of `T`. -/
theorem undershoot_of_gt (hv : ContinuousOn v (Ioo p q)) (hpos : ∀ y ∈ Ioo p q, 0 < v y)
    {x : ℝ → ℝ} (hx : IsRollout1D v x) (hin : ∀ ρ ∈ Icc (0 : ℝ) 1, x ρ ∈ Ioo p q)
    {T : ℝ} (hT : T ∈ Ioo p q) (h : 1 < ∫ y in x 0..T, 1 / v y) : x 1 < T := by
  by_contra hle
  push Not at hle
  have h0 := hin 0 (by norm_num)
  have h1 := hin 1 (by norm_num)
  have hsplit := intervalIntegral.integral_add_adjacent_intervals
    (inv_intervalIntegrable hv hpos h0 hT) (inv_intervalIntegrable hv hpos hT h1)
  have hnn : 0 ≤ ∫ y in T..x 1, 1 / v y := intervalIntegral.integral_nonneg hle fun u hu =>
    (one_div_pos.2 (hpos u (ordConnected_Ioo.out hT h1 hu))).le
  rw [travel_time hv hpos hx hin] at hsplit
  linarith

/-- **`Φ` is strictly increasing**: rollouts in `I` keep the order of their starts. -/
theorem rollout_strictMono (hv : ContinuousOn v (Ioo p q)) (hpos : ∀ y ∈ Ioo p q, 0 < v y)
    {x y : ℝ → ℝ} (hx : IsRollout1D v x) (hy : IsRollout1D v y)
    (hxin : ∀ ρ ∈ Icc (0 : ℝ) 1, x ρ ∈ Ioo p q) (hyin : ∀ ρ ∈ Icc (0 : ℝ) 1, y ρ ∈ Ioo p q)
    (h : x 0 < y 0) : x 1 < y 1 := by
  by_contra hle
  push Not at hle
  have hx0 := hxin 0 (by norm_num); have hx1 := hxin 1 (by norm_num)
  have hy0 := hyin 0 (by norm_num); have hy1 := hyin 1 (by norm_num)
  have e1 := intervalIntegral.integral_add_adjacent_intervals
    (inv_intervalIntegrable hv hpos hx0 hy0) (inv_intervalIntegrable hv hpos hy0 hx1)
  have e2 := intervalIntegral.integral_add_adjacent_intervals
    (inv_intervalIntegrable hv hpos hy0 hy1) (inv_intervalIntegrable hv hpos hy1 hx1)
  have hp1 : 0 < ∫ t in x 0..y 0, 1 / v t :=
    intervalIntegral.intervalIntegral_pos_of_pos_on (inv_intervalIntegrable hv hpos hx0 hy0)
      (fun u hu => one_div_pos.2 (hpos u (ordConnected_Ioo.out hx0 hy0 (Ioo_subset_Icc_self hu))))
      h
  have hp2 : 0 ≤ ∫ t in y 1..x 1, 1 / v t := intervalIntegral.integral_nonneg hle fun u hu =>
    (one_div_pos.2 (hpos u (ordConnected_Ioo.out hy1 hx1 hu))).le
  rw [travel_time hv hpos hx hxin] at e1
  rw [travel_time hv hpos hy hyin] at e2
  linarith

/-- **The rollout stays in `I`** when its speed there is positive and at most `M` and
`x(0) + M` is still inside `I`. -/
theorem rollout_stays_in {M : ℝ} (hM : 0 ≤ M) (hpos : ∀ y ∈ Ioo p q, 0 < v y)
    (hle : ∀ y ∈ Ioo p q, v y ≤ M) {x : ℝ → ℝ} (hx : IsRollout1D v x)
    (h0 : x 0 ∈ Ioo p q) (hq : x 0 + M < q) : ∀ ρ ∈ Icc (0 : ℝ) 1, x ρ ∈ Ioo p q := by
  by_contra hcon
  push Not at hcon
  set S := {ρ | ρ ∈ Icc (0 : ℝ) 1 ∧ x ρ ∉ Ioo p q}
  have hne : S.Nonempty := let ⟨ρ, h1, h2⟩ := hcon; ⟨ρ, h1, h2⟩
  have hcl : IsClosed S := by
    exact hx.1.preimage_isClosed_of_isClosed isClosed_Icc
      (isOpen_Ioo (a := p) (b := q)).isClosed_compl
  have hbdd : BddBelow S := ⟨0, fun ρ hρ => hρ.1.1⟩
  set τ := sInf S
  have hτ : τ ∈ S := hcl.csInf_mem hne hbdd
  have hτ0 : 0 < τ := by
    rcases eq_or_lt_of_le hτ.1.1 with h | h
    · exfalso; apply hτ.2; rw [← h]; exact h0
    · exact h
  have hbefore : ∀ ρ ∈ Ico (0 : ℝ) τ, x ρ ∈ Ioo p q := by
    intro ρ hρ
    by_contra hn
    have : ρ ∈ S := ⟨⟨hρ.1, (hρ.2.le.trans hτ.1.2)⟩, hn⟩
    exact absurd (csInf_le hbdd this) (not_le.2 hρ.2)
  obtain ⟨c, hc, hslope⟩ := exists_hasDerivAt_eq_slope x (fun ρ => v (x ρ)) hτ0
    (hx.1.mono (Icc_subset_Icc le_rfl hτ.1.2))
    (fun ρ hρ => hx.2 ρ ⟨hρ.1, hρ.2.trans_le hτ.1.2⟩)
  have hxc := hbefore c ⟨hc.1.le, hc.2⟩
  have hv1 := hpos _ hxc
  have hv2 := hle _ hxc
  rw [hslope, sub_zero] at hv1 hv2
  have hlow : x 0 < x τ := by
    have := (div_pos_iff_of_pos_right hτ0).1 hv1; linarith
  have hup : x τ ≤ x 0 + M := by
    rw [div_le_iff₀ hτ0] at hv2
    nlinarith [hτ.1.2]
  exact hτ.2 ⟨h0.1.trans hlow, hup.trans_lt hq⟩

end Travel

/-! ### The data of Proposition 2(a) -/

section Prop2a

variable {δ : ℝ}

/-- The map `T`: `+2` on the first block, `+4` on the second. -/
noncomputable def Tmap (a : ℝ) : ℝ := if a < 1 / 2 then a + 2 else a + 4

def block1 (δ : ℝ) : Set ℝ := Icc (-δ) δ
def block2 (δ : ℝ) : Set ℝ := Icc (1 - δ) (1 + δ)

lemma Tmap_block1 (hδ : δ < 1 / 2) {a : ℝ} (ha : a ∈ block1 δ) : Tmap a = a + 2 := by
  have : a < 1 / 2 := lt_of_le_of_lt ha.2 hδ
  unfold Tmap; exact ite_eq_left this

lemma Tmap_block2 (hδ : δ < 1 / 2) {a : ℝ} (ha : a ∈ block2 δ) : Tmap a = a + 4 := by
  have : ¬ a < 1 / 2 := by have := ha.1; linarith
  unfold Tmap; exact ite_eq_right this

/-- `T` sends the first block below the second: `T a ≤ 2 + δ < 5 - δ ≤ T b`. -/
theorem T_block_images (hδ : δ < 1 / 2) {a b : ℝ} (ha : a ∈ block1 δ) (hb : b ∈ block2 δ) :
    Tmap a ≤ 2 + δ ∧ 2 + δ < 5 - δ ∧ 5 - δ ≤ Tmap b := by
  rw [Tmap_block1 hδ ha, Tmap_block2 hδ hb]
  exact ⟨by linarith [ha.2], by linarith, by linarith [hb.1]⟩

/-- `T` is increasing on the support of `P_S` (hence, in 1-D, the Brenier map). -/
theorem T_strictMonoOn (hδ : δ < 1 / 2) : StrictMonoOn Tmap (block1 δ ∪ block2 δ) := by
  intro a ha b hb hab
  rcases ha with ha | ha <;> rcases hb with hb | hb
  · rw [Tmap_block1 hδ ha, Tmap_block1 hδ hb]; linarith
  · have := T_block_images hδ ha hb; linarith
  · exfalso; have := ha.1; have := hb.2; linarith
  · rw [Tmap_block2 hδ ha, Tmap_block2 hδ hb]; linarith

/-- At phase `ρ` the groups lie in `[2ρ - δ, 2ρ + δ]` and `[1 + 4ρ - δ, 1 + 4ρ + δ]`, which are
disjoint: every first-group state is below every second-group state. -/
theorem groups_disjoint_at_phase (hδ : δ < 1 / 2) {a b ρ : ℝ} (ha : a ∈ block1 δ)
    (hb : b ∈ block2 δ) (hρ : 0 ≤ ρ) : a + 2 * ρ < b + 4 * ρ := by
  have := ha.2; have := hb.1; linarith

/-- The straight demonstration from `s`. -/
noncomputable def demo2a (s ρ : ℝ) : ℝ := s + ρ * (Tmap s - s)

/-- **No label conflict.**  Two demonstrations at the same state and phase have the same
start (hence the same velocity `T s - s ∈ {2, 4}`), so `C = 0` and `v̄` is `2` on the region
swept by the first group and `4` on the region swept by the second. -/
theorem prop2a_no_conflict (hδ : δ < 1 / 2) {s s' ρ : ℝ} (hs : s ∈ block1 δ ∪ block2 δ)
    (hs' : s' ∈ block1 δ ∪ block2 δ) (hρ : 0 ≤ ρ) (h : demo2a s ρ = demo2a s' ρ) : s = s' := by
  unfold demo2a at h
  rcases hs with hs | hs <;> rcases hs' with hs' | hs'
  · rw [Tmap_block1 hδ hs, Tmap_block1 hδ hs'] at h; linarith
  · rw [Tmap_block1 hδ hs, Tmap_block2 hδ hs'] at h
    have := groups_disjoint_at_phase hδ hs hs' hρ; linarith
  · rw [Tmap_block2 hδ hs, Tmap_block1 hδ hs'] at h
    have := groups_disjoint_at_phase hδ hs' hs hρ; linarith
  · rw [Tmap_block2 hδ hs, Tmap_block2 hδ hs'] at h; linarith

/-- **Occupation of one pass.**  A pass `a + cρ`, `ρ ∈ [0,1]`, at speed `c > 0` spends time
`1/c` per unit length: for `φ` vanishing outside `(a, a + c)`,
`∫₀¹ φ(a + cρ) dρ = (1/c) ∫ φ`. -/
theorem occupation_per_demo {c a : ℝ} (hc : 0 < c) {φ : ℝ → ℝ}
    (hφ0 : ∀ x ∉ Ioo a (a + c), φ x = 0) :
    ∫ ρ in (0 : ℝ)..1, φ (a + c * ρ) = (1 / c) * ∫ x, φ x := by
  have h := intervalIntegral.integral_comp_mul_add (a := 0) (b := 1) φ hc.ne' a
  simp only [mul_zero, zero_add, mul_one] at h
  have e : (fun ρ => φ (a + c * ρ)) = fun ρ => φ (c * ρ + a) := by funext ρ; rw [add_comm]
  rw [e, h, smul_eq_mul, one_div, intervalIntegral.integral_of_le (by linarith)]
  congr 1
  rw [setIntegral_eq_integral_of_forall_compl_eq_zero]
  intro x hx
  apply hφ0
  intro hx'
  exact hx ⟨hx'.1, by rw [add_comm]; exact hx'.2.le⟩

/-- **Occupation density of a group.**  A group of mass `w` with starts uniform on
`[s₀ - δ, s₀ + δ]` moving at speed `c` has occupation density `w / c` on the region
`(s₀ + δ, s₀ + c - δ)` that all its demonstrations cross. -/
theorem group_occupation {w c s₀ : ℝ} (hδ : 0 < δ) (hc : 0 < c) {φ : ℝ → ℝ}
    (hφ0 : ∀ x ∉ Ioo (s₀ + δ) (s₀ + c - δ), φ x = 0) :
    w * ((2 * δ)⁻¹ * ∫ a in (s₀ - δ)..(s₀ + δ), ∫ ρ in (0 : ℝ)..1, φ (a + c * ρ)) =
      (w / c) * ∫ x, φ x := by
  have hconst : ∀ a ∈ uIcc (s₀ - δ) (s₀ + δ),
      ∫ ρ in (0 : ℝ)..1, φ (a + c * ρ) = (1 / c) * ∫ x, φ x := by
    intro a ha
    rw [uIcc_of_le (by linarith)] at ha
    refine occupation_per_demo hc fun x hx => hφ0 x fun hx' => hx ⟨?_, ?_⟩
    · linarith [ha.2, hx'.1]
    · linarith [ha.1, hx'.2]
  rw [intervalIntegral.integral_congr hconst, intervalIntegral.integral_const, smul_eq_mul]
  field_simp
  ring

/-- The two densities on the overlap `(1 + δ, 2 - δ)`: `μ₁ = ½ · ½`, `μ₂ = ½ · ¼`. -/
theorem occupation_values : (1 / 2 : ℝ) / 2 = 1 / 4 ∧ (1 / 2 : ℝ) / 4 = 1 / 8 := by norm_num

/-- `v̄_free = (2μ₁ + 4μ₂)/(μ₁ + μ₂)` (Eq. (8) with two non-overlapping groups) is `2` where
only the first group passes, `4` where only the second does, and strictly between where
both pass. -/
theorem vfree_convex {μ₁ μ₂ : ℝ} (h1 : 0 ≤ μ₁) (h2 : 0 ≤ μ₂) (hs : 0 < μ₁ + μ₂) :
    (μ₂ = 0 → (2 * μ₁ + 4 * μ₂) / (μ₁ + μ₂) = 2) ∧
    (μ₁ = 0 → (2 * μ₁ + 4 * μ₂) / (μ₁ + μ₂) = 4) ∧
    (0 < μ₁ → 0 < μ₂ → 2 < (2 * μ₁ + 4 * μ₂) / (μ₁ + μ₂) ∧
      (2 * μ₁ + 4 * μ₂) / (μ₁ + μ₂) < 4) := by
  refine ⟨fun h => ?_, fun h => ?_, fun p1 p2 => ⟨?_, ?_⟩⟩
  · subst h; rw [div_eq_iff hs.ne']; ring
  · subst h; rw [div_eq_iff hs.ne']; ring
  · rw [lt_div_iff₀ hs]; linarith
  · rw [div_lt_iff₀ hs]; linarith

/-- On the overlap, `v̄_free = 8/3 ≠ v̄ ∈ {2, 4}`. -/
theorem vfree_overlap : (2 * (1 / 4) + 4 * (1 / 8) : ℝ) / (1 / 4 + 1 / 8) = 8 / 3 := by norm_num

/-- `A → ¼ (2 - 8/3)² + ⅛ (4 - 8/3)² = ⅓` as `δ → 0`. -/
theorem aliasing_limit : (1 / 4 : ℝ) * (2 - 8 / 3) ^ 2 + 1 / 8 * (4 - 8 / 3) ^ 2 = 1 / 3 := by
  norm_num

/-- The profile of `v̄_free` derived above, on `I = (-δ, 5 + δ)`. -/
structure VfreeProfile (δ : ℝ) (v : ℝ → ℝ) : Prop where
  cont : ContinuousOn v (Ioo (-δ) (5 + δ))
  low : ∀ y ∈ Ioc (-δ) (1 - δ), v y = 2
  mid : ∀ y ∈ Ioo (1 - δ) (2 + δ), 2 < v y ∧ v y < 4
  high : ∀ y ∈ Ico (2 + δ) (5 + δ), v y = 4

lemma VfreeProfile.bounds (_hδ : 0 < δ) (_hδ' : δ < 1 / 2) {v : ℝ → ℝ} (h : VfreeProfile δ v)
    {y : ℝ} (hy : y ∈ Ioo (-δ) (5 + δ)) : 2 ≤ v y ∧ v y ≤ 4 := by
  rcases le_or_gt y (1 - δ) with h1 | h1
  · rw [h.low y ⟨hy.1, h1⟩]; norm_num
  rcases lt_or_ge y (2 + δ) with h2 | h2
  · have := h.mid y ⟨h1, h2⟩; exact ⟨this.1.le, this.2.le⟩
  · rw [h.high y ⟨h2, hy.2⟩]; norm_num

/-- **Overshoot.**  For a start `a ∈ (-δ, δ]` of the first group, the rollout of `v̄_free`
reaches `T(a) = a + 2` before `ρ = 1` and moves on: `Φ(a) > T(a)`. -/
theorem prop2a_overshoot (hδ : 0 < δ) (hδ' : δ < 1 / 2) {v : ℝ → ℝ} (hv : VfreeProfile δ v)
    {x : ℝ → ℝ} (hx : IsRollout1D v x) (ha : x 0 ∈ Ioc (-δ) δ) :
    Tmap (x 0) < x 1 := by
  have hpos : ∀ y ∈ Ioo (-δ) (5 + δ), 0 < v y := fun y hy =>
    by linarith [(hv.bounds hδ hδ' hy).1]
  have hin := rollout_stays_in (M := 4) (by norm_num) hpos
    (fun y hy => (hv.bounds hδ hδ' hy).2) hx ⟨ha.1, by linarith [ha.2]⟩ (by linarith [ha.2])
  set a := x 0
  rw [Tmap_block1 hδ' ⟨ha.1.le, ha.2⟩]
  have hsub : Icc a (a + 2) ⊆ Ioo (-δ) (5 + δ) := fun y hy =>
    ⟨ha.1.trans_le hy.1, by linarith [hy.2, ha.2]⟩
  refine overshoot_of_lt hv.cont hpos hx hin (hsub ⟨by linarith, le_rfl⟩) ?_
  have hlt : ∫ y in a..a + 2, 1 / v y < ∫ y in a..a + 2, (1 / 2 : ℝ) := by
    apply intervalIntegral.integral_lt_integral_of_continuousOn_of_le_of_exists_lt (by linarith)
    · exact (inv_continuousOn hv.cont hpos).mono hsub
    · exact continuousOn_const
    · intro y hy
      have := (hv.bounds hδ hδ' (hsub (Ioc_subset_Icc_self hy))).1
      rw [div_le_div_iff₀ (by linarith) (by norm_num)]; linarith
    · refine ⟨(1 - δ + (a + 2)) / 2, ⟨by linarith [ha.1, ha.2], by linarith [ha.1, ha.2]⟩, ?_⟩
      have := (hv.mid ((1 - δ + (a + 2)) / 2)
        ⟨by linarith [ha.1, ha.2], by linarith [ha.1, ha.2]⟩).1
      rw [div_lt_div_iff₀ (by linarith) (by norm_num)]; linarith
  simpa using hlt

/-- **Undershoot.**  For a start `b ∈ [1-δ, 1+δ)` of the second group, `Φ(b) < T(b)`. -/
theorem prop2a_undershoot (hδ : 0 < δ) (hδ' : δ < 1 / 2) {v : ℝ → ℝ} (hv : VfreeProfile δ v)
    {x : ℝ → ℝ} (hx : IsRollout1D v x) (hb : x 0 ∈ Ico (1 - δ) (1 + δ)) :
    x 1 < Tmap (x 0) := by
  have hpos : ∀ y ∈ Ioo (-δ) (5 + δ), 0 < v y := fun y hy =>
    by linarith [(hv.bounds hδ hδ' hy).1]
  have hin := rollout_stays_in (M := 4) (by norm_num) hpos
    (fun y hy => (hv.bounds hδ hδ' hy).2) hx ⟨by linarith [hb.1], by linarith [hb.2]⟩
    (by linarith [hb.2])
  set b := x 0
  rw [Tmap_block2 hδ' ⟨hb.1, hb.2.le⟩]
  have hsub : Icc b (b + 4) ⊆ Ioo (-δ) (5 + δ) := fun y hy =>
    ⟨by linarith [hy.1, hb.1], by linarith [hy.2, hb.2]⟩
  refine undershoot_of_gt hv.cont hpos hx hin (hsub ⟨by linarith, le_rfl⟩) ?_
  have hlt : ∫ y in b..b + 4, (1 / 4 : ℝ) < ∫ y in b..b + 4, 1 / v y := by
    apply intervalIntegral.integral_lt_integral_of_continuousOn_of_le_of_exists_lt (by linarith)
    · exact continuousOn_const
    · exact (inv_continuousOn hv.cont hpos).mono hsub
    · intro y hy
      have h := hv.bounds hδ hδ' (hsub (Ioc_subset_Icc_self hy))
      rw [div_le_div_iff₀ (by norm_num) (by linarith)]; linarith
    · have hcin : (b + (2 + δ)) / 2 ∈ Icc b (b + 4) :=
        ⟨by linarith [hb.1, hb.2], by linarith [hb.1, hb.2]⟩
      refine ⟨(b + (2 + δ)) / 2, hcin, ?_⟩
      have := (hv.mid ((b + (2 + δ)) / 2) ⟨by linarith [hb.1, hb.2], by linarith [hb.2]⟩).2
      have hp := hpos _ (hsub hcin)
      rw [div_lt_div_iff₀ (by norm_num) hp]; linarith
  simpa using hlt

/-- **The distribution functions differ (abstract form).**  If `Φ` and `T` are strictly
increasing on a full-measure set `S`, `Φ(b) < T(x)` for the points `x ∈ S ∩ (c, b]`, and
that set has positive mass, then `Pr(T(x₀) ≤ Φ(b)) < Pr(Φ(x₀) ≤ Φ(b))`. -/
theorem cdf_lt_of_undershoot (P : Measure ℝ) [IsFiniteMeasure P] {S : Set ℝ}
    (hS : MeasurableSet S) (hSc : P Sᶜ = 0) {Φ T : ℝ → ℝ} (hΦ : StrictMonoOn Φ S)
    (hT : StrictMonoOn T S) {b c : ℝ} (hb : b ∈ S) (hcb : c < b)
    (hTc : ∀ x ∈ S, c < x → x ≤ b → Φ b < T x) (hpos : 0 < P (S ∩ Ioc c b)) :
    P {x | T x ≤ Φ b} < P {x | Φ x ≤ Φ b} := by
  have e1 : P {x | Φ x ≤ Φ b} = P (Iic b ∩ S) := by
    rw [← measure_inter_conull hSc]
    congr 1; ext x
    simp only [mem_inter_iff, mem_ofPred_eq, mem_Iic]
    exact ⟨fun h => ⟨hΦ.le_iff_le h.2 hb |>.1 h.1, h.2⟩,
      fun h => ⟨hΦ.le_iff_le h.2 hb |>.2 h.1, h.2⟩⟩
  have e2 : P {x | T x ≤ Φ b} ≤ P (Iic c ∩ S) := by
    rw [← measure_inter_conull hSc]
    apply measure_mono
    intro x ⟨hx, hxS⟩
    refine ⟨?_, hxS⟩
    by_contra hcx
    simp only [mem_Iic, not_le] at hcx
    simp only [mem_ofPred_eq] at hx
    rcases le_or_gt x b with hxb | hxb
    · exact absurd hx (not_le.2 (hTc x hxS hcx hxb))
    · have h1 := hTc b hb hcb le_rfl
      have h2 := hT hb hxS hxb
      linarith
  have e3 : P (Iic b ∩ S) = P (Iic c ∩ S) + P (S ∩ Ioc c b) := by
    rw [← measure_union]
    · congr 1; ext x
      simp only [mem_inter_iff, mem_Iic, mem_union, mem_Ioc]
      constructor
      · intro ⟨h1, h2⟩
        rcases le_or_gt x c with h | h
        · exact Or.inl ⟨h, h2⟩
        · exact Or.inr ⟨h2, h, h1⟩
      · rintro (⟨h1, h2⟩ | ⟨h1, h2, h3⟩)
        · exact ⟨h1.trans hcb.le, h2⟩
        · exact ⟨h3, h1⟩
    · rw [Set.disjoint_left]
      intro x hx hx'
      exact absurd hx.1 (not_le.2 hx'.2.1)
    · exact hS.inter measurableSet_Ioc
  rw [e1, e3]
  calc P {x | T x ≤ Φ b} ≤ P (Iic c ∩ S) := e2
    _ < P (Iic c ∩ S) + P (S ∩ Ioc c b) :=
        ENNReal.lt_add_right (measure_ne_top _ _) hpos.ne'

/-- `P_S = ½ U[-δ, δ] + ½ U[1-δ, 1+δ]`, i.e. density `1/(4δ)` on the two blocks. -/
noncomputable def PS (δ : ℝ) : Measure ℝ :=
  (1 / (4 * δ)).toNNReal • volume.restrict (block1 δ ∪ block2 δ)

instance (δ : ℝ) : IsFiniteMeasure (PS δ) := by
  unfold PS
  have : IsFiniteMeasure (volume.restrict (block1 δ ∪ block2 δ)) :=
    isFiniteMeasure_restrict.2 (measure_union_ne_top (by simp [block1]) (by simp [block2]))
  infer_instance

/-- **`Φ_# P_S ≠ P_G`.**  Let `X s` be a rollout of `v̄_free` from each start `s` of `P_S`
and `Φ(s) = X s 1`.  For every `b ∈ (1 - δ, 1 + δ)`,
`F_G(Φ(b)) = Pr(T(x₀) ≤ Φ(b)) < Pr(Φ(x₀) ≤ Φ(b)) = F_{Φ_# P_S}(Φ(b))`. -/
theorem prop2a_not_PG (hδ : 0 < δ) (hδ' : δ < 1 / 2) {v : ℝ → ℝ} (hv : VfreeProfile δ v)
    (X : ℝ → ℝ → ℝ) (hX : ∀ s ∈ block1 δ ∪ block2 δ, IsRollout1D v (X s) ∧ X s 0 = s)
    {b : ℝ} (hb : b ∈ Ioo (1 - δ) (1 + δ)) :
    PS δ {x | Tmap x ≤ X b 1} < PS δ {x | X x 1 ≤ X b 1} := by
  set S := block1 δ ∪ block2 δ
  have hSm : MeasurableSet S := measurableSet_Icc.union measurableSet_Icc
  have hpos : ∀ y ∈ Ioo (-δ) (5 + δ), 0 < v y := fun y hy =>
    by linarith [(hv.bounds hδ hδ' hy).1]
  -- the two endpoints `-δ`, `1 + δ` are a null set; work on `S' = S \ {-δ, 1 + δ}`
  set S' := S \ {-δ, 1 + δ}
  have hS'm : MeasurableSet S' := hSm.diff (Set.toFinite _).measurableSet
  have hin : ∀ s ∈ S', ∀ ρ ∈ Icc (0 : ℝ) 1, X s ρ ∈ Ioo (-δ) (5 + δ) := by
    intro s hs
    have hs0 := (hX s hs.1).2
    have hne1 : s ≠ -δ := fun h => hs.2 (Or.inl h)
    have hne2 : s ≠ 1 + δ := fun h => hs.2 (Or.inr h)
    refine rollout_stays_in (M := 4) (by norm_num) hpos (fun y hy => (hv.bounds hδ hδ' hy).2)
      (hX s hs.1).1 ?_ ?_
    · rw [hs0]
      rcases hs.1 with h | h
      · exact ⟨lt_of_le_of_ne h.1 (Ne.symm hne1), by linarith [h.2]⟩
      · exact ⟨by linarith [h.1], by linarith [h.2]⟩
    · rw [hs0]
      rcases hs.1 with h | h
      · linarith [h.2]
      · linarith [lt_of_le_of_ne h.2 hne2]
  have hS'c : PS δ S'ᶜ = 0 := by
    unfold PS
    rw [Measure.smul_apply, Measure.restrict_apply hS'm.compl]
    have : volume (S'ᶜ ∩ S) = 0 := by
      apply measure_mono_null (t := {-δ, 1 + δ})
      · intro x ⟨hx1, hx2⟩
        by_contra hne
        exact hx1 ⟨hx2, hne⟩
      · exact (Set.toFinite _).measure_zero _
    rw [this, smul_zero]
  have hΦ : StrictMonoOn (fun s => X s 1) S' := by
    intro s hs t ht hst
    have h1 := hX s hs.1; have h2 := hX t ht.1
    refine rollout_strictMono hv.cont hpos h1.1 h2.1 (hin s hs) (hin t ht) ?_
    rw [h1.2, h2.2]; exact hst
  have hT : StrictMonoOn Tmap S' := (T_strictMonoOn hδ').mono sdiff_subset
  have hbS : b ∈ S := Or.inr ⟨hb.1.le, hb.2.le⟩
  have hbS' : b ∈ S' := by
    refine ⟨hbS, ?_⟩
    rintro (h | h)
    · linarith [hb.1]
    · rw [Set.mem_singleton_iff] at h; linarith [hb.2]
  have hund : X b 1 < Tmap b := by
    have h := prop2a_undershoot hδ hδ' hv (hX b hbS).1
      (by rw [(hX b hbS).2]; exact ⟨hb.1.le, hb.2⟩)
    rwa [(hX b hbS).2] at h
  rw [Tmap_block2 hδ' ⟨hb.1.le, hb.2.le⟩] at hund
  set c := max (1 - δ) (X b 1 - 4)
  have hcb : c < b := max_lt hb.1 (by linarith)
  refine cdf_lt_of_undershoot (PS δ) hS'm hS'c hΦ hT hbS' hcb ?_ ?_
  · intro x _ hcx hxb
    have hx2 : x ∈ block2 δ := ⟨(le_max_left _ _).trans hcx.le, hxb.trans hb.2.le⟩
    rw [Tmap_block2 hδ' hx2]
    linarith [le_max_right (1 - δ) (X b 1 - 4)]
  · unfold PS
    rw [Measure.smul_apply, Measure.restrict_apply (hS'm.inter measurableSet_Ioc),
      ENNReal.smul_def, smul_eq_mul]
    refine ENNReal.mul_pos (by simp; positivity) ?_
    have hsub : Ioo c b ⊆ S' ∩ Ioc c b ∩ S := by
      intro x hx
      have hx2 : x ∈ block2 δ := ⟨(le_max_left _ _).trans hx.1.le, hx.2.le.trans hb.2.le⟩
      refine ⟨⟨⟨Or.inr hx2, ?_⟩, hx.1, hx.2.le⟩, Or.inr hx2⟩
      rintro (h | h)
      · linarith [hx2.1]
      · rw [Set.mem_singleton_iff] at h; linarith [hx.2, hb.2]
    refine (lt_of_lt_of_le ?_ (measure_mono hsub)).ne'
    rw [Real.volume_Ioo]; simp [hcb]

/-- **The limit `δ → 0`.**  `v̄_free` is `2` on `[0,1)`, `8/3` on `(1,2)`, `4` on `(2,5]`.  The
rollout from `0` spends `1/2` on `[0,1]` and `3/8` on `[1,2]` and ends at `2 + 4·⅛ = 2.5`;
the rollout from `1` spends `3/8` on `[1,2]` and ends at `2 + 4·⅝ = 4.5`. -/
theorem prop2a_limit :
    (1 : ℝ) / 2 = 1 / 2 ∧ (1 : ℝ) / (8 / 3) = 3 / 8 ∧
    (2 : ℝ) + 4 * (1 - 1 / 2 - 3 / 8) = 2.5 ∧ (2 : ℝ) + 4 * (1 - 3 / 8) = 4.5 ∧
    (2 : ℝ) < 2.5 ∧ (4.5 : ℝ) < 5 := by
  norm_num

end Prop2a

/-! ### Proposition 2(b): stall -/

section Prop2b

/-- The demonstration `x_ρ = x₀ + sin(πρ)`. -/
noncomputable def demo2b (x0 ρ : ℝ) : ℝ := x0 + sin (π * ρ)

theorem prop2b_velocity (x0 ρ : ℝ) : HasDerivAt (demo2b x0) (π * cos (π * ρ)) ρ := by
  have := ((hasDerivAt_id' ρ).const_mul π).sin.const_add x0
  exact this.congr_deriv (by ring)

/-- **No label conflict.**  At phase `ρ` all demonstrations are shifted by the same
`sin(πρ)`, so distinct starts never meet at the same phase; every demonstration has
velocity `π cos(πρ)`, so `C = 0` and `v̄(x, ρ) = π cos(πρ)`. -/
theorem prop2b_no_conflict {x0 x0' ρ : ℝ} (h : demo2b x0 ρ = demo2b x0' ρ) : x0 = x0' := by
  unfold demo2b at h; linarith

/-- `v̄(x, ρ) = π cos(πρ)` is `0`-Lipschitz in `x`, and each demonstration is its rollout. -/
theorem prop2b_demo_is_rollout (x0 : ℝ) :
    LipschitzInState (fun (_ : ℝ) ρ => π * cos (π * ρ)) (fun _ => 0) 0 1 ∧
    IsRollout (fun (_ : ℝ) ρ => π * cos (π * ρ)) (demo2b x0) 0 1 := by
  refine ⟨fun ρ _ x y => by simp, ?_, ?_⟩
  · exact fun ρ _ => (prop2b_velocity x0 ρ).continuousAt.continuousWithinAt
  · exact fun ρ _ => (prop2b_velocity x0 ρ).hasDerivWithinAt

/-- **`v̄_free = 0`.**  Each demonstration goes out and comes back, so its contribution to
the time-integrated flux `∫ φ · μ v̄_free` vanishes for every continuous `φ`
(a pass at speed `c` spends time `∝ 1/c` near each state and contributes `±1`; the outward
and return passes cancel).  Equivalently `μ v̄_free = F_S - F_G = 0`. -/
theorem prop2b_flux_zero (x0 : ℝ) {φ : ℝ → ℝ} (hφ : Continuous φ) :
    ∫ ρ in (0 : ℝ)..1, φ (demo2b x0 ρ) * (π * cos (π * ρ)) = 0 := by
  have h := intervalIntegral.integral_comp_mul_deriv (f := demo2b x0)
    (f' := fun ρ => π * cos (π * ρ)) (g := φ) (a := 0) (b := 1)
    (fun ρ _ => prop2b_velocity x0 ρ) (by fun_prop) hφ
  simp only [Function.comp] at h
  rw [h]
  simp [demo2b]

/-- **Stall.**  The rollouts of `v̄_free = 0` never move: the constant path is a rollout and,
the zero field being Lipschitz, it is the only one. -/
theorem prop2b_stall {z : ℝ → ℝ} (hz : IsRollout (fun (_ : ℝ) (_ : ℝ) => (0 : ℝ)) z 0 1) :
    EqOn z (fun _ => z 0) (Icc 0 1) := by
  refine rollouts_eq_of_eq (L := fun _ => 0) continuous_const (fun _ _ _ _ => by simp) hz
    ⟨continuousOn_const, fun ρ _ => hasDerivWithinAt_const _ _ _⟩ rfl

/-- At `ρ = ½` the demonstrations sit in `[1 - δ, 1 + δ]`, disjoint from the support
`[-δ, δ]` of `P_S` where the stalled rollouts are. -/
theorem prop2b_supports_disjoint {δ : ℝ} (hδ' : δ < 1 / 2) :
    Disjoint (Icc (-δ) δ) ((fun x0 => demo2b x0 (1 / 2)) '' Icc (-δ) δ) := by
  rw [Set.disjoint_left]
  rintro x hx ⟨x0, hx0, rfl⟩
  simp only [demo2b, show π * (1 / 2) = π / 2 by ring, sin_pi_div_two] at hx
  linarith [hx.2, hx0.1]

/-- For `ρ ∈ (0, 1)` every demonstration is shifted by `sin(πρ) > 0`, so `p_ρ ≠ P_S`. -/
theorem prop2b_shift_pos {ρ : ℝ} (hρ : ρ ∈ Ioo (0 : ℝ) 1) : 0 < sin (π * ρ) :=
  sin_pos_of_pos_of_lt_pi (by nlinarith [pi_pos, hρ.1]) (by nlinarith [pi_pos, hρ.2])

/-- **`A > 0`.**  Every occupied state is visited at phases `ρ` and `1 - ρ` with opposite
velocities, nonzero for `ρ ≠ ½`, so a phase-free policy cannot fit both. -/
theorem prop2b_aliasing (x0 ρ : ℝ) (hρ : ρ ∈ Ioo (0 : ℝ) (1 / 2)) :
    demo2b x0 ρ = demo2b x0 (1 - ρ) ∧
    π * cos (π * (1 - ρ)) = -(π * cos (π * ρ)) ∧ π * cos (π * ρ) ≠ 0 := by
  refine ⟨?_, ?_, ?_⟩
  · simp only [demo2b, mul_sub, mul_one, sin_pi_sub]
  · rw [mul_sub, mul_one, cos_pi_sub]; ring
  · refine mul_ne_zero pi_ne_zero (cos_pos_of_mem_Ioo ⟨?_, ?_⟩).ne'
    · nlinarith [pi_pos, hρ.1]
    · nlinarith [pi_pos, hρ.2]

end Prop2b

end Phaseformer
