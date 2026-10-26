import Mathlib

/-!
# Grönwall bounds for rollouts (App. B.4, paragraph "Grönwall"; used throughout)

This file formalizes the paragraph "Grönwall" at the start of App. B.4:

* `gronwall_rollouts` — Eq. (13): if `pol` is `L(ρ)`-Lipschitz in the robot state, two
  rollouts satisfy `‖Φ(x) - Φ(y)‖ ≤ exp (∫_{ρ₀}^{ρ₁} L) ‖x - y‖`.
  We assume `L` continuous (the paper only needs it integrable); every use in the
  appendix (`L` bounded on a compact phase interval) satisfies this.
* `rollouts_eq_of_eq` — uniqueness of rollouts (used in Lemma 1(b), Lemma 2(c), Thm 4(iii)).
* `closedLoop_lipschitz` — closed-loop dynamics `ẋ = f(x, pol(x,ρ))` are
  `(L_x + L_u L(ρ))`-Lipschitz, so Eq. (13) holds with `K = exp ∫ (L_x + L_u L)`.
* `euler_lipschitz` — Euler rollouts: `K = ∏ₖ (1 + Δρ (L_x + L_u L(ρₖ)))`.
* `unicycle_Lu`, `unicycle_Lx` — for the unicycle `ẋ = B(θ) u`, `L_u = 1` and
  `L_x = |v|`.

Rollouts follow Mathlib's ODE conventions: continuous on `[ρ₀, ρ₁]`, with a right
derivative on `[ρ₀, ρ₁)`.
-/

namespace Phaseformer

open Set Real
open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `z` is a rollout of the (time-dependent) field `pol` on `[ρ₀, ρ₁]`:
`dz/dρ = pol(z, ρ)` (right derivative), `z` continuous on the closed interval. -/
def IsRollout (pol : E → ℝ → E) (z : ℝ → E) (ρ₀ ρ₁ : ℝ) : Prop :=
  ContinuousOn z (Icc ρ₀ ρ₁) ∧
    ∀ ρ ∈ Ico ρ₀ ρ₁, HasDerivWithinAt z (pol (z ρ) ρ) (Ici ρ) ρ

/-- `pol` is `L(ρ)`-Lipschitz in the robot state for `ρ ∈ [ρ₀, ρ₁]`. -/
def LipschitzInState (pol : E → ℝ → E) (L : ℝ → ℝ) (ρ₀ ρ₁ : ℝ) : Prop :=
  ∀ ρ ∈ Icc ρ₀ ρ₁, ∀ x y : E, ‖pol x ρ - pol y ρ‖ ≤ L ρ * ‖x - y‖

/-- **Eq. (13) (Grönwall).**  If `pol` is `L(ρ)`-Lipschitz with `L` continuous, any two
rollouts `z, z'` satisfy `‖z ρ - z' ρ‖ ≤ exp (∫_{ρ₀}^{ρ} L) · ‖z ρ₀ - z' ρ₀‖`. -/
theorem gronwall_rollouts {pol : E → ℝ → E} {L : ℝ → ℝ} {ρ₀ ρ₁ : ℝ}
    (hL : Continuous L) (hLip : LipschitzInState pol L ρ₀ ρ₁)
    {z z' : ℝ → E} (hz : IsRollout pol z ρ₀ ρ₁) (hz' : IsRollout pol z' ρ₀ ρ₁) :
    ∀ ρ ∈ Icc ρ₀ ρ₁, ‖z ρ - z' ρ‖ ≤ exp (∫ s in ρ₀..ρ, L s) * ‖z ρ₀ - z' ρ₀‖ := by
  set G : ℝ → ℝ := fun ρ => ∫ s in ρ₀..ρ, L s with hG
  have hGd : ∀ ρ, HasDerivAt G (L ρ) ρ := fun ρ =>
    (hL.integral_hasStrictDerivAt ρ₀ ρ).hasDerivAt
  -- `F ρ = ‖z ρ - z' ρ‖² · exp (-2 G ρ)` is non-increasing
  set F : ℝ → ℝ := fun ρ => ‖z ρ - z' ρ‖ ^ 2 * exp (-(2 * G ρ)) with hF
  have hFc : ContinuousOn F (Icc ρ₀ ρ₁) := by
    apply ContinuousOn.mul
    · exact ((hz.1.sub hz'.1).norm).pow 2
    · exact (continuous_exp.comp
        (continuous_const.mul (continuous_iff_continuousAt.2 fun ρ => (hGd ρ).continuousAt)).neg)
        |>.continuousOn
  have hFd : ∀ ρ ∈ Ico ρ₀ ρ₁, HasDerivWithinAt F
      ((2 * ⟪z ρ - z' ρ, pol (z ρ) ρ - pol (z' ρ) ρ⟫) * exp (-(2 * G ρ)) +
        ‖z ρ - z' ρ‖ ^ 2 * (exp (-(2 * G ρ)) * (-(2 * L ρ)))) (Ici ρ) ρ := by
    intro ρ hρ
    have h1 := ((hz.2 ρ hρ).sub (hz'.2 ρ hρ)).norm_sq
    have h2 : HasDerivWithinAt (fun ρ => exp (-(2 * G ρ)))
        (exp (-(2 * G ρ)) * (-(2 * L ρ))) (Ici ρ) ρ :=
      ((((hGd ρ).const_mul 2).neg).exp).hasDerivWithinAt
    exact h1.mul h2
  have hFle : ∀ ρ ∈ Ico ρ₀ ρ₁,
      (2 * ⟪z ρ - z' ρ, pol (z ρ) ρ - pol (z' ρ) ρ⟫) * exp (-(2 * G ρ)) +
        ‖z ρ - z' ρ‖ ^ 2 * (exp (-(2 * G ρ)) * (-(2 * L ρ))) ≤ 0 := by
    intro ρ hρ
    have hcs : ⟪z ρ - z' ρ, pol (z ρ) ρ - pol (z' ρ) ρ⟫ ≤ L ρ * ‖z ρ - z' ρ‖ ^ 2 := by
      calc ⟪z ρ - z' ρ, pol (z ρ) ρ - pol (z' ρ) ρ⟫
          ≤ ‖z ρ - z' ρ‖ * ‖pol (z ρ) ρ - pol (z' ρ) ρ‖ := real_inner_le_norm _ _
        _ ≤ ‖z ρ - z' ρ‖ * (L ρ * ‖z ρ - z' ρ‖) :=
            mul_le_mul_of_nonneg_left (hLip ρ (Ico_subset_Icc_self hρ) _ _) (norm_nonneg _)
        _ = L ρ * ‖z ρ - z' ρ‖ ^ 2 := by ring
    have he := exp_pos (-(2 * G ρ))
    nlinarith
  have key := image_le_of_deriv_right_le_deriv_boundary hFc hFd (B := fun _ => F ρ₀)
    (B' := fun _ => 0) le_rfl continuousOn_const
    (fun _ _ => hasDerivWithinAt_const _ _ _) hFle
  intro ρ hρ
  have hk := key hρ
  simp only [hF, hG, intervalIntegral.integral_same, mul_zero, neg_zero, exp_zero,
    mul_one] at hk
  -- unpack `‖d‖² e^{-2G} ≤ ‖d₀‖²`
  have he : 0 < exp (-(2 * G ρ)) := exp_pos _
  have hsq : ‖z ρ - z' ρ‖ ^ 2 ≤ (exp (G ρ) * ‖z ρ₀ - z' ρ₀‖) ^ 2 := by
    have : exp (-(2 * G ρ)) * exp (2 * G ρ) = 1 := by rw [← exp_add]; simp
    have h2 : exp (2 * G ρ) = exp (G ρ) ^ 2 := by rw [← exp_nat_mul]; push_cast; ring_nf
    calc ‖z ρ - z' ρ‖ ^ 2 = ‖z ρ - z' ρ‖ ^ 2 * exp (-(2 * G ρ)) * exp (2 * G ρ) := by
          rw [mul_assoc, this, mul_one]
      _ ≤ ‖z ρ₀ - z' ρ₀‖ ^ 2 * exp (2 * G ρ) :=
          mul_le_mul_of_nonneg_right hk (exp_pos _).le
      _ = (exp (G ρ) * ‖z ρ₀ - z' ρ₀‖) ^ 2 := by rw [h2]; ring
  exact (pow_le_pow_iff_left₀ (norm_nonneg _)
    (mul_nonneg (exp_pos _).le (norm_nonneg _)) two_ne_zero).1 hsq

/-- **Uniqueness of rollouts.**  Two rollouts of a Lipschitz field that agree at `ρ₀`
agree on `[ρ₀, ρ₁]`.  (Lemma 1(b): flow-matching inference and the closed-loop rollout
solve the same initial-value problem, so they coincide; Lemma 2(c); Theorem 4(iii).) -/
theorem rollouts_eq_of_eq {pol : E → ℝ → E} {L : ℝ → ℝ} {ρ₀ ρ₁ : ℝ}
    (hL : Continuous L) (hLip : LipschitzInState pol L ρ₀ ρ₁)
    {z z' : ℝ → E} (hz : IsRollout pol z ρ₀ ρ₁) (hz' : IsRollout pol z' ρ₀ ρ₁)
    (h0 : z ρ₀ = z' ρ₀) : EqOn z z' (Icc ρ₀ ρ₁) := by
  intro ρ hρ
  have := gronwall_rollouts hL hLip hz hz' ρ hρ
  rw [h0, sub_self, norm_zero, mul_zero] at this
  exact sub_eq_zero.1 (norm_le_zero_iff.1 this)

/-- **Closed-loop dynamics.**  If `f` is `L_x`-Lipschitz in the state and `L_u`-Lipschitz in
the action, and `pol` is `L`-Lipschitz, then `x ↦ f(x, pol x)` is `(L_x + L_u L)`-Lipschitz. -/
theorem closedLoop_lipschitz {X U : Type*} [NormedAddCommGroup X] [NormedAddCommGroup U]
    {F : Type*} [NormedAddCommGroup F]
    (f : X → U → F) (pol : X → U) {Lx Lu L : ℝ} (hLu : 0 ≤ Lu)
    (hfx : ∀ x y u, ‖f x u - f y u‖ ≤ Lx * ‖x - y‖)
    (hfu : ∀ y u u', ‖f y u - f y u'‖ ≤ Lu * ‖u - u'‖)
    (hpol : ∀ x y, ‖pol x - pol y‖ ≤ L * ‖x - y‖) (x y : X) :
    ‖f x (pol x) - f y (pol y)‖ ≤ (Lx + Lu * L) * ‖x - y‖ := by
  calc ‖f x (pol x) - f y (pol y)‖
      = ‖(f x (pol x) - f y (pol x)) + (f y (pol x) - f y (pol y))‖ := by congr 1; abel
    _ ≤ ‖f x (pol x) - f y (pol x)‖ + ‖f y (pol x) - f y (pol y)‖ := norm_add_le _ _
    _ ≤ Lx * ‖x - y‖ + Lu * (L * ‖x - y‖) :=
        add_le_add (hfx _ _ _) ((hfu _ _ _).trans (mul_le_mul_of_nonneg_left (hpol _ _) hLu))
    _ = (Lx + Lu * L) * ‖x - y‖ := by ring

/-- One Euler step `x ↦ x + Δρ g(x)` of a `c`-Lipschitz field is `(1 + Δρ c)`-Lipschitz. -/
theorem euler_step_lipschitz {g : E → E} {c Δ : ℝ} (hΔ : 0 ≤ Δ)
    (hg : ∀ x y, ‖g x - g y‖ ≤ c * ‖x - y‖) (x y : E) :
    ‖(x + Δ • g x) - (y + Δ • g y)‖ ≤ (1 + Δ * c) * ‖x - y‖ := by
  calc ‖(x + Δ • g x) - (y + Δ • g y)‖ = ‖(x - y) + Δ • (g x - g y)‖ := by
        congr 1; rw [smul_sub]; abel
    _ ≤ ‖x - y‖ + Δ * ‖g x - g y‖ := by
        refine (norm_add_le _ _).trans (le_of_eq ?_)
        rw [norm_smul, Real.norm_of_nonneg hΔ]
    _ ≤ ‖x - y‖ + Δ * (c * ‖x - y‖) := by gcongr; exact hg x y
    _ = (1 + Δ * c) * ‖x - y‖ := by ring

/-- The Euler rollout map with steps `g 0, g 1, …` and step size `Δ`. -/
def eulerMap (g : ℕ → E → E) (Δ : ℝ) : ℕ → E → E
  | 0 => id
  | n + 1 => fun x => eulerMap g Δ n x + Δ • g n (eulerMap g Δ n x)

/-- **Euler rollouts (Eq. (17), second form).**  If the field used at step `k` is
`c k`-Lipschitz (for closed-loop dynamics, `c k = L_x + L_u L(ρ_k)`), the `n`-step Euler
rollout map is `∏_{k<n} (1 + Δρ c k)`-Lipschitz. -/
theorem euler_lipschitz {g : ℕ → E → E} {c : ℕ → ℝ} {Δ : ℝ} (hΔ : 0 ≤ Δ)
    (hc : ∀ k, 0 ≤ c k) (hg : ∀ k x y, ‖g k x - g k y‖ ≤ c k * ‖x - y‖) (n : ℕ) (x y : E) :
    ‖eulerMap g Δ n x - eulerMap g Δ n y‖ ≤ (∏ k ∈ Finset.range n, (1 + Δ * c k)) * ‖x - y‖ := by
  induction n with
  | zero => simp [eulerMap]
  | succ n ih =>
    simp only [eulerMap, Finset.prod_range_succ]
    calc _ ≤ (1 + Δ * c n) * ‖eulerMap g Δ n x - eulerMap g Δ n y‖ :=
          euler_step_lipschitz hΔ (hg n) _ _
      _ ≤ (1 + Δ * c n) * ((∏ k ∈ Finset.range n, (1 + Δ * c k)) * ‖x - y‖) := by
          gcongr
          nlinarith [hc n]
      _ = _ := by ring

/-- **Unicycle, `L_u = 1`.**  The columns of `B(θ)` are orthonormal, so
`‖B(θ)(u - u')‖² = ‖u - u'‖²` with `u = (v, ω)`, `B(θ)u = (v cos θ, v sin θ, ω)`. -/
theorem unicycle_Lu (θ v v' ω ω' : ℝ) :
    (v * cos θ - v' * cos θ) ^ 2 + (v * sin θ - v' * sin θ) ^ 2 + (ω - ω') ^ 2 =
      (v - v') ^ 2 + (ω - ω') ^ 2 := by
  have := sin_sq_add_cos_sq θ
  linear_combination (v - v') ^ 2 * this

/-- `(cos θ - cos θ')² + (sin θ - sin θ')² ≤ (θ - θ')²`. -/
theorem chord_le_arc (θ θ' : ℝ) :
    (cos θ - cos θ') ^ 2 + (sin θ - sin θ') ^ 2 ≤ (θ - θ') ^ 2 := by
  have h1 : (cos θ - cos θ') ^ 2 + (sin θ - sin θ') ^ 2 = 2 - 2 * cos (θ - θ') := by
    rw [cos_sub]; nlinarith [sin_sq_add_cos_sq θ, sin_sq_add_cos_sq θ']
  have h2 : cos (θ - θ') = 1 - 2 * sin ((θ - θ') / 2) ^ 2 := by
    have h := cos_two_mul ((θ - θ') / 2)
    rw [show 2 * ((θ - θ') / 2) = θ - θ' by ring] at h
    rw [h]; nlinarith [sin_sq_add_cos_sq ((θ - θ') / 2)]
  have h3 : |sin ((θ - θ') / 2)| ≤ |(θ - θ') / 2| := abs_sin_le_abs
  have h4 : sin ((θ - θ') / 2) ^ 2 ≤ ((θ - θ') / 2) ^ 2 := by
    rw [← sq_abs, ← sq_abs ((θ - θ') / 2)]
    exact pow_le_pow_left₀ (abs_nonneg _) h3 2
  rw [h1, h2]; nlinarith

/-- **Unicycle, `L_x = |v|`.**  Only the first column of `B(θ)` depends on the state, and
`‖B(θ)u - B(θ')u‖² = v² ‖(cos θ - cos θ', sin θ - sin θ')‖² ≤ v² (θ - θ')²`. -/
theorem unicycle_Lx (θ θ' v ω : ℝ) :
    (v * cos θ - v * cos θ') ^ 2 + (v * sin θ - v * sin θ') ^ 2 + (ω - ω) ^ 2 ≤
      (|v| * |θ - θ'|) ^ 2 := by
  have := chord_le_arc θ θ'
  rw [mul_pow, sq_abs, sq_abs]
  nlinarith [sq_nonneg v]

end Phaseformer
