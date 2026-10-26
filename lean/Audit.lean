import PhaseformerProofs

/-!
Run `lake env lean Audit.lean` after `lake build`.

Each line should print only the three standard axioms of Lean + Mathlib
(`propext`, `Classical.choice`, `Quot.sound`).  A `sorryAx` here would mean an
unfinished proof; there are none.
-/

open Phaseformer

-- B.1
#print axioms pythagoras_nested
#print axioms eq7_L2
#print axioms eq9_L2
#print axioms eq7_minimizer
#print axioms fiberMean_tower
#print axioms lemma1a
#print axioms lemma1b
#print axioms lemma1c_straight_velocity
#print axioms lemma1c_sin_not_straight
-- B.2
#print axioms average_velocity_first
#print axioms prop1_weak_continuity
#print axioms prop1_phase_free_identity
#print axioms phase_free_identity_1d
#print axioms regularity_vbar_zero
#print axioms regularity_path_norm
#print axioms gauss_lipschitz_le_one
#print axioms gauss_rollout
#print axioms gauss_rollout_unique
#print axioms gauss_rollout_returns
#print axioms gauss_field_half
#print axioms gauss_no_crossing
#print axioms gauss_pushforward
#print axioms gauss_covariance
-- B.3
#print axioms T_strictMonoOn
#print axioms prop2a_no_conflict
#print axioms occupation_per_demo
#print axioms group_occupation
#print axioms vfree_convex
#print axioms vfree_overlap
#print axioms aliasing_limit
#print axioms travel_time
#print axioms rollout_stays_in
#print axioms rollout_strictMono
#print axioms prop2a_overshoot
#print axioms prop2a_undershoot
#print axioms cdf_lt_of_undershoot
#print axioms prop2a_not_PG
#print axioms prop2a_limit
#print axioms prop2b_no_conflict
#print axioms prop2b_demo_is_rollout
#print axioms prop2b_flux_zero
#print axioms prop2b_stall
#print axioms prop2b_supports_disjoint
#print axioms prop2b_shift_pos
#print axioms prop2b_aliasing
-- B.4
#print axioms gronwall_rollouts
#print axioms rollouts_eq_of_eq
#print axioms closedLoop_lipschitz
#print axioms euler_lipschitz
#print axioms unicycle_Lu
#print axioms unicycle_Lx
#print axioms lemma2a_branch_average
#print axioms fiberMean_const
#print axioms lemma2b_lipschitz_lower
#print axioms lemma2b_not_integrable
#print axioms lemma2c_one_branch
#print axioms prop3a_undecided
#print axioms prop3a_prob
#print axioms prop3b_segment
#print axioms prop3b_interval_1d
#print axioms prop3_gauss_too_wide
#print axioms phi2_bounds
#print axioms prop3_gauss_prob
-- B.5
#print axioms jensen_effort
#print axioms jensen_eq_iff
#print axioms straight_of_const_velocity
#print axioms minimizer_attains_bounds
#print axioms thm4i_straight
#print axioms thm4ii
#print axioms gradient_convex_monotone
#print axioms eq21
#print axioms no_crossing
#print axioms velocity_lipschitz
#print axioms inv_min_eq_max
#print axioms idealField_on_demo
#print axioms idealField_lipschitz
#print axioms demo_unique_rollout
-- B.6
#print axioms min_of_pointwise_min
#print axioms pointwise_min_of_min
#print axioms thm5_ii_bound
#print axioms thm5_ii_between
#print axioms g_unique_min
#print axioms thm5_iii_iff
#print axioms thm5_ii_limit
#print axioms absDev_convex
#print axioms absDev_lipschitz
#print axioms median_of_const
#print axioms median_of_symmetric
#print axioms J_split
#print axioms thm5_s_zero
#print axioms thm5_i
#print axioms thm5_along
#print axioms thm5_ii
#print axioms thm5_iii
#print axioms ex_symmetric_tie
#print axioms ex_shift_against
#print axioms ex_bias
#print axioms stationarity
#print axioms realizability_score
