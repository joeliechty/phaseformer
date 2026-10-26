"""End-to-end planning behaviour."""

import numpy as np
import pytest
import torch

from skate_planner.planner import SkateLimits, SkatePlanner, SkateWeights
from skate_planner.skate_model import lateral_slip, rollout
from skate_planner.world import Circle, ObstacleField, scenario

D = torch.double


@pytest.fixture(scope="module")
def slalom():
    return scenario("slalom")


def _planner(world, **kwargs):
    kwargs.setdefault("horizon", 45)
    kwargs.setdefault("dt", 0.2)
    kwargs.setdefault("radius", 0.3)
    return SkatePlanner.for_world(world, cell_size=0.1, **kwargs)


def test_plans_a_collision_free_path(slalom):
    world, start, goal = slalom
    solution = _planner(world).plan(
        world, start, goal, n_starts=8, rng=np.random.default_rng(0)
    )
    assert solution.feasible.any(), "no restart produced an executable, collision-free path"
    best = solution.best_index
    assert solution.feasible[best]
    assert solution.diagnostics.min_clearance[best] >= 0.3
    assert solution.diagnostics.goal_error_xy[best] < 0.25


def test_controls_stay_within_limits(slalom):
    world, start, goal = slalom
    limits = SkateLimits(v_max=1.5, v_min=0.0, omega_max=1.2)
    solution = _planner(world, limits=limits).plan(
        world, start, goal, n_starts=8, rng=np.random.default_rng(0)
    )
    controls = solution.best_controls
    # HingeCost is a soft penalty, so allow a small violation rather than asserting a hard box.
    assert controls[:, 0].max().item() <= limits.v_max + 0.15
    assert controls[:, 0].min().item() >= limits.v_min - 0.15
    assert controls[:, 1].abs().max().item() <= limits.omega_max + 0.15


def test_structural_mode_has_exactly_zero_slip_in_the_rollout(slalom):
    """Controls have no lateral component, so anything rolled out from them cannot slip."""
    world, start, goal = slalom
    solution = _planner(world).plan(world, start, goal, n_starts=4, rng=np.random.default_rng(0))
    poses = rollout(
        solution.poses[:, 0], solution.controls, 0.2
    )
    assert lateral_slip(poses, 0.2).abs().max().item() < 1e-12


def test_multistart_is_at_least_as_good_as_a_single_start(slalom):
    world, start, goal = slalom
    planner = _planner(world)
    single = planner.plan(world, start, goal, n_starts=1)
    multi = planner.plan(world, start, goal, n_starts=8, rng=np.random.default_rng(0))
    assert multi.feasible.sum() >= single.feasible.sum()
    if single.feasible.any() and multi.feasible.any():
        best_multi = multi.diagnostics.path_length[multi.best_index]
        best_single = single.diagnostics.path_length[single.best_index]
        assert best_multi <= best_single + 1e-6


def test_penalty_mode_agrees_with_structural_mode(slalom):
    """Cross-check the custom cost against Theseus's own GPMotionModel + Nonholonomic."""
    world, start, goal = slalom
    seed = np.random.default_rng(0)
    structural = _planner(world, twist_mode="structural").plan(
        world, start, goal, n_starts=8, rng=seed
    )
    penalty = _planner(world, twist_mode="penalty").plan(
        world, start, goal, n_starts=8, rng=np.random.default_rng(0)
    )
    assert penalty.feasible.any()
    # Both should find a comparably short route through the same field. Agreement here is the
    # evidence that the custom SkateDynamics residual is right, independent of check_jacobians.
    assert penalty.diagnostics.path_length[penalty.best_index] == pytest.approx(
        structural.diagnostics.path_length[structural.best_index], rel=0.15
    )


def test_penalty_mode_uses_lateral_velocity_it_should_not_have(slalom):
    """The distinction that justifies preferring the structural formulation.

    Here the lateral twist component is a free variable that is merely penalized, so the
    optimizer spends a little of it. In structural mode there is no such variable to spend.
    """
    world, start, goal = slalom
    planner = _planner(world, twist_mode="penalty")
    solution = planner.plan(world, start, goal, n_starts=4, rng=np.random.default_rng(0))
    lateral = torch.stack(
        [solution.info.best_solution[f"vel_{i}"][:, 1] for i in range(planner.horizon)]
    )
    assert lateral.abs().max().item() > 0.0


def test_position_goal_leaves_heading_free():
    world = ObstacleField(bounds=(0.0, 0.0, 8.0, 8.0), circles=[Circle(4.0, 4.0, 1.0)])
    start = np.array([1.0, 1.0, 0.0])
    goal = np.array([7.0, 7.0, 0.0])
    solution = _planner(world, goal_mode="position").plan(world, start, goal, n_starts=4,
                                                          rng=np.random.default_rng(0))
    assert solution.diagnostics.goal_error_xy[solution.best_index] < 0.25


def test_unknown_input_key_raises_instead_of_warning(slalom):
    """Theseus only warns on unknown keys, which silently skips the update."""
    world, start, goal = slalom
    planner = _planner(world)
    with pytest.raises(KeyError, match="Unknown input keys"):
        planner._validate_inputs({"pose_0": torch.zeros(1, 4, dtype=D), "typo_here": None})


def test_missing_required_input_raises(slalom):
    world, _, _ = slalom
    planner = _planner(world)
    with pytest.raises(KeyError, match="Missing required inputs"):
        planner._validate_inputs({"start": torch.zeros(1, 4, dtype=D)})


def test_grid_shape_mismatch_is_reported_clearly(slalom):
    world, start, goal = slalom
    planner = _planner(world)
    with pytest.raises(ValueError, match="rasterizes to"):
        planner.plan(world, start, goal, cell_size=0.2)


def test_rejects_bad_modes(slalom):
    world, _, _ = slalom
    with pytest.raises(ValueError, match="twist_mode"):
        _planner(world, twist_mode="nonsense")
    with pytest.raises(ValueError, match="goal_mode"):
        _planner(world, goal_mode="nonsense")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_runs_on_gpu(slalom):
    world, start, goal = slalom
    solution = _planner(world, device="cuda").plan(
        world, start, goal, n_starts=8, rng=np.random.default_rng(0)
    )
    assert solution.poses.device.type == "cuda"
    assert solution.feasible.any()
