import numpy as np
import pytest
import torch

from fm_bc.skate import SCENARIOS, GenerationConfig, generate, get_scenario
from skate_planner.initializers import interpolate

CLEARANCE = 0.3 + 0.15  # planner radius + safety margin


def test_waypoint_chain_seeds():
    d = torch.double
    start = torch.tensor([[5.0, 1.0, np.pi / 2]], dtype=d)
    goal = torch.tensor([[1.6, 6.6, 0.0]], dtype=d)
    via = torch.tensor([[4.0, 4.0]], dtype=d)
    one, _ = interpolate(start, goal, 50, 0.15, via=via)
    chain_of_one, _ = interpolate(start, goal, 50, 0.15, via=via[:, None])
    torch.testing.assert_close(one, chain_of_one)  # a one-point chain is the quadratic Bezier
    s = torch.linspace(0, 1, 51, dtype=d)[:, None]
    bezier = (1 - s) ** 2 * start[0, :2] + 2 * (1 - s) * s * via[0] + s**2 * goal[0, :2]
    torch.testing.assert_close(one[0, :, :2], bezier)

    chain = torch.tensor([[[5.0, 3.6], [3.0, 4.4]]], dtype=d)
    poses, _ = interpolate(start, goal, 50, 0.15, via=chain)
    torch.testing.assert_close(poses[0, 0], start[0])
    torch.testing.assert_close(poses[0, -1], goal[0])
    # The two pieces join at the midpoint of the edge between the waypoints.
    torch.testing.assert_close(poses[0, 25, :2], chain[0].mean(dim=0))

    # A free goal heading arrives along the path instead of spinning to goal[:, 2].
    free, ctrl = interpolate(start, goal, 50, 0.15, via=chain, goal_heading=False)
    torch.testing.assert_close(free[0, -1, :2], goal[0, :2])
    d = free[0, -1, :2] - free[0, -2, :2]
    torch.testing.assert_close(free[0, -1, 2], torch.atan2(d[1], d[0]))
    _, forced = interpolate(start, goal, 50, 0.15, via=chain)
    assert ctrl[0, -1, 1].abs() < 1.0 < forced[0, -1, 1].abs()


@pytest.mark.parametrize("name", SCENARIOS)
def test_sample_shapes(name):
    spec = get_scenario(name)
    starts, goals, vias, modes = spec.sample(16, np.random.default_rng(0))
    assert starts.shape == (16, 3) and goals.shape == (16, 3) and modes.shape == (16,)
    assert vias.ndim == 3 and vias.shape[0] == 16 and vias.shape[2] == 2


@pytest.mark.parametrize("name", SCENARIOS)
def test_distribution_means_are_free(name):
    spec = get_scenario(name)
    points = np.stack([spec.start.mean[:2], *[g.mean[:2] for g in spec.goals]])
    assert spec.world.is_free(points, clearance=CLEARANCE).all()


@pytest.mark.parametrize("name,n_modes", [("single_path", 1), ("dual_goal", 2), ("dual_path", 2)])
def test_generate_small(name, n_modes):
    spec = get_scenario(name)
    assert spec.n_modes == n_modes
    cfg = GenerationConfig(chunk=24, max_rounds=6, seed=1)
    ds = generate(spec, n_target=8, config=cfg, verbose=False)

    assert len(ds) == 8
    assert sorted(set(ds.mode.tolist())) == list(range(n_modes))
    # Every kept trajectory really is collision-free on the exact geometry...
    assert (spec.world.analytic_sdf(ds.poses[..., :2]) >= 0.3 - 1e-6).all()
    # ...ends near its sampled goal, and is in the homotopy class it is labelled with.
    assert (np.linalg.norm(ds.poses[:, -1, :2] - ds.goals[:, :2], axis=1) <= cfg.goal_tol).all()
    np.testing.assert_array_equal(spec.label_mode(ds.poses), ds.mode)
    # Twists are the skate model's [v, 0, omega].
    np.testing.assert_array_equal(ds.twists[..., 1], 0.0)


def test_generation_defaults_are_todays_planner():
    cfg = GenerationConfig()
    assert cfg.sdf == "exact" and cfg.seed_goal_heading is False


def test_generate_with_the_pre_2eaf39d_planner():
    cfg = GenerationConfig(chunk=24, max_rounds=6, seed=1, sdf="edt", seed_goal_heading=True)
    ds = generate(get_scenario("single_path"), n_target=4, config=cfg, verbose=False)
    assert len(ds) == 4
    assert ds.config["sdf"] == "edt" and ds.config["seed_goal_heading"] is True
