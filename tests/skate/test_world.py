"""Obstacle fields and SDFs, including the Theseus grid-layout conventions we rely on."""

import numpy as np
import pytest
import theseus as th
import torch

from skate_planner.world import Box, Circle, ObstacleField, random_field, scenario

D = torch.double


def test_circle_sdf_signs():
    circle = Circle(1.0, 2.0, 0.5)
    pts = np.array([[1.0, 2.0], [1.5, 2.0], [2.0, 2.0]])
    assert circle.sdf(pts).tolist() == pytest.approx([-0.5, 0.0, 0.5])


def test_box_sdf_inside_outside_and_corner():
    box = Box(0.0, 0.0, 1.0, 2.0)
    assert box.sdf(np.array([0.0, 0.0])) == pytest.approx(-1.0)  # nearest face
    assert box.sdf(np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert box.sdf(np.array([4.0, 6.0])) == pytest.approx(5.0)  # 3-4-5 from the corner


def test_boundary_counts_as_an_obstacle():
    world = ObstacleField(bounds=(0.0, 0.0, 10.0, 10.0))
    assert world.analytic_sdf(np.array([5.0, 5.0])) == pytest.approx(5.0)
    assert world.analytic_sdf(np.array([-1.0, 5.0])) == pytest.approx(-1.0)


def test_grid_sdf_matches_analytic_within_a_cell():
    world, _, _ = scenario("slalom")
    cell = 0.05
    grid = world.grid_sdf(cell)
    centers, _ = world.cell_centers(cell)
    exact = world.analytic_sdf(centers)
    # The grid samples the exact field at its nodes (an EDT used to be off by up to a cell).
    np.testing.assert_allclose(grid.data, exact, atol=1e-12)
    assert np.sign(grid.data[exact > cell]).min() > 0


def test_grid_sdf_edt_is_the_old_distance_transform():
    world, _, _ = scenario("slalom")
    cell = 0.1
    exact, edt = world.grid_sdf(cell), world.grid_sdf(cell, method="edt")
    assert edt.data.shape == exact.data.shape and edt.origin.tolist() == exact.origin.tolist()
    # It measures to the nearest occupied node, not the surface: in free space it over-reports
    # clearance, never under-reports it, and by at most a cell diagonal.
    free = exact.data > 0
    over = edt.data[free] - exact.data[free]
    assert over.min() >= -1e-12 and over.max() <= cell * np.sqrt(2)
    assert over.max() > cell / 2
    with pytest.raises(ValueError, match="method"):
        world.grid_sdf(cell, method="nearest")


def test_grid_sdf_shape_and_origin():
    world = ObstacleField(bounds=(0.0, 0.0, 1.0, 2.0))
    grid = world.grid_sdf(0.5)
    assert grid.data.shape == (5, 3)  # rows advance in +y, cols in +x
    assert grid.origin.tolist() == [0.0, 0.0]


def test_theseus_sdf_grid_is_row_y_col_x():
    """Pin the layout Collision2D assumes; the planner feeds it our grid directly."""
    rows, cols = 5, 6
    r_idx, c_idx = torch.meshgrid(
        torch.arange(rows, dtype=D), torch.arange(cols, dtype=D), indexing="ij"
    )
    data = (c_idx + 100.0 * r_idx).unsqueeze(0)
    sdf = th.eb.SignedDistanceField2D(
        origin=torch.zeros(1, 2, dtype=D),
        cell_size=torch.ones(1, 1, dtype=D),
        sdf_data=data,
    )
    # Points are (batch, 2, num_points) with rows = (x, y).
    dist, jac = sdf.signed_distance(torch.tensor([[[3.0], [1.0]]], dtype=D))
    assert dist.shape == (1, 1)
    assert jac.shape == (1, 1, 2)
    assert dist.item() == pytest.approx(103.0)  # x=3 -> col, y=1 -> row
    assert jac[0, 0].tolist() == pytest.approx([1.0, 100.0])  # d/dx, d/dy


def test_collision2d_uses_our_sign_convention():
    """Collision cost is positive only where clearance falls below cost_eps."""
    world = ObstacleField(bounds=(0.0, 0.0, 10.0, 10.0), circles=[Circle(5.0, 5.0, 1.0)])
    grid = world.grid_sdf(0.05)
    data = torch.as_tensor(grid.data, dtype=D)[None]
    origin = torch.as_tensor(grid.origin, dtype=D)[None]
    cell = torch.tensor([[grid.cell_size]], dtype=D)
    eps = torch.tensor([[0.5]], dtype=D)

    def error_at(x, y):
        pose = th.SE2(x_y_theta=torch.tensor([[x, y, 0.0]], dtype=D))
        cost = th.eb.Collision2D(
            pose, origin, data, cell, eps, th.ScaleCostWeight(torch.tensor(1.0, dtype=D))
        )
        return cost.error().item()

    assert error_at(5.0, 5.0) > 0.0  # inside the obstacle
    assert error_at(5.0, 6.2) > 0.0  # within the 0.5 m margin
    assert error_at(5.0, 8.0) == pytest.approx(0.0)  # far away, hinge is flat


def test_sample_free_respects_clearance():
    world, _, _ = scenario("slalom")
    pts = world.sample_free(50, clearance=0.4, rng=np.random.default_rng(1))
    assert pts.shape == (50, 2)
    assert world.analytic_sdf(pts).min() >= 0.4


def test_sample_free_fails_loudly_when_impossible():
    world = ObstacleField(bounds=(0.0, 0.0, 1.0, 1.0), circles=[Circle(0.5, 0.5, 5.0)])
    with pytest.raises(RuntimeError, match="too cluttered"):
        world.sample_free(5, clearance=0.1, rng=np.random.default_rng(0))


def test_scenarios_actually_block_the_straight_line():
    """A scenario whose direct path is already free would not test avoidance at all."""
    for name in ("slalom", "clutter"):
        world, start, goal = scenario(name)
        line = np.stack(
            [np.linspace(start[0], goal[0], 300), np.linspace(start[1], goal[1], 300)], axis=-1
        )
        assert world.analytic_sdf(line).min() < 0.0, f"{name} straight line is not blocked"


def test_random_field_keeps_endpoints_clear():
    world = random_field(
        n_circles=12, keep_clear=[(1.0, 1.0), (9.0, 9.0)], clear_radius=1.0,
        rng=np.random.default_rng(3),
    )
    assert world.analytic_sdf(np.array([[1.0, 1.0], [9.0, 9.0]])).min() > 0.0


def test_unknown_scenario_raises():
    with pytest.raises(ValueError, match="Unknown scenario"):
        scenario("nope")
