"""Kinematics: does the model actually describe a skate?"""

import numpy as np
import pytest
import theseus as th
import torch

from skate_planner.skate_model import (
    LATERAL_INDEX,
    body_twist,
    lateral_slip,
    rollout,
    se2_tensor,
    wrap_angle,
    xytheta,
)

D = torch.double


def test_body_twist_zeroes_the_lateral_slot():
    controls = torch.tensor([[1.5, -0.4], [0.0, 2.0]], dtype=D)
    twist = body_twist(controls)
    assert torch.equal(twist[:, LATERAL_INDEX], torch.zeros(2, dtype=D))
    assert torch.equal(twist[:, 0], controls[:, 0])
    assert torch.equal(twist[:, 2], controls[:, 1])


def test_se2_tangent_ordering_is_translation_then_rotation():
    """Guards the assumption the dynamics residual is built on."""
    for tangent, expect_xy, expect_theta in (
        ([1.0, 0.0, 0.0], [1.0, 0.0], 0.0),
        ([0.0, 1.0, 0.0], [0.0, 1.0], 0.0),
        ([0.0, 0.0, 0.5], [0.0, 0.0], 0.5),
    ):
        g = th.SE2.exp_map(torch.tensor([tangent], dtype=D))
        assert torch.allclose(g.xy().tensor[0], torch.tensor(expect_xy, dtype=D), atol=1e-12)
        assert g.theta()[0].item() == pytest.approx(expect_theta, abs=1e-12)


def test_se2_tensor_round_trip():
    xyt = torch.tensor([[1.2, -3.4, 2.9], [0.0, 0.0, -1.1]], dtype=D)
    assert torch.allclose(xytheta(se2_tensor(xyt)), xyt, atol=1e-14)


def test_constant_turn_traces_an_exact_circle():
    """v and omega held constant must produce a circle of radius v/omega."""
    # omega is chosen so a whole number of steps closes the circle exactly; otherwise the
    # "returns to start" check below is testing rounding, not kinematics.
    v, dt, steps_per_rev = 1.4, 0.05, 180
    omega = 2 * np.pi / (steps_per_rev * dt)
    radius = v / omega
    controls = torch.tensor([[[v, omega]]], dtype=D).repeat(1, steps_per_rev, 1)
    poses = rollout(torch.zeros(1, 3, dtype=D), controls, dt)

    # Center is perpendicular-left of the start heading at distance r.
    center = torch.tensor([0.0, radius], dtype=D)
    distances = torch.linalg.norm(poses[0, :, :2] - center, dim=-1)
    assert torch.allclose(distances, torch.full_like(distances, radius), atol=1e-12)

    # After a full revolution it must return to the start pose.
    assert torch.allclose(poses[0, -1, :2], torch.zeros(2, dtype=D), atol=1e-10)
    assert torch.cos(poses[0, -1, 2]).item() == pytest.approx(1.0, abs=1e-10)


def test_zero_yaw_rate_gives_a_straight_line():
    controls = torch.tensor([[[2.0, 0.0]]], dtype=D).repeat(1, 20, 1)
    start = torch.tensor([[0.5, -1.0, 0.6]], dtype=D)
    poses = rollout(start, controls, 0.1)
    headings = poses[0, :, 2]
    assert torch.allclose(headings, torch.full_like(headings, 0.6), atol=1e-12)
    # Every point lies on the ray from the start along theta.
    offsets = poses[0, :, :2] - start[0, :2]
    cross = offsets[:, 0] * np.sin(0.6) - offsets[:, 1] * np.cos(0.6)
    assert torch.allclose(cross, torch.zeros_like(cross), atol=1e-12)


def test_rollout_has_no_lateral_slip():
    """The defining property of the skate model, to machine precision."""
    torch.manual_seed(0)
    controls = torch.randn(4, 30, 2, dtype=D)
    start = torch.randn(4, 3, dtype=D)
    poses = rollout(start, controls, 0.15)
    assert lateral_slip(poses, 0.15).abs().max().item() < 1e-12


def test_substeps_add_samples_without_moving_the_knots():
    """Exact arc integration composes: Exp(dt*xi) == Exp((dt/m)*xi)^m."""
    controls = torch.tensor([[[1.1, 0.9], [0.7, -1.3], [1.5, 0.2]]], dtype=D)
    start = torch.tensor([[0.3, 0.2, -0.4]], dtype=D)
    coarse = rollout(start, controls, 0.2)
    dense = rollout(start, controls, 0.2, substeps=5)
    assert dense.shape[1] == 3 * 5 + 1
    assert torch.allclose(dense[:, ::5], coarse, atol=1e-12)


def test_wrap_angle():
    angles = torch.tensor([0.0, np.pi - 1e-9, -np.pi + 1e-9, 3 * np.pi], dtype=D)
    wrapped = wrap_angle(angles)
    assert bool((wrapped.abs() <= np.pi + 1e-9).all())
    assert torch.allclose(torch.cos(wrapped), torch.cos(angles), atol=1e-12)


def test_rollout_rejects_bad_substeps():
    with pytest.raises(ValueError, match="substeps"):
        rollout(torch.zeros(1, 3, dtype=D), torch.zeros(1, 2, 2, dtype=D), 0.1, substeps=0)
