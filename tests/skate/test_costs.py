"""Cost functions, above all their analytic jacobians.

A wrong jacobian does not crash: it makes the optimizer converge slowly to the wrong answer,
which is far harder to notice than an exception. These are the highest-value tests in the suite.
"""

import pytest
import theseus as th
import torch
from theseus.utils import check_jacobians

from skate_planner.costs import SkateDynamics, XYGoal
from skate_planner.skate_model import body_twist, rollout

D = torch.double


def _weight(value: float = 1.0) -> th.ScaleCostWeight:
    return th.ScaleCostWeight(torch.tensor(value, dtype=D))


def _dynamics(dt: float = 0.2) -> SkateDynamics:
    return SkateDynamics(
        th.SE2(x_y_theta=torch.zeros(1, 3, dtype=D), name="pa"),
        th.SE2(x_y_theta=torch.zeros(1, 3, dtype=D), name="pb"),
        th.Vector(2, name="u", dtype=D),
        dt,
        _weight(),
    )


def test_skate_dynamics_jacobians_match_autograd():
    """check_jacobians raises if analytic and autograd jacobians disagree."""
    check_jacobians(_dynamics(), num_checks=50, tol=1e-6)


def test_xy_goal_jacobians_match_autograd():
    check_jacobians(
        XYGoal(
            th.SE2(x_y_theta=torch.zeros(1, 3, dtype=D), name="var"),
            th.Point2(tensor=torch.zeros(1, 2, dtype=D), name="target"),
            _weight(),
        ),
        num_checks=50,
        tol=1e-6,
    )


def test_dynamics_residual_vanishes_on_an_exactly_integrated_arc():
    dt = 0.2
    controls = torch.tensor([[[1.3, 0.7]]], dtype=D)
    poses = rollout(torch.tensor([[0.4, -0.2, 0.5]], dtype=D), controls, dt)
    cost = SkateDynamics(
        th.SE2(x_y_theta=poses[:, 0]), th.SE2(x_y_theta=poses[:, 1]),
        th.Vector(tensor=controls[:, 0]), dt, _weight(),
    )
    assert cost.error().abs().max().item() < 1e-14


def test_dt_is_not_silently_downcast_to_float32():
    """A float32 dt caps the residual at ~4e-9 even inside a float64 objective."""
    cost = _dynamics(dt=0.2)
    assert cost.dt.tensor.dtype == torch.double
    assert cost.dt.tensor.item() == 0.2  # exact float64 0.2, not 0.20000000298...


def test_lateral_row_is_what_penalizes_side_slip():
    """Sliding the second pose purely sideways must show up only in the lateral row."""
    dt = 0.2
    pose_a = th.SE2(x_y_theta=torch.zeros(1, 3, dtype=D))
    controls = torch.tensor([[1.0, 0.0]], dtype=D)
    pose_b = pose_a.retract(dt * body_twist(controls))
    slipped = pose_b.retract(torch.tensor([[0.0, 0.05, 0.0]], dtype=D))

    clean = SkateDynamics(pose_a, pose_b, th.Vector(tensor=controls), dt, _weight()).error()
    dirty = SkateDynamics(pose_a, slipped, th.Vector(tensor=controls), dt, _weight()).error()

    assert clean.abs().max().item() < 1e-14
    assert dirty[0, 1].item() == pytest.approx(0.05, abs=1e-12)
    assert dirty[0, 0].abs().item() < 1e-12
    assert dirty[0, 2].abs().item() < 1e-12


def test_dynamics_rejects_wrong_control_dimension():
    with pytest.raises(ValueError, match="2-dof"):
        SkateDynamics(
            th.SE2(x_y_theta=torch.zeros(1, 3, dtype=D)),
            th.SE2(x_y_theta=torch.zeros(1, 3, dtype=D)),
            th.Vector(3, dtype=D),
            0.2,
            _weight(),
        )


def test_cost_functions_are_copyable():
    """Theseus copies cost functions when vectorizing, so _copy_impl must work."""
    for cost in (_dynamics(), XYGoal(
        th.SE2(x_y_theta=torch.zeros(1, 3, dtype=D)),
        th.Point2(tensor=torch.zeros(1, 2, dtype=D)), _weight(),
    )):
        clone = cost.copy(new_name="clone")
        assert clone.name == "clone"
        assert clone.dim() == cost.dim()
