"""Custom Theseus cost functions for the skate planner.

Only two live here. Everything else the planner needs already exists in Theseus and is reused
verbatim: :class:`theseus.embodied.Collision2D` (SE2-aware, chains the ``(B, 2, 3)`` jacobian of
``SE2.xy()`` internally), :class:`theseus.embodied.HingeCost`, :class:`theseus.Difference` and
:class:`theseus.Between`.
"""

from __future__ import annotations

from typing import Optional, Union

import theseus as th
import torch

from .skate_model import body_twist, twist_selector

__all__ = ["SkateDynamics", "XYGoal"]


class SkateDynamics(th.CostFunction):
    r"""Residual between consecutive poses and the skate twist that should connect them.

    .. math::
        e = \mathrm{Log}(T_a^{-1} T_b) - \Delta t \, [v, 0, \omega]^\top

    Three rows, one per SE(2) tangent direction. Row 1 -- the lateral one -- is the whole point:
    it compares how far the robot moved *sideways* against a commanded lateral velocity that is a
    hard-coded zero. Driving it down forbids side-slip. Because the zero is structural rather
    than a variable with a penalty on it, there is no slip weight to trade off against progress.

    Because ``Log(T_a^{-1} T_b)`` is exactly the twist of the circular arc joining the two poses,
    this residual vanishes precisely when ``T_b`` is the exact arc-integration of ``(v, omega)``
    from ``T_a`` over ``dt`` -- not a first-order approximation of it.

    Jacobians are analytic. This is the hot factor (one copy per timestep), and the control block
    is the constant ``-dt * [[1, 0], [0, 0], [0, 1]]``, so there is little reason to pay for
    autograd here.
    """

    def __init__(
        self,
        pose_a: th.SE2,
        pose_b: th.SE2,
        ctrl: th.Vector,
        dt: Union[float, torch.Tensor, th.Variable],
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        if not isinstance(pose_a, th.SE2) or not isinstance(pose_b, th.SE2):
            raise ValueError("SkateDynamics expects SE2 poses.")
        if ctrl.dof() != 2:
            raise ValueError(
                f"SkateDynamics expects a 2-dof control [v, omega], got dof {ctrl.dof()}."
            )
        self.pose_a = pose_a
        self.pose_b = pose_b
        self.ctrl = ctrl
        # dtype must be supplied at construction, not cast afterwards: th.as_variable() on a
        # plain Python float defaults to float32, and casting 0.2f up to float64 preserves
        # 0.20000000298..., capping this residual's accuracy at ~4e-9 inside a float64 objective.
        self.dt = th.as_variable(dt, dtype=pose_a.dtype, device=pose_a.device)
        # as_variable() returns a Variable argument untouched, so still normalize shape/dtype.
        self.dt.tensor = self.dt.tensor.to(
            dtype=pose_a.dtype, device=pose_a.device
        ).view(-1, 1)
        self.register_optim_vars(["pose_a", "pose_b", "ctrl"])
        self.register_aux_vars(["dt"])

    def dim(self) -> int:
        return 3

    def _scaled_twist(self) -> torch.Tensor:
        """``dt * [v, 0, omega]``, shape ``(B, 3)``."""
        return self.dt.tensor * body_twist(self.ctrl.tensor)

    def error(self) -> torch.Tensor:
        return th.local(self.pose_a, self.pose_b) - self._scaled_twist()

    def jacobians(self) -> tuple[list[torch.Tensor], torch.Tensor]:
        pose_jacobians: list[torch.Tensor] = []
        diff = th.local(self.pose_a, self.pose_b, jacobians=pose_jacobians)
        error = diff - self._scaled_twist()

        selector = twist_selector(dtype=error.dtype, device=error.device)
        jac_ctrl = (-self.dt.tensor.view(-1, 1, 1) * selector).expand(
            error.shape[0], 3, 2
        )
        # .contiguous() because callers (e.g. Theseus's vectorized weighting) may write in place.
        return [pose_jacobians[0], pose_jacobians[1], jac_ctrl.contiguous()], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "SkateDynamics":
        return SkateDynamics(
            self.pose_a.copy(),
            self.pose_b.copy(),
            self.ctrl.copy(),
            self.dt.copy(),
            self.weight.copy(),
            name=new_name,
        )


class XYGoal(th.CostFunction):
    """Position-only goal for an SE(2) pose, leaving final heading free.

    Modeled on the private ``_XYDifference`` in Theseus's own bundled motion-planning example.
    Use this instead of :class:`theseus.Difference` when you care where the robot ends up but
    not which way it faces -- constraining heading unnecessarily is a common way to make an
    otherwise easy problem infeasible.
    """

    def __init__(
        self,
        var: th.SE2,
        target: th.Point2,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        if not isinstance(var, th.SE2) or not isinstance(target, th.Point2):
            raise ValueError("XYGoal expects an SE2 var and a Point2 target.")
        self.var = var
        self.target = target
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])

    def dim(self) -> int:
        return 2

    def error(self) -> torch.Tensor:
        return self.var.xy().tensor - self.target.tensor

    def jacobians(self) -> tuple[list[torch.Tensor], torch.Tensor]:
        jac: list[torch.Tensor] = []
        xy = self.var.xy(jacobians=jac)
        return [jac[0]], xy.tensor - self.target.tensor

    def _copy_impl(self, new_name: Optional[str] = None) -> "XYGoal":
        return XYGoal(
            self.var.copy(), self.target.copy(), self.weight.copy(), name=new_name
        )
