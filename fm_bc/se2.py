"""Batched SE(2) Lie-group operations in plain torch.

Poses are ``(..., 3)`` tensors ``[x, y, theta]``; twists are ``(..., 3)`` body-frame tangent
vectors ``[rho_x, rho_y, theta]`` -- the same ordering and convention as Theseus's SE2
(``th.local(a, b) == Log(a^-1 b)``, ``pose.retract(xi) == pose @ Exp(xi)``), which
``tests/test_se2.py`` checks directly.

These exist because the divergence and score estimators need *all pairs* of poses at every
phase step (``B^2`` logs per step). Building Theseus objects for a million pairs is slow; the
closed forms below are a few vectorized trig calls and run on the GPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

__all__ = ["wrap_angle", "se2_log", "se2_exp_retract", "pairwise_twists", "twist_norm",
           "legacy_half_cot"]

# Below this |theta| the closed forms switch to their Taylor expansions, whose truncation error
# (~theta^4) is far under float64 resolution here.
_SMALL = 1e-4

# True only inside legacy_half_cot().
_LEGACY_HALF_COT = False


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles into ``(-pi, pi]``."""
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def _half_cot_half(theta: torch.Tensor) -> torch.Tensor:
    """``(theta/2) * cot(theta/2)``, the diagonal of ``V(theta)^-1``; ``-> 1`` as ``theta -> 0``."""
    small = theta.abs() < _SMALL
    safe = torch.where(small, torch.ones_like(theta), theta)
    if _LEGACY_HALF_COT:
        exact = 0.5 * safe * torch.sin(safe) / (1.0 - torch.cos(safe))
    else:
        # sin/(1 - cos) == cot(theta/2), but 1 - cos cancels to 0 in float32 just above _SMALL.
        exact = 0.5 * safe / torch.tan(0.5 * safe)
    return torch.where(small, 1.0 - theta**2 / 12.0, exact)


@contextmanager
def legacy_half_cot() -> Iterator[None]:
    """Inside the block, compute ``(theta/2) cot(theta/2)`` as ``sin / (1 - cos)``, its form
    before commit ``8c0f517``.

    The two forms agree in exact arithmetic. In float64 they differ in the last bits -- more
    near the small-angle cutoff, where ``1 - cos`` cancels -- and that reaches the kNN divergence
    and kernel score (by up to ~3e-8). The shipped skate datasets' targets were computed the old
    way, so ``scripts/skate/reproduce_datasets.py`` uses this to rebuild them bit for bit. Not
    thread-safe: it flips a module flag.
    """
    global _LEGACY_HALF_COT
    previous, _LEGACY_HALF_COT = _LEGACY_HALF_COT, True
    try:
        yield
    finally:
        _LEGACY_HALF_COT = previous


def se2_log(pose_a: torch.Tensor, pose_b: torch.Tensor) -> torch.Tensor:
    """Body twist ``Log(T_a^-1 T_b)``, i.e. the constant twist that carries ``a`` to ``b`` in
    unit time, expressed in ``a``'s frame. Broadcasts over leading dims."""
    dth = wrap_angle(pose_b[..., 2] - pose_a[..., 2])
    c, s = torch.cos(pose_a[..., 2]), torch.sin(pose_a[..., 2])
    dx = pose_b[..., 0] - pose_a[..., 0]
    dy = pose_b[..., 1] - pose_a[..., 1]
    # Relative translation in a's body frame.
    tx = c * dx + s * dy
    ty = -s * dx + c * dy
    # rho = V(theta)^-1 t, with V^-1 = [[a, h], [-h, a]], a = (th/2)cot(th/2), h = th/2.
    a = _half_cot_half(dth)
    h = 0.5 * dth
    return torch.stack([a * tx + h * ty, -h * tx + a * ty, dth], dim=-1)


def se2_exp_retract(pose: torch.Tensor, twist: torch.Tensor) -> torch.Tensor:
    """``T @ Exp(xi)``: move along the body twist ``xi`` from ``pose``. Broadcasts."""
    th = twist[..., 2]
    small = th.abs() < _SMALL
    safe = torch.where(small, torch.ones_like(th), th)
    sin_over = torch.where(small, 1.0 - th**2 / 6.0, torch.sin(safe) / safe)
    cos_over = torch.where(small, th / 2.0, (1.0 - torch.cos(safe)) / safe)
    # t = V(theta) rho, with V = [[sin/th, -(1-cos)/th], [(1-cos)/th, sin/th]].
    tx = sin_over * twist[..., 0] - cos_over * twist[..., 1]
    ty = cos_over * twist[..., 0] + sin_over * twist[..., 1]
    c, s = torch.cos(pose[..., 2]), torch.sin(pose[..., 2])
    return torch.stack(
        [
            pose[..., 0] + c * tx - s * ty,
            pose[..., 1] + s * tx + c * ty,
            wrap_angle(pose[..., 2] + th),
        ],
        dim=-1,
    )


def pairwise_twists(poses: torch.Tensor, others: torch.Tensor | None = None) -> torch.Tensor:
    """All-pairs body twists: ``out[i, j] = Log(T_i^-1 T_j)``, shape ``(B, B', 3)``.

    Row ``i`` is expressed in pose ``i``'s body frame -- the same tangent space a
    ``pose.retract`` perturbation of pose ``i`` lives in.
    """
    others = poses if others is None else others
    return se2_log(poses[:, None, :], others[None, :, :])


def twist_norm(twist: torch.Tensor, rot_scale: float = 1.0) -> torch.Tensor:
    """``sqrt(rho_x^2 + rho_y^2 + (rot_scale * theta)^2)``.

    ``rot_scale`` (metres per radian) is the one free choice in a left-invariant SE(2) metric:
    how many metres of translation one radian of heading is worth. ``1`` is the "unit twist"
    convention.
    """
    weights = twist.new_tensor([1.0, 1.0, rot_scale**2])
    return torch.sqrt((twist**2 * weights).sum(dim=-1))
