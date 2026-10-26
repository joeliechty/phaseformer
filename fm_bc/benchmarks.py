"""SE(2) particle flows whose divergence is known exactly, for validating the estimators.

Particles sit at ``c @ Exp(xi_i(t))`` with ``xi_i(t) = exp(R t) xi_i(0)`` componentwise: the flow
is linear in exponential coordinates around ``c``, so its divergence is ``sum(R)`` everywhere,
for every particle, at every time. (Measured against the Haar measure the Exp Jacobian adds a
correction of order ``theta^2 / 12``, negligible at the heading spreads used here.)
"""

from __future__ import annotations

import numpy as np
import torch

from .se2 import se2_exp_retract

__all__ = ["linear_flow", "kde_attenuation"]


def linear_flow(
    rates: tuple[float, float, float],
    n: int = 1024,
    sigma: tuple[float, float, float] = (0.3, 0.3, 0.15),
    t_end: float = 2.0,
    n_steps: int = 21,
    seed: int = 0,
    center: tuple[float, float, float] = (2.0, 1.0, 0.5),
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, float]:
    """``(poses (n, n_steps, 3), dt)`` for a flow of divergence ``sum(rates)``.

    ``sigma`` is the initial Gaussian spread per tangent direction ``[rho_x, rho_y, theta]``;
    the defaults match the start distributions of the Fig. 6 datasets.
    """
    g = torch.Generator().manual_seed(seed)
    xi = torch.randn(n, 3, generator=g, dtype=torch.double)
    xi = xi * torch.tensor(sigma, dtype=torch.double)
    r = torch.tensor(rates, dtype=torch.double)
    c = torch.tensor(center, dtype=torch.double).expand(n, 3)
    times = torch.linspace(0.0, t_end, n_steps, dtype=torch.double)
    poses = torch.stack([se2_exp_retract(c, xi * torch.exp(r * t)) for t in times], dim=1)
    return poses.to(device), float(times[1] - times[0])


def kde_attenuation(u: np.ndarray | float) -> np.ndarray | float:
    """Fraction of the true divergence a fixed-bandwidth Gaussian KDE recovers.

    For a Gaussian marginal of spread ``sigma`` contracting or expanding at rate ``a``
    (``x_t = e^{a t} x_0``), the KDE sees the smoothed density ``N(0, sigma^2 + h^2)``. Its
    log-rate along a particle, averaged over particles, is
    ``-a sigma^2 (sigma^2 + 2 h^2) / (sigma^2 + h^2)^2``, so with ``u = h / sigma``

    .. math:: \\widehat{\\nabla\\cdot v} / \\nabla\\cdot v = (1 + 2u^2) / (1 + u^2)^2,

    which is 1 at ``u -> 0``, 3/4 at ``u = 1`` and ``~2/u^2`` for ``u >> 1``.
    """
    u2 = np.asarray(u, dtype=float) ** 2
    return (1.0 + 2.0 * u2) / (1.0 + u2) ** 2
