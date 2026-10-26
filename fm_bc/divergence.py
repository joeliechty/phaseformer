"""Divergence and score of a trajectory distribution, estimated from samples on SE(2).

This is the SE(2) port of the estimators in ``robomimic/utils/divergence_utils.py``
(``_compute_divergence_via_neighbors`` and ``_compute_score_via_neighbors``). We never see the
demonstrator's flow field, only samples from it, so neither quantity can be differentiated
directly. Instead:

**Divergence** is the rate at which a small volume of particles expands. The distance ``r_k`` from
each state to its k-th nearest neighbour at the same phase is a local length scale, so
``d log(vol)/dt ~= D * d log(r_k)/dt`` with ``D`` the manifold dimension (3 for SE(2)). The time
derivative is taken with a Savitzky-Golay filter, which smooths the kNN noise while
differentiating. Because the neighbours are local, this is robust to multimodality: particles
in one bundle never see the other.

**Score** ``grad log p`` is the mean-shift vector of a Gaussian kernel density built from the
states at the same phase, computed in each state's own body tangent space (``Log(T_i^-1 T_j)``),
so it lives in the same coordinates as the policy's twist outputs.

**KDE divergence** is the alternative: by the continuity equation, ``d/dt ln p = -div(v)``
along each particle, so the rate of the Gaussian-KDE log-density works too. It is smoother than
kNN but attenuated toward zero wherever the distribution is narrower than the bandwidth (see
:func:`divergence_kde`). kNN's scale follows the local sample spacing instead, so it stays
accurate down to far smaller spreads -- though it too attenuates once the spread falls below
the neighbour spacing.

``metric="xy"`` runs the same estimators on positions alone (Euclidean, D = 2), as a reference
that should agree in sign with the SE(2) numbers.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from scipy.signal import savgol_filter

from .se2 import pairwise_twists, se2_log, twist_norm

__all__ = [
    "manifold_dim",
    "state_distances",
    "divergence_knn",
    "log_density_kde",
    "divergence_kde",
    "score_kernel",
    "score_kernel_at",
]

Metric = Literal["se2", "xy"]


def manifold_dim(metric: Metric) -> int:
    if metric == "se2":
        return 3
    if metric == "xy":
        return 2
    raise ValueError(f"metric must be 'se2' or 'xy', got {metric!r}")


def _tangent(poses: torch.Tensor, others: torch.Tensor, metric: Metric) -> torch.Tensor:
    """Vectors from each pose to each other pose, ``(B, B', D)``, in the metric's coordinates."""
    if metric == "se2":
        return pairwise_twists(poses, others)
    manifold_dim(metric)
    return others[None, :, :2] - poses[:, None, :2]


def state_distances(
    poses: torch.Tensor,
    metric: Metric = "se2",
    rot_scale: float = 1.0,
    others: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise distances between ``(B, 3)`` poses, ``(B, B')``."""
    others = poses if others is None else others
    if metric == "se2":
        return twist_norm(pairwise_twists(poses, others), rot_scale)
    manifold_dim(metric)
    return torch.cdist(poses[:, :2], others[:, :2])


def _kth_neighbour_distance(
    poses: torch.Tensor, k: int, metric: Metric, rot_scale: float, chunk: int
) -> torch.Tensor:
    """Distance from each pose to its k-th nearest neighbour (self excluded), ``(B,)``.

    Rows are processed in chunks so memory stays ``chunk * B`` rather than ``B^2``.
    """
    out = []
    for lo in range(0, poses.shape[0], chunk):
        d = state_distances(poses[lo : lo + chunk], metric, rot_scale, others=poses)
        # Column 0 of the sorted row is the point itself at distance 0.
        out.append(torch.topk(d, k + 1, dim=1, largest=False).values[:, k])
    return torch.cat(out)


def _rate(series: torch.Tensor, dt: float, window_length: int, polyorder: int) -> torch.Tensor:
    """Savitzky-Golay time derivative of ``(M, T)`` per-particle series along ``T``."""
    n_steps = series.shape[1]
    window = min(window_length, n_steps if n_steps % 2 else n_steps - 1)
    d = savgol_filter(
        series.cpu().numpy(),
        window_length=window,
        polyorder=min(polyorder, window - 1),
        deriv=1,
        delta=dt,
        axis=1,
        mode="interp",
    )
    return torch.from_numpy(np.ascontiguousarray(d)).to(series)


def divergence_knn(
    poses: torch.Tensor,
    dt: float,
    k: int = 16,
    window_length: int = 9,
    polyorder: int = 2,
    metric: Metric = "se2",
    rot_scale: float = 1.0,
    chunk: int = 1024,
) -> torch.Tensor:
    """Flow divergence at every sample, ``(M, T)``, in 1/s.

    Parameters
    ----------
    poses:
        ``(M, T, 3)`` trajectories on a shared phase grid (sample ``m`` at step ``t`` is a draw
        from the phase-``t`` marginal).
    dt:
        Time between phase steps, so the result is per second rather than per phase.
    k:
        Neighbour rank. Small ``k`` is local (good for multimodal bundles) but noisier.
    window_length, polyorder:
        Savitzky-Golay filter settings; the window is clipped to the trajectory length.
    """
    if poses.shape[0] <= k:
        raise ValueError(f"Need more than k={k} samples per phase, got {poses.shape[0]}.")
    log_r = torch.stack(
        [
            torch.log(_kth_neighbour_distance(poses[:, t], k, metric, rot_scale, chunk) + 1e-8)
            for t in range(poses.shape[1])
        ],
        dim=1,
    )
    # D * dlog(r_k)/dt is exactly -dln(p)/dt for the kNN density estimate p ~ k / (M r_k^D).
    return _rate(log_r, dt, window_length, polyorder) * manifold_dim(metric)


def log_density_kde(
    poses: torch.Tensor,
    bandwidth: float = 0.2,
    metric: Metric = "se2",
    rot_scale: float = 1.0,
    leave_one_out: bool = True,
    chunk: int = 1024,
) -> torch.Tensor:
    """Unnormalized Gaussian-KDE log-density at each of ``(B, 3)`` poses, ``(B,)``.

    Same kernel and metric as :func:`score_kernel`. The normalization
    ``-ln M - (D/2) ln(2 pi h^2)`` is omitted: it is constant as long as ``M`` and ``h`` are.
    ``leave_one_out`` drops each point's own kernel, which otherwise adds a constant ``exp(0)``
    to every sum and flattens the estimate wherever few neighbours fall within ``h``.
    """
    weights = poses.new_tensor([1.0, 1.0, rot_scale**2])
    if metric == "xy":
        weights = weights[:2]
    out = []
    for lo in range(0, poses.shape[0], chunk):
        xi = _tangent(poses[lo : lo + chunk], poses, metric)
        logits = -(xi**2 * weights).sum(dim=-1) / (2 * bandwidth**2)
        if leave_one_out:
            rows = torch.arange(logits.shape[0], device=logits.device)
            logits[rows, lo + rows] = -torch.inf
        out.append(torch.logsumexp(logits, dim=1))
    return torch.cat(out)


def divergence_kde(
    poses: torch.Tensor,
    dt: float,
    bandwidth: float = 0.2,
    window_length: int = 9,
    polyorder: int = 2,
    metric: Metric = "se2",
    rot_scale: float = 1.0,
    leave_one_out: bool = True,
    chunk: int = 1024,
) -> torch.Tensor:
    """Flow divergence at every sample, ``(M, T)``, in 1/s, from a Gaussian KDE.

    Dividing the continuity equation ``dp/dt + div(p v) = 0`` by ``p`` gives the material
    derivative ``d/dt ln p(x_t, t) = -div(v)`` along each particle (the instantaneous change of
    variables of Chen et al. 2018), so the divergence is minus the Savitzky-Golay rate of the
    KDE log-density along each trajectory. No ``D`` multiplier is needed: the density carries
    the dimension itself.

    Caveat: a fixed-bandwidth KDE estimates the density of ``p_t`` *convolved with the kernel*.
    Wherever the distribution is narrower than ``h`` in some direction, the smoothed density
    barely changes as that direction contracts or expands, so the divergence is attenuated
    toward zero. The bandwidth must stay fixed over phase for the omitted normalization to drop
    out of the time derivative.
    """
    log_p = torch.stack(
        [
            log_density_kde(poses[:, t], bandwidth, metric, rot_scale, leave_one_out, chunk)
            for t in range(poses.shape[1])
        ],
        dim=1,
    )
    return -_rate(log_p, dt, window_length, polyorder)


def score_kernel_at(
    queries: torch.Tensor,
    samples: torch.Tensor,
    bandwidth: float = 0.2,
    metric: Metric = "se2",
    rot_scale: float = 1.0,
    chunk: int = 1024,
) -> torch.Tensor:
    """Kernel estimate of ``grad log p`` at arbitrary ``(Q, 3)`` query poses, ``(Q, D)``.

    ``samples`` is the ``(M, 3)`` sample set every query is scored against, or ``(Q, M, 3)`` for
    a set per query (e.g. each query's own phase). The kernel, metric and units are those of
    :func:`score_kernel`, in each query's body tangent space.
    """
    weights = queries.new_tensor([1.0, 1.0, rot_scale**2])
    if metric == "xy":
        weights = weights[:2]
    h2 = bandwidth**2
    per_query = samples.dim() == 3
    rows = []
    for lo in range(0, queries.shape[0], chunk):
        q = queries[lo : lo + chunk]
        s = samples[lo : lo + chunk] if per_query else samples[None]
        if metric == "se2":
            xi = se2_log(q[:, None, :], s)  # (b, M, D)
        else:
            manifold_dim(metric)
            xi = s[..., :2] - q[:, None, :2]
        wxi = xi * weights
        # softmax == normalized kernel weights, without underflow for far-apart points.
        w = torch.softmax(-(xi * wxi).sum(dim=-1) / (2 * h2), dim=1)
        rows.append((w[..., None] * wxi).sum(dim=1) / h2)
    return torch.cat(rows)


def score_kernel(
    poses: torch.Tensor,
    bandwidth: float = 0.2,
    metric: Metric = "se2",
    rot_scale: float = 1.0,
    chunk: int = 1024,
) -> torch.Tensor:
    """Kernel estimate of ``grad log p_t`` at every sample, ``(M, T, D)``.

    With ``xi_ij = Log(T_i^-1 T_j)`` and kernel ``exp(-|xi|_W^2 / 2h^2)``,
    ``W = diag(1, 1, rot_scale^2)``, the gradient of the log KDE at ``T_i`` in its body tangent
    space is ``sum_j w_ij W xi_ij / h^2`` with ``w_ij`` the normalized kernel weights. Units are
    1/m for the translational rows and 1/rad for the rotational one.
    """
    return torch.stack([
        score_kernel_at(poses[:, t], poses[:, t], bandwidth, metric, rot_scale, chunk)
        for t in range(poses.shape[1])
    ], dim=1)
