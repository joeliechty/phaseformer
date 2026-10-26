"""The estimators against flows whose divergence is known in closed form."""

import math

import pytest
import torch

from fm_bc.benchmarks import kde_attenuation, linear_flow
from fm_bc.divergence import divergence_kde, divergence_knn, score_kernel
from fm_bc.se2 import se2_exp_retract


def _cloud(n=400, sigma=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, generator=g, dtype=torch.double) * sigma


def _left_moved(n=200):
    """``T_i(t) = g(t) T_i``: relative poses ``T_i^-1 T_j`` never change, so neither does any
    distance or density."""
    base = torch.cat([_cloud(n, 1.0)[:, :2], _cloud(n, 0.5, 1)[:, 2:]], dim=1)
    times = torch.linspace(0, 3, 25, dtype=torch.double)
    g_twist = torch.tensor([1.0, 0.3, 0.7], dtype=torch.double)
    traj = []
    for t in times:
        g = se2_exp_retract(torch.zeros(3, dtype=torch.double), t * g_twist)
        c, s = torch.cos(g[2]), torch.sin(g[2])
        xy = base[:, :2] @ torch.stack([torch.stack([c, s]), torch.stack([-s, c])]) + g[:2]
        traj.append(torch.cat([xy, (base[:, 2:] + g[2])], dim=1))
    return torch.stack(traj, dim=1), float(times[1] - times[0])


def test_common_left_motion_has_zero_divergence():
    poses, dt = _left_moved()
    assert divergence_knn(poses, dt).abs().max() < 1e-6
    assert divergence_kde(poses, dt).abs().max() < 1e-6


@pytest.mark.parametrize("metric,dim", [("se2", 3), ("xy", 2)])
def test_isotropic_tangent_scaling(metric, dim):
    # Particles at c * Exp(e^{a t} xi_i): an isotropic expansion at rate a in all 3 tangent
    # directions, so the flow's divergence is 3a (and 2a for its xy projection).
    a = 0.4
    xi = _cloud(500, 0.02)
    center = torch.tensor([2.0, -1.0, 0.8], dtype=torch.double)
    times = torch.linspace(0, 2, 21, dtype=torch.double)
    poses = torch.stack(
        [se2_exp_retract(center.expand(500, 3), math.exp(a * t) * xi) for t in times], dim=1
    )
    div = divergence_knn(poses, dt=float(times[1] - times[0]), metric=metric)
    assert div.mean().item() == pytest.approx(dim * a, rel=0.05)


def test_knn_anisotropic_compression():
    # Only y contracts: the D multiplier is still right, because D * dlog(r_k)/dt is the rate
    # of the kNN *density* estimate, not an isotropy assumption.
    poses, dt = linear_flow((0.0, -0.4, 0.0))
    assert divergence_knn(poses, dt, k=16).mean().item() == pytest.approx(-0.4, rel=0.05)


def test_kde_recovers_compression_when_bandwidth_is_small():
    poses, dt = linear_flow((0.0, -0.4, 0.0))
    assert divergence_kde(poses, dt, bandwidth=0.1).mean().item() == pytest.approx(-0.4, rel=0.1)


def test_kde_attenuates_when_bandwidth_exceeds_spread():
    # Heading spread is 0.15 rad; a 0.4 kernel smooths it away, so contracting only heading
    # barely changes the smoothed density. This is the KDE's known bias, pinned here on purpose.
    poses, dt = linear_flow((0.0, 0.0, -0.4))
    assert abs(divergence_kde(poses, dt, bandwidth=0.4).mean().item()) < 0.25 * 0.4


def test_kde_attenuation_matches_gaussian_theory():
    # A short window keeps sigma nearly constant, so the closed form at the mid-window spread
    # should predict the recovered fraction.
    rate, t_end, s0 = -0.3, 1.0, 0.15
    poses, dt = linear_flow((rate, 0.0, 0.0), sigma=(s0, 0.3, 0.15), t_end=t_end, n_steps=11)
    sigma_mid = s0 * math.exp(rate * t_end / 2)
    ratio = divergence_kde(poses, dt, bandwidth=0.2).mean().item() / rate
    assert ratio == pytest.approx(kde_attenuation(0.2 / sigma_mid), abs=0.1)


def test_score_points_to_mode():
    xi = _cloud(800, 0.3, seed=5)
    center = torch.tensor([1.0, 1.0, 0.0], dtype=torch.double)
    poses = se2_exp_retract(center.expand(800, 3), xi)[:, None, :]
    score = score_kernel(poses, bandwidth=0.15)[:, 0]
    # For a Gaussian the true score is -xi / sigma^2: toward the mode in the tangent space.
    cos = torch.nn.functional.cosine_similarity(score, -xi, dim=1)
    far = xi.norm(dim=1) > 0.3  # the direction is ill-defined right at the mode
    assert cos[far].mean() > 0.9
