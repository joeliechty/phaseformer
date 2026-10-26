import pytest
import theseus as th
import torch

from fm_bc import se2
from fm_bc.se2 import pairwise_twists, se2_exp_retract, se2_log, twist_norm


def _poses(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.randn(n, 2, generator=g, dtype=torch.double) * 3
    th_ = (torch.rand(n, 1, generator=g, dtype=torch.double) * 2 - 1) * torch.pi
    return torch.cat([xy, th_], dim=1)


def test_log_matches_theseus_local():
    a, b = _poses(64, 0), _poses(64, 1)
    ours = se2_log(a, b)
    ref = th.local(th.SE2(x_y_theta=a), th.SE2(x_y_theta=b))
    torch.testing.assert_close(ours, ref, atol=1e-9, rtol=1e-9)


def test_retract_matches_theseus():
    a = _poses(64, 2)
    xi = torch.randn(64, 3, dtype=torch.double)
    xi[:8, 2] = 1e-7  # exercise the small-angle branch
    ours = se2_exp_retract(a, xi)
    ref = th.SE2(x_y_theta=a).retract(xi)
    ref_xyt = torch.stack(
        [ref.tensor[:, 0], ref.tensor[:, 1], torch.atan2(ref.tensor[:, 3], ref.tensor[:, 2])], -1
    )
    torch.testing.assert_close(ours, ref_xyt, atol=1e-9, rtol=1e-9)


def test_log_antisymmetric_and_roundtrip():
    a, b = _poses(64, 3), _poses(64, 4)
    # Log(g^-1) = -Log(g), so pairwise distances are symmetric.
    torch.testing.assert_close(se2_log(b, a), -se2_log(a, b), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(twist_norm(se2_log(b, a)), twist_norm(se2_log(a, b)))
    torch.testing.assert_close(se2_exp_retract(a, se2_log(a, b)), b, atol=1e-9, rtol=1e-9)


def test_pairwise_shapes_and_diagonal():
    p = _poses(10)
    xi = pairwise_twists(p)
    assert xi.shape == (10, 10, 3)
    torch.testing.assert_close(xi.diagonal(dim1=0, dim2=1).T, torch.zeros(10, 3, dtype=p.dtype))


def test_legacy_half_cot_is_the_same_function_and_always_resets():
    theta = torch.linspace(-3.0, 3.0, 2001, dtype=torch.double)
    now = se2._half_cot_half(theta)
    with se2.legacy_half_cot():
        old = se2._half_cot_half(theta)
    # Equal in exact arithmetic; in float64 the old form loses bits where 1 - cos cancels, near
    # the small-angle cutoff.
    torch.testing.assert_close(old, now, atol=1e-9, rtol=0)
    assert torch.equal(se2._half_cot_half(theta), now)
    with pytest.raises(RuntimeError), se2.legacy_half_cot():
        raise RuntimeError
    assert not se2._LEGACY_HALF_COT
