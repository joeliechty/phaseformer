"""Push-T MMD to the held-out demonstrations."""

from __future__ import annotations

import math

import pytest
import torch

from fm_bc.pusht.evaluation import per_phase_mmd, phase_grid, state_mmd


def test_mmd_is_zero_for_identical_sets_and_grows_with_a_shift():
    torch.manual_seed(0)
    a = torch.rand(30, 5, dtype=torch.float64) * 400
    assert state_mmd(a, a) == pytest.approx(0.0, abs=1e-12)
    near, far = a.clone(), a.clone()
    near[:, 2] += 20
    far[:, 2] += 200
    assert 0 < state_mmd(a, near) < state_mmd(a, far)


def test_heading_wraps():
    a = torch.tensor([[100.0, 100.0, 200.0, 200.0, 0.01]])
    b = torch.tensor([[100.0, 100.0, 200.0, 200.0, 2 * math.pi - 0.01]])
    assert state_mmd(a, b) < 1e-3


def test_phase_grid_reads_each_trajectory_on_its_own_clock():
    short = torch.arange(3, dtype=torch.float64)[:, None].repeat(1, 5)   # steps 0, 1, 2
    long = torch.arange(11, dtype=torch.float64)[:, None].repeat(1, 5)   # steps 0..10
    g = phase_grid([short, long], n_grid=3)
    assert g.shape == (3, 2, 5)
    assert g[:, 0, 0].tolist() == [0, 1, 2] and g[:, 1, 0].tolist() == [0, 5, 10]
    r = per_phase_mmd([short, long], [short, long], n_grid=3)
    assert r["mmd_mean"] == pytest.approx(0.0, abs=1e-12) and len(r["mmd_per_phase"]) == 3
