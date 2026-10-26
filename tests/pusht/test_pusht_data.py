"""Push-T data pipeline: container invariants, splits, normalizers and the phase grid.

Everything here runs on a tiny synthetic fixture, so the suite needs neither the 30 MB download
nor a GPU. One test additionally pins the property that makes that possible: the Push-T modules
must import without Theseus.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fm_bc.se2 import wrap_angle
from fm_bc.pusht.constants import DT, WORKSPACE_PX
from fm_bc.pusht.data import PushTEpisodes, derive_indices
from fm_bc.pusht.grid import resample_to_phase_grid
from fm_bc.pusht.normalize import MODES, normalizer_for, stats_from_data
from fm_bc.pusht.prepared import PushTPrepared, prepare
from fm_bc.pusht.splits import (
    TEST, TRAIN, UNUSED, VAL, episode_split, ratio_split, subset_episodes, val_mask,
)

LENGTHS = (7, 4, 11, 4, 9)


def _toy(lengths=LENGTHS, seed: int = 0) -> PushTEpisodes:
    """Synthetic episodes: the agent walks a noisy ramp, the block drifts, and episode 2's
    heading deliberately crosses the 2*pi -> 0 seam."""
    g = torch.Generator().manual_seed(seed)
    n = sum(lengths)
    t = torch.arange(n, dtype=torch.float32)[:, None]
    state = torch.cat(
        [
            50 + 3 * t + torch.rand(n, 2, generator=g),  # agent xy
            200 + 2 * t + torch.rand(n, 2, generator=g),  # block xy
            torch.zeros(n, 1),  # heading, filled per-episode below
        ],
        dim=1,
    )
    ends = torch.tensor(lengths).cumsum(0)
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    for e, (s, t_end) in enumerate(zip(starts.tolist(), ends.tolist())):
        # Episode 2 sweeps through 2*pi; the others stay in a safe band.
        base = 6.0 if e == 2 else 0.5 + e
        state[s:t_end, 4] = (base + 0.4 * torch.arange(t_end - s, dtype=torch.float32)) % (2 * math.pi)
    action = state[:, :2] + torch.randn(n, 2, generator=g)
    return PushTEpisodes.from_arrays(state, action.clamp(0, WORKSPACE_PX), ends)


# -- container ---------------------------------------------------------------------------


def test_derived_fields_match_episode_ends():
    ep = _toy()
    assert ep.n_episodes == len(LENGTHS)
    assert len(ep) == sum(LENGTHS)
    assert ep.lengths.tolist() == list(LENGTHS)
    for e, length in enumerate(LENGTHS):
        sl = ep.episode_slice(e)
        assert ep.step_index[sl].tolist() == list(range(length))
        assert ep.episode_index[sl].tolist() == [e] * length
        assert ep.phase[sl][0] == 0.0 and ep.phase[sl][-1] == 1.0
    ep.check()


def test_check_rejects_corrupted_index_field():
    ep = _toy()
    ep.step_index = ep.step_index.clone()
    ep.step_index[3] += 1
    with pytest.raises(ValueError, match="step_index"):
        ep.check()


def test_check_rejects_non_monotone_episode_ends():
    ep = _toy()
    ep.episode_ends = torch.tensor([7, 5, 18, 22, 31])
    with pytest.raises(ValueError, match="strictly increasing"):
        ep.check()


def test_agent_delta_is_forward_difference_with_edge_hold():
    ep = _toy()
    for e in range(ep.n_episodes):
        s, t = int(ep.episode_starts[e]), int(ep.episode_ends[e])
        want = ep.state[s + 1:t, :2] - ep.state[s:t - 1, :2]
        assert torch.allclose(ep.agent_delta[s:t - 1], want, atol=1e-5)
        # The final step has no t+1: it repeats the previous delta rather than reading as a stop.
        assert torch.allclose(ep.agent_delta[t - 1], ep.agent_delta[t - 2], atol=1e-5)
        assert ep.agent_delta[t - 1].abs().sum() > 0


def test_vel_action_is_action_minus_agent_position():
    ep = _toy()
    assert torch.allclose(ep.vel_action, ep.action - ep.state[:, :2], atol=1e-5)


def test_single_step_episode_gets_zero_phase_and_delta():
    starts, index, step, phase = derive_indices(torch.tensor([1, 4]), 4)
    assert phase[0] == 0.0  # no denominator to divide by
    assert phase[1:].tolist() == [0.0, 0.5, 1.0]


def test_save_load_round_trip(tmp_path: Path):
    ep = _toy()
    path = tmp_path / "episodes.npz"
    ep.save(path)
    back = PushTEpisodes.load(path)
    for name, value in ep.__dict__.items():
        if isinstance(value, torch.Tensor):
            got = getattr(back, name)
            assert got.dtype == value.dtype, name
            assert torch.allclose(got, value, atol=0), name
    assert back.meta == ep.meta


def test_load_rejects_a_corrupted_artifact(tmp_path: Path):
    ep = _toy()
    path = tmp_path / "bad.npz"
    ep.save(path)
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["phase"] = arrays["phase"] + 0.1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="phase"):
        PushTEpisodes.load(path)


# -- splits ------------------------------------------------------------------------------


def test_split_reproduces_diffusion_policy_counts():
    # 206 episodes at val_ratio 0.02 -> 4 validation; the rest downsampled to 90 training.
    split = episode_split(206, val_ratio=0.02, seed=42, max_train_episodes=90)
    assert (split == VAL).sum() == 4
    assert (split == TRAIN).sum() == 90
    assert (split == UNUSED).sum() == 206 - 94


def test_split_is_deterministic_and_disjoint():
    a = episode_split(206)
    b = episode_split(206)
    assert np.array_equal(a, b)
    assert not np.any((a == TRAIN) & (a == VAL))


def test_val_mask_matches_the_reference_formula():
    for n, ratio in ((206, 0.02), (50, 0.1), (10, 0.5)):
        expected = min(max(1, round(n * ratio)), n - 1)
        assert val_mask(n, ratio).sum() == expected


def test_max_train_episodes_none_keeps_every_non_val_episode():
    split = episode_split(206, max_train_episodes=None)
    assert (split == TRAIN).sum() == 202
    assert (split == UNUSED).sum() == 0


def test_subset_episodes_never_returns_a_validation_id():
    split = episode_split(206)
    train = np.flatnonzero(split == TRAIN)
    val = np.flatnonzero(split == VAL)
    picked = subset_episodes(train, 20, np.random.default_rng(0))
    assert len(picked) == 20
    assert len(np.intersect1d(picked, val)) == 0
    assert np.array_equal(picked, np.sort(picked))


# -- normalizers -------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_normalizer_round_trips(mode: str):
    x = torch.randn(200, 3) * 50 + 100
    n = normalizer_for(mode, stats_from_data(x), box=(0.0, WORKSPACE_PX))
    assert torch.allclose(n.invert(n.apply(x)), x, atol=1e-3)


def test_box_mode_maps_the_workspace_onto_the_unit_interval():
    x = torch.tensor([[0.0, 0.0], [WORKSPACE_PX, WORKSPACE_PX]])
    n = normalizer_for("box", stats_from_data(x), box=(0.0, WORKSPACE_PX))
    out = n.apply(x)
    assert torch.allclose(out[0], torch.full((2,), -1.0), atol=1e-4)
    assert torch.allclose(out[1], torch.full((2,), 1.0), atol=1e-4)


@pytest.mark.parametrize("mode", MODES)
def test_constant_column_stays_finite(mode: str):
    x = torch.cat([torch.randn(100, 1), torch.full((100, 1), 7.0)], dim=1)
    n = normalizer_for(mode, stats_from_data(x), box=None)
    out = n.apply(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(n.invert(out), x, atol=1e-3)


def test_scale_offset_follows_the_reference_convention():
    x = torch.randn(50, 2) * 10
    n = normalizer_for("gaussian", stats_from_data(x), box=None)
    # raw == scale * normalized + offset, with scale a denominator.
    assert torch.allclose(n.apply(x), (x - n.offset) / n.scale, atol=1e-5)


# -- phase grid --------------------------------------------------------------------------


def test_grid_endpoints_are_exact():
    ep = _toy()
    grid = resample_to_phase_grid(ep, n_grid=16)
    for e in range(ep.n_episodes):
        s, t = int(ep.episode_starts[e]), int(ep.episode_ends[e])
        # Positions land exactly; the heading agrees modulo 2*pi because the grid re-emits it
        # in (-pi, pi] while the source is [0, 2*pi).
        assert torch.allclose(grid.grid_state[e, 0, :4], ep.state[s, :4], atol=1e-3)
        assert torch.allclose(grid.grid_state[e, -1, :4], ep.state[t - 1, :4], atol=1e-3)
        for got, want in ((grid.grid_state[e, 0, 4], ep.state[s, 4]),
                          (grid.grid_state[e, -1, 4], ep.state[t - 1, 4])):
            assert abs(math.remainder(float(got - want), 2 * math.pi)) < 1e-4


def test_grid_phase_is_monotone_and_spans_the_unit_interval():
    grid = resample_to_phase_grid(_toy(), n_grid=16)
    assert bool((grid.grid_phase.diff() > 0).all())
    assert grid.grid_phase[0] == 0.0 and grid.grid_phase[-1] == 1.0


def test_identity_resample_reproduces_the_episode():
    """An episode whose length already equals n_grid must survive unchanged."""
    ep = _toy(lengths=(9, 9))
    grid = resample_to_phase_grid(ep, n_grid=9)
    assert torch.allclose(grid.grid_state[0, :, :4], ep.state[:9, :4], atol=1e-3)


def test_linear_ramp_resamples_to_a_linear_ramp():
    ep = _toy(lengths=(5,))
    ep.state[:, 0] = torch.linspace(0.0, 40.0, 5)
    grid = resample_to_phase_grid(ep, n_grid=9)
    assert torch.allclose(grid.grid_state[0, :, 0], torch.linspace(0.0, 40.0, 9), atol=1e-3)


def test_grid_dt_is_per_episode():
    """A single global dt would silently mis-scale every short episode's divergence."""
    ep = _toy()
    n_grid = 16
    grid = resample_to_phase_grid(ep, n_grid=n_grid)
    want = (grid.lengths - 1).float() * DT / (n_grid - 1)
    assert torch.allclose(grid.grid_dt, want, atol=1e-6)
    assert float(grid.grid_dt.max() / grid.grid_dt.min()) > 1.0


def test_heading_takes_the_short_way_across_the_seam():
    """Episode 2 sweeps through 2*pi; the resampled heading must follow the short arc.

    The check is on the *wrapped* difference between consecutive grid points. The raw stored
    difference necessarily shows a ~2*pi step wherever the trajectory crosses the (-pi, pi]
    boundary -- that is a property of storing a wrapped angle, not of the interpolation. What
    would signal a real bug is a wrapped step near pi, meaning the geodesic went the long way
    round. The fixture advances 0.4 rad per source step and is upsampled here, so every wrapped
    step should be well under that.
    """
    grid = resample_to_phase_grid(_toy(), n_grid=32)
    theta = grid.grid_state[..., 4]
    assert bool((theta.abs() <= math.pi + 1e-4).all())
    step = wrap_angle(theta.diff(dim=1))
    assert float(step.abs().max()) < 0.4 + 1e-3


def test_grid_rejects_a_degenerate_episode():
    ep = _toy(lengths=(1, 5))
    with pytest.raises(ValueError, match="at least 2 steps"):
        resample_to_phase_grid(ep, n_grid=8)


# -- prepared artifact -------------------------------------------------------------------


def test_prepare_round_trip(tmp_path: Path):
    ep = _toy()
    prep = prepare(ep, n_grid=8, val_ratio=0.2, seed=0, max_train_episodes=3)
    path = tmp_path / "prepared.npz"
    prep.save(path)
    back = PushTPrepared.load(path)
    assert torch.equal(back.split, prep.split)
    assert torch.allclose(back.grid.grid_state, prep.grid.grid_state, atol=0)
    assert torch.allclose(back.grid.grid_dt, prep.grid.grid_dt, atol=0)
    assert sorted(back.norm) == sorted(prep.norm)
    for key in prep.norm:
        for mode in prep.norm[key]:
            assert torch.allclose(back.norm[key][mode].scale, prep.norm[key][mode].scale)
    assert back.meta == prep.meta


def test_normalizer_stats_ignore_validation_episodes():
    """Stats fitted on a split must not move when a held-out episode's values change."""
    ep = _toy()
    prep = prepare(ep, n_grid=8, val_ratio=0.2, seed=0, max_train_episodes=4)
    val = int(prep.val_ids[0])
    s, t = int(ep.episode_starts[val]), int(ep.episode_ends[val])
    ep.state[s:t, :2] += 1000.0  # wildly out of range, but held out
    again = prepare(ep, n_grid=8, val_ratio=0.2, seed=0, max_train_episodes=4)
    assert torch.allclose(prep.normalizer("agent_pos", "minmax").scale,
                          again.normalizer("agent_pos", "minmax").scale, atol=1e-5)


def test_grid_slices_feed_both_divergence_metrics():
    grid = resample_to_phase_grid(_toy(), n_grid=8)
    assert grid.agent_positions().shape[-1] == 2  # metric="xy", D = 2
    assert grid.block_poses().shape[-1] == 3  # metric="se2", D = 3


# -- import hygiene ----------------------------------------------------------------------


def test_pusht_imports_without_theseus():
    """The Push-T path must import with torch alone -- that is what lets it run off the GPU box.

    A meta_path hook makes `theseus` and `skate_planner` unimportable, which is the situation on
    any machine where theseus-ai's cp38-cp310 wheels do not apply.
    """
    code = """
import sys

class Block:
    def find_module(self, name, path=None):
        return self if name.split('.')[0] in ('theseus', 'skate_planner') else None
    def load_module(self, name):
        raise ImportError(f'{name} is blocked')
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('theseus', 'skate_planner'):
            raise ImportError(f'{name} is blocked')
        return None

sys.meta_path.insert(0, Block())
import fm_bc
import fm_bc.se2
import fm_bc.windows
import fm_bc.skate  # lazy: resolves nothing until a name is used
import fm_bc.pusht.data
import fm_bc.pusht.grid
import fm_bc.pusht.prepared
import fm_bc.pusht.chunks
import fm_bc.pusht.progress
import fm_bc.pusht.progress_training
import fm_bc.pusht.policy
import fm_bc.pusht.policy_training
import fm_bc.pusht.sim
import fm_bc.pusht.evaluation
import fm_bc.pusht.phase_mlp
import fm_bc.pusht.baselines
print('ok')
"""
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root)
    assert out.returncode == 0, f"stdout={out.stdout}\nstderr={out.stderr}"
    assert "ok" in out.stdout


def test_ratio_split_uses_every_episode_at_70_20_10():
    split = ratio_split(206, (0.7, 0.2, 0.1), seed=42)
    assert [(split == k).sum() for k in (TRAIN, VAL, TEST, UNUSED)] == [144, 41, 21, 0]
    assert np.array_equal(split, ratio_split(206, (0.7, 0.2, 0.1), seed=42))
    assert not np.array_equal(split, ratio_split(206, (0.7, 0.2, 0.1), seed=0))
    with pytest.raises(ValueError):
        ratio_split(206, (0.7, 0.2, 0.2))


def test_prepared_ratio_split_fits_normalizers_on_train_and_holds_out_test():
    ep = _toy(lengths=(7, 4, 11, 4, 9, 6, 8, 5, 10, 6))
    prep = prepare(ep, n_grid=8, split_mode="ratio", fractions=(0.6, 0.2, 0.2), seed=1)
    assert prep.meta["split"]["n_test"] == 2 and prep.meta["split"]["n_val"] == 2
    assert torch.equal(prep.held_out_ids, prep.test_ids)
    train_steps = torch.isin(ep.episode_index, prep.train_ids)
    want = ep.state[train_steps, :2].min(0).values.to(torch.float64)
    got = prep.normalizer("agent_pos", "minmax")
    assert torch.allclose(got.invert(torch.full((2,), -0.999999)).to(torch.float64), want, atol=1e-3)
