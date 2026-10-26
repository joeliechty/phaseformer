"""Episode-aware action-chunk windows.

The property that matters is the one the fixed-horizon version never had to worry about: a
window must never read a step belonging to a *different* episode. It is asserted over every
window in the fixture rather than a sample, because a single leaked index would corrupt training
silently.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fm_bc.pusht.chunks import ACT_DIM, OBS_DIM, embed_state, make_windows
from fm_bc.pusht.constants import WORKSPACE_PX
from fm_bc.pusht.normalize import normalizer_for, stats_from_data
from fm_bc.windows import episode_window_indices, window_indices

from test_pusht_data import LENGTHS, _toy  # noqa: E402 -- pytest puts tests/ on sys.path

TO, TP = 2, 16


def _indices(ep, to: int = TO, tp: int = TP):
    return episode_window_indices(ep.episode_starts, ep.episode_ends, ep.episode_index, to, tp)


def test_one_window_per_step():
    ep = _toy()
    obs, act = _indices(ep)
    assert obs.shape == (len(ep), TO)
    assert act.shape == (len(ep), TP)


@pytest.mark.parametrize("to,tp", [(1, 1), (2, 16), (3, 8), (4, 4)])
def test_no_window_ever_leaves_its_own_episode(to: int, tp: int):
    ep = _toy()
    obs, act = _indices(ep, to, tp)
    lo = ep.episode_starts[ep.episode_index][:, None]
    hi = (ep.episode_ends[ep.episode_index] - 1)[:, None]
    for name, idx in (("obs", obs), ("act", act)):
        assert bool((idx >= lo).all()), f"{name} reached before its episode start"
        assert bool((idx <= hi).all()), f"{name} reached past its episode end"


def test_episode_boundaries_edge_pad_with_their_own_samples():
    ep = _toy()
    obs, act = _indices(ep)
    for e in range(ep.n_episodes):
        s, t = int(ep.episode_starts[e]), int(ep.episode_ends[e])
        # First step of the episode: the whole observation window is the episode's first sample,
        # never the previous episode's last.
        assert obs[s].tolist() == [s] * TO
        # Its chunk opens with the same padded past, then advances into the episode.
        assert act[s].tolist()[:TO] == [s] * TO
        assert act[s].tolist()[TO] == min(s + 1, t - 1)
        # Last step: the chunk opens at the OLDEST observed step, k - To + 1 -- not at k -- and
        # then saturates on the episode's own final sample.
        assert act[t - 1].tolist() == [max(s, t - TO)] + [t - 1] * (TP - 1)


def test_interior_windows_are_contiguous():
    ep = _toy()
    obs, act = _indices(ep)
    k = int(ep.episode_starts[2]) + 5  # comfortably inside the 11-step episode
    assert obs[k].tolist() == [k - 1, k]
    assert act[k][:3].tolist() == [k - 1, k, k + 1]


def test_matches_the_fixed_horizon_convention_for_equal_length_episodes():
    """Ties the new code to the already-tested skate convention.

    Observations are index-for-index identical. Actions differ in exactly one documented way:
    the fixed-horizon helper clamps them to ``n_steps - 1``, because the skate task stores
    ``N+1`` poses against only ``N`` controls. Push-T stores a state *and* an action at every
    step, so its actions clamp to ``L - 1``. Same padding rule, one more reachable index.
    """
    length = 9
    ep = _toy(lengths=(length, length))
    obs, act = _indices(ep)
    ref_obs, _ = window_indices(np.arange(length), length - 1, TO, TP)
    want_act = np.clip(np.arange(length)[:, None] - TO + 1 + np.arange(TP), 0, length - 1)
    for e in range(2):
        start = int(ep.episode_starts[e])
        assert (obs[start:start + length] - start).numpy().tolist() == ref_obs.tolist()
        assert (act[start:start + length] - start).numpy().tolist() == want_act.tolist()


def test_make_windows_shapes_and_gather_consistency():
    ep = _toy()
    w = make_windows(ep, to=TO, tp=TP)
    assert w["obs"].shape == (len(ep), TO, 5)
    assert w["act"].shape == (len(ep), TP, ACT_DIM)
    assert w["agent_delta"].shape == (len(ep), TP, 2)
    assert torch.allclose(w["obs"], ep.state[w["obs_idx"]], atol=0)
    assert torch.allclose(w["act"], ep.action[w["act_idx"]], atol=0)


def test_episode_ids_select_exactly_those_episodes():
    ep = _toy()
    ids = torch.tensor([1, 3])
    w = make_windows(ep, to=TO, tp=TP, episode_ids=ids)
    assert w["obs"].shape[0] == sum(LENGTHS[i] for i in ids.tolist())
    kept = ep.episode_index[w["obs_idx"][:, -1]]
    assert set(kept.tolist()) == set(ids.tolist())


def test_embed_state_dimension_and_angle_wrapping():
    ep = _toy()
    pos = normalizer_for("box", stats_from_data(ep.state[:, :2]), box=(0.0, WORKSPACE_PX))
    emb = embed_state(ep.state, pos)
    assert emb.shape == (len(ep), OBS_DIM)
    # cos/sin stay on the unit circle, so the 2*pi seam is not a discontinuity in the input.
    assert torch.allclose(emb[:, 4] ** 2 + emb[:, 5] ** 2, torch.ones(len(ep)), atol=1e-5)


def test_window_indices_rejects_degenerate_horizons():
    ep = _toy()
    with pytest.raises(ValueError, match="to >= 1"):
        _indices(ep, to=0, tp=4)
