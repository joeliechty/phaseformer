"""gym-pusht rollouts: exact state placement, the inverse PD command and the nominal clock."""

from __future__ import annotations

import numpy as np
import pytest
import torch

gym_pusht = pytest.importorskip("gym_pusht")  # optional: the simulator is not a core dependency

from fm_bc.pusht.policy import PushTMLP  # noqa: E402
from fm_bc.pusht.sim import (  # noqa: E402
    NominalClock, make_env, pd_coefficients, place_state, rollout_from_states, rollout_policy,
)


def test_placement_reproduces_the_dataset_pose_exactly():
    env = make_env()
    for i in range(100):  # whatever state the env was left in
        env.np_random = np.random.default_rng(i)
        state = [100.0 + i, 400.0, 300.0, 200.0, 0.4 + 0.05 * i]
        obs = place_state(env, state)
        assert np.allclose(obs, state, atol=1e-6)


def test_inverse_pd_achieves_the_requested_displacement():
    env = make_env()
    alpha, beta, _ = pd_coefficients(env)
    obs = place_state(env, [100.0, 400.0, 400.0, 100.0, 0.3])  # pusher well clear of the block
    rng = np.random.default_rng(0)
    for _ in range(10):
        dx = rng.uniform(-15, 15, 2)
        x, v = obs[:2].copy(), np.asarray(env.agent.velocity)
        obs, *_ = env.step(x + (dx - beta * v) / alpha)
        assert np.allclose(obs[:2], x + dx, atol=1e-6)


def test_nominal_clock_is_monotone_saturates_and_keeps_stepping():
    clock = NominalClock.from_episodes(torch.tensor([50, 101, 200]))
    assert clock.median_length == 101 and clock.dphase == pytest.approx(0.01)
    phases = [clock.phase(t) for t in range(150)]
    assert phases[0] == 0.0 and phases[100] == pytest.approx(1.0) and phases[149] == 1.0
    assert all(b >= a for a, b in zip(phases, phases[1:]))


def test_rollout_runs_and_reports_the_protocol_fields():
    pos = {"scale": [250.0, 250.0], "offset": [256.0, 256.0]}
    torch.manual_seed(0)
    model = PushTMLP(pos, hidden=(8,))
    r = rollout_policy(model, NominalClock(120), seeds=(100000, 100001), max_steps=5)
    assert r["states"].shape == (2, 6, 5) and r["actions"].shape == (2, 5, 2)
    assert ((r["score"] >= 0) & (r["score"] <= 1)).all()
    assert (r["actions"] >= 0).all() and (r["actions"] <= 512).all()


def test_stretched_clock_keeps_the_median_step():
    clock = NominalClock(101, phase_length=201)
    assert clock.dphase == pytest.approx(0.01)
    assert clock.phase(100) == pytest.approx(0.5) and clock.phase(200) == 1.0
    assert NominalClock(101).phase_length == 101


def test_rollout_from_states_starts_exactly_and_runs_each_demo_length():
    pos = {"scale": [250.0, 250.0], "offset": [256.0, 256.0]}
    torch.manual_seed(0)
    model = PushTMLP(pos, hidden=(8,))
    starts = [[100.0, 400.0, 300.0, 200.0, 0.4], [400.0, 100.0, 200.0, 300.0, 2.0]]
    trajs = rollout_from_states(model, starts, [4, 7])
    assert [t.shape for t in trajs] == [(4, 5), (7, 5)]
    assert np.allclose(trajs[0][0], starts[0], atol=1e-6)


def test_chunk_rollout_executes_action_steps_per_replan():
    from fm_bc.pusht.baselines import make_pusht_baseline  # noqa: PLC0415
    from fm_bc.pusht.sim import rollout_chunk_policy  # noqa: PLC0415

    norm = {"scale": [256.0, 256.0], "offset": [256.0, 256.0]}
    torch.manual_seed(0)
    model = make_pusht_baseline("mip", pos_norm=norm, act_norm=norm, obs_steps=2, horizon=8,
                                action_steps=4, emb_dim=16, n_layers=2, timestep_emb_dim=8)
    r = rollout_chunk_policy(model, seeds=(100000, 100001), max_steps=6)
    assert r["states"].shape == (2, 7, 5) and r["actions"].shape == (2, 6, 2)
    assert ((r["score"] >= 0) & (r["score"] <= 1)).all()
    assert (r["actions"] >= 0).all() and (r["actions"] <= 512).all()


def test_temporal_ensemble_weights_match_the_skate_definition():
    pytest.importorskip("theseus")
    from fm_bc.skate.evaluation import temporal_ensemble_weights as skate_weights  # noqa: PLC0415
    from fm_bc.pusht.sim import temporal_ensemble_weights  # noqa: PLC0415

    assert np.allclose(temporal_ensemble_weights(5, 0.01), skate_weights(5, 0.01).numpy())


def test_closed_loop_replans_every_step():
    from fm_bc.pusht.baselines import make_pusht_baseline  # noqa: PLC0415
    from fm_bc.pusht.sim import rollout_chunk_policy  # noqa: PLC0415

    norm = {"scale": [256.0, 256.0], "offset": [256.0, 256.0]}
    torch.manual_seed(0)
    model = make_pusht_baseline("mip", pos_norm=norm, act_norm=norm, obs_steps=2, horizon=8,
                                action_steps=4, emb_dim=16, n_layers=2, timestep_emb_dim=8).eval()
    calls = []
    sample = model.sample
    model.sample = lambda obs, gen=None: (calls.append(1), sample(obs, gen))[1]
    r = rollout_chunk_policy(model, seeds=(100000, 100001), max_steps=6, temporal_ensemble=0.01)
    assert len(calls) == 6  # one per step, batched over both envs
    assert r["states"].shape == (2, 7, 5)
