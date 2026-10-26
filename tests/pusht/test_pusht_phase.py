"""The supervised phase estimator and its monotone rollout wrapper."""

from __future__ import annotations

import pytest
import torch

from fm_bc.pusht.phase_mlp import MonotonePhase, PhaseClockTransformer, PhaseMLP, load_phase_mlp

POS = {"scale": [250.0, 250.0], "offset": [256.0, 256.0]}


def test_estimate_is_a_probability_and_accepts_a_window():
    torch.manual_seed(0)
    model = PhaseMLP(POS, hidden=(8, 8))
    state = torch.rand(4, 5) * 300
    rho = model(state)
    assert rho.shape == (4,) and ((rho > 0) & (rho < 1)).all()
    assert torch.allclose(model(state[:, None]), rho)  # a 1-step window is the same call


def test_wrapper_never_moves_backwards_and_never_freezes():
    torch.manual_seed(0)
    clock = MonotonePhase(PhaseMLP(POS, hidden=(8, 8)), eps=0.01)
    clock.reset()
    state = torch.rand(5) * 300
    phases, steps = [], []
    for t in range(50):
        rho, drho = clock.advance(t, state)  # the same state every step: the estimate cannot rise
        phases.append(rho)
        steps.append(drho)
    assert all(b >= a for a, b in zip(phases, phases[1:]))
    assert all(d >= 0.01 - 1e-9 for d in steps[1:])
    assert max(phases) <= 1.0


def test_wrapper_follows_an_estimate_that_rises_faster_than_eps():
    class Ramp(PhaseMLP):
        def forward(self, state, start=None):  # rho = x / 512
            state = state[:, -1] if state.dim() == 3 else state
            return state[:, 0] / 512.0

    clock = MonotonePhase(Ramp(POS, hidden=(4,)), eps=1e-4)
    clock.reset()
    rho0, _ = clock.advance(0, torch.tensor([51.2, 0.0, 0.0, 0.0, 0.0]))
    rho1, drho = clock.advance(1, torch.tensor([256.0, 0.0, 0.0, 0.0, 0.0]))
    assert rho0 == pytest.approx(0.1) and rho1 == pytest.approx(0.5)
    assert drho == pytest.approx(0.4)


def test_smoothing_low_passes_the_estimate():
    class Spike(PhaseMLP):
        def forward(self, state, start=None):
            state = state[:, -1] if state.dim() == 3 else state
            return state[:, 0] / 512.0

    clock = MonotonePhase(Spike(POS, hidden=(4,)), eps=1e-4, smooth=0.25)
    clock.reset()
    clock.advance(0, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))
    rho, _ = clock.advance(1, torch.tensor([512.0, 0.0, 0.0, 0.0, 0.0]))
    assert rho == pytest.approx(0.25)  # a quarter of the way to the spike


def test_without_the_ratchet_the_clock_is_the_estimate():
    class Ramp(PhaseMLP):
        def forward(self, state, start=None):  # rho = x / 512
            state = state[:, -1] if state.dim() == 3 else state
            return state[:, 0] / 512.0

    clock = MonotonePhase(Ramp(POS, hidden=(4,)), eps=0.01, step=0.008, ratchet=False)
    clock.reset()
    phases = [clock.advance(t, torch.tensor([x, 0.0, 0.0, 0.0, 0.0]))
              for t, x in enumerate([256.0, 256.0, 128.0])]
    assert [rho for rho, _ in phases] == pytest.approx([0.5, 0.5, 0.25])  # stands, then backs up
    assert all(drho == 0.008 for _, drho in phases)  # the fixed step, whatever the estimate does
    with pytest.raises(ValueError):  # its own increment would be 0, then negative
        MonotonePhase(Ramp(POS, hidden=(4,)), ratchet=False)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = PhaseMLP(POS, hidden=(8, 8))
    model.save(tmp_path / "m.pt", metrics={"rmse": 0.1})
    loaded, extra = load_phase_mlp(tmp_path / "m.pt")
    state = torch.rand(3, 5) * 300
    assert torch.allclose(loaded(state), model(state)) and extra["metrics"]["rmse"] == 0.1


def test_start_pose_variant_uses_the_episode_start():
    torch.manual_seed(0)
    model = PhaseMLP(POS, hidden=(8, 8), use_start=True)
    state, start = torch.rand(4, 5) * 300, torch.rand(1, 5) * 300
    rho = model(state, start)
    assert rho.shape == (4,) and ((rho > 0) & (rho < 1)).all()
    assert not torch.allclose(rho, model(state, torch.rand(1, 5) * 300))  # the start matters
    with pytest.raises(ValueError):
        model(state)


def test_wrapper_holds_the_start_state_of_the_episode():
    torch.manual_seed(0)
    clock = MonotonePhase(PhaseMLP(POS, hidden=(8, 8), use_start=True), eps=1e-4)
    clock.reset()
    first = torch.tensor([10.0, 20.0, 30.0, 40.0, 0.5])
    clock.advance(0, first)
    clock.advance(1, torch.rand(5) * 300)
    assert torch.allclose(clock.start.flatten(), first)
    clock.reset()
    assert clock.start is None


# -- the transformer clock: the robot and the object as separate tokens -------------------------

_TINY = {"hidden_dim": 16, "num_layers": 1, "num_heads": 2}


def test_transformer_clock_is_a_probability_and_accepts_a_window():
    torch.manual_seed(0)
    model = PhaseClockTransformer(POS, **_TINY).eval()
    state = torch.rand(4, 5) * 300
    rho = model(state)
    assert rho.shape == (4,) and ((rho > 0) & (rho < 1)).all()
    assert torch.allclose(model(state[:, None]), rho)  # a 1-step window is the same call


def test_transformer_clock_tokens_are_robot_object_and_start():
    torch.manual_seed(0)
    model = PhaseClockTransformer(POS, use_start=True, **_TINY).eval()
    state, start = torch.rand(4, 5) * 300, torch.rand(1, 5) * 300
    tokens = model.tokens(state, start)
    assert tokens.shape == (4, 4, 16)  # readout, robot, object now, object at the start
    moved = state.clone()
    moved[:, :2] += 10.0  # the agent alone moves: only the robot token may change
    changed = (model.tokens(moved, start) - tokens).abs().amax(dim=(0, 2)) > 0
    assert changed.tolist() == [False, True, False, False]
    turned = state.clone()
    turned[:, 2:] += 0.3  # the block alone moves: only the current-object token may change
    changed = (model.tokens(turned, start) - tokens).abs().amax(dim=(0, 2)) > 0
    assert changed.tolist() == [False, False, True, False]
    # One projection serves the object now and at the start; the slot embedding tells them apart.
    same = model.tokens(state, state)
    assert torch.allclose(same[:, 2] - model.slot_emb[0, 2], same[:, 3] - model.slot_emb[0, 3])


def test_transformer_clock_needs_and_uses_the_start():
    torch.manual_seed(0)
    model = PhaseClockTransformer(POS, use_start=True, **_TINY).eval()
    state, start = torch.rand(4, 5) * 300, torch.rand(1, 5) * 300
    rho = model(state, start)
    assert rho.shape == (4,) and ((rho > 0) & (rho < 1)).all()
    assert not torch.allclose(rho, model(state, torch.rand(1, 5) * 300))
    with pytest.raises(ValueError):
        model(state)


def test_transformer_clock_round_trips_and_drives_the_wrapper(tmp_path):
    torch.manual_seed(0)
    model = PhaseClockTransformer(POS, use_start=True, **_TINY).eval()
    model.save(tmp_path / "xf.pt", metrics={"rmse": 0.01})
    loaded, extra = load_phase_mlp(tmp_path / "xf.pt")
    state, start = torch.rand(3, 5) * 300, torch.rand(1, 5) * 300
    assert type(loaded) is PhaseClockTransformer and extra["metrics"]["rmse"] == 0.01
    assert torch.allclose(loaded(state, start), model(state, start))
    # The MLP still loads as the MLP: dispatch is on the checkpoint's kind.
    PhaseMLP(POS, hidden=(8, 8)).save(tmp_path / "mlp.pt")
    assert type(load_phase_mlp(tmp_path / "mlp.pt")[0]) is PhaseMLP
    clock = MonotonePhase(loaded, eps=0.01)
    clock.reset()
    phases = [clock.advance(t, torch.rand(5) * 300)[0] for t in range(5)]
    assert all(b > a for a, b in zip(phases, phases[1:]))


@pytest.mark.parametrize("hidden_dim, num_layers, n_params, mlp_twin, mlp_params", [
    (104, 2, 263_849, (512, 512), 268_801),
    (148, 2, 531_765, (724, 724), 533_589),
    (104, 4, 526_137, (512, 512, 512), 531_457),
    (148, 4, 1_061_309, (724, 724, 724), 1_058_489),
])
def test_transformer_clock_sizes_match_their_mlp_twins(hidden_dim, num_layers, n_params,
                                                       mlp_twin, mlp_params):
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(PhaseClockTransformer(POS, hidden_dim, num_layers, use_start=True)) == n_params
    assert n(PhaseMLP(POS, mlp_twin, use_start=True)) == mlp_params
    assert abs(n_params / mlp_params - 1) < 0.03
