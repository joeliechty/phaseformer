"""Closed-loop Push-T rollouts in ``gym-pusht``, with a phase clock and an inverse PD command.

Three pieces sit between a phase-velocity policy (:class:`fm_bc.pusht.policy.PushTMLP`) and the
simulator:

**Placement.** The dataset's block pose ``(block_x, block_y, theta)`` is the pymunk *body-origin*
pose -- the demonstrations' keypoints lie on the env's T polygon in that frame to ~1 px. gym-pusht's
``reset(options={"reset_to_state": ...})`` keeps a legacy order (position, then angle) that under
pymunk 6.11 rotates the body about its centre of gravity and misplaces the block by ~60 px, so
:func:`place_state` sets the angle first. With that, replaying a demonstration's actions tracks its
recorded block position to a median 0.03 px over 20 steps.

**Phase clock.** The policy's ``rho = t / (L - 1)`` is an oracle on demonstrations; a rollout does
not know ``L``. :class:`NominalClock` substitutes the median training-episode length:
``rho_t = min(t / (L_med - 1), 1)`` and a constant ``drho = 1 / (L_med - 1)``. The step size stays
nominal after ``rho`` saturates at 1, so the robot never freezes; the clock is open loop and
strictly monotone, the zero-parameter baseline any state-based phase estimate has to beat.

**Command.** The env's action is not a displacement but the target of a PD loop (``k_p = 100``,
``k_v = 20``, ten 10 ms substeps per 100 ms control step) on a kinematic pusher, so a target at
``x + dx`` covers only part of ``dx`` and the pusher carries momentum -- the ~2-step lag between
``action[t]`` and the agent in the data. The policy predicts the *achieved* displacement
``dx = u * drho`` (it is trained on ``agent_delta``), so ``command="inverse_pd"`` inverts that loop:
it is linear, ``x' = x + alpha (a - x) + beta v``, and contacts never act on a kinematic body, so
``a = x + (dx - beta v) / alpha`` achieves ``dx`` exactly (up to the action-space clip). It reads
the pusher's velocity from the simulator, the analogue of a robot's joint-velocity sensing.
``command="target"`` is the naive ``a = x + dx``.

**Protocol** (Diffusion Policy's ``pusht_keypoints_runner``): episodes of at most 300 steps from
``env.reset(seed=s)``, the score of an episode is its maximum reward ``clip(coverage / 0.95, 0, 1)``
and it succeeds once coverage exceeds 0.95. DP tests on seeds 100000..100049; gym-pusht draws its
own initial states from those seeds, so the states match DP's protocol but not DP's exact envs.

``gymnasium`` and ``gym_pusht`` are imported lazily, so importing this module needs torch only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from .constants import WORKSPACE_PX

__all__ = ["make_env", "place_state", "pd_coefficients", "NominalClock", "rollout_policy",
           "rollout_from_states", "rollout_chunk_policy", "temporal_ensemble_weights",
           "COMMANDS", "MAX_STEPS", "TEST_SEEDS"]

COMMANDS = ("inverse_pd", "target")
MAX_STEPS = 300  # DP's env_runner max_steps
TEST_SEEDS = tuple(range(100000, 100050))  # DP's n_test = 50 from test_start_seed


def make_env():
    """The raw (unwrapped) gym-pusht env with state observations."""
    import gymnasium as gym  # noqa: PLC0415 -- optional dependency
    import gym_pusht  # noqa: F401, PLC0415 -- registers the env

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gym.make("gym_pusht/PushT-v0", obs_type="state",
                        disable_env_checker=True).unwrapped


def place_state(env, state) -> np.ndarray:
    """Put the env in the dataset's ``[agent_x, agent_y, block_x, block_y, theta]`` exactly.

    Angle before position, so the body origin lands on the requested point (see the module
    docstring for why ``reset_to_state`` does not). Returns the observation.
    """
    state = np.asarray(state, dtype=np.float64)
    # Reset near the requested state, not to a random one: pymunk keeps the contacts of the state
    # it leaves, and a random reset's overlaps would push the placed block by up to ~1.5 px.
    env.reset(options={"reset_to_state": state})
    env.agent.position = list(state[:2])
    env.agent.velocity = (0.0, 0.0)
    env.block.angle = float(state[4])
    env.block.position = list(state[2:4])
    env.space.step(env.dt)
    return np.asarray(env.get_obs(), dtype=np.float64)


def pd_coefficients(env) -> tuple[float, float, float]:
    """``(alpha, beta, gamma)`` of one control step of the env's PD loop, per axis.

    ``x' = x + alpha (a - x) + beta v`` and ``v' = ... `` -- only the position map is needed. Read
    off by stepping the same semi-implicit update the env runs, with unit inputs, so it tracks the
    env's ``k_p``, ``k_v``, ``dt`` and substep count rather than restating them. ``gamma`` is the
    fraction of the target gap covered by the velocity after the step (reported for diagnostics).
    """
    n = int(round(1 / (env.dt * env.control_hz)))

    def step(gap: float, v: float) -> tuple[float, float]:
        x = 0.0
        for _ in range(n):
            v += (env.k_p * (gap - x) - env.k_v * v) * env.dt
            x += v * env.dt
        return x, v

    alpha, gamma = step(1.0, 0.0)
    beta, _ = step(0.0, 1.0)
    return alpha, beta, gamma


@dataclass
class NominalClock:
    """``rho_t = min(t / (L_phase - 1), 1)``, ``drho = 1 / (L_med - 1)`` -- see the module docstring.

    ``phase_length`` (default ``median_length``) stretches how slowly ``rho`` advances without
    changing the step size ``dx = u * drho``, which stays the median demonstration's.
    """

    median_length: float
    phase_length: float | None = None

    def __post_init__(self) -> None:
        if self.phase_length is None:
            self.phase_length = self.median_length

    @classmethod
    def from_episodes(cls, lengths: torch.Tensor, phase_length: float | None = None) -> "NominalClock":
        return cls(float(lengths.to(torch.float64).median()), phase_length)

    @property
    def dphase(self) -> float:
        return 1.0 / (self.median_length - 1.0)

    def phase(self, t: int) -> float:
        return min(t / (self.phase_length - 1.0), 1.0)

    def reset(self) -> None:
        """No state to reset -- the clock is open loop (the interface a state-based clock needs)."""

    def advance(self, t: int, state) -> tuple[float, float]:
        """``(rho fed to the policy, drho that scales the step)`` at step ``t``."""
        return self.phase(t), self.dphase


@torch.no_grad()
def rollout_policy(
    model,
    clock: NominalClock,
    seeds=TEST_SEEDS,
    max_steps: int = MAX_STEPS,
    command: str = "inverse_pd",
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """Roll ``model`` out closed loop from each seed's initial state.

    Returns per seed ``score`` (max reward), ``success``, ``steps`` and ``final_coverage``, and
    the padded ``states (S, max_steps + 1, 5)`` / ``actions (S, max_steps, 2)`` with ``length`` --
    a finished episode's trailing entries repeat its last state.

    ``clock`` is any object with ``reset()`` and ``advance(t, state) -> (rho, drho)``:
    :class:`NominalClock` (open loop) or :class:`fm_bc.pusht.phase_mlp.MonotonePhase`
    (a state-based estimate, bounded below so it never freezes).
    """
    if command not in COMMANDS:
        raise ValueError(f"command must be one of {COMMANDS}, got {command!r}")
    env = make_env()
    alpha, beta, _ = pd_coefficients(env)
    model.eval().to(device)
    scale = model.pos_scale.to(device)
    n = len(seeds)
    states = np.zeros((n, max_steps + 1, 5))
    actions = np.zeros((n, max_steps, 2))
    out = {k: np.zeros(n) for k in ("score", "success", "steps", "final_coverage")}
    for i, seed in enumerate(seeds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obs, _ = env.reset(seed=int(seed))
        states[i, 0] = obs
        best, t = 0.0, 0
        info = {"coverage": 0.0}
        clock.reset()
        for t in range(max_steps):
            o = torch.as_tensor(obs, dtype=torch.float32, device=device)[None, None]
            phase, dphase = clock.advance(t, obs)
            rho = torch.tensor([phase], dtype=torch.float32, device=device)
            u = model(o, rho)[0]  # (2,) normalized per unit phase
            dx = (u * scale).cpu().numpy() * dphase  # px
            x = np.asarray(obs[:2], dtype=np.float64)
            if command == "inverse_pd":
                v = np.asarray(env.agent.velocity, dtype=np.float64)
                a = x + (dx - beta * v) / alpha
            else:
                a = x + dx
            a = np.clip(a, 0.0, WORKSPACE_PX)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obs, reward, terminated, _, info = env.step(a)
            actions[i, t] = a
            states[i, t + 1] = obs
            best = max(best, float(reward))
            if terminated:
                break
        states[i, t + 2:] = states[i, t + 1]
        out["score"][i], out["success"][i] = best, float(info["coverage"] > env.success_threshold)
        out["steps"][i], out["final_coverage"][i] = t + 1, info["coverage"]
    return {**out, "states": states, "actions": actions, "seeds": np.asarray(seeds)}


@torch.no_grad()
def rollout_from_states(
    model,
    initial_states,
    lengths,
    command: str = "inverse_pd",
    device: str | torch.device = "cpu",
) -> list[np.ndarray]:
    """Roll ``model`` out from given states with each demonstration's own clock.

    Episode ``i`` starts at ``initial_states[i]`` (placed exactly, :func:`place_state`) and runs
    ``lengths[i] - 1`` steps with ``rho_k = k / (L_i - 1)`` and ``drho = 1 / (L_i - 1)``, the phase
    the policy was trained on, without stopping at success -- so the ``(L_i, 5)`` trajectories it
    returns share the demonstrations' phase axis (:mod:`fm_bc.pusht.evaluation`).
    """
    if command not in COMMANDS:
        raise ValueError(f"command must be one of {COMMANDS}, got {command!r}")
    env = make_env()
    alpha, beta, _ = pd_coefficients(env)
    model.eval().to(device)
    scale = model.pos_scale.to(device)
    out = []
    for s0, length in zip(initial_states, lengths):
        length = int(length)
        dphase = 1.0 / max(length - 1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obs = place_state(env, s0)
        traj = [obs]
        for k in range(length - 1):
            o = torch.as_tensor(obs, dtype=torch.float32, device=device)[None, None]
            rho = torch.tensor([k * dphase], dtype=torch.float32, device=device)
            dx = (model(o, rho)[0] * scale).cpu().numpy() * dphase
            x = np.asarray(obs[:2], dtype=np.float64)
            if command == "inverse_pd":
                a = x + (dx - beta * np.asarray(env.agent.velocity, dtype=np.float64)) / alpha
            else:
                a = x + dx
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obs, *_ = env.step(np.clip(a, 0.0, WORKSPACE_PX))
            traj.append(obs)
        out.append(np.asarray(traj, dtype=np.float64))
    return out


def temporal_ensemble_weights(n: int, m: float) -> np.ndarray:
    """ACT's ``exp(-m i)`` for ``n`` predictions of one step, oldest first, normalized.

    Restated from :func:`fm_bc.skate.evaluation.temporal_ensemble_weights` (that module imports
    ``skate_planner``); a test pins the two to each other.
    """
    w = np.exp(-m * np.arange(n, dtype=np.float64))
    return w / w.sum()


@torch.no_grad()
def rollout_chunk_policy(
    model,
    seeds=TEST_SEEDS,
    max_steps: int = MAX_STEPS,
    device: str | torch.device = "cpu",
    sample_seed: int = 0,
    temporal_ensemble: float | None = None,
) -> dict[str, np.ndarray]:
    """DP's protocol for a chunk baseline (:mod:`fm_bc.pusht.baselines`), open or closed loop.

    The chunk *is* the environment's action -- absolute PD targets -- so there is no phase clock
    and no inverse PD. A chunk is sampled from the last ``obs_steps`` states and its future part
    ``chunk[To-1:]`` covers this step onwards (the window convention of
    :func:`fm_bc.pusht.chunks.make_windows`).

    **Open loop** (``temporal_ensemble=None``), the policy as trained: execute ``action_steps`` of
    each chunk before re-planning. **Closed loop** (``temporal_ensemble=m``), ACT's temporal
    ensembling: re-plan every step and execute the ``exp(-m i)``-weighted mean of every chunk
    still covering it -- the same two modes the skate baselines are reported in.

    Every seed runs in lockstep so the sampler is batched over the envs, which is what makes
    Diffusion Policy's 100-step chain affordable. Returns the fields of :func:`rollout_policy`.
    """
    envs = [make_env() for _ in seeds]
    model.eval().to(device)
    to, ta = model.obs_steps, model.action_steps
    n = len(seeds)
    states = np.zeros((n, max_steps + 1, 5))
    actions = np.zeros((n, max_steps, 2))
    out = {k: np.zeros(n) for k in ("score", "success", "steps", "final_coverage")}
    history = []
    for i, (env, seed) in enumerate(zip(envs, seeds)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obs, _ = env.reset(seed=int(seed))
        states[i, 0] = obs
        history.append([np.asarray(obs, dtype=np.float64)] * to)
    gen = torch.Generator(device=device).manual_seed(sample_seed)
    active = np.ones(n, dtype=bool)
    queries: list[tuple[int, np.ndarray]] = []
    plan = np.zeros((n, ta, 2))
    for t in range(max_steps):
        if not active.any():
            break
        if temporal_ensemble is not None or t % ta == 0:
            window = torch.as_tensor(np.stack([np.stack(h[-to:]) for h in history]),
                                     dtype=torch.float32, device=device)
            future = model.sample(window, gen)[:, to - 1:].cpu().numpy()  # (n, Tp-To+1, 2)
            if temporal_ensemble is None:
                plan = future[:, :ta]
            else:
                queries.append((t, future))
                queries = [(q, f) for q, f in queries if t - q < f.shape[1]]
        if temporal_ensemble is None:
            a_all = plan[:, t % ta]
        else:
            preds = [f[:, t - q] for q, f in queries if 0 <= t - q < f.shape[1]]
            w = temporal_ensemble_weights(len(preds), temporal_ensemble)
            a_all = np.einsum("i,ibc->bc", w, np.stack(preds))
        for i, env in enumerate(envs):
            if not active[i]:
                states[i, t + 1] = states[i, t]
                history[i].append(states[i, t])
                continue
            a = np.clip(a_all[i], 0.0, WORKSPACE_PX)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obs, reward, terminated, _, info = env.step(a)
            actions[i, t] = a
            states[i, t + 1] = obs
            history[i].append(np.asarray(obs, dtype=np.float64))
            out["score"][i] = max(out["score"][i], float(reward))
            out["steps"][i] = t + 1
            out["final_coverage"][i] = info["coverage"]
            out["success"][i] = float(info["coverage"] > env.success_threshold)
            if terminated:
                active[i] = False
    for i in range(n):
        last = int(out["steps"][i])
        states[i, last + 1:] = states[i, last]
    return {**out, "states": states, "actions": actions, "seeds": np.asarray(seeds)}
