"""Distribution-level evaluation of Push-T rollouts: MMD to the held-out demonstrations.

The skate metric (:func:`fm_bc.skate.evaluation.mmd`), restated for Push-T so this subtree stays
theseus-free. Each policy is rolled out from the start states of the test demonstrations, each
for its demonstration's own length with the demonstration's clock ``rho = k / (L - 1)``
(:func:`fm_bc.pusht.sim.rollout_from_states`), so rollout and demonstration share a phase axis.
Both sets are read off on a common phase grid (nearest step) and compared at every grid point with
a squared MMD (biased V-statistic, Gaussian kernel of fixed bandwidth) on the full state embedded
as ``[agent_x, agent_y, block_x, block_y, l cos theta, l sin theta]`` in px, with
``l = ROT_SCALE_PX_PER_RAD``. ``mmd_mean`` averages over the grid, ``mmd_final`` is ``rho = 1``.

**Bandwidth** 100 px: the demonstrations' median pairwise distance per phase falls from ~300 px
at the start to ~130 px at the end, and at 100 px a disjoint set of demonstrations (validation vs
test) scores 0.061 while the frozen start states score 0.218 -- the useful range.
"""

from __future__ import annotations

import torch

from .constants import ROT_SCALE_PX_PER_RAD

__all__ = ["MMD_BANDWIDTH_PX", "N_PHASE_GRID", "embed_state", "state_mmd", "phase_grid",
           "per_phase_mmd"]

MMD_BANDWIDTH_PX = 100.0
N_PHASE_GRID = 128


def embed_state(states: torch.Tensor, rot_scale: float = ROT_SCALE_PX_PER_RAD) -> torch.Tensor:
    """``(..., 5)`` states -> ``(..., 6)`` Euclidean embedding (the heading as a scaled chord)."""
    th = states[..., 4:5]
    return torch.cat([states[..., :4], rot_scale * torch.cos(th), rot_scale * torch.sin(th)], -1)


def state_mmd(a: torch.Tensor, b: torch.Tensor, bandwidth: float = MMD_BANDWIDTH_PX) -> float:
    """Squared MMD (biased V-statistic) between state sets ``(M, 5)`` and ``(M', 5)``."""
    ea, eb = embed_state(a.double()), embed_state(b.double())
    gamma = 1.0 / (2.0 * bandwidth**2)

    def k(p, q):
        return torch.exp(-gamma * torch.cdist(p, q).pow(2)).mean()

    return float(k(ea, ea) + k(eb, eb) - 2.0 * k(ea, eb))


def phase_grid(trajectories, n_grid: int = N_PHASE_GRID) -> torch.Tensor:
    """A list of ``(L_i, 5)`` trajectories -> ``(n_grid, N, 5)`` states at ``rho = j / (n_grid - 1)``
    (nearest step of each trajectory's own ``k / (L_i - 1)``)."""
    rho = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float64)
    out = []
    for s in trajectories:
        s = torch.as_tensor(s)
        out.append(s[torch.round(rho * (len(s) - 1)).long()])
    return torch.stack(out, 1)


def per_phase_mmd(rollouts, demos, bandwidth: float = MMD_BANDWIDTH_PX,
                  n_grid: int = N_PHASE_GRID) -> dict:
    """``mmd_mean``, ``mmd_final`` and ``mmd_per_phase`` between two lists of trajectories."""
    a, b = phase_grid(rollouts, n_grid), phase_grid(demos, n_grid)
    curve = [state_mmd(a[j], b[j], bandwidth) for j in range(n_grid)]
    return {"mmd_mean": sum(curve) / n_grid, "mmd_final": curve[-1], "mmd_per_phase": curve,
            "bandwidth_px": bandwidth, "n_grid": n_grid}
