"""The skate navigation task: flow-matching BC on planner-generated SE(2) demonstrations.

A first-order nonholonomic ("skate") robot drives through the three Fig. 6 obstacle scenarios.
Its demonstrations are *generated* by the Theseus planner in :mod:`skate_planner`, all at the same
fixed horizon, so every demo shares the phase grid ``k / N`` and the divergence and score targets
need no phase binning.

Quick start::

    from fm_bc.skate import get_scenario, generate, add_div_and_score

    ds = generate(get_scenario("dual_path"), n_target=512)
    add_div_and_score(ds)
    ds.save("data/dual_path.npz")

Everything here needs Theseus, through :mod:`skate_planner`. Submodules are still imported
**lazily** (PEP 562), as in :mod:`fm_bc.pusht`, so ``import fm_bc.skate`` itself stays cheap and
imports on a machine without Theseus.
"""

from __future__ import annotations

import importlib

# Public name -> the submodule that defines it.
_EXPORTS = {
    "GenerationConfig": ".dataset",
    "TrajectoryDataset": ".dataset",
    "generate": ".dataset",
    "add_div_and_score": ".dataset",
    "SCENARIOS": ".scenarios",
    "EXTRA_SCENARIOS": ".scenarios",
    "Gaussian2D": ".scenarios",
    "ScenarioSpec": ".scenarios",
    "get_scenario": ".scenarios",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Resolve a public name to its submodule on first access, then cache it in ``globals()``."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_EXPORTS[name], __name__)
    globals()[name] = value = getattr(module, name)
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
