"""Normalization as a ``(scale, offset)`` pair, in robomimic's convention.

    normalized = (raw - offset) / scale        raw = scale * normalized + offset

``scale`` is a **denominator**. Expressing every mode -- analytic box, data min/max, gaussian --
as the same pair means apply and invert are one function each, and switching modes never touches
a call site.

Three modes:

``box``
    The analytic workspace box ``[0, 512]^2``. This is the Push-T analogue of the skate task's
    control box (see :mod:`fm_bc.skate.chunks`), and the argument carries over unchanged: the
    scale is identical for every subset and seed, so ``--n-train`` sweeps stay comparable; a
    sample clipped to ``[-1, 1]`` is exactly a command the environment would accept untouched; and
    it cannot leak validation data, because it never looks at the data.
``minmax``
    Empirical per-dimension min/max. Uses more of the ``[-1, 1]`` range -- the demonstrations
    occupy far less than the full 512 box -- and is what Diffusion Policy's own ``LinearNormalizer``
    does, so it is the mode to pick when matching published numbers.
``gaussian``
    Per-dimension mean/std.

All three are computed and written, so the choice is a config flag rather than a re-run. Note
one asymmetry: ``vel_action`` and ``agent_delta`` are displacements, bounded by the env's
per-step motion rather than by the workspace, so they have no meaningful analytic box and fall
back to ``minmax`` regardless of the requested mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["MODES", "Normalizer", "NO_BOX_KEYS", "stats_from_data", "normalizer_for"]

MODES = ("box", "minmax", "gaussian")

# Keys with no meaningful analytic box: displacements, not positions.
NO_BOX_KEYS = ("vel_action", "agent_delta")

# robomimic's constants (robomimic/utils/dataset.py::action_stats_to_normalization_stats).
_OUT_MIN, _OUT_MAX = -0.999999, 0.999999
_RANGE_EPS = 1e-4
_STD_EPS = 1e-6


@dataclass(frozen=True)
class Normalizer:
    """A per-dimension affine map and its inverse."""

    scale: torch.Tensor
    offset: torch.Tensor

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset.to(x)) / self.scale.to(x)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.to(x) + self.offset.to(x)

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {"scale": self.scale, "offset": self.offset}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(scale=torch.as_tensor(d["scale"]), offset=torch.as_tensor(d["offset"]))


def stats_from_data(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-dimension ``{min, max, mean, std}`` over every leading axis of ``x``."""
    flat = x.reshape(-1, x.shape[-1]).to(torch.float64)
    return {
        "min": flat.min(dim=0).values,
        "max": flat.max(dim=0).values,
        "mean": flat.mean(dim=0),
        "std": flat.std(dim=0, unbiased=False),
    }


def _linear(lo: torch.Tensor, hi: torch.Tensor) -> Normalizer:
    """Map ``[lo, hi]`` onto ``[-0.999999, 0.999999]``, guarding degenerate dimensions.

    A dimension whose range is below ``range_eps`` (a constant column) would otherwise divide by
    ~0; robomimic widens it to the full output range instead, which maps the constant to the
    midpoint and keeps the transform finite and invertible.
    """
    degenerate = (hi - lo) < _RANGE_EPS
    mid, half = (lo + hi) / 2, (_OUT_MAX - _OUT_MIN) / 2
    lo = torch.where(degenerate, mid - half, lo)
    hi = torch.where(degenerate, mid + half, hi)
    scale = (hi - lo) / (_OUT_MAX - _OUT_MIN)
    return Normalizer(scale=scale.to(torch.float32),
                      offset=(lo - scale * _OUT_MIN).to(torch.float32))


def normalizer_for(
    mode: str,
    stats: dict[str, torch.Tensor],
    box: tuple[float, float] | None = None,
) -> Normalizer:
    """Build the normalizer for ``mode`` from per-dimension ``stats``.

    ``box`` is the analytic ``(lo, hi)`` applied to every dimension; when it is ``None`` the
    ``box`` mode falls back to ``minmax`` (see :data:`NO_BOX_KEYS`).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "gaussian":
        std = stats["std"].clamp(min=_STD_EPS)
        return Normalizer(scale=std.to(torch.float32),
                          offset=stats["mean"].to(torch.float32))
    if mode == "box" and box is not None:
        lo = torch.full_like(stats["min"], box[0])
        hi = torch.full_like(stats["max"], box[1])
        return _linear(lo, hi)
    return _linear(stats["min"], stats["max"])
