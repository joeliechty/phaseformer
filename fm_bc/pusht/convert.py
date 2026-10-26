"""Read the Diffusion Policy Push-T replay buffer (a zarr store) into :class:`PushTEpisodes`.

This is the only module that knows about zarr, and it imports it *inside* the function: zarr is
a one-time build dependency for producing ``pusht_episodes.npz``, not a runtime dependency of
anything downstream.

Only the subset of the zarr API that is identical in 2.x and 3.x is used -- ``zarr.open(store,
mode="r")`` then ``root[key][:]`` -- so one reader covers both. zarr 3 still reads
zarr-format-v2 stores, and the version split falls out of the Python pin on its own (zarr 3.0+
needs py>=3.11, and this repo's Theseus env is py3.10).

Everything read is asserted against the layout published in the Diffusion Policy README, so a
moved or re-encoded artifact fails loudly on the first run instead of silently producing a
misshapen dataset.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import torch

from .constants import (
    ACTION_BOX, DT, FPS, GOAL_POSE, N_KEYPOINTS, ROT_SCALE_PX_PER_RAD, SOURCE_URL,
    STATE_KEYS, SUCCESS_COVERAGE, WORKSPACE_PX,
)
from .data import PushTEpisodes

__all__ = ["from_zarr", "CONVERTER_VERSION"]

CONVERTER_VERSION = 1

_EXPECTED = {
    "data/state": 5,
    "data/action": 2,
    "data/keypoint": (N_KEYPOINTS, 2),
    "data/n_contacts": 1,
}


def from_zarr(
    store: str | Path,
    keypoints: bool = True,
    n_contacts: bool = True,
    max_episodes: int | None = None,
    zip_sha256: str | None = None,
) -> PushTEpisodes:
    """Convert ``pusht_cchi_v7_replay.zarr`` into a :class:`PushTEpisodes`.

    ``max_episodes`` truncates to the first N episodes, for smoke tests. ``img`` is deliberately
    never read: it is 25650x96x96x3 (~708 MB) and this pipeline is low-dimensional. To add it
    later, write it as a separate memmap keyed by the same ``episode_ends``.
    """
    import zarr  # noqa: PLC0415 -- deliberately lazy; see the module docstring

    store = Path(store)
    if not store.exists():
        raise FileNotFoundError(
            f"no zarr store at {store}. Run scripts/pusht/download_pusht.sh first."
        )
    root = zarr.open(str(store), mode="r")

    ends = torch.as_tensor(root["meta/episode_ends"][:], dtype=torch.int64)
    if max_episodes is not None:
        ends = ends[:max_episodes]
    n_steps = int(ends[-1])

    def read(key: str) -> torch.Tensor:
        arr = torch.as_tensor(root[key][:n_steps])
        want = _EXPECTED[key]
        got = tuple(arr.shape[1:])
        if got != (want if isinstance(want, tuple) else (want,)):
            raise ValueError(f"{key}: expected trailing shape {want}, got {got}")
        return arr.to(torch.float32)

    state, action = read("data/state"), read("data/action")
    meta = {
        "source_url": SOURCE_URL,
        "zip_sha256": zip_sha256,
        "zarr_store": store.name,
        "converter_version": CONVERTER_VERSION,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "fps": FPS,
        "dt": DT,
        "workspace_px": WORKSPACE_PX,
        "action_box": [list(ACTION_BOX[0]), list(ACTION_BOX[1])],
        "goal_pose_px": list(GOAL_POSE),
        "success_coverage": SUCCESS_COVERAGE,
        "state_keys": list(STATE_KEYS),
        "theta_convention": "[0, 2pi) as released; the phase grid re-emits it in (-pi, pi]",
        "action_semantics": "absolute agent target position in pixels, not a delta",
        "agent_delta_last": "edge-held: each episode's final step repeats the previous delta",
        "img": "not stored (~708 MB); add as a separate memmap keyed by the same episode_ends",
        "suggested_rot_scale_px_per_rad": ROT_SCALE_PX_PER_RAD,
        "n_episodes": int(ends.numel()),
        "n_steps": n_steps,
    }
    ds = PushTEpisodes.from_arrays(
        state=state,
        action=action,
        episode_ends=ends,
        keypoint=read("data/keypoint") if keypoints else None,
        n_contacts=read("data/n_contacts") if n_contacts else None,
        meta=meta,
    )
    lengths = ds.lengths.to(torch.float64)
    ds.meta["length_stats"] = {
        "min": int(lengths.min()), "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 2),
    }
    ds.check()
    return ds


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, for recording which zip an artifact came from."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
