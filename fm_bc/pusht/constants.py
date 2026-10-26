"""Fixed physical constants of the Push-T task (Chi et al. 2023, RSS).

Everything here is a property of the environment, not of any particular dataset or run, and is
taken from ``diffusion_policy/env/pusht/pusht_env.py``.
"""

from __future__ import annotations

import math

__all__ = [
    "ACTION_BOX", "DT", "FPS", "GOAL_POSE", "N_KEYPOINTS", "N_TEE_POINTS",
    "ROT_SCALE_PX_PER_RAD", "SOURCE_URL", "STATE_KEYS", "SUCCESS_COVERAGE", "TEE_CENTROIDS",
    "TEE_POINTS", "TEE_VERTICES", "WORKSPACE_PX",
]

SOURCE_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"

# The env's window_size: both the agent and the block live in a 512x512 pixel frame, and the
# action space is exactly this box.
WORKSPACE_PX = 512.0
ACTION_BOX = ((0.0, 0.0), (WORKSPACE_PX, WORKSPACE_PX))

FPS = 10  # control_hz; the sim runs at 100 Hz with 10 substeps per control step
DT = 1.0 / FPS

# [x, y, theta]. Fixed for every episode, so it is a constant rather than a per-episode field.
GOAL_POSE = (256.0, 256.0, math.pi / 4)

# reward = clip(coverage / SUCCESS_COVERAGE, 0, 1); done = coverage > SUCCESS_COVERAGE.
# NOTE: no frame of the *released* demonstrations reaches this -- the human operators stop just
# short (max coverage 0.9489), so a `success` column derived from the demos is identically
# False. Do not use it as a training label.
SUCCESS_COVERAGE = 0.95

STATE_KEYS = ("agent_x", "agent_y", "block_x", "block_y", "block_theta")
N_KEYPOINTS = 9  # the zarr's `keypoint` array is (T, 9, 2)

# The T block's own geometry, from `add_tee(scale=30, length=4)`: a 120x30 crossbar and a 30x90
# stem, eight vertices in outline order. These are in the pymunk BODY-ORIGIN frame -- the frame
# the dataset's (block_x, block_y, block_theta) is in -- so a point rides the block under
# `body_xy + R(theta) @ local` (`fm_bc.pusht.chunks.tee_points`, and `in_world` in plot_pusht).
#
# NOTE these are NOT the zarr's nine `keypoint`s. Those are furthest-point-sampled from a
# rendered mask of the block with a random perturbation, so they are a stored artifact of one
# sampling run and cannot be recovered from the geometry. The vertices can, which is why a model
# that wants points on the block computes these instead of reading `keypoint`.
TEE_VERTICES = (
    (-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (15.0, 30.0),
    (15.0, 120.0), (-15.0, 120.0), (-15.0, 30.0), (-60.0, 30.0),
)
# Centroids of the two rectangles the T is built from (crossbar, then stem). Interior points, so
# a policy attending over the outline alone cannot place the body's bulk.
TEE_CENTROIDS = ((0.0, 15.0), (0.0, 75.0))
TEE_POINTS = TEE_VERTICES + TEE_CENTROIDS  # the rigidly attached points a policy may tokenize
N_TEE_POINTS = len(TEE_POINTS)  # 10

# Metres-per-radian analogue for the left-invariant SE(2) metric in `fm_bc.se2.twist_norm`, used
# by the kNN divergence and the kernel score.
#
# This is the one free constant in that metric and it MUST be matched to the workspace scale.
# Here translations are O(100 px) and rotations O(pi rad), so the repo's skate default of 1.0
# would make the metric ~99.9% translation -- the mirror image of the reference implementation's
# ~99% orientation failure, and equally wrong. 100 px/rad is the T-block's characteristic
# radius: one radian of block rotation then counts the same as 100 px of block translation.
ROT_SCALE_PX_PER_RAD = 100.0
