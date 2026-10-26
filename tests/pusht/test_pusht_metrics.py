"""The terminal-error metrics the Push-T violin figures report.

``scripts/pusht/plot_pusht.py`` is not a package, so the module is loaded by path. Only the metrics
are exercised here -- they decide reported numbers and have closed forms to check against;
the drawing does not.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "plot_pusht", Path(__file__).resolve().parents[2] / "scripts" / "pusht" / "plot_pusht.py")
plot_pusht = importlib.util.module_from_spec(_spec)
sys.modules["plot_pusht"] = plot_pusht
_spec.loader.exec_module(plot_pusht)

from fm_bc.pusht.constants import GOAL_POSE  # noqa: E402


def test_add_is_zero_at_the_goal_and_equals_a_pure_translation():
    goal = np.array(GOAL_POSE)
    assert plot_pusht.add_error(goal) == pytest.approx(0.0, abs=1e-9)
    for shift in (1.0, 10.0, 137.0):
        moved = goal + np.array([shift * 0.6, shift * 0.8, 0.0])  # a 3-4-5 displacement
        assert plot_pusht.add_error(moved) == pytest.approx(shift)


def test_add_of_a_pure_rotation_is_the_mean_radius_chord():
    """Rotating in place moves every body point along a chord: |x| * 2 sin(dtheta / 2)."""
    goal = np.array(GOAL_POSE)
    radius = np.linalg.norm(plot_pusht.TEE_POINTS, axis=-1).mean()
    for deg in (0.5, 5.0, 45.0, 180.0):
        turned = goal + np.array([0.0, 0.0, math.radians(deg)])
        expected = radius * 2 * math.sin(math.radians(deg) / 2)
        assert plot_pusht.add_error(turned) == pytest.approx(expected, rel=1e-6)


def test_add_keeps_the_leading_shape_of_the_poses():
    poses = np.tile(np.array(GOAL_POSE), (4, 7, 1))
    assert plot_pusht.add_error(poses).shape == (4, 7)


def test_add_points_fill_the_block_rather_than_trace_it():
    """The average runs over the T's area: the points lie inside it and span both limbs."""
    pts = plot_pusht.TEE_POINTS
    assert len(pts) > 100
    crossbar = pts[pts[:, 1] <= 30]
    stem = pts[pts[:, 1] > 30]
    assert len(crossbar) > len(stem) > 0  # the crossbar is the larger limb, 3600 px^2 to 2700
    assert np.abs(pts[:, 0]).max() > 55 and pts[:, 1].max() > 115  # out to both extremes
    assert np.all(np.abs(stem[:, 0]) <= 15)  # nothing in the notches beside the stem


def test_goal_error_wraps_the_orientation_the_short_way():
    goal = np.array(GOAL_POSE)
    turned = goal + np.array([0.0, 0.0, 2 * math.pi - math.radians(10)])
    pos, rot = plot_pusht.goal_error(turned)
    assert pos == pytest.approx(0.0, abs=1e-9)
    assert rot == pytest.approx(10.0)
