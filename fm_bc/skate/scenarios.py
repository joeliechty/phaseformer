"""The three navigation scenarios of Fig. 6, as start/goal *distributions* over one fixed world.

A) ``single_path``  one start, one goal, one corridor. The two boxes are flush against opposite
   walls, so every collision-free path threads the same S-shaped gap between them -- there is
   exactly one homotopy class. The case deterministic BC is expected to handle.
B) ``dual_goal``    one start, two goals either side of a block hanging from the ceiling, 50/50.
C) ``dual_path``    one start, one goal just beyond a small central block: go left or right,
   50/50. The classic trap for a deterministic policy, which averages the two into the block.

In B and C the start sits at the bottom of a lane walled off by two floor blocks, so both
modes drive the same straight segment first and only diverge once the lane ends -- the pose
alone does not reveal the mode until the split (``scripts/skate/mode_separability.py`` measures
this). An earlier version (open floor, wide via points) let the demos commit to a side within
one control step, so a phase-free state-feedback policy never had to average two modes.

A *mode* is a homotopy class. Each mode has a via point that seeds the optimizer on the correct
side, and :meth:`ScenarioSpec.label_mode` re-derives the mode from the finished trajectory so the
dataset can reject solves that hopped to the other side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skate_planner.world import Box, ObstacleField

__all__ = ["Gaussian2D", "ScenarioSpec", "SCENARIOS", "EXTRA_SCENARIOS", "get_scenario"]

BOUNDS = (0.0, 0.0, 10.0, 8.0)


@dataclass(frozen=True)
class Gaussian2D:
    """Isotropic Gaussian over position, optionally with a heading."""

    x: float
    y: float
    sigma: float = 0.3
    theta: float = 0.0
    sigma_theta: float = 0.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, 3)`` samples ``[x, y, theta]``."""
        xy = rng.normal([self.x, self.y], self.sigma, size=(n, 2))
        th = rng.normal(self.theta, self.sigma_theta, size=(n, 1)) if self.sigma_theta else (
            np.full((n, 1), self.theta)
        )
        return np.concatenate([xy, th], axis=1)

    @property
    def mean(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])


@dataclass
class ScenarioSpec:
    """A world plus the distributions demonstrations are drawn from.

    ``goal_of_mode[m]`` is the index into ``goals`` that mode ``m`` drives to, and
    ``vias[m]`` is the seed via point -- or a chain of waypoints, the same length for every
    mode -- that puts the optimizer on mode ``m``'s side.
    ``split_box`` is the obstacle whose left/right side defines the mode (``None`` for a
    single-mode scenario).

    The remaining fields default to the Fig. 6 scenarios' behaviour. ``goal_mode="pose"`` makes
    the planner constrain the final heading too. ``reach_radius`` is the success tolerance
    around a goal (``None``: 3 of the goal's sigmas). ``via_jitter`` overrides the generator's
    jitter of the via points. ``deterministic`` plans each mode once and repeats that one
    trajectory, so every demonstration of a mode is bit-identical.
    """

    name: str
    title: str
    world: ObstacleField
    start: Gaussian2D
    goals: list[Gaussian2D]
    goal_of_mode: list[int]
    vias: list  # per mode: (x, y), or a sequence of (x, y) waypoints
    mode_names: list[str]
    split_box: Box | None = None
    mode_weights: list[float] = field(default_factory=list)
    goal_mode: str = "position"
    reach_radius: float | None = None
    via_jitter: float | None = None
    deterministic: bool = False

    def __post_init__(self):
        if not self.mode_weights:
            self.mode_weights = [1.0 / self.n_modes] * self.n_modes

    @property
    def n_modes(self) -> int:
        return len(self.vias)

    def sample(
        self, n: int, rng: np.random.Generator, modes: np.ndarray | None = None,
        via_jitter: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draw ``n`` problems: ``(starts (n,3), goals (n,3), vias (n,K,2), modes (n,))``, with
        every waypoint jittered independently."""
        if modes is None:
            modes = rng.choice(self.n_modes, size=n, p=self.mode_weights)
        chains = [np.asarray(v, dtype=float).reshape(-1, 2) for v in self.vias]
        k = len(chains[0])
        if any(len(c) != k for c in chains):
            raise ValueError(f"{self.name}: every mode needs the same number of via points")
        starts = self.start.sample(n, rng)
        goals = np.empty((n, 3))
        vias = np.empty((n, k, 2))
        for m in range(self.n_modes):
            idx = np.flatnonzero(modes == m)
            goals[idx] = self.goals[self.goal_of_mode[m]].sample(len(idx), rng)
            vias[idx] = chains[m] + rng.normal(0.0, via_jitter, size=(len(idx), k, 2))
        return starts, goals, vias, modes

    def label_mode(self, poses: np.ndarray) -> np.ndarray:
        """Homotopy class of each ``(M, T, 3)`` trajectory; ``-1`` if it cannot be decided.

        With a ``split_box`` the class is which side of the box the path passes: the median x
        of the samples level with the box, compared against the box center. For
        ``dual_goal`` the side must also agree with the goal reached (a path that reaches the
        left goal by going round the right of the box is labelled ``-1``).
        """
        m = poses.shape[0]
        if self.split_box is None:
            return np.zeros(m, dtype=int)
        box = self.split_box
        level = np.abs(poses[..., 1] - box.cy) <= box.hy
        labels = np.full(m, -1, dtype=int)
        for i in range(m):
            if level[i].any():
                labels[i] = int(np.median(poses[i, level[i], 0]) > box.cx)
        if len(self.goals) > 1:
            final = poses[:, -1, :2]
            centers = np.stack([g.mean[:2] for g in self.goals])
            reached = np.argmin(np.linalg.norm(final[:, None] - centers[None], axis=-1), axis=1)
            goal_mode = np.array([self.goal_of_mode.index(g) for g in reached])
            labels = np.where(labels == goal_mode, labels, -1)
        return labels


def _single_path() -> ScenarioSpec:
    # Big box flush to the left wall, small box flush to the right wall. Their inner ends
    # overlap in x, so the straight line from start to goal clips both corners and the
    # only way through is the band between them.
    world = ObstacleField(
        bounds=BOUNDS,
        boxes=[
            Box(cx=3.0, cy=5.5, hx=3.0, hy=0.7),  # x in [0, 6.0], y in [4.8, 6.2]
            Box(cx=7.0, cy=2.4, hx=3.0, hy=0.4),  # x in [4.0, 10], y in [2.0, 2.8]
        ],
    )
    return ScenarioSpec(
        name="single_path",
        title="A) single start, single goal, single path",
        world=world,
        start=Gaussian2D(2.0, 1.0, sigma=0.3, theta=np.pi / 2, sigma_theta=0.15),
        goals=[Gaussian2D(8.2, 6.8, sigma=0.3)],
        goal_of_mode=[0],
        vias=[(5.0, 3.8)],
        mode_names=["corridor"],
    )


# B and C share a start lane: two blocks flush to the floor and side walls leave a 1.4 m lane
# up the middle, so every demonstration -- whichever mode it is headed for -- drives the same
# straight segment first and only splits after the lane ends. The lane is narrow enough that
# a demo cannot lean toward its side early (a 2.2 m lane still let it commit within a step or
# two), and the start is tight (0.08 m, 0.05 rad) so it fits the lane and carries no heading
# hint about the mode.
_LANE_TOP = 3.3
_LANE_BLOCKS = [
    Box(cx=2.15, cy=_LANE_TOP / 2, hx=2.15, hy=_LANE_TOP / 2),  # x in [0, 4.3], y in [0, 3.3]
    Box(cx=7.85, cy=_LANE_TOP / 2, hx=2.15, hy=_LANE_TOP / 2),  # x in [5.7, 10], y in [0, 3.3]
]
_LANE_START = Gaussian2D(5.0, 0.9, sigma=0.08, theta=np.pi / 2, sigma_theta=0.05)


def _dual_goal() -> ScenarioSpec:
    # A block hanging from the ceiling caps the lane exit; one goal on each side of it.
    box = Box(cx=5.0, cy=6.65, hx=1.9, hy=1.35)  # x in [3.1, 6.9], y in [5.3, 8.0]
    return ScenarioSpec(
        name="dual_goal",
        title="B) single start, dual goal, dual path",
        world=ObstacleField(bounds=BOUNDS, boxes=[*_LANE_BLOCKS, box]),
        start=_LANE_START,
        goals=[Gaussian2D(1.6, 6.6, sigma=0.25), Gaussian2D(8.4, 6.6, sigma=0.25)],
        goal_of_mode=[0, 1],
        # Up the lane, then out to the side under the block.
        vias=[[(5.0, 3.6), (3.0, 4.4)], [(5.0, 3.6), (7.0, 4.4)]],
        mode_names=["left goal", "right goal"],
        split_box=box,
    )


# A small block above the lane exit, the goal just beyond it: loop round either side.
_DUAL_PATH_BOX = Box(cx=5.0, cy=5.35, hx=1.1, hy=0.45)  # x in [3.9, 6.1], y in [4.9, 5.8]
# Up the lane, then wide round the block.
_DUAL_PATH_VIAS = [[(5.0, 3.5), (2.4, 5.35)], [(5.0, 3.5), (7.6, 5.35)]]


def _dual_path() -> ScenarioSpec:
    box = _DUAL_PATH_BOX
    return ScenarioSpec(
        name="dual_path",
        title="Single Start, Single Goal, Dual Path",
        world=ObstacleField(bounds=BOUNDS, boxes=[*_LANE_BLOCKS, box]),
        start=_LANE_START,
        goals=[Gaussian2D(5.0, 7.0, sigma=0.2)],
        goal_of_mode=[0, 0],
        vias=_DUAL_PATH_VIAS,
        mode_names=["left", "right"],
        split_box=box,
    )


_FIXED_START = Gaussian2D(5.0, 0.9, sigma=0.0, theta=np.pi / 2)
_FIXED_GOAL = Gaussian2D(5.0, 7.0, sigma=0.0, theta=np.pi / 2)


def _point_to_point(name: str, title: str, start: Gaussian2D, via_jitter: float,
                    deterministic: bool) -> ScenarioSpec:
    # C's world and block driven to one goal *pose* (heading up), for the point-to-point side
    # experiments (EXTRA_SCENARIOS, not the paper).
    box = _DUAL_PATH_BOX
    return ScenarioSpec(
        name=name,
        title=title,
        world=ObstacleField(bounds=BOUNDS, boxes=[*_LANE_BLOCKS, box]),
        start=start,
        goals=[_FIXED_GOAL],
        goal_of_mode=[0, 0],
        vias=_DUAL_PATH_VIAS,
        mode_names=["left", "right"],
        split_box=box,
        goal_mode="pose",
        reach_radius=0.6,  # C's 3 sigma, so success is comparable
        via_jitter=via_jitter,
        deterministic=deterministic,
    )


def _dual_path_fixed() -> ScenarioSpec:
    # One start pose, one goal pose, no via jitter: the demos are exactly two trajectories, a
    # left and a right one, each repeated.
    return _point_to_point("dual_path_fixed", "Fixed Start/Goal Pose, Dual Path",
                           _FIXED_START, via_jitter=0.0, deterministic=True)


def _dual_path_fixed_via() -> ScenarioSpec:
    # The same fixed start and goal pose, with C's via jitter back: the demos spread in the
    # middle of the path but share both ends.
    return _point_to_point("dual_path_fixed_via", "Fixed Start/Goal Pose + Via Jitter",
                           _FIXED_START, via_jitter=0.3, deterministic=False)


def _dual_path_noisy_start() -> ScenarioSpec:
    # C's start noise (the lane start), the fixed goal pose, no via jitter: every demo is the
    # planner's one optimum from its own start.
    return _point_to_point("dual_path_noisy_start", "Noisy Start, Fixed Goal Pose",
                           _LANE_START, via_jitter=0.0, deterministic=False)


# Start-noise mixtures: p% of the demos from dual_path_noisy_start, the rest the two repeated
# dual_path_fixed trajectories (scripts/skate/build_start_mixes.py assembles them; they are never
# planned as such). The spec is dual_path_noisy_start's under its own name.
MIX_PERCENTS = (5, 10, 15)  # 20-50% were dropped from the sweep


def _start_mix(pct: int):
    def build() -> ScenarioSpec:
        spec = _dual_path_noisy_start()
        spec.name = f"dual_path_mix{pct:02d}"
        spec.title = f"{pct}% Noisy Starts, Rest Fixed"
        return spec
    return build


def _mix_only(pct: int):
    # The noisy-start demos of dual_path_mix{pct} on their own, without the fixed copies.
    def build() -> ScenarioSpec:
        spec = _dual_path_noisy_start()
        spec.name = f"dual_path_mixonly{pct:02d}"
        spec.title = f"The {pct}% Mixture's Noisy Starts Alone"
        return spec
    return build


# Start-spread sets (scripts/skate/build_start_spread.py): C's start shape scaled to sigma, with
# the heading sigma scaled alike (0.05 rad at 8 cm), no via jitter and the fixed goal pose.
# "spread" assigns the sides at random, exactly half each; "sorted" sends a start left of centre
# left. Every set's starts come from one shared draw of standard normals in antithetic pairs
# (common random numbers, exactly symmetric about the centre line), so the sets differ only in
# sigma and in how the sides are assigned. They are planned
# from those starts, never sampled from the spec, whose start is there to describe them.
SPREAD_CM = (2, 4, 8, 12)
SORTED_CM = (2, 8)


def _start_spread(cm: int, sides: str):
    def build() -> ScenarioSpec:
        s = cm / 100
        start = Gaussian2D(5.0, 0.9, sigma=s, theta=np.pi / 2, sigma_theta=0.05 * s / 0.08)
        title = f"{cm} cm Start Spread, {'Sorted' if sides == 'sorted' else 'Random'} Sides"
        return _point_to_point(f"dual_path_{sides}{cm:02d}", title, start, via_jitter=0.0,
                               deterministic=False)
    return build


START_SPREAD_SCENARIOS = (*(f"dual_path_spread{c:02d}" for c in SPREAD_CM),
                          *(f"dual_path_sorted{c:02d}" for c in SORTED_CM),
                          *(f"dual_path_mixonly{p:02d}" for p in MIX_PERCENTS))

_BUILDERS = {"single_path": _single_path, "dual_goal": _dual_goal, "dual_path": _dual_path,
             "dual_path_fixed": _dual_path_fixed, "dual_path_fixed_via": _dual_path_fixed_via,
             "dual_path_noisy_start": _dual_path_noisy_start,
             **{f"dual_path_mix{p:02d}": _start_mix(p) for p in MIX_PERCENTS},
             **{f"dual_path_mixonly{p:02d}": _mix_only(p) for p in MIX_PERCENTS},
             **{f"dual_path_spread{c:02d}": _start_spread(c, "spread") for c in SPREAD_CM},
             **{f"dual_path_sorted{c:02d}": _start_spread(c, "sorted") for c in SORTED_CM}}
# The paper's three scenarios, which every "all" means; EXTRA_SCENARIOS are side experiments.
SCENARIOS = ("single_path", "dual_goal", "dual_path")
EXTRA_SCENARIOS = ("dual_path_fixed", "dual_path_fixed_via", "dual_path_noisy_start",
                   *(f"dual_path_mix{p:02d}" for p in MIX_PERCENTS), *START_SPREAD_SCENARIOS)


def get_scenario(name: str) -> ScenarioSpec:
    try:
        return _BUILDERS[name]()
    except KeyError:
        raise ValueError(f"Unknown scenario {name!r}; expected one of {tuple(_BUILDERS)}.") from None
