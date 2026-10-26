"""A skate-model motion planner built on Theseus.

Plans a first-order nonholonomic ("skate") robot through a 2D obstacle field by solving a
nonlinear least-squares trajectory optimization with Theseus's Levenberg-Marquardt optimizer.

Quick start::

    from skate_planner import SkatePlanner, scenario, plot_solution

    world, start, goal = scenario("slalom")
    planner = SkatePlanner.for_world(world, cell_size=0.1)
    solution = planner.plan(world, start, goal, n_starts=8)
    print(solution.report())
    plot_solution(world, solution, radius=planner.radius, save="plan.png")
"""

from .costs import SkateDynamics, XYGoal
from .initializers import controls_from_poses, interpolate, multistart_seeds
from .planner import SkateLimits, SkatePlanner, SkateSolution, SkateWeights
from .skate_model import (
    TrajectoryDiagnostics,
    body_twist,
    evaluate,
    integrate,
    lateral_slip,
    rollout,
    se2_tensor,
    xytheta,
)
from .viz import animate, plot_multistart, plot_solution
from .world import Box, Circle, GridSDF, ObstacleField, random_field, scenario

__all__ = [
    "SkatePlanner",
    "SkateLimits",
    "SkateWeights",
    "SkateSolution",
    "SkateDynamics",
    "XYGoal",
    "ObstacleField",
    "Circle",
    "Box",
    "GridSDF",
    "random_field",
    "scenario",
    "body_twist",
    "integrate",
    "rollout",
    "lateral_slip",
    "evaluate",
    "TrajectoryDiagnostics",
    "se2_tensor",
    "xytheta",
    "interpolate",
    "multistart_seeds",
    "controls_from_poses",
    "plot_solution",
    "plot_multistart",
    "animate",
]

__version__ = "0.1.0"
