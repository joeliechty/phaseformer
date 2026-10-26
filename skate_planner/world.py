"""2D obstacle fields and the signed distance fields the planner optimizes against.

Sign convention throughout: the SDF is **positive in free space** and **negative inside
obstacles**, which is what ``theseus.embodied.Collision2D`` expects -- its error is
``clamp(cost_eps - sdf(xy), min=0)``, so a larger SDF means safer.

Two backends are provided:

``ObstacleField.analytic_sdf``
    Exact closed-form distance, used for ground-truth checks and for honest post-hoc
    clearance reporting. No discretization error.

``ObstacleField.grid_sdf``
    The exact field sampled on a grid, suitable for handing to Theseus as an aux variable.
    This is what the planner actually optimizes against, because Theseus's SDF lookup is a
    bilinear interpolation over a grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["Circle", "Box", "ObstacleField", "GridSDF", "random_field", "scenario"]


@dataclass(frozen=True)
class Circle:
    """A disc obstacle centered at ``(cx, cy)``."""

    cx: float
    cy: float
    r: float

    def sdf(self, xy: np.ndarray) -> np.ndarray:
        """Signed distance for query points ``xy`` of shape ``(..., 2)``."""
        center = np.asarray([self.cx, self.cy], dtype=xy.dtype)
        return np.linalg.norm(xy - center, axis=-1) - self.r


@dataclass(frozen=True)
class Box:
    """An axis-aligned box, given by center and half-extents."""

    cx: float
    cy: float
    hx: float
    hy: float

    def sdf(self, xy: np.ndarray) -> np.ndarray:
        center = np.asarray([self.cx, self.cy], dtype=xy.dtype)
        half = np.asarray([self.hx, self.hy], dtype=xy.dtype)
        d = np.abs(xy - center) - half
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
        # Inside the box every component of d is negative; the largest one is the
        # (negative) distance to the closest face.
        inside = np.minimum(np.max(d, axis=-1), 0.0)
        return outside + inside


@dataclass(frozen=True)
class GridSDF:
    """A rasterized signed distance field, in the layout Theseus wants.

    Attributes
    ----------
    origin:
        World ``(x, y)`` coordinate of cell ``[0, 0]``, shape ``(2,)``.
    data:
        Signed distances, shape ``(rows, cols)``. Row index advances in ``+y``,
        column index advances in ``+x``.
    cell_size:
        Side length of one cell, in world units.
    """

    origin: np.ndarray
    data: np.ndarray
    cell_size: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    def extent(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` for ``matplotlib.imshow(..., extent=...)``."""
        rows, cols = self.data.shape
        x0, y0 = float(self.origin[0]), float(self.origin[1])
        return (
            x0 - 0.5 * self.cell_size,
            x0 + (cols - 0.5) * self.cell_size,
            y0 - 0.5 * self.cell_size,
            y0 + (rows - 0.5) * self.cell_size,
        )


@dataclass
class ObstacleField:
    """A rectangular world containing circular and box obstacles.

    ``bounds`` is ``(xmin, ymin, xmax, ymax)``. The world boundary itself is treated as an
    obstacle by :meth:`analytic_sdf` and :meth:`occupancy` so the planner cannot escape the
    map.
    """

    bounds: tuple[float, float, float, float]
    circles: list[Circle] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)

    # -- exact geometry ---------------------------------------------------------------

    def analytic_sdf(self, xy: np.ndarray) -> np.ndarray:
        """Exact signed distance to the nearest obstacle *or* world boundary.

        ``xy`` has shape ``(..., 2)``; the result has shape ``(...)``.
        """
        xy = np.asarray(xy, dtype=float)
        return np.minimum(self._obstacle_sdf(xy), self._boundary_sdf(xy))

    def _obstacle_sdf(self, xy: np.ndarray) -> np.ndarray:
        """Distance to the nearest obstacle, ignoring the world boundary.

        ``+inf`` when there are no obstacles, so it composes under ``minimum``.
        """
        obstacles = (*self.circles, *self.boxes)
        if not obstacles:
            return np.full(xy.shape[:-1], np.inf)
        return np.min(np.stack([obs.sdf(xy) for obs in obstacles], axis=0), axis=0)

    def _boundary_sdf(self, xy: np.ndarray) -> np.ndarray:
        """Distance to the world edge: positive inside the box, negative outside."""
        xmin, ymin, xmax, ymax = self.bounds
        dx = np.minimum(xy[..., 0] - xmin, xmax - xy[..., 0])
        dy = np.minimum(xy[..., 1] - ymin, ymax - xy[..., 1])
        return np.minimum(dx, dy)

    def is_free(self, xy: np.ndarray, clearance: float = 0.0) -> np.ndarray:
        """Boolean mask of points at least ``clearance`` away from every obstacle."""
        return self.analytic_sdf(xy) >= clearance

    # -- rasterization ----------------------------------------------------------------

    def cell_centers(self, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Grid of cell-center coordinates, plus the origin of cell ``[0, 0]``.

        Returns ``(centers, origin)`` where ``centers`` has shape ``(rows, cols, 2)``.
        """
        xmin, ymin, xmax, ymax = self.bounds
        cols = int(np.ceil((xmax - xmin) / cell_size)) + 1
        rows = int(np.ceil((ymax - ymin) / cell_size)) + 1
        xs = xmin + cell_size * np.arange(cols)
        ys = ymin + cell_size * np.arange(rows)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")  # both (rows, cols)
        centers = np.stack([gx, gy], axis=-1)
        origin = np.asarray([xmin, ymin], dtype=float)
        return centers, origin

    def occupancy(self, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Boolean occupancy grid (``True`` == inside an obstacle) and the grid origin."""
        centers, origin = self.cell_centers(cell_size)
        return self._obstacle_sdf(centers) < 0.0, origin

    def grid_sdf(self, cell_size: float, method: str = "exact") -> GridSDF:
        """Sample the exact signed distance field at the grid nodes.

        Every obstacle here -- and the world boundary -- has a closed-form SDF, so the grid
        holds exact values and the only remaining error is Theseus' bilinear interpolation
        between nodes (millimetres at ``cell_size=0.1``).

        This replaced a Euclidean distance transform of the occupancy grid, which measures
        distance to the nearest *occupied node* rather than to the surface: it over-reported
        clearance by up to a cell (0.1-0.15 m) next to a box, and unevenly -- whether a node
        lying exactly on an edge counts as occupied comes down to float rounding -- so mirror-
        image corners were cut by different amounts. The boundary is included (as before) so
        the optimizer cannot drive off the map, since the collision cost only ever sees this
        grid.

        ``method="edt"`` still builds that old field, for one purpose: the shipped
        ``single_path`` demonstrations were planned against it, and
        ``scripts/skate/reproduce_datasets.py`` needs it to rebuild them bit for bit.
        """
        if method == "edt":
            return self._grid_sdf_edt(cell_size)
        if method != "exact":
            raise ValueError(f"method must be 'exact' or 'edt', got {method!r}")
        centers, origin = self.cell_centers(cell_size)
        return GridSDF(origin=origin, data=self.analytic_sdf(centers), cell_size=cell_size)

    def _grid_sdf_edt(self, cell_size: float) -> GridSDF:
        """The field before commit ``2eaf39d``, verbatim: a Euclidean distance transform of the
        occupancy grid, run once on free and once on occupied space and subtracted, with the
        world boundary folded in analytically. :meth:`grid_sdf` says why it was replaced."""
        from scipy.ndimage import distance_transform_edt  # noqa: PLC0415 -- this path only

        occ, origin = self.occupancy(cell_size)
        if occ.any():
            free_to_obstacle = distance_transform_edt(~occ, sampling=cell_size)
            obstacle_to_free = distance_transform_edt(occ, sampling=cell_size)
            obstacle_field = np.asarray(free_to_obstacle) - np.asarray(obstacle_to_free)
        else:
            obstacle_field = np.full(occ.shape, np.inf)

        centers, _ = self.cell_centers(cell_size)
        data = np.minimum(obstacle_field, self._boundary_sdf(centers))
        return GridSDF(origin=origin, data=data, cell_size=cell_size)

    # -- sampling ---------------------------------------------------------------------

    def sample_free(
        self,
        n: int,
        clearance: float = 0.0,
        rng: np.random.Generator | None = None,
        max_tries: int = 200,
    ) -> np.ndarray:
        """Rejection-sample ``n`` points with at least ``clearance`` of free space.

        Raises ``RuntimeError`` rather than returning fewer points than asked for, so a
        world that is too cluttered to plan in fails loudly instead of silently degrading
        the multi-start seeds.
        """
        rng = rng or np.random.default_rng()
        xmin, ymin, xmax, ymax = self.bounds
        out: list[np.ndarray] = []
        for _ in range(max_tries):
            batch = rng.uniform([xmin, ymin], [xmax, ymax], size=(max(n * 4, 32), 2))
            out.extend(batch[self.is_free(batch, clearance)])
            if len(out) >= n:
                return np.stack(out[:n])
        raise RuntimeError(
            f"Could only find {len(out)}/{n} free points with clearance {clearance}; "
            "the world is too cluttered or the clearance is too large."
        )


def random_field(
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
    n_circles: int = 8,
    radius_range: tuple[float, float] = (0.4, 1.0),
    keep_clear: Sequence[tuple[float, float]] = (),
    clear_radius: float = 1.0,
    rng: np.random.Generator | None = None,
) -> ObstacleField:
    """Random disc-obstacle field, keeping ``keep_clear`` points (start/goal) unobstructed."""
    rng = rng or np.random.default_rng()
    xmin, ymin, xmax, ymax = bounds
    circles: list[Circle] = []
    guard = np.asarray(keep_clear, dtype=float).reshape(-1, 2)
    while len(circles) < n_circles:
        r = float(rng.uniform(*radius_range))
        cx = float(rng.uniform(xmin + r, xmax - r))
        cy = float(rng.uniform(ymin + r, ymax - r))
        if guard.size:
            if np.any(np.linalg.norm(guard - [cx, cy], axis=-1) < r + clear_radius):
                continue
        circles.append(Circle(cx, cy, r))
    return ObstacleField(bounds=bounds, circles=circles)


def scenario(name: str = "slalom") -> tuple[ObstacleField, np.ndarray, np.ndarray]:
    """A few fixed worlds, returned as ``(field, start_xytheta, goal_xytheta)``.

    These are deterministic so tests and examples have something stable to plan through.
    """
    if name == "slalom":
        # Every obstacle straddles the straight line y = 4, so the direct path is blocked and
        # the robot has to weave. There are two distinct ways around (over/under), which is
        # what makes this a useful multi-start test rather than a warm-up.
        obstacles = ObstacleField(
            bounds=(0.0, 0.0, 12.0, 8.0),
            circles=[
                Circle(3.0, 3.3, 1.2),
                Circle(6.0, 4.7, 1.2),
                Circle(9.0, 3.3, 1.2),
            ],
        )
        return obstacles, np.array([0.8, 4.0, 0.0]), np.array([11.2, 4.0, 0.0])

    if name == "gap":
        # A wall with a single 1.2 m opening: the planner has to commit to the gap.
        obstacles = ObstacleField(
            bounds=(0.0, 0.0, 12.0, 8.0),
            boxes=[Box(6.0, 1.7, 0.4, 1.7), Box(6.0, 6.3, 0.4, 1.7)],
        )
        return obstacles, np.array([1.0, 4.0, 0.0]), np.array([11.0, 4.0, 0.0])

    if name == "clutter":
        obstacles = random_field(
            bounds=(0.0, 0.0, 12.0, 8.0),
            n_circles=10,
            radius_range=(0.35, 0.85),
            keep_clear=[(1.0, 1.0), (11.0, 7.0)],
            clear_radius=0.9,
            rng=np.random.default_rng(0),
        )
        return obstacles, np.array([1.0, 1.0, 0.0]), np.array([11.0, 7.0, np.pi / 2])

    raise ValueError(f"Unknown scenario {name!r}; expected 'slalom', 'gap' or 'clutter'.")
