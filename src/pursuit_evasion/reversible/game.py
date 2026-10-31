"""Exact held-motion events and explicit map geometry for reversible cars."""

from dataclasses import dataclass
import numpy as np

from ..game import _first_zero
from .model import joint_step
from .events_geometry import first_wall_time, first_capture_time


@dataclass(frozen=True)
class WorldMap:
    """Square boundary and optional axis-aligned rectangles; None means free space.

    Rectangle coordinates are (xmin, xmax, ymin, ymax). Vehicles are points.
    """

    half_width: float | None = 3.0
    obstacles: tuple = ()

    def __post_init__(self):
        if self.half_width is not None and (
            not np.isfinite(self.half_width) or self.half_width <= 0
        ):
            raise ValueError("Square half-width must be positive, or None.")
        for box in self.obstacles:
            if (
                len(box) != 4
                or not np.isfinite(box).all()
                or box[0] >= box[1]
                or box[2] >= box[3]
            ):
                raise ValueError("Invalid rectangle.")

    def clearance(self, points):
        p = np.asarray(points, float)
        if p.shape[-1] != 2:
            raise ValueError("Expected points[...,2].")
        out = (
            np.full(p.shape[:-1], np.inf)
            if self.half_width is None
            else self.half_width - np.max(abs(p), axis=-1)
        )
        for xmin, xmax, ymin, ymax in self.obstacles:
            q = abs(p - np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])) - np.array(
                [(xmax - xmin) / 2, (ymax - ymin) / 2]
            )
            signed = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(
                np.max(q, axis=-1), 0
            )
            out = np.minimum(out, signed)
        return out

    def to_dict(self):
        return dict(
            half_width=self.half_width, obstacles=[list(x) for x in self.obstacles]
        )


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    elapsed: float
    event: str | None

    @property
    def terminal(self):
        return self.event is not None

    @property
    def physical_capture(self):
        return self.event == "capture"

    @property
    def pursuer_success(self):
        return self.event in ("capture", "evader_boundary_fault")


def _measures(config, world):
    return [
        (lambda z: world.clearance(z[..., :2]), config.pursuer.max_abs_speed),
        (
            lambda z: np.linalg.norm(z[..., :2] - z[..., 4:6], axis=-1)
            - config.capture_radius,
            config.pursuer.max_abs_speed + config.evader.max_abs_speed,
        ),
        (lambda z: world.clearance(z[..., 4:6]), config.evader.max_abs_speed),
    ]


def advance(state, p_control, e_control, config, duration=None, world=None):
    if not config.point_vehicles:
        raise ValueError(
            "Point-game events cannot handle footprints; use footprint_map.advance."
        )
    s = np.asarray(state, float)
    if s.shape != (8,):
        raise ValueError("Expected one eight-dimensional state.")
    dt = config.control_dt if duration is None else float(duration)
    world = WorldMap(config.arena_half_width) if world is None else world

    def pose(t):
        return joint_step(s, p_control, e_control, t, config)

    endpoint = pose(dt)  # validates all inputs before event handling
    if (
        world.half_width is None
        and not world.obstacles
        and np.linalg.norm(s[:2] - s[4:6])
        > config.capture_radius
        + (config.pursuer.max_abs_speed + config.evader.max_abs_speed) * dt
        + 1e-12
    ):
        return Transition(endpoint, dt, None)
    roots = [
        first_wall_time(s[:4], p_control, dt, config.pursuer, world),
        first_capture_time(s, p_control, e_control, dt, config),
        first_wall_time(s[4:], e_control, dt, config.evader, world),
    ]
    names = ("pursuer_boundary_fault", "capture", "evader_boundary_fault")
    found = [(root, i, names[i]) for i, root in enumerate(roots) if root is not None]
    if not found:
        return Transition(endpoint, dt, None)
    first = min(x[0] for x in found)
    t, _, name = min((x for x in found if x[0] <= first + 2e-8), key=lambda x: x[1])
    return Transition(pose(t), t, name)


def predict_pairs(state, p_controls, e_controls, config, duration=None, world=None):
    """First-event successors for all pairs, codes0:none1:capture2:Pwall3:Ewall.

    A speed bound prunes noncrossing intervals. Bisection locates ordinary
    crossings; adaptive scalar search covers possible enter-and-exit intervals.
    Returned states are at the first event (or the full duration when continuing).
    """
    if not config.point_vehicles:
        raise ValueError("Point-game predictions cannot handle footprints.")
    s = np.asarray(state, float)
    P = np.asarray(p_controls, float)
    E = np.asarray(e_controls, float)
    if (
        s.shape != (8,)
        or P.ndim != 2
        or E.ndim != 2
        or P.shape[1] != 2
        or E.shape[1] != 2
    ):
        raise ValueError("Expected state[8], P[n,2], E[m,2].")
    if not len(P) or not len(E):
        raise ValueError("Empty action catalogue.")
    dt = config.control_dt if duration is None else float(duration)
    world = WorldMap(config.arena_half_width) if world is None else world
    shape = (len(P), len(E))

    def poses(t):
        return joint_step(s, P[:, None, :], E[None, :, :], t, config)

    endpoint = poses(dt)
    if (
        world.half_width is None
        and not world.obstacles
        and np.linalg.norm(s[:2] - s[4:6])
        > config.capture_radius
        + (config.pursuer.max_abs_speed + config.evader.max_abs_speed) * dt
        + 1e-12
    ):
        return endpoint, np.full(shape, dt), np.zeros(shape, dtype=int)
    initial = np.broadcast_to(s, endpoint.shape)
    p_roots = np.array(
        [first_wall_time(s[:4], u, dt, config.pursuer, world) for u in P], object
    )
    e_roots = np.array(
        [first_wall_time(s[4:], u, dt, config.evader, world) for u in E], object
    )
    p_roots = np.array([np.inf if x is None else x for x in p_roots], float)
    e_roots = np.array([np.inf if x is None else x for x in e_roots], float)
    wall_p = np.broadcast_to(p_roots[:, None], shape)
    wall_e = np.broadcast_to(e_roots[None, :], shape)
    measure = (
        lambda z: np.linalg.norm(z[..., :2] - z[..., 4:6], axis=-1)
        - config.capture_radius
    )
    root = np.full(shape, np.inf)
    va = measure(initial)
    root[va <= 0] = 0
    left = 0.0
    vp = np.maximum(abs(s[3]), abs(endpoint[..., 3]))
    ve = np.maximum(abs(s[7]), abs(endpoint[..., 7]))
    accel = (
        abs(P[:, None, 0])
        + vp**2 * abs(P[:, None, 1])
        + abs(E[None, :, 0])
        + ve**2 * abs(E[None, :, 1])
    )
    for right in np.linspace(0, dt, 5)[1:]:
        vb = measure(poses(right))
        active = ~np.isfinite(root)
        crossing = active & (vb <= 0)
        if crossing.any():
            lo = np.full(shape, left)
            hi = np.full(shape, right)
            for _ in range(22):
                mid = (lo + hi) / 2
                vm = measure(poses(mid))
                hi = np.where(crossing & (vm <= 0), mid, hi)
                lo = np.where(crossing & (vm > 0), mid, lo)
            root[crossing] = (lo[crossing] + hi[crossing]) / 2
        middle = poses((left + right) / 2)
        velocity_p = middle[..., 3, None] * np.stack(
            [np.cos(middle[..., 2]), np.sin(middle[..., 2])], axis=-1
        )
        velocity_e = middle[..., 7, None] * np.stack(
            [np.cos(middle[..., 6]), np.sin(middle[..., 6])], axis=-1
        )
        bound = np.minimum(
            config.pursuer.max_abs_speed + config.evader.max_abs_speed,
            np.linalg.norm(velocity_e - velocity_p, axis=-1)
            + accel * (right - left) / 2,
        )
        ambiguous = active & ~crossing & (va + vb <= bound * (right - left))
        for i, j in np.argwhere(ambiguous):
            beginning = joint_step(s, P[i], E[j], left, config)
            found = first_capture_time(
                beginning, P[i], E[j], float(right - left), config
            )
            if found is not None:
                root[i, j] = left + found
        left = float(right)
        va = vb
    roots = [wall_p, root, wall_e]
    first = np.minimum(np.minimum(roots[0], roots[1]), roots[2])
    terminal = np.isfinite(first)
    code = np.where(
        roots[0] <= first + 2e-8, 2, np.where(roots[1] <= first + 2e-8, 1, 3)
    )
    times = np.where(terminal, first, dt)
    return poses(times), times, np.where(terminal, code, 0)
