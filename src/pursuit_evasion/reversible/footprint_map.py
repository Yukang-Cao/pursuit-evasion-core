"""Oriented rectangular footprints in an axis-aligned obstacle map.

Positions refer to footprint centers. A signed separation is used as an
implicit collision function, not as an exact Euclidean distance everywhere.
"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Footprint:
    length: float
    width: float

    def __post_init__(self):
        if (
            not np.isfinite([self.length, self.width]).all()
            or min(self.length, self.width) <= 0
        ):
            raise ValueError("Footprint dimensions must be positive and finite")

    @property
    def radius(self):
        return 0.5 * np.hypot(self.length, self.width)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FootprintMap:
    half_width: float = 2.5
    obstacles: tuple = ()
    pursuer: Footprint = Footprint(0.8, 0.5)
    evader: Footprint = Footprint(0.4, 0.25)

    def __post_init__(self):
        if not np.isfinite(self.half_width) or self.half_width <= 0:
            raise ValueError("Positive square half-width required")
        for box in self.obstacles:
            if (
                len(box) != 4
                or not np.isfinite(box).all()
                or box[0] >= box[1]
                or box[2] >= box[3]
            ):
                raise ValueError("Obstacles are [xmin,xmax,ymin,ymax]")

    def to_dict(self):
        return dict(
            half_width=self.half_width,
            obstacles=[list(b) for b in self.obstacles],
            pursuer=self.pursuer.to_dict(),
            evader=self.evader.to_dict(),
            pose_reference="rectangle_center",
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["half_width"],
            tuple(tuple(b) for b in data["obstacles"]),
            Footprint(**data["pursuer"]),
            Footprint(**data["evader"]),
        )

    def clearance(self, pose, role="pursuer"):
        """Positive iff the rectangle has no contact with any obstacle/wall."""
        s = np.asarray(pose, dtype=float)
        if s.shape[-1] < 3 or not np.isfinite(s).all():
            raise ValueError("Pose requires finite x,y,heading")
        body = getattr(self, role)
        x, y, h = s[..., 0], s[..., 1], s[..., 2]
        c, sn = np.cos(h), np.sin(h)
        ac, ass = abs(c), abs(sn)
        hl, hw = body.length / 2, body.width / 2
        ex, ey = hl * ac + hw * ass, hl * ass + hw * ac
        result = np.minimum(
            self.half_width - abs(x) - ex, self.half_width - abs(y) - ey
        )
        for xmin, xmax, ymin, ymax in self.obstacles:
            ox, oy = (xmin + xmax) / 2, (ymin + ymax) / 2
            hx, hy = (xmax - xmin) / 2, (ymax - ymin) / 2
            dx, dy = x - ox, y - oy
            # Separating axes: world x/y and the two body axes.
            distances = np.stack(
                [
                    abs(dx) - hx - ex,
                    abs(dy) - hy - ey,
                    abs(dx * c + dy * sn) - hl - hx * ac - hy * ass,
                    abs(-dx * sn + dy * c) - hw - hx * ass - hy * ac,
                ],
                axis=-1,
            )
            result = np.minimum(result, distances.max(axis=-1))
        return result

    def point_sdf(self, xy):
        """Point signed-clearance image for the obstacles and square room."""
        xy = np.asarray(xy, float)
        x, y = xy[..., 0], xy[..., 1]
        result = self.half_width - np.maximum(abs(x), abs(y))
        for xmin, xmax, ymin, ymax in self.obstacles:
            qx = abs(x - (xmin + xmax) / 2) - (xmax - xmin) / 2
            qy = abs(y - (ymin + ymax) / 2) - (ymax - ymin) / 2
            value = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(
                np.maximum(qx, qy), 0
            )
            result = np.minimum(result, value)
        return result

    def surfaces(self, state, capture_radius):
        """Existing first-fault-loses game: capture OR E fault, avoid P fault."""
        s = np.asarray(state, float)
        cap = np.linalg.norm(s[..., :2] - s[..., 4:6], axis=-1) - capture_radius
        pc = self.clearance(s[..., :4], "pursuer")
        ec = self.clearance(s[..., 4:], "evader")
        target = np.minimum(cap, ec)
        avoid = -pc + 1e-6
        return np.maximum(target, avoid), avoid

    def event(self, state, capture_radius):
        s = np.asarray(state, float)
        if self.clearance(s[:4], "pursuer") <= 0:
            return "pursuer_collision"
        if np.linalg.norm(s[:2] - s[4:6]) <= capture_radius:
            return "capture"
        if self.clearance(s[4:], "evader") <= 0:
            return "evader_collision"
        return None


@dataclass
class FootprintStep:
    state: np.ndarray
    elapsed: float
    event: str | None

    @property
    def terminal(self):
        return self.event is not None


def advance(state, p_control, e_control, config, geometry, duration=None):
    """Held-control step with continuous capture and footprint collision events."""
    from ..game import _first_zero
    from .model import joint_step, vehicle_step

    if config.point_vehicles:
        raise ValueError("Footprint events require point_vehicles=False.")
    duration = config.control_dt if duration is None else float(duration)
    s = np.asarray(state, float)
    p = np.asarray(p_control, float)
    e = np.asarray(e_control, float)
    end = joint_step(s, p, e, duration, config)
    initial = geometry.event(s, config.capture_radius)
    if initial:
        return FootprintStep(s.copy(), 0.0, initial)
    speeds = [max(abs(s[k + 3]), abs(end[k + 3])) for k in (0, 4)]
    events = []
    for offset, control, role, priority, speed in [
        (0, p, "pursuer", 0, speeds[0]),
        (4, e, "evader", 2, speeds[1]),
    ]:
        pose = s[offset : offset + 4]
        body = getattr(geometry, role)
        radius = body.radius
        for xmin, xmax, ymin, ymax in geometry.obstacles:
            center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
            obstacle_radius = 0.5 * np.hypot(xmax - xmin, ymax - ymin)
            radius = max(
                radius,
                np.linalg.norm(pose[:2] - center) + speed * duration + obstacle_radius,
            )
        lipschitz = speed + radius * speed * abs(control[1])
        if float(geometry.clearance(pose, role)) > lipschitz * duration:
            continue
        t = _first_zero(
            lambda t: float(
                geometry.clearance(
                    vehicle_step(pose, control, t, getattr(config, role)), role
                )
            ),
            lipschitz,
            duration,
            time_tolerance=1e-9,
        )
        if t is not None:
            events.append((t, priority, role + "_collision"))
    if np.linalg.norm(s[:2] - s[4:6]) - config.capture_radius <= sum(speeds) * duration:

        def separation(t):
            z = joint_step(s, p, e, t, config)
            return float(np.linalg.norm(z[:2] - z[4:6]) - config.capture_radius)

        t = _first_zero(separation, sum(speeds), duration, time_tolerance=1e-9)
        if t is not None:
            events.append((t, 1, "capture"))
    if not events:
        return FootprintStep(end, duration, None)
    first = min(x[0] for x in events)
    selected = min((x for x in events if x[0] <= first + 1e-8), key=lambda x: x[1])
    return FootprintStep(
        joint_step(s, p, e, selected[0], config), float(selected[0]), selected[2]
    )
