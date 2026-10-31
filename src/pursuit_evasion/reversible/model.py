"""Shared reversible-car contract.

World state: [xP,yP,thetaP,vP,xE,yE,thetaE,vE].
Relative state: [x,y,thetaE-thetaP,vP,vE].
Each command: [acceleration (m/s²), curvature (1/m)].
Curvature is commanded directly; signed speed changes through bounded acceleration.
"""

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class VehicleLimits:
    speed_min: float
    speed_max: float
    acceleration: float
    max_curvature: float

    def __post_init__(self):
        values = (self.speed_min, self.speed_max, self.acceleration, self.max_curvature)
        if not np.isfinite(values).all() or not self.speed_min < 0 < self.speed_max:
            raise ValueError("Signed finite speed bounds must straddle zero.")
        if self.acceleration <= 0 or self.max_curvature <= 0:
            raise ValueError("Acceleration and curvature bounds must be positive.")

    @property
    def max_abs_speed(self):
        return max(abs(self.speed_min), self.speed_max)

    @property
    def minimum_turn_radius(self):
        return 1.0 / self.max_curvature


@dataclass(frozen=True)
class ReversibleConfig:
    name: str = "reversible_cars"
    pursuer: VehicleLimits = VehicleLimits(-1.0, 2.0, 2.0, 1.0)
    evader: VehicleLimits = VehicleLimits(-0.5, 1.0, 1.0, 2.0)
    capture_radius: float = 0.3
    control_dt: float = 0.05
    arena_half_width: float = 3.0
    boundary_rule: str = "first_fault_loses"
    point_vehicles: bool = True

    def __post_init__(self):
        values = (self.capture_radius, self.control_dt, self.arena_half_width)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError(
                "Radius, control interval and square extent must be positive."
            )
        if self.boundary_rule != "first_fault_loses" or not isinstance(
            self.point_vehicles, bool
        ):
            raise ValueError(
                "Use first_fault_loses and an explicit point/footprint flag."
            )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        for role in ("pursuer", "evader"):
            if role in data:
                data[role] = VehicleLimits(**data[role])
        return cls(**data)


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi


def integrated_speed(speed, acceleration, duration, limits):
    """Final clipped speed and exact signed distance under held acceleration.

    Acceleration ceases at a speed limit. Integration splits analytically at that
    hitting time. Crossing zero does not reset heading or teleport position.
    """
    speed, acceleration, duration = np.broadcast_arrays(
        np.asarray(speed, float),
        np.asarray(acceleration, float),
        np.asarray(duration, float),
    )
    target = np.where(acceleration >= 0, limits.speed_max, limits.speed_min)
    hit = np.divide(
        target - speed,
        acceleration,
        out=np.full_like(speed, np.inf),
        where=abs(acceleration) > 1e-14,
    )
    active = np.minimum(duration, np.maximum(0.0, hit))
    final = np.clip(speed + acceleration * duration, limits.speed_min, limits.speed_max)
    distance = (
        speed * active + 0.5 * acceleration * active**2 + final * (duration - active)
    )
    return final, distance


def vehicle_step(states, controls, duration, limits):
    """Exact held acceleration/curvature propagation, supporting batch arrays.

    The trajectory follows a constant-curvature arc parameterized by signed
    distance, including speed saturation and reversal during the interval.
    Inputs must be admissible; this function does not silently clip controls.
    """
    s, u = np.asarray(states, float), np.asarray(controls, float)
    if s.shape[-1] != 4 or u.shape[-1] != 2:
        raise ValueError("Expected state[...,4] and control[...,2].")
    shape = np.broadcast_shapes(s.shape[:-1], u.shape[:-1], np.shape(duration))
    s, u = np.broadcast_to(s, shape + (4,)), np.broadcast_to(u, shape + (2,))
    dt = np.broadcast_to(np.asarray(duration, float), shape)
    if (
        not np.isfinite(s).all()
        or not np.isfinite(u).all()
        or not np.isfinite(dt).all()
    ):
        raise ValueError("State, command and duration must be finite.")
    if (dt < 0).any():
        raise ValueError("Duration must be nonnegative.")
    if (
        (s[..., 3] < limits.speed_min - 1e-9) | (s[..., 3] > limits.speed_max + 1e-9)
    ).any():
        raise ValueError("Initial speed is outside the model bounds.")
    if (abs(u[..., 0]) > limits.acceleration + 1e-9).any() or (
        abs(u[..., 1]) > limits.max_curvature + 1e-9
    ).any():
        raise ValueError("Control exceeds model bounds.")
    v, distance = integrated_speed(s[..., 3], u[..., 0], dt, limits)
    turn = u[..., 1] * distance
    chord = distance * np.sinc(turn / (2 * np.pi))
    midpoint = s[..., 2] + turn / 2
    return np.stack(
        (
            s[..., 0] + chord * np.cos(midpoint),
            s[..., 1] + chord * np.sin(midpoint),
            wrap_angle(s[..., 2] + turn),
            v,
        ),
        axis=-1,
    )


def joint_step(states, p_controls, e_controls, duration, config):
    states = np.asarray(states, float)
    if states.shape[-1] != 8:
        raise ValueError("World state must have eight components.")
    p = vehicle_step(states[..., :4], p_controls, duration, config.pursuer)
    e = vehicle_step(states[..., 4:], e_controls, duration, config.evader)
    shape = np.broadcast_shapes(p.shape[:-1], e.shape[:-1])
    return np.concatenate(
        (np.broadcast_to(p, shape + (4,)), np.broadcast_to(e, shape + (4,))), axis=-1
    )


def relative_state(states):
    s = np.asarray(states, float)
    if s.shape[-1] != 8 or not np.isfinite(s).all():
        raise ValueError("Expected finite world states[...,8].")
    dx, dy = s[..., 4] - s[..., 0], s[..., 5] - s[..., 1]
    c, q = np.cos(s[..., 2]), np.sin(s[..., 2])
    return np.stack(
        (
            c * dx + q * dy,
            -q * dx + c * dy,
            wrap_angle(s[..., 6] - s[..., 2]),
            s[..., 3],
            s[..., 7],
        ),
        axis=-1,
    )


def _acceleration(v, a, limits):
    return np.where(
        ((v <= limits.speed_min) & (a < 0)) | ((v >= limits.speed_max) & (a > 0)),
        0.0,
        a,
    )


def world_rhs(states, p_controls, e_controls, config):
    s, p, e = np.asarray(states), np.asarray(p_controls), np.asarray(e_controls)
    terms = (
        s[..., 3] * np.cos(s[..., 2]),
        s[..., 3] * np.sin(s[..., 2]),
        s[..., 3] * p[..., 1],
        _acceleration(s[..., 3], p[..., 0], config.pursuer),
        s[..., 7] * np.cos(s[..., 6]),
        s[..., 7] * np.sin(s[..., 6]),
        s[..., 7] * e[..., 1],
        _acceleration(s[..., 7], e[..., 0], config.evader),
    )
    return np.stack(np.broadcast_arrays(*terms), axis=-1)


def relative_rhs(states, p_controls, e_controls, config):
    r, p, e = np.asarray(states), np.asarray(p_controls), np.asarray(e_controls)
    x, y, psi, vp, ve = [r[..., k] for k in range(5)]
    wp, we = vp * p[..., 1], ve * e[..., 1]
    terms = (
        ve * np.cos(psi) - vp + wp * y,
        ve * np.sin(psi) - wp * x,
        we - wp,
        _acceleration(vp, p[..., 0], config.pursuer),
        _acceleration(ve, e[..., 0], config.evader),
    )
    return np.stack(np.broadcast_arrays(*terms), axis=-1)
