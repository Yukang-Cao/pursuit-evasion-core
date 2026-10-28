"""Shared physical contract: two fixed-speed, bounded-turn-rate planar cars.

Joint state order: [xP, yP, thetaP, xE, yE, thetaE].
Commands are angular velocities in rad/s, not steering angles or curvature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Vehicle:
    speed: float
    max_turn_rate: float

    def __post_init__(self):
        if (
            not np.isfinite([self.speed, self.max_turn_rate]).all()
            or self.speed <= 0
            or self.max_turn_rate <= 0
        ):
            raise ValueError("Fixed speed and turn bound must be finite and positive.")

    @property
    def minimum_turn_radius(self) -> float:
        return self.speed / self.max_turn_rate


@dataclass(frozen=True)
class GameConfig:
    name: str = "square6d"
    arena_half_width: float = 3.0
    capture_radius: float = 0.3
    control_dt: float = 0.05
    pursuer: Vehicle = Vehicle(1.0, 1.0)
    evader: Vehicle = Vehicle(0.3, 2.0)
    boundary_rule: str = "first_fault_loses"
    point_vehicles: bool = True

    def __post_init__(self):
        scalars = [self.arena_half_width, self.capture_radius, self.control_dt]
        if not np.isfinite(scalars).all() or min(scalars) <= 0:
            raise ValueError(
                "Map size, capture radius and command interval must be finite and positive."
            )
        if self.boundary_rule != "first_fault_loses" or not self.point_vehicles:
            raise ValueError(
                "This version implements point vehicles and first_fault_loses only."
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GameConfig":
        d = dict(data)
        for name in ("pursuer", "evader"):
            if name in d:
                d[name] = Vehicle(**d[name])
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "GameConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def vehicle_step(pose, turn_rate, duration, speed):
    """Exact constant-command arc, with broadcasting and a stable zero-turn limit."""
    pose = np.asarray(pose, dtype=float)
    turn_rate, duration = np.broadcast_arrays(turn_rate, duration)
    angle = turn_rate * duration
    distance = speed * duration * np.sinc(angle / (2 * np.pi))
    mid_heading = pose[..., 2] + 0.5 * angle
    x, y = pose[..., 0] + distance * np.cos(mid_heading), pose[
        ..., 1
    ] + distance * np.sin(mid_heading)
    return np.stack(
        np.broadcast_arrays(x, y, wrap_angle(pose[..., 2] + angle)), axis=-1
    )


def joint_step(state, p_turn, e_turn, duration, config: GameConfig):
    """Propagate without event stopping. Use game.advance for executed motion."""
    state = np.asarray(state, dtype=float)
    p = vehicle_step(state[..., :3], p_turn, duration, config.pursuer.speed)
    e = vehicle_step(state[..., 3:], e_turn, duration, config.evader.speed)
    shape = np.broadcast_shapes(p.shape[:-1], e.shape[:-1])
    return np.concatenate(
        (np.broadcast_to(p, shape + (3,)), np.broadcast_to(e, shape + (3,))), axis=-1
    )


def vector_field(state, p_turn, e_turn, config: GameConfig):
    """World-frame derivative, for checking PDE adapters against the simulator."""
    s = np.asarray(state, dtype=float)
    return np.stack(
        np.broadcast_arrays(
            config.pursuer.speed * np.cos(s[..., 2]),
            config.pursuer.speed * np.sin(s[..., 2]),
            p_turn,
            config.evader.speed * np.cos(s[..., 5]),
            config.evader.speed * np.sin(s[..., 5]),
            e_turn,
        ),
        axis=-1,
    )


def hamiltonian(state, gradient, config: GameConfig):
    """min_P max_E gradient dot dynamics for the declared continuous control sets."""
    s, g = np.asarray(state), np.asarray(gradient)
    drift = config.pursuer.speed * (
        np.cos(s[..., 2]) * g[..., 0] + np.sin(s[..., 2]) * g[..., 1]
    ) + config.evader.speed * (
        np.cos(s[..., 5]) * g[..., 3] + np.sin(s[..., 5]) * g[..., 4]
    )
    return (
        drift
        - config.pursuer.max_turn_rate * abs(g[..., 2])
        + config.evader.max_turn_rate * abs(g[..., 5])
    )


def relative_state(state):
    """Diagnostic coordinates only; walls prevent using this as the whole game state."""
    s = np.asarray(state, dtype=float)
    dx, dy, a = s[..., 3] - s[..., 0], s[..., 4] - s[..., 1], s[..., 2]
    return np.stack(
        (
            np.cos(a) * dx + np.sin(a) * dy,
            -np.sin(a) * dx + np.cos(a) * dy,
            wrap_angle(s[..., 5] - a),
        ),
        axis=-1,
    )
