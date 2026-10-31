"""Rigid coordinate transforms for a fixed local planning frame."""

from dataclasses import dataclass
import numpy as np
from .model import wrap_angle


@dataclass(frozen=True)
class PlanningFrame:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0

    def __post_init__(self):
        if not np.isfinite([self.x, self.y, self.heading]).all():
            raise ValueError("Frame origin and heading must be finite")

    def to_local(self, states):
        s = np.asarray(states, float)
        if s.ndim < 1 or s.shape[-1] != 8 or not np.isfinite(s).all():
            raise ValueError("Expected finite eight-component game states")
        out = s.copy()
        c = np.cos(self.heading)
        sn = np.sin(self.heading)
        for k in (0, 4):
            dx = s[..., k] - self.x
            dy = s[..., k + 1] - self.y
            out[..., k] = c * dx + sn * dy
            out[..., k + 1] = -sn * dx + c * dy
            out[..., k + 2] = wrap_angle(s[..., k + 2] - self.heading)
        return out

    def to_world(self, states):
        s = np.asarray(states, float)
        if s.ndim < 1 or s.shape[-1] != 8 or not np.isfinite(s).all():
            raise ValueError("Expected finite eight-component game states")
        out = s.copy()
        c = np.cos(self.heading)
        sn = np.sin(self.heading)
        for k in (0, 4):
            out[..., k] = c * s[..., k] - sn * s[..., k + 1] + self.x
            out[..., k + 1] = sn * s[..., k] + c * s[..., k + 1] + self.y
            out[..., k + 2] = wrap_angle(s[..., k + 2] + self.heading)
        return out

    def sdf_image(self, geometry, extent=6.0, size=128):
        q = np.linspace(-extent, extent, size)
        x, y = np.meshgrid(q, q)
        c = np.cos(self.heading)
        sn = np.sin(self.heading)
        points = np.stack([c * x - sn * y + self.x, sn * x + c * y + self.y], -1)
        return geometry.point_sdf(points).astype(np.float32)
