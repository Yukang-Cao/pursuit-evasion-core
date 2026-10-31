"""Rectangular capture in bounded arenas of circular BARN obstacles.

Map occupancy is interpreted as the original equal-radius cylinders. Positions
refer to vehicle rectangle centers. Capture is rectangle contact, never a wall
fault. Both cars retain four dynamic states [x,y,heading,signed speed].
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from ..reversible.footprint_map import Footprint


def rectangle_separation(p, e, p_body=Footprint(0.8, 0.5), e_body=Footprint(0.4, 0.25)):
    """Positive iff the oriented rectangles are separated; <=0 means contact.

    This separating-axis implicit function has the exact contact sign. Its
    positive magnitude is not generally the Euclidean distance between bodies.
    """
    p, e = np.asarray(p), np.asarray(e)
    d = e[..., :2] - p[..., :2]
    delta = e[..., 2] - p[..., 2]
    ca, sa = np.abs(np.cos(delta)), np.abs(np.sin(delta))
    values = []
    for pose, own, other, flip in [(p, p_body, e_body, 1), (e, e_body, p_body, -1)]:
        c, s = np.cos(pose[..., 2]), np.sin(pose[..., 2])
        x = d[..., 0] * c + d[..., 1] * s
        y = -d[..., 0] * s + d[..., 1] * c
        values.extend(
            [
                abs(x) - own.length / 2 - (other.length * ca + other.width * sa) / 2,
                abs(y) - own.width / 2 - (other.length * sa + other.width * ca) / 2,
            ]
        )
    return np.maximum.reduce(values)


@dataclass
class CircleArena:
    circles: np.ndarray
    half_width: float
    pursuer: Footprint = Footprint(0.8, 0.5)
    evader: Footprint = Footprint(0.4, 0.25)
    source: dict | None = None

    def __post_init__(self):
        self.circles = np.asarray(self.circles, np.float64).reshape(-1, 3)
        if not np.isfinite(self.circles).all() or np.any(self.circles[:, 2] <= 0):
            raise ValueError("Finite circles[x,y,r] with positive radii required")
        if not np.isfinite(self.half_width) or self.half_width <= 0:
            raise ValueError("Positive bounded arena size required")

    def to_dict(self):
        return dict(
            circles=self.circles.tolist(),
            half_width=self.half_width,
            pursuer=self.pursuer.to_dict(),
            evader=self.evader.to_dict(),
            source=self.source,
            capture_rule="oriented_rectangle_contact",
            pose_reference="rectangle_center",
        )

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["circles"],
            d["half_width"],
            Footprint(**d["pursuer"]),
            Footprint(**d["evader"]),
            d.get("source"),
        )

    @classmethod
    def from_barn(cls, grid_file, *, base_resolution=0.15, scale=2):
        grid_file = Path(grid_file)
        grid = np.load(grid_file, allow_pickle=False)
        if (
            grid.ndim != 2
            or grid.shape[0] != grid.shape[1]
            or not np.isin(grid, [0, 1]).all()
        ):
            raise ValueError("Expected original square binary BARN grid")
        res = float(base_resolution * scale)
        row, col = np.nonzero(grid)
        size = grid.shape[0]
        centers = np.stack(
            [
                (col + 0.5 - size / 2) * res,
                (size / 2 - row - 0.5) * res,
                np.full(len(row), res / 2),
            ],
            -1,
        )
        return cls(
            centers,
            size * res / 2,
            source=dict(
                kind="BARN occupancy cylinders",
                map_file=grid_file.name,
                sha256=hashlib.sha256(grid_file.read_bytes()).hexdigest(),
                original_resolution=base_resolution,
                scale=scale,
                cell_size=res,
                grid_shape=list(grid.shape),
                boundary="added enclosing square walls",
                coordinates="columns+x, rows-y; map centered at origin",
            ),
        )

    def clearance(self, poses, role="pursuer"):
        """Exact signed rectangle/circle or rectangle/wall separation."""
        s = np.asarray(poses, float)
        body = getattr(self, role)
        c, sn = np.cos(s[..., 2]), np.sin(s[..., 2])
        ex = (body.length * abs(c) + body.width * abs(sn)) / 2
        ey = (body.length * abs(sn) + body.width * abs(c)) / 2
        out = np.minimum(
            self.half_width - abs(s[..., 0]) - ex, self.half_width - abs(s[..., 1]) - ey
        )
        for x, y, r in self.circles:
            dx, dy = x - s[..., 0], y - s[..., 1]
            qx = abs(dx * c + dy * sn) - body.length / 2
            qy = abs(-dx * sn + dy * c) - body.width / 2
            value = (
                np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
                + np.minimum(np.maximum(qx, qy), 0)
                - r
            )
            out = np.minimum(out, value)
        return out

    def point_clearance(self, xy):
        s = np.asarray(xy, float)
        out = self.half_width - abs(s).max(axis=-1)
        for x, y, r in self.circles:
            out = np.minimum(out, np.hypot(s[..., 0] - x, s[..., 1] - y) - r)
        return out

    def capture_value(self, state):
        s = np.asarray(state)
        return rectangle_separation(s[..., :4], s[..., 4:], self.pursuer, self.evader)
