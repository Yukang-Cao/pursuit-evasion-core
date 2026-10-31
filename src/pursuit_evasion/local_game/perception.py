"""Ideal 360-degree range scans and rolling observed maps, refreshed in body frame.

Only ray returns and traversed cells update memory. Full geometry is used by
this simulator to render the scan, not copied into the actor observation.
"""

from dataclasses import dataclass
import numpy as np
from .native import scan_ranges
from ..reversible.planning_frame import PlanningFrame


@dataclass(frozen=True)
class SensorSettings:
    range: float = 6.0
    beams: int = 720
    memory_resolution: float = 0.05
    extent: float = 6.0
    image_size: int = 192


class LidarMemory:
    def __init__(self, arena, settings=SensorSettings()):
        self.arena = arena
        self.settings = settings
        self.origin = np.array([-arena.half_width - 0.5, -arena.half_width - 0.5])
        n = int(np.ceil((2 * arena.half_width + 1) / settings.memory_resolution))
        self.cells = np.zeros((n, n), np.uint8)  # 0unknown,1observedfree,2hit
        self.points = np.full((n, n, 2), np.nan)
        self.last_ranges = None
        self.scans = 0

    def observe(self, pose):
        cfg = self.settings
        pose = np.asarray(pose)
        ranges = scan_ranges(self.arena, pose, cfg.beams, cfg.range)
        angles = pose[2] + np.arange(cfg.beams) * 2 * np.pi / cfg.beams
        r = np.arange(0, cfg.range, cfg.memory_resolution * 0.5)
        distance = np.broadcast_to(r, (cfg.beams, len(r)))
        valid = distance < ranges[:, None] - cfg.memory_resolution * 0.5
        x = pose[0] + distance * np.cos(angles)[:, None]
        y = pose[1] + distance * np.sin(angles)[:, None]
        ix = np.floor((x - self.origin[0]) / cfg.memory_resolution).astype(int)
        iy = np.floor((y - self.origin[1]) / cfg.memory_resolution).astype(int)
        valid &= (
            (ix >= 0)
            & (iy >= 0)
            & (ix < self.cells.shape[1])
            & (iy < self.cells.shape[0])
        )
        flat = iy[valid] * self.cells.shape[1] + ix[valid]
        flat = np.unique(flat)
        current = self.cells.reshape(-1)
        current[flat[current[flat] != 2]] = 1
        x = pose[0] + ranges * np.cos(angles)
        y = pose[1] + ranges * np.sin(angles)
        ix = np.floor((x - self.origin[0]) / cfg.memory_resolution).astype(int)
        iy = np.floor((y - self.origin[1]) / cfg.memory_resolution).astype(int)
        valid = (
            (ranges < cfg.range - 1e-8)
            & (ix >= 0)
            & (iy >= 0)
            & (ix < self.cells.shape[1])
            & (iy < self.cells.shape[0])
        )
        self.cells[iy[valid], ix[valid]] = 2
        keys = iy[valid] * self.cells.shape[1] + ix[valid]
        _, first = np.unique(keys, return_index=True)
        self.points.reshape(-1, 2)[keys[first]] = np.stack(
            [x[valid][first], y[valid][first]], -1
        )
        self.last_ranges = ranges
        self.scans += 1

    def observed_arena(self):
        from .geometry import CircleArena

        points = self.points[self.cells == 2]
        circles = np.c_[points, np.full(len(points), 1e-9)]
        return CircleArena(
            circles,
            self.arena.half_width,
            self.arena.pursuer,
            self.arena.evader,
            source={"kind": "observed static surface points plus known arena boundary"},
        )

    def image(self, pose, *, ideal=False):
        from scipy.ndimage import distance_transform_edt

        cfg = self.settings
        frame = PlanningFrame(*np.asarray(pose)[:3])
        q = np.linspace(-cfg.extent, cfg.extent, cfg.image_size)
        x, y = np.meshgrid(q, q)
        c, s = np.cos(frame.heading), np.sin(frame.heading)
        wx = frame.x + c * x - s * y
        wy = frame.y + s * x + c * y
        if ideal:
            sdf = self.arena.point_clearance(np.stack([wx, wy], -1)).astype(np.float32)
            return np.stack([sdf, np.ones_like(sdf)]), frame
        ix = np.floor((wx - self.origin[0]) / cfg.memory_resolution).astype(int)
        iy = np.floor((wy - self.origin[1]) / cfg.memory_resolution).astype(int)
        valid = (
            (ix >= 0)
            & (iy >= 0)
            & (ix < self.cells.shape[1])
            & (iy < self.cells.shape[0])
        )
        status = np.zeros(ix.shape, np.uint8)
        status[valid] = self.cells[iy[valid], ix[valid]]
        # Forward-rasterize hits so resampling does not erase thin observed surfaces.
        gy, gx = np.nonzero(self.cells == 2)
        dx = self.origin[0] + (gx + 0.5) * cfg.memory_resolution - frame.x
        dy = self.origin[1] + (gy + 0.5) * cfg.memory_resolution - frame.y
        px = np.rint(
            (c * dx + s * dy + cfg.extent) / (2 * cfg.extent) * (cfg.image_size - 1)
        ).astype(int)
        py = np.rint(
            (-s * dx + c * dy + cfg.extent) / (2 * cfg.extent) * (cfg.image_size - 1)
        ).astype(int)
        good = (px >= 0) & (py >= 0) & (px < cfg.image_size) & (py < cfg.image_size)
        status[py[good], px[good]] = 2
        occupied = status == 2
        pixel = 2 * cfg.extent / (cfg.image_size - 1)
        sdf = (
            (
                (distance_transform_edt(~occupied) - distance_transform_edt(occupied))
                * pixel
            ).astype(np.float32)
            if occupied.any()
            else np.full(x.shape, cfg.extent, np.float32)
        )
        return np.stack([sdf, (status > 0).astype(np.float32)]), frame


def observed_clearances(image, states, extent, p_body, e_body):
    """Observed-map feature only; NOT the simulator's collision predicate.

    Sample the visible-obstacle SDF on a fixed rectangular stencil. Unknown
    cells remain distinguishable in the network's separate known-mask channel.
    """
    s = np.asarray(states, float)
    shape = s.shape[:-1]
    s = s.reshape(-1, 8)
    n = image.shape[-1]
    results = []
    for offset, body in [(0, p_body), (4, e_body)]:
        a, b = np.meshgrid(
            np.linspace(-body.length / 2, body.length / 2, 5),
            np.linspace(-body.width / 2, body.width / 2, 3),
        )
        a = a.ravel()
        b = b.ravel()
        c = np.cos(s[:, offset + 2, None])
        q = np.sin(s[:, offset + 2, None])
        x = s[:, offset, None] + a * c - b * q
        y = s[:, offset + 1, None] + a * q + b * c
        gx = np.clip((x / extent + 1) * 0.5 * (n - 1), 0, n - 1)
        gy = np.clip((y / extent + 1) * 0.5 * (n - 1), 0, n - 1)
        ix = np.minimum(gx.astype(int), n - 2)
        iy = np.minimum(gy.astype(int), n - 2)
        u = gx - ix
        v = gy - iy
        z = (
            (1 - u) * (1 - v) * image[0, iy, ix]
            + u * (1 - v) * image[0, iy, ix + 1]
            + (1 - u) * v * image[0, iy + 1, ix]
            + u * v * image[0, iy + 1, ix + 1]
        )
        results.append(z.min(axis=1).reshape(shape))
    return results
