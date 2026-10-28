"""Grid and target construction shared by numerical backends."""

from dataclasses import asdict, dataclass
import numpy as np

from ..model import GameConfig


@dataclass(frozen=True)
class GridSpec:
    positions: int = 17
    headings: int = 16
    extent: float = 3.5

    def __post_init__(self):
        if self.positions < 5 or self.headings < 4 or self.extent <= 0:
            raise ValueError(
                "Need at least five position samples and four periodic headings."
            )

    @property
    def shape(self):
        n, h = self.positions, self.headings
        return (n, n, h, n, n, h)

    @property
    def states(self):
        return self.positions**4 * self.headings**2

    @property
    def axes(self):
        x = np.linspace(-self.extent, self.extent, self.positions)
        theta = -np.pi + np.arange(self.headings) * 2 * np.pi / self.headings
        return (x, x, theta, x, x, theta)

    @property
    def spacings(self):
        d = 2 * self.extent / (self.positions - 1)
        h = 2 * np.pi / self.headings
        return (d, d, h, d, d, h)

    def to_dict(self):
        return dict(
            **asdict(self),
            shape=list(self.shape),
            states=self.states,
            spacings=list(self.spacings),
            periodic_axes=[2, 5],
            state_order=["xP", "yP", "thetaP", "xE", "yE", "thetaE"],
        )


def target_and_avoid(grid: GridSpec, config: GameConfig, dtype=np.float32):
    """Target: capture OR E boundary fault. Avoid: P boundary fault.

    A tiny signed-distance margin makes a P state exactly on the wall losing
    under the zero-sublevel convention. Query code applies exact event rules.
    The domain must contain the physical square and an exterior padding band.
    """
    if grid.extent <= config.arena_half_width:
        raise ValueError(
            "The numerical domain must extend outside the physical square."
        )
    x = np.asarray(grid.axes[0], dtype=dtype)
    xp, yp = x[:, None, None, None, None, None], x[None, :, None, None, None, None]
    xe, ye = x[None, None, None, :, None, None], x[None, None, None, None, :, None]
    capture = np.sqrt((xp - xe) ** 2 + (yp - ye) ** 2) - config.capture_radius
    e_fault = config.arena_half_width - np.maximum(abs(xe), abs(ye))
    target = np.broadcast_to(np.minimum(capture, e_fault), grid.shape).copy()
    avoid = np.broadcast_to(
        np.maximum(abs(xp), abs(yp)) - config.arena_half_width + 1e-6, grid.shape
    ).copy()
    return target.astype(dtype, copy=False), avoid.astype(dtype, copy=False)


def output_times(horizon, output_dt):
    if horizon <= 0 or output_dt <= 0:
        raise ValueError("Horizon and output spacing must be positive.")
    times = np.minimum(
        np.arange(int(np.ceil(horizon / output_dt)) + 1) * output_dt, horizon
    )
    return np.unique(times)


def global_rate_bound(grid: GridSpec, config: GameConfig):
    dx, _, dh, *_ = grid.spacings
    return (
        2 * (config.pursuer.speed + config.evader.speed) / dx
        + (config.pursuer.max_turn_rate + config.evader.max_turn_rate) / dh
    )
