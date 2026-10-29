"""C-Free-Uniform MPPI navigation adapted to the shared reversible-car model.

The tracking objective, Gaussian sampling, and raw-noise softmax update come
from the user's existing implementation. This is a documented NumPy adaptation,
not an import of its ROS/perception stack. See docs/reversible-mppi.md.
"""

from dataclasses import dataclass
import time
import numpy as np

from ..game import _first_zero
from ._cfree_mppi import navigation_costs, raw_noise_update
from .events_geometry import first_wall_time
from .game import WorldMap
from .model import vehicle_step


@dataclass(frozen=True)
class MPPISettings:
    samples: int = 512
    horizon: float = 1.5
    iterations: int = 1
    temperature: float = 0.5
    noise_std: tuple[float, float] | None = None
    sampler: str = "gaussian"
    log_noise_sigma: float = 1.0
    goal_tolerance: float = 0.15
    distance_weight: float = 1.0
    terminal_weight: float = 5.0
    obstacle_penalty: float = 2000.0


class ReversibleMPPI:
    """Position-goal tracker; it does not solve a nested pursuer/evader game.

    ``clearance(points[...,2])`` returns signed distance to blocked space, positive
    in the free region. It must be 1-Lipschitz for the interval safety certificate.
    The default is the configured square. Supply an all-infinity callback for
    free space. Capture and exact physical wall events are owned by the simulator.
    """

    def __init__(
        self, config, settings=MPPISettings(), seed=0, *, role="pursuer", clearance=None
    ):
        if role not in ("pursuer", "evader"):
            raise ValueError("Role must be pursuer or evader.")
        self.config = config
        self.settings = settings
        self.limits = getattr(config, role)
        self.seed = int(seed)
        self.clearance = clearance
        owner = getattr(clearance, "__self__", None)
        self._map = (
            WorldMap(config.arena_half_width)
            if clearance is None
            else owner if isinstance(owner, WorldMap) else None
        )
        self.steps = round(settings.horizon / config.control_dt)
        if self.steps < 1 or not np.isclose(
            self.steps * config.control_dt, settings.horizon
        ):
            raise ValueError(
                "Horizon must be a positive integer number of control steps."
            )
        if settings.samples < 10 or settings.iterations < 1:
            raise ValueError("Use at least ten samples and one update iteration.")
        positive = (
            settings.temperature,
            settings.goal_tolerance,
            settings.distance_weight,
            settings.terminal_weight,
            settings.obstacle_penalty,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError(
                "Navigation cost and sampling parameters must be positive."
            )
        self.upper = np.array([self.limits.acceleration, self.limits.max_curvature])
        self.lower = -self.upper
        self.sigma = np.asarray(
            settings.noise_std if settings.noise_std is not None else self.upper / 2,
            dtype=float,
        )
        if (
            self.sigma.shape != (2,)
            or not np.isfinite(self.sigma).all()
            or (self.sigma < 0).any()
        ):
            raise ValueError(
                "Noise standard deviation must contain two nonnegative values."
            )
        if settings.sampler not in ("gaussian", "log_mppi"):
            raise ValueError("sampler must be gaussian or log_mppi")
        if (
            not np.isfinite(settings.log_noise_sigma)
            or not 0 <= settings.log_noise_sigma <= 3
        ):
            raise ValueError("log_noise_sigma must be between zero and three")
        self.reset()

    def sample_noise(self, shape):
        """Signed normal × log-normal perturbations for the log-MPPI option.

        The log multiplier has log-mean zero and explicit log-space std. Thus
        noise_std is the normal-factor scale, not the final NLN standard
        deviation. The latter is noise_std * exp(log_noise_sigma**2), before
        control clipping. The objective and raw-noise update are unchanged.
        """
        noise = self.rng.normal(size=shape) * self.sigma
        if self.settings.sampler == "log_mppi":
            noise *= self.rng.lognormal(0.0, self.settings.log_noise_sigma, size=shape)
        return noise

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.nominal = np.zeros((self.steps, 2))
        self.last = {}
        self._exact_wall_checks = 0
        self._clear_interval_certificates = 0
        self._skipped_terminal_intervals = 0

    def _clearance(self, points):
        points = np.asarray(points)
        if self.clearance is None:
            result = self.config.arena_half_width - np.max(abs(points), axis=-1)
        else:
            result = np.broadcast_to(
                np.asarray(self.clearance(points), float), points.shape[:-1]
            )
        if np.isnan(result).any():
            raise ValueError("Map clearance returned NaN.")
        return result

    def _wall_time(self, pose, control):
        self._exact_wall_checks += 1
        if self._map is not None:
            return first_wall_time(
                pose, control, self.config.control_dt, self.limits, self._map
            )
        # Retain the callback API for arbitrary signed-distance maps. Ordinary
        # WorldMap callbacks use the analytic square/rectangle geometry above.
        return _first_zero(
            lambda t: float(
                self._clearance(vehicle_step(pose, control, t, self.limits)[:2])
            ),
            self.limits.max_abs_speed,
            self.config.control_dt,
            time_tolerance=1e-10,
        )

    def _rollout(self, pose, commands, target):
        """Return costs, validity, safe-prefix lengths, paths for [sample,time,2]."""
        pose = np.asarray(pose, float)
        commands = np.asarray(commands, float)
        if (
            pose.shape != (4,)
            or commands.ndim != 3
            or commands.shape[1:] != (self.steps, 2)
        ):
            raise ValueError("Expected pose[4] and commands[sample,time,2].")
        count = len(commands)
        paths = np.empty((self.steps + 1, count, 4), float)
        paths[0] = pose
        blocked = np.zeros((self.steps + 1, count), bool)
        clearance = self._clearance(paths[0, :, :2])
        blocked[0] = clearance <= 0
        blocked_seen = blocked[0].copy()
        goal_seen = (
            np.sum((paths[0, :, :2] - target) ** 2, axis=1)
            <= self.settings.goal_tolerance**2
        )
        for step in range(self.steps):
            before = paths[step]
            after = vehicle_step(
                before, commands[:, step], self.config.control_dt, self.limits
            )
            paths[step + 1] = after
            next_clearance = self._clearance(after[:, :2])
            # A failed distance certificate means "check the actual arc", not
            # "reject the trajectory". The previous implementation incorrectly
            # rejected some safe motions accelerating away from a nearby wall.
            active = ~blocked_seen & (~goal_seen if step else True)
            self._skipped_terminal_intervals += int((~active).sum())
            interval_speed = np.maximum(abs(before[:, 3]), abs(after[:, 3]))
            certified = (
                (clearance > 0)
                & (next_clearance > 0)
                & (clearance + next_clearance > interval_speed * self.config.control_dt)
            )
            self._clear_interval_certificates += int((active & certified).sum())
            endpoint_hit = active & ((clearance <= 0) | (next_clearance <= 0))
            blocked[step + 1, endpoint_hit] = True
            ambiguous = active & ~certified & ~endpoint_hit
            for i in np.flatnonzero(ambiguous):
                blocked[step + 1, i] = (
                    self._wall_time(before[i], commands[i, step]) is not None
                )
            blocked_seen |= blocked[step + 1]
            goal_seen |= (
                np.sum((after[:, :2] - target) ** 2, axis=1)
                <= self.settings.goal_tolerance**2
            ) & ~blocked_seen
            clearance = next_clearance
        result = navigation_costs(
            paths,
            target,
            blocked,
            goal_tolerance=self.settings.goal_tolerance,
            distance_weight=self.settings.distance_weight,
            terminal_weight=self.settings.terminal_weight,
            obstacle_penalty=self.settings.obstacle_penalty,
        )
        safe_steps = np.minimum(result.blocked_step - 1, self.steps)
        # A waypoint ending the navigation cost does not shorten the physical
        # command currently executed by the game. Protect that whole first
        # interval even if the target is already reached at its beginning.
        first_interval_blocked = np.any(blocked[:2], axis=0)
        safe_steps = np.where(first_interval_blocked, 0, safe_steps)
        safe_steps = np.where(blocked[0], -1, safe_steps)
        valid = ~result.blocked & ~first_interval_blocked
        return result.costs, valid, safe_steps, np.swapaxes(paths, 0, 1)

    def command(self, pose, target, diagnostics=True):
        begin = time.perf_counter()
        pose, target = np.asarray(pose, float), np.asarray(target, float)
        if pose.shape != (4,) or target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("Expected pose[4] and finite position target[2].")
        self._exact_wall_checks = 0
        self._clear_interval_certificates = 0
        self._skipped_terminal_intervals = 0
        # Preserve the source controller's circular warm-start shift.
        self.nominal = np.roll(self.nominal, -1, axis=0)
        fallback = "none"
        for _ in range(self.settings.iterations):
            noise = self.sample_noise((self.steps, self.settings.samples, 2))
            candidates = np.clip(
                self.nominal[:, None, :] + noise, self.lower, self.upper
            )
            # The dynamics changed from direct positive speed to acceleration.
            # Explicit constant candidates retain basic forward/reverse/turn
            # options even when the nominal starts at zero acceleration.
            candidates[:, 0] = self.nominal
            controls = np.array(
                [
                    [a, k]
                    for a in (-self.limits.acceleration, 0.0, self.limits.acceleration)
                    for k in (
                        -self.limits.max_curvature,
                        0.0,
                        self.limits.max_curvature,
                    )
                ]
            )
            candidates[:, 1:10] = controls[None, :, :]
            noise[:, :10] = candidates[:, :10] - self.nominal[:, None, :]
            costs, valid, prefixes, paths = self._rollout(
                pose, np.swapaxes(candidates, 0, 1), target
            )
            if valid.any():
                self.nominal, weights = raw_noise_update(
                    self.nominal,
                    noise,
                    np.where(valid, costs, np.inf),
                    self.lower,
                    self.upper,
                    self.settings.temperature,
                )
            else:
                # No feasible full navigation prediction was found. Prefer
                # longest valid prefix, then the source navigation objective.
                longest = prefixes.max()
                selected = int(np.argmin(np.where(prefixes == longest, costs, np.inf)))
                self.nominal = candidates[:, selected].copy()
                weights = np.zeros(self.settings.samples)
                weights[selected] = 1
                fallback = "longest_valid_prefix"
        final_cost, final_valid, _, final_path = self._rollout(
            pose, self.nominal[None], target
        )
        if not final_valid[0] and valid.any():
            selected = int(np.argmin(np.where(valid, costs, np.inf)))
            self.nominal = candidates[:, selected].copy()
            final_cost, final_valid, _, final_path = self._rollout(
                pose, self.nominal[None], target
            )
            fallback = "best_valid_sample"
        command = self.nominal[0].copy()
        self.last = {
            "controller": "cfree_uniform_mppi_numpy_adaptation",
            "sampler": self.settings.sampler,
            "normal_noise_scale": self.sigma.tolist(),
            "log_noise_sigma": (
                self.settings.log_noise_sigma
                if self.settings.sampler == "log_mppi"
                else None
            ),
            "command_order": ["acceleration", "curvature"],
            "target": target.tolist(),
            "samples": self.settings.samples,
            "horizon": self.settings.horizon,
            "navigation_cost": float(final_cost[0]),
            "valid_samples": int(valid.sum()),
            "selected_valid": bool(final_valid[0]),
            "fallback": fallback,
            "effective_samples": float(1.0 / np.sum(weights**2)),
            "goal_tolerance": self.settings.goal_tolerance,
            "planning_clearance_rule": "safe distance certificate; exact ambiguous-interval check",
            "exact_wall_checks": self._exact_wall_checks,
            "clear_interval_certificates": self._clear_interval_certificates,
            "skipped_terminal_intervals": self._skipped_terminal_intervals,
            "wall_seconds": time.perf_counter() - begin,
        }
        if diagnostics:
            self.last.update(
                selected_controls=self.nominal.copy(),
                selected_rollout=final_path[0].copy(),
                sampled_controls=np.swapaxes(candidates, 0, 1).copy(),
                sampled_rollouts=paths.copy(),
                sampled_costs=costs.copy(),
                sampled_valid=valid.copy(),
                weights=weights.copy(),
            )
        return command
