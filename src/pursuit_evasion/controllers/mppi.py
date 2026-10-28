"""A small NumPy MPPI target tracker for fixed-speed cars.

The tracker samples P controls only. It is not a second nested P/E minimax
planner. Its cost is target tracking; game outcomes are evaluated by game.py.
"""

from dataclasses import asdict, dataclass
import time
import numpy as np

from ..model import GameConfig, vehicle_step


@dataclass(frozen=True)
class MPPISettings:
    samples: int = 256
    horizon: float = 1.5
    iterations: int = 1
    noise_std: float = 0.7
    temperature: float = 0.6
    running_distance_weight: float = 2.0
    terminal_distance_weight: float = 5.0
    control_weight: float = 0.03
    safety_margin: float = 0.02


class MPPITracker:
    def __init__(self, config: GameConfig, settings=MPPISettings(), seed=0):
        self.config = config
        self.settings = settings
        self.seed = int(seed)
        if settings.samples < 4 or settings.temperature <= 0:
            raise ValueError("Invalid MPPI sampling settings.")
        self.steps = max(1, round(settings.horizon / config.control_dt))
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.nominal = np.zeros(self.steps)
        self.last = {}

    def _rollout(self, pose, commands, target):
        c = self.config
        s = self.settings
        count = len(commands)
        states = np.broadcast_to(np.asarray(pose), (count, 3)).copy()
        costs = np.zeros(count)
        initial_clearance = c.arena_half_width - np.max(abs(np.asarray(pose)[:2]))
        valid = np.full(
            count,
            initial_clearance >= s.safety_margin + c.pursuer.speed * c.control_dt / 2,
            bool,
        )
        safe_steps = np.zeros(count, int)
        paths = [states.copy()]
        limit = c.arena_half_width - s.safety_margin
        for k in range(self.steps):
            states = vehicle_step(states, commands[:, k], c.control_dt, c.pursuer.speed)
            # A speed-radius margin protects the whole interval, not just its endpoint.
            inside = (
                np.max(abs(states[:, :2]), axis=1)
                <= limit - c.pursuer.speed * c.control_dt / 2
            )
            valid &= inside
            safe_steps += valid
            distance2 = np.sum((states[:, :2] - target) ** 2, axis=1)
            costs += c.control_dt * (
                s.running_distance_weight * distance2
                + s.control_weight * commands[:, k] ** 2
            )
            paths.append(states.copy())
        costs += s.terminal_distance_weight * np.sum(
            (states[:, :2] - target) ** 2, axis=1
        )
        return costs, valid, safe_steps, np.stack(paths, axis=1)

    def command(self, pose, target, diagnostics=True):
        begin = time.perf_counter()
        c = self.config
        s = self.settings
        self.nominal = np.concatenate((self.nominal[1:], self.nominal[-1:]))
        fallback = False
        for _ in range(s.iterations):
            noise = self.rng.normal(0, s.noise_std, size=(s.samples, self.steps))
            # Nominal, straight, full-left and full-right witnesses are retained.
            candidates = np.clip(
                self.nominal[None, :] + noise,
                -c.pursuer.max_turn_rate,
                c.pursuer.max_turn_rate,
            )
            candidates[0] = self.nominal
            candidates[1] = 0
            candidates[2] = c.pursuer.max_turn_rate
            candidates[3] = -c.pursuer.max_turn_rate
            costs, valid, safe_steps, paths = self._rollout(
                pose, candidates, np.asarray(target)
            )
            if not valid.any():
                # A recorded fallback, not a claim that a safe full rollout exists.
                ranking = costs - 1e4 * safe_steps
                selected = int(np.argmin(ranking))
                self.nominal = candidates[selected].copy()
                weights = np.zeros(s.samples)
                weights[selected] = 1.0
                fallback = True
            else:
                minimum = float(costs[valid].min())
                weights = np.where(
                    valid,
                    np.exp(np.clip(-(costs - minimum) / s.temperature, -700, 0)),
                    0.0,
                )
                weights /= weights.sum()
                effective_noise = candidates - self.nominal[None, :]
                self.nominal = np.clip(
                    self.nominal + (weights[:, None] * effective_noise).sum(axis=0),
                    -c.pursuer.max_turn_rate,
                    c.pursuer.max_turn_rate,
                )
        nominal_cost, nominal_valid, _, nominal_path = self._rollout(
            pose, self.nominal[None, :], np.asarray(target)
        )
        # Averaging controls can cross an obstacle even when every contributing
        # trajectory is valid. Recheck the result and use a disclosed witness if needed.
        if not nominal_valid[0] and valid.any():
            selected = int(np.argmin(np.where(valid, costs, np.inf)))
            self.nominal = candidates[selected].copy()
            nominal_path = paths[selected : selected + 1]
            nominal_cost = costs[selected : selected + 1]
            fallback = True
        command = float(self.nominal[0])
        self.last = dict(
            target=np.asarray(target).tolist(),
            command=command,
            samples=s.samples,
            steps=self.steps,
            valid_samples=int(valid.sum()),
            sampling_fallback=fallback,
            nominal_cost=float(nominal_cost[0]),
            milliseconds=1000 * (time.perf_counter() - begin),
            settings=asdict(s),
        )
        if diagnostics:
            self.last["nominal_rollout"] = nominal_path[0].tolist()
            best = np.argsort(np.where(valid, costs, np.inf))[:6]
            self.last["sampled_rollouts"] = paths[best].tolist()
        return command
