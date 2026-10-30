"""Declared development evaders for reversible cars; none is map-optimal.

Both policies observe P's current command, and both execute only one control
interval. The one-second prediction holds that P command and each sampled E
command fixed; this is a benchmark policy assumption, not an execution hold.
"""

import time
import numpy as np

from ..game import _first_zero
from .game import WorldMap, predict_pairs
from .model import joint_step, relative_state, vehicle_step


def _finite_list(values):
    return [float(x) if np.isfinite(x) else None for x in np.asarray(values)]


class ReactiveEvader:
    """Nine constant acceleration/curvature responses with map checks.

    ``separation`` retains the former benchmark's geometric objective,
    2*min(minimum separation,0.6)+final separation, with terminal events ranked
    before geometric utility. ``dp`` maximizes one-step numerical capture time.
    An unresolved time uses the terminal signed field as an explicit heuristic;
    out-of-domain fields trigger the separation fallback. Neither is an exact
    worst-case evader for the constrained continuous game.
    """

    def __init__(self, config, world=None, table=None, mode="separation", horizon=1.0):
        if mode not in ("separation", "dp"):
            raise ValueError("Mode must be separation or dp.")
        if not np.isfinite(horizon) or horizon < config.control_dt:
            raise ValueError("Prediction horizon must cover at least one control step.")
        if mode == "dp" and table is None:
            raise ValueError("DP response requires a relative value table.")
        self.config = config
        self.world = WorldMap(config.arena_half_width) if world is None else world
        self.table = table
        self.mode = mode
        self.horizon = float(horizon)
        if table is not None and hasattr(table, "assert_compatible"):
            table.assert_compatible(config)
        a, k = config.evader.acceleration, config.evader.max_curvature
        # Neutral first makes exact ties deterministic without a steering cost.
        self.actions = np.array(
            [
                [0, 0],
                [-a, 0],
                [a, 0],
                [0, -k],
                [0, k],
                [-a, -k],
                [-a, k],
                [a, -k],
                [a, k],
            ],
            dtype=float,
        )
        self.last = {}

    def reset(self):
        self.last = {}

    def _wall_times(self, state):
        """First wall times for E's own motions, independent of imagined P capture.

        A vectorized endpoint-distance bound certifies ordinary clear intervals.
        Potential contacts use the same adaptive first-zero calculation as the
        game simulator. A later imagined P fault cannot hide an unsafe E plan.
        """
        count = max(1, int(np.ceil(self.horizon / (self.config.control_dt / 4))))
        times = np.linspace(0, self.horizon, count + 1)
        paths = vehicle_step(
            state[4:], self.actions[:, None, :], times[None, :], self.config.evader
        )
        clear = self.world.clearance(paths[..., :2])
        speed = np.maximum(abs(paths[:, :-1, 3]), abs(paths[:, 1:, 3]))
        margin = speed * np.diff(times)[None, :] / 2
        clear_intervals = np.minimum(clear[:, :-1], clear[:, 1:]) > margin
        certified = (clear[:, 0] > 0) & np.all(clear_intervals, axis=1)
        # A straight line in the empty square stays inside when its extreme
        # positions do. Under constant acceleration those occur at the horizon
        # endpoints and at a possible zero-speed reversal. This exact check
        # avoids expensive generic refinement for legal wall-parallel motion.
        if self.world.half_width is not None and not self.world.obstacles:
            for i in np.flatnonzero((self.actions[:, 1] == 0) & ~certified):
                a = self.actions[i, 0]
                reversal = -state[7] / a if a else 0.0
                critical = np.array(
                    [0.0, self.horizon, np.clip(reversal, 0, self.horizon)]
                )
                points = vehicle_step(
                    state[4:], self.actions[i], critical, self.config.evader
                )
                certified[i] = np.all(self.world.clearance(points[:, :2]) > 0)
        hits = np.full(len(self.actions), np.inf)
        for i in np.flatnonzero(~certified):
            result = _first_zero(
                lambda t: float(
                    self.world.clearance(
                        vehicle_step(state[4:], self.actions[i], t, self.config.evader)[
                            :2
                        ]
                    )
                ),
                self.config.evader.max_abs_speed,
                self.horizon,
                time_tolerance=1e-9,
            )
            if result is not None:
                hits[i] = result
        return hits

    def _preview(self, state, p_control):
        """First-event-aware geometric utility for the held-command prediction."""
        end, elapsed, codes = predict_pairs(
            state,
            np.asarray(p_control)[None],
            self.actions,
            self.config,
            duration=self.horizon,
            world=self.world,
        )
        end, elapsed, codes = end[0], elapsed[0], codes[0]
        initial_gap = np.linalg.norm(state[:2] - state[4:6])
        minimum = np.full(len(self.actions), initial_gap)
        steps = max(1, int(np.ceil(self.horizon / self.config.control_dt)))
        for t in np.linspace(0, self.horizon, steps + 1)[1:]:
            predicted = joint_step(
                state, p_control, self.actions, float(t), self.config
            )
            gap = np.linalg.norm(predicted[:, :2] - predicted[:, 4:6], axis=1)
            minimum = np.where(t <= elapsed + 1e-10, np.minimum(minimum, gap), minimum)
        final = np.linalg.norm(end[:, :2] - end[:, 4:6], axis=1)
        minimum = np.minimum(minimum, final)
        utility = 2 * np.minimum(minimum, self.config.capture_radius + 0.3) + final
        return utility, codes, elapsed

    def _fallback(self, state, wall_times, eligible):
        longest = np.max(wall_times[eligible])
        pool = eligible & np.isclose(wall_times, longest, atol=1e-8, rtol=0)
        braking = -np.sign(state[7]) * self.config.evader.acceleration
        # Prefer braking among equally long feasible prefixes. No safety claim
        # is made when every constant response has a predicted wall contact.
        merit = -abs(self.actions[:, 0] - braking)
        return int(np.argmax(np.where(pool, merit, -np.inf)))

    def _terminal_choice(self, pool, codes, elapsed, utility):
        wins = pool & (codes == 2)
        if wins.any():
            return int(np.argmax(np.where(wins, utility, -np.inf))), "predicted_P_fault"
        continuing = pool & (codes == 0)
        if continuing.any():
            return None, continuing
        # All available moves lose under this prediction. Later loss is better
        # for E; post-capture separation must not reverse that ordering.
        return int(np.argmax(np.where(pool, elapsed, -np.inf))), "latest_predicted_loss"

    def command(self, state, p_control):
        begin = time.perf_counter()
        state, p_control = np.asarray(state, float), np.asarray(p_control, float)
        if state.shape != (8,) or p_control.shape != (2,):
            raise ValueError("Expected state[8] and current P command[2].")
        # Shared propagation validates finiteness, speed and control bounds.
        joint_step(state, p_control, self.actions, 0, self.config)
        # In free space, a speed bound can certify that every preview remains
        # beyond the capped minimum-distance term. Its contribution is then
        # identical for all E candidates, so endpoint ranking is exactly the
        # original policy, without repeated intermediate geometry checks.
        cap = self.config.capture_radius + 0.3
        if (
            self.mode == "separation"
            and self.world.half_width is None
            and not self.world.obstacles
            and np.linalg.norm(state[:2] - state[4:6])
            > cap
            + (self.config.pursuer.max_abs_speed + self.config.evader.max_abs_speed)
            * self.horizon
            + 1e-12
        ):
            endpoints = joint_step(
                state, p_control, self.actions, self.horizon, self.config
            )
            utility = 2 * cap + np.linalg.norm(
                endpoints[:, :2] - endpoints[:, 4:6], axis=1
            )
            selected = int(np.argmax(utility))
            command = self.actions[selected].copy()
            self.last = dict(
                policy="reactive_reversible_separation",
                command=command.tolist(),
                selected_index=selected,
                selection_basis="separation",
                fallback=False,
                valid_candidates=len(self.actions),
                observes_current_P_command=True,
                prediction_horizon=self.horizon,
                execution_interval=self.config.control_dt,
                P_prediction="current P command held only inside the heuristic prediction",
                candidate_controls=self.actions.tolist(),
                candidate_wall_times=[None] * len(self.actions),
                candidate_utilities=utility.tolist(),
                predicted_event_codes=[0] * len(self.actions),
                predicted_event_times=[self.horizon] * len(self.actions),
                first_interval_event_codes=[0] * len(self.actions),
                first_interval_event_times=[self.config.control_dt] * len(self.actions),
                exact_free_space_shortcut=True,
                wall_seconds=time.perf_counter() - begin,
            )
            return command
        wall_times = self._wall_times(state)
        valid = ~np.isfinite(wall_times)
        utility, preview_codes, preview_times = self._preview(state, p_control)
        successors, elapsed, codes = predict_pairs(
            state,
            p_control[None],
            self.actions,
            self.config,
            duration=self.config.control_dt,
            world=self.world,
        )
        successors, elapsed, codes = successors[0], elapsed[0], codes[0]
        immediate_wins = codes == 2
        immediate_continuing = codes == 0
        values = fields = None
        if immediate_wins.any():
            selected = int(np.argmax(np.where(immediate_wins, utility, -np.inf)))
            basis = "first_interval_P_fault"
            fallback = False
        elif not immediate_continuing.any():
            selected = int(np.argmax(elapsed))
            basis = "all_sampled_first_interval_losses"
            fallback = True
        elif not (valid & immediate_continuing).any():
            selected = self._fallback(state, wall_times, immediate_continuing)
            basis = "longest_wall_safe_prefix_then_braking"
            fallback = True
        elif self.mode == "separation":
            pool = valid & immediate_continuing
            selected, choice = self._terminal_choice(
                pool, preview_codes, preview_times, utility
            )
            if selected is None:
                selected = int(np.argmax(np.where(choice, utility, -np.inf)))
                basis = "separation"
            else:
                basis = choice
            fallback = False
        else:
            # First-interval terminal events have already been resolved above.
            # The longer wall screen is only a policy heuristic for continuing
            # motions, not a claim that E must hold its command for a second.
            pool = valid & immediate_continuing
            fallback = False
            r = relative_state(successors)
            values = np.asarray(self.table.value(r), float)
            fields = np.asarray(self.table.field(r, self.table.horizon), float)
            if np.any(pool & ~np.isfinite(fields)):
                selected = int(np.argmax(np.where(pool, utility, -np.inf)))
                basis = "outside_table_separation"
                fallback = True
            else:
                unknown = pool & ~np.isfinite(values)
                if unknown.any():
                    selected = int(np.argmax(np.where(unknown, fields, -np.inf)))
                    basis = "unresolved_signed_field"
                    fallback = True
                else:
                    total = elapsed + values
                    selected = int(np.argmax(np.where(pool, total, -np.inf)))
                    basis = "one_step_DP_time"
        command = self.actions[selected].copy()
        self.last = dict(
            policy="reactive_reversible_" + self.mode,
            command=command.tolist(),
            selected_index=selected,
            selection_basis=basis,
            fallback=fallback,
            valid_candidates=int(valid.sum()),
            observes_current_P_command=True,
            prediction_horizon=self.horizon,
            execution_interval=self.config.control_dt,
            P_prediction="current P command held only inside the heuristic prediction",
            candidate_controls=self.actions.tolist(),
            candidate_wall_times=_finite_list(wall_times),
            candidate_utilities=utility.tolist(),
            predicted_event_codes=preview_codes.tolist(),
            predicted_event_times=preview_times.tolist(),
            first_interval_event_codes=codes.tolist(),
            first_interval_event_times=elapsed.tolist(),
            wall_seconds=time.perf_counter() - begin,
        )
        if values is not None:
            self.last.update(
                DP_times=_finite_list(values), terminal_fields=_finite_list(fields)
            )
        return command
