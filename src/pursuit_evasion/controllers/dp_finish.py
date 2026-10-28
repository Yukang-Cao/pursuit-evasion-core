"""One-step minimax extraction from a numerical partial DP table."""

import time
import numpy as np

from ..model import joint_step
from ..game import advance, clearance


class DPFinish:
    def __init__(self, table, config=None, p_actions=5, e_actions=9):
        self.table = table
        self.config = table.config if config is None else config
        # A PDE table does not depend on controller sample period; all physical
        # parameters must still match. Changing the hold time requires replay tests.
        a = self.config.to_dict()
        b = table.config.to_dict()
        for key in ("name", "control_dt"):
            a.pop(key, None)
            b.pop(key, None)
        if a != b:
            raise ValueError("Controller and table physical models differ.")
        self.P = np.linspace(
            -self.config.pursuer.max_turn_rate,
            self.config.pursuer.max_turn_rate,
            p_actions,
        )
        self.E = np.linspace(
            -self.config.evader.max_turn_rate,
            self.config.evader.max_turn_rate,
            e_actions,
        )
        self.last = {}

    def command(self, state, remaining=None):
        begin = time.perf_counter()
        c = self.config
        dt = c.control_dt
        z = joint_step(
            np.asarray(state)[None, None, :], self.P[:, None], self.E[None, :], dt, c
        )
        values = self.table.value(z)
        cost = dt + values
        near_capture = (
            np.linalg.norm(np.asarray(state)[:2] - np.asarray(state)[3:5])
            <= c.capture_radius + (c.pursuer.speed + c.evader.speed) * dt + 1e-8
        )
        near_wall = (
            clearance(np.asarray(state)[:3], c) <= c.pursuer.speed * dt
            or clearance(np.asarray(state)[3:], c) <= c.evader.speed * dt
        )
        kinds = np.full(cost.shape, "continuation", object)
        if near_capture or near_wall:
            for i, u in enumerate(self.P):
                for j, v in enumerate(self.E):
                    event = advance(state, float(u), float(v), c)
                    if event.terminal:
                        cost[i, j] = event.elapsed if event.pursuer_success else np.inf
                        kinds[i, j] = event.event
        cost = np.where(np.isfinite(cost), cost, np.inf)
        worst = cost.max(axis=1)
        limit = self.table.times[-1] if remaining is None else float(remaining)
        admissible = worst <= limit + 1e-7
        i = int(np.argmin(np.where(admissible, worst, np.inf)))
        self.last = dict(
            available=bool(admissible.any()),
            remaining_budget=limit,
            resolved_rows=int(admissible.sum()),
            action_count_P=len(self.P),
            action_count_E=len(self.E),
            worst_costs=[float(x) if np.isfinite(x) else None for x in worst],
            milliseconds=1000 * (time.perf_counter() - begin),
        )
        if not admissible.any():
            return None
        j = int(np.argmax(cost[i]))
        self.last.update(
            command=float(self.P[i]),
            estimated_worst_time=float(worst[i]),
            worst_tested_E=float(self.E[j]),
            worst_kind=str(kinds[i, j]),
        )
        return float(self.P[i])
