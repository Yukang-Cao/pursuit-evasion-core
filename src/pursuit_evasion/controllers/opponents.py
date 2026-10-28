"""Declared development opponents. None is called an optimal evader."""

import numpy as np
from ..model import vehicle_step


class ReactiveEvader:
    """Choose a constant-turn response using a one-second geometric lookahead.

    The P prediction holds its current candidate command during this lookahead.
    Thus this benchmark E observes that command; keep this convention identical
    across P controllers. Both execute only one control interval.
    """

    def __init__(self, config, horizon=1.0, actions=17):
        self.config = config
        self.horizon = horizon
        self.actions = np.linspace(
            -config.evader.max_turn_rate, config.evader.max_turn_rate, actions
        )
        self.last = {}

    def command(self, state, p_turn):
        c = self.config
        s = np.asarray(state)
        count = len(self.actions)
        e = np.broadcast_to(s[3:], (count, 3)).copy()
        p = s[:3].copy()
        valid = np.ones(count, bool)
        min_separation = np.full(count, np.inf)
        min_clearance = np.full(count, np.inf)
        for _ in range(max(1, round(self.horizon / c.control_dt))):
            p = vehicle_step(p, p_turn, c.control_dt, c.pursuer.speed)
            e = vehicle_step(e, self.actions, c.control_dt, c.evader.speed)
            gap = np.linalg.norm(e[:, :2] - p[:2], axis=1)
            clear = c.arena_half_width - np.max(abs(e[:, :2]), axis=1)
            valid &= clear > c.evader.speed * c.control_dt / 2
            min_separation = np.minimum(min_separation, gap)
            min_clearance = np.minimum(min_clearance, clear)
        final = np.linalg.norm(e[:, :2] - p[:2], axis=1)
        utility = (
            2 * np.minimum(min_separation, c.capture_radius + 0.3)
            + final
            + 0.05 * min_clearance
        )
        if valid.any():
            utility = np.where(valid, utility, -np.inf)
        else:
            utility = min_clearance
        i = int(np.argmax(utility))
        self.last = dict(
            policy="reactive_constant_turn_bank",
            command=float(self.actions[i]),
            valid_candidates=int(valid.sum()),
            observes_current_P_command=True,
        )
        return float(self.actions[i])
