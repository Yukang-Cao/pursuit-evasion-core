"""Baseline 1: lion-guided target tracking, with a local numerical DP finish."""

import time
import numpy as np

from .dp_finish import DPFinish
from .lion import LionTarget
from .mppi import MPPITracker, MPPISettings


class LionMPPIDP:
    def __init__(
        self,
        config,
        table,
        *,
        seed=0,
        mppi_settings=MPPISettings(),
        lion_step=1.0,
        target_period=0.25,
        use_dp=True,
    ):
        self.config = config
        self.table = table
        self.use_dp = use_dp
        if use_dp and table is None:
            raise ValueError("DP finishing requires a local table.")
        self.dp = DPFinish(table, config) if use_dp else None
        self.lion = LionTarget(lion_step, target_period)
        self.tracker = MPPITracker(config, mppi_settings, seed)
        self.last = {}

    def reset(self, state):
        self.lion.reset(state)
        self.tracker.reset()
        self.deadline = None
        self.cooldown_until = 0.0
        self.handoffs = 0
        self.failed_finishes = 0

    def command(self, state, elapsed, diagnostics=True):
        begin = time.perf_counter()
        rejected = None
        u = None
        if self.use_dp and elapsed >= self.cooldown_until:
            remaining = None if self.deadline is None else self.deadline - elapsed
            u = self.dp.command(state, remaining)
            if u is not None:
                if self.deadline is None:
                    self.deadline = elapsed + self.table.times[-1]
                    self.handoffs += 1
                self.last = dict(
                    mode="dp",
                    control_owner="dp",
                    dp=dict(self.dp.last),
                    deadline=self.deadline,
                    handoffs=self.handoffs,
                    failed_finishes=self.failed_finishes,
                    milliseconds=1000 * (time.perf_counter() - begin),
                )
                return u
            if self.deadline is not None:
                rejected = "dp_unresolved_or_deadline_exceeded"
                self.failed_finishes += 1
                self.deadline = None
                self.cooldown_until = elapsed + 0.5
        goal = self.lion.target_at(state, elapsed)
        u = self.tracker.command(np.asarray(state)[:3], goal, diagnostics)
        self.last = dict(
            mode="approach",
            control_owner="teacher",
            lion=dict(self.lion.last),
            mppi=dict(self.tracker.last),
            rejected_finish=rejected,
            handoffs=self.handoffs,
            failed_finishes=self.failed_finishes,
            milliseconds=1000 * (time.perf_counter() - begin),
        )
        return u
