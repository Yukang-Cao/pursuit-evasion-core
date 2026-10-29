"""B1 and direct game-value controls with one shared wall-feasibility check.

The wall check asks whether the first command has a feasible braking continuation.
It does not claim that the free-space capture value is correct in the map.
"""

from itertools import product
import time
import numpy as np

from ..controllers.lion import LionTarget
from ..game import _first_zero
from .model import vehicle_step, relative_state
from .game import WorldMap, predict_pairs
from .events_geometry import first_wall_time
from .mppi import ReversibleMPPI, MPPISettings


def action_catalogue(limits):
    return np.array(
        list(
            product(
                [-limits.acceleration, 0.0, limits.acceleration],
                [-limits.max_curvature, 0.0, limits.max_curvature],
            )
        )
    )


def _interval_safe(pose, control, dt, limits, world):
    return first_wall_time(pose, control, dt, limits, world) is None


def braking_safe_mask(pose, commands, config, world):
    """A sufficient P feasibility check with three explicit braking continuations.

    Execute the requested first command for dt, then brake toward zero speed with
    straight/maximum-left/maximum-right curvature. A stopped state is a permitted
    kinematic continuation here. This is not a capture guarantee or exact viability
    kernel. The same check is applied to all compared P controllers.
    """
    pose = np.asarray(pose, float)
    commands = np.asarray(commands, float)
    if world.half_width is None and not world.obstacles:
        return np.ones(len(commands), bool)
    limits = config.pursuer
    valid = np.zeros(len(commands), bool)
    for i, u in enumerate(commands):
        if not _interval_safe(pose, u, config.control_dt, limits, world):
            continue
        start = vehicle_step(pose, u, config.control_dt, limits)
        if abs(start[3]) < 1e-12:
            valid[i] = True
            continue
        duration = abs(start[3]) / limits.acceleration
        acceleration = -np.sign(start[3]) * limits.acceleration
        for curvature in (0.0, -limits.max_curvature, limits.max_curvature):
            if _interval_safe(
                start, [acceleration, curvature], duration, limits, world
            ):
                valid[i] = True
                break
    return valid


def table_states(table, world_states):
    if getattr(table, "state_dimension", 5) == 8:
        return np.asarray(world_states)
    return relative_state(world_states)


def dp_decision(
    state, table, config, world, p_actions=None, e_actions=None, guard=True
):
    """P-first finite-hold minimax; unresolved time rows use signed-field guidance.

    E boundary faults terminate as E losses but are not physical captures.
    A finite row is an estimated time under the numerical table, not a certificate.
    """
    P = action_catalogue(config.pursuer) if p_actions is None else np.asarray(p_actions)
    E = action_catalogue(config.evader) if e_actions is None else np.asarray(e_actions)
    successors, elapsed, events = predict_pairs(state, P, E, config, world=world)
    query = table_states(table, successors)
    values = np.asarray(table.value(query))
    costs = elapsed + values
    costs = np.where((events == 1) | (events == 3), elapsed, costs)
    costs = np.where(events == 2, np.inf, costs)
    costs = np.where(np.isnan(costs), np.inf, costs)
    continuation_safe = (
        braking_safe_mask(np.asarray(state)[:4], P, config, world)
        if guard
        else np.ones(len(P), bool)
    )
    terminal_success = np.all((events == 1) | (events == 3), axis=1)
    safe = continuation_safe | terminal_success
    criterion = costs
    field_available = True
    worst = costs.max(axis=1)
    resolved = np.isfinite(worst) & safe
    if resolved.any():
        ranking = np.where(resolved, worst, np.inf)
        kind = "time"
        known = True
    else:
        # Signed payoff provides a declared fallback ranking, never a fabricated
        # capture-time value for an unresolved state.
        horizon = float(table.times[-1])
        field = np.asarray(table.field(query, max(0.0, horizon - config.control_dt)))
        field = np.where((events == 1) | (events == 3), -1e6, field)
        field = np.where(events == 2, 1e6, field)
        field = np.where(np.isfinite(field), field, np.inf)
        field_available = bool(np.any(safe & np.isfinite(field).all(axis=1)))
        criterion = field
        ranking = np.where(safe, field.max(axis=1), np.inf)
        kind = "signed_field"
        known = False
    if not safe.any():
        # Explicit failure to find a feasible continuation. Do not pretend a
        # finite field or a clipped command makes the state safe.
        idx = int(np.argmin(costs.max(axis=1)))
    else:
        minimum = np.min(ranking)
        tied = np.flatnonzero(np.isclose(ranking, minimum, rtol=0, atol=1e-7))
        # Deterministic tie convention, not an extra cost term: retain straight
        # steering first, then smaller acceleration magnitude.
        idx = int(min(tied, key=lambda j: (abs(P[j, 1]), abs(P[j, 0]), j)))
    return P[idx].copy(), dict(
        kind=kind,
        resolved=known,
        estimated_cost=float(worst[idx]),
        safe_commands=int(safe.sum()),
        selected_index=idx,
        worst_e_index=int(np.argmax(criterion[idx])),
        guidance_available=field_available,
        candidate_costs=worst.tolist(),
        guard_accepts=bool(safe[idx]),
        terminal_success=bool(terminal_success[idx]),
        braking_feasible=bool(continuation_safe[idx]),
    )


class Baseline1:
    """Lion-guided MPPI approach; known numerical DP row enables the finish.

    Modes: 'hybrid', 'approach' (DP disabled), 'dp' (use any available field
    guidance; outside-domain states use the same approach fallback).
    There is no strict countdown or claim of a certified handoff.
    """

    def __init__(
        self,
        config,
        world=None,
        table=None,
        mode="hybrid",
        seed=0,
        mppi_settings=None,
        lookahead=1.0,
    ):
        if mode not in ("hybrid", "approach", "dp"):
            raise ValueError("Unknown controller mode.")
        self.config = config
        self.world = WorldMap(config.arena_half_width) if world is None else world
        self.table = table
        self.mode = mode
        if table is not None:
            if hasattr(table, "assert_compatible"):
                table.assert_compatible(config)
            if getattr(table, "state_dimension", 5) == 8:
                tc = table.config
                if (
                    tc.pursuer != config.pursuer
                    or tc.evader != config.evader
                    or tc.capture_radius != config.capture_radius
                    or self.world.half_width != tc.arena_half_width
                    or self.world.obstacles
                ):
                    raise ValueError(
                        "Fixed-map table does not match the physical game/map."
                    )
        self.tracker = ReversibleMPPI(
            config,
            mppi_settings or MPPISettings(),
            seed,
            clearance=self.world.clearance,
        )
        self.lion = LionTarget(step_length=lookahead, update_period=config.control_dt)
        self.last = {}
        self.ready = False

    def reset(self, state):
        self.tracker.reset()
        self.lion.reset(np.asarray(state)[[0, 1, 2, 4, 5, 6]])
        self.ready = True
        self.last = {}

    def command(self, state, t=0.0, diagnostics=False):
        start = time.perf_counter()
        s = np.asarray(state, float)
        if not self.ready:
            self.reset(s)
        decision = None
        owner = "lion_mppi"
        value = np.nan
        if self.mode != "approach" and self.table is not None:
            q = table_states(self.table, s)
            value = float(self.table.value(q))
            # Finite signed queries distinguish an in-domain unresolved state
            # from a state outside this table's represented spatial domain.
            inside = np.isfinite(
                float(self.table.field(q, float(self.table.times[-1])))
            )
            if inside and (np.isfinite(value) or self.mode == "dp"):
                u, decision = dp_decision(s, self.table, self.config, self.world)
                if (
                    decision["guard_accepts"]
                    and decision["guidance_available"]
                    and (decision["resolved"] or self.mode == "dp")
                ):
                    owner = "dp" if decision["resolved"] else "dp_field_guide"
                    self.last = dict(
                        control_owner=owner,
                        dp_value=value,
                        dp=decision,
                        seconds=time.perf_counter() - start,
                    )
                    return u
        goal = self.lion.target_at(s[[0, 1, 2, 4, 5, 6]], t)
        u = self.tracker.command(s[:4], goal, diagnostics=diagnostics)
        guard = braking_safe_mask(s[:4], u[None], self.config, self.world)
        shielded = False
        guard_failure = False
        if not guard[0]:
            P = action_catalogue(self.config.pursuer)
            safe = braking_safe_mask(s[:4], P, self.config, self.world)
            if safe.any():
                scale = np.array(
                    [
                        self.config.pursuer.acceleration,
                        self.config.pursuer.max_curvature,
                    ]
                )
                index = int(
                    np.argmin(
                        np.where(safe, np.sum(((P - u) / scale) ** 2, axis=1), np.inf)
                    )
                )
                u = P[index].copy()
                shielded = True
            else:
                guard_failure = True
        self.last = dict(
            control_owner=owner,
            dp_value=value,
            dp=decision,
            lion=dict(self.lion.last),
            mppi=dict(self.tracker.last),
            shielded=shielded,
            guard_failure=guard_failure,
            seconds=time.perf_counter() - start,
        )
        return u
