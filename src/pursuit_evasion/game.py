"""Continuous event stopping for held car commands.

Capture and boundary faults are separate outcomes. The first terminal event
wins. Numerically simultaneous events prioritize P's fault, then capture, then
E's fault. This convention must match the solver target/avoid construction.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .model import GameConfig, joint_step


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    elapsed: float
    event: str | None

    @property
    def terminal(self) -> bool:
        return self.event is not None

    @property
    def physical_capture(self) -> bool:
        return self.event == "capture"

    @property
    def pursuer_success(self) -> bool:
        return self.event in ("capture", "evader_boundary_fault")


def clearance(pose, config: GameConfig):
    return config.arena_half_width - np.max(np.abs(np.asarray(pose)[..., :2]), axis=-1)


def _first_zero(
    f: Callable[[float], float],
    lipschitz: float,
    end: float,
    time_tolerance: float = 1e-8,
) -> float | None:
    """Find first nonpositive value using a speed bound to prune safe intervals.

    An interval with positive endpoints cannot cross zero if
    f(a)+f(b) > L*(b-a). This avoids relying only on sampled endpoints.
    The final root is numerical, with the stated tolerance; not an exact proof.
    """
    fa = float(f(0.0))
    if fa <= 0:
        return 0.0
    fb = float(f(end))
    stack = [(0.0, float(end), fa, fb)]
    while stack:
        a, b, va, vb = stack.pop()
        if va <= 0:
            return a
        if va + vb > lipschitz * (b - a):
            continue
        mid = (a + b) / 2
        vm = float(f(mid))
        if b - a <= time_tolerance:
            if min(va, vm, vb) <= lipschitz * time_tolerance:
                return mid
            continue
        # Stack is LIFO: inspect the earlier half first.
        stack.append((mid, b, vm, vb))
        stack.append((a, mid, va, vm))
    return None


def advance(
    state,
    p_turn: float,
    e_turn: float,
    config: GameConfig,
    duration: float | None = None,
) -> Transition:
    """Execute one held command pair, stopping at the first capture or wall fault."""
    s = np.asarray(state, dtype=float)
    if s.shape != (6,) or not np.isfinite(s).all():
        raise ValueError("Expected a finite six-dimensional joint state.")
    dt = config.control_dt if duration is None else float(duration)
    if dt < 0 or not np.isfinite(dt):
        raise ValueError("Duration must be finite and nonnegative.")
    if (
        abs(p_turn) > config.pursuer.max_turn_rate + 1e-12
        or abs(e_turn) > config.evader.max_turn_rate + 1e-12
    ):
        raise ValueError("Turn command exceeds the declared vehicle bound.")

    def pose(t):
        return joint_step(s, p_turn, e_turn, t, config)

    functions = [
        (
            "pursuer_boundary_fault",
            lambda t: clearance(pose(t)[:3], config),
            config.pursuer.speed,
        ),
        (
            "capture",
            lambda t: np.linalg.norm(pose(t)[:2] - pose(t)[3:5])
            - config.capture_radius,
            config.pursuer.speed + config.evader.speed,
        ),
        (
            "evader_boundary_fault",
            lambda t: clearance(pose(t)[3:], config),
            config.evader.speed,
        ),
    ]
    events = []
    for priority, (name, function, bound) in enumerate(functions):
        root = _first_zero(function, bound, dt)
        if root is not None:
            events.append((root, priority, name))
    if not events:
        return Transition(pose(dt), dt, None)
    earliest = min(t for t, _, _ in events)
    tied = [event for event in events if event[0] <= earliest + 2e-8]
    elapsed, _, event = min(tied, key=lambda x: x[1])
    return Transition(pose(elapsed), elapsed, event)
