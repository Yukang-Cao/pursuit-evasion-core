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


def predict_pairs(state, p_turns, e_turns, config: GameConfig, duration=None):
    """Batch successors and first events for a pair of command catalogues.

    Codes: 0 none, 1 capture, 2 P fault, 3 E fault. Positive-endpoint intervals
    that could contain a crossing use the scalar adaptive detector. Ordinary
    crossings use vector bisection, avoiding a Python root search per pair.
    """
    dt = config.control_dt if duration is None else float(duration)
    p_turns = np.asarray(p_turns, dtype=float)
    e_turns = np.asarray(e_turns, dtype=float)
    if p_turns.ndim != 1 or e_turns.ndim != 1:
        raise ValueError("Command catalogues must be vectors.")
    if (
        np.max(abs(p_turns)) > config.pursuer.max_turn_rate + 1e-12
        or np.max(abs(e_turns)) > config.evader.max_turn_rate + 1e-12
    ):
        raise ValueError("Command catalogue exceeds turn limits.")
    shape = (len(p_turns), len(e_turns))
    P, E = p_turns[:, None], e_turns[None, :]
    s = np.asarray(state, dtype=float)

    def poses(t):
        return joint_step(s, P, E, t, config)

    functions = [
        (lambda z: clearance(z[..., :3], config), config.pursuer.speed),
        (
            lambda z: np.linalg.norm(z[..., :2] - z[..., 3:5], axis=-1)
            - config.capture_radius,
            config.pursuer.speed + config.evader.speed,
        ),
        (lambda z: clearance(z[..., 3:], config), config.evader.speed),
    ]
    roots = []
    for measure, bound in functions:
        root = np.full(shape, np.inf)
        left = 0.0
        va = np.broadcast_to(measure(poses(0.0)), shape).copy()
        root[va <= 0] = 0.0
        for right in np.linspace(0, dt, 9)[1:]:
            vb = measure(poses(right))
            active = ~np.isfinite(root)
            crossing = active & (vb <= 0)
            if crossing.any():
                lo, hi = np.full(shape, left), np.full(shape, right)
                for _ in range(22):
                    mid = (lo + hi) / 2
                    vm = measure(poses(mid))
                    hi = np.where(crossing & (vm <= 0), mid, hi)
                    lo = np.where(crossing & (vm > 0), mid, lo)
                root[crossing] = (lo[crossing] + hi[crossing]) / 2
            ambiguous = active & ~crossing & (va + vb <= bound * (right - left))
            for i, j in np.argwhere(ambiguous):

                def f(t):
                    z = joint_step(s, p_turns[i], e_turns[j], left + t, config)
                    return float(measure(z))

                found = _first_zero(f, bound, float(right - left))
                if found is not None:
                    root[i, j] = left + found
            va, left = vb, float(right)
        roots.append(root)
    earliest = np.minimum(np.minimum(roots[0], roots[1]), roots[2])
    finite = np.isfinite(earliest)
    codes = np.where(
        roots[0] <= earliest + 2e-8, 2, np.where(roots[1] <= earliest + 2e-8, 1, 3)
    )
    return poses(dt), np.where(finite, earliest, dt), np.where(finite, codes, 0)
