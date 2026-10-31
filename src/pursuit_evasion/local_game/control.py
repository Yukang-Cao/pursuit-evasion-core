"""Matched P-min/E-max decisions for the same finite-body capture objective."""

import numpy as np
from .native import GeometryEngine, capture_times
from ..reversible.map_game_field import first_crossing


def admissible_controls(pose, engine, dt):
    b = engine.limits
    acceleration = [-b.acceleration, 0.0, b.acceleration]
    stop = -float(pose[3]) / dt
    if abs(stop) <= b.acceleration and min(abs(stop - x) for x in acceleration) > 1e-9:
        acceleration.append(stop)
    controls = np.array(
        [[a, k] for a in acceleration for k in [-b.max_curvature, 0.0, b.max_curvature]]
    )
    good, states = engine.actions(pose, controls, dt)
    good = good[0]
    if not good.any():
        raise RuntimeError("No admissible held command despite the braking invariant")
    return controls[good], states[0, good]


def pair_predictions(state, config, arena, p_engine=None, e_engine=None):
    p_engine = p_engine or GeometryEngine(
        arena, config.pursuer, "pursuer", threads=1, control_dt=config.control_dt
    )
    e_engine = e_engine or GeometryEngine(
        arena, config.evader, "evader", threads=1, control_dt=config.control_dt
    )
    p, pn = admissible_controls(state[:4], p_engine, config.control_dt)
    e, en = admissible_controls(state[4:], e_engine, config.control_dt)
    states = np.concatenate(
        [
            np.broadcast_to(pn[:, None], (len(p), len(e), 4)),
            np.broadcast_to(en[None], (len(p), len(e), 4)),
        ],
        -1,
    )
    contact = capture_times(state, p, e, config.control_dt, p_engine, e_engine)
    return p, e, states, contact


def costs(source, states, contact, dt, exit_radius=3.0):
    values, valid, shape = source.fields(states)
    tau = first_crossing(source.times, values, valid).reshape(shape)
    time = dt + tau
    margin = values[-1].reshape(shape)
    outside = np.linalg.norm(states[..., :2] - states[..., 4:6], axis=-1) > exit_radius
    time = np.where(outside | ~np.isfinite(time), np.inf, time)
    margin = np.where(outside | ~np.isfinite(margin), exit_radius, margin)
    captured = np.isfinite(contact)
    time = np.where(captured, contact, time)
    margin = np.where(captured, -exit_radius, margin)
    return time, margin, valid.reshape(shape)


def choose_e(times, margins, controls):
    maximum = times.max()
    candidates = np.flatnonzero(
        np.isinf(times)
        if np.isinf(maximum)
        else np.isclose(times, maximum, atol=1e-7, rtol=0)
    )
    if len(candidates) > 1 and np.isinf(maximum):
        worst = margins[candidates].max()
        candidates = candidates[
            np.isclose(margins[candidates], worst, atol=1e-7, rtol=0)
        ]
    return int(
        min(candidates, key=lambda i: (abs(controls[i, 1]), abs(controls[i, 0]), i))
    )


def choose_p(times, margins, p_controls, e_controls):
    replies = np.array([choose_e(t, m, e_controls) for t, m in zip(times, margins)])
    row = np.arange(len(replies))
    worst_t = times[row, replies]
    worst_m = margins[row, replies]
    minimum = worst_t.min()
    candidates = np.flatnonzero(
        np.isinf(worst_t)
        if np.isinf(minimum)
        else np.isclose(worst_t, minimum, atol=1e-7, rtol=0)
    )
    if len(candidates) > 1 and np.isinf(minimum):
        best = worst_m[candidates].min()
        candidates = candidates[
            np.isclose(worst_m[candidates], best, atol=1e-7, rtol=0)
        ]
    index = int(
        min(candidates, key=lambda i: (abs(p_controls[i, 1]), abs(p_controls[i, 0]), i))
    )
    return index, replies, worst_t, worst_m
