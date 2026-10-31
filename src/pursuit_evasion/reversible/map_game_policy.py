"""Inspectable one-step local game policies for the footprint-map pilot.

This is a diagnostic local selector, not the global B1 approach controller.
All dynamics/events use footprint_map.advance; no point-collision fallback.
"""

import numpy as np
from .footprint_map import advance
from .controllers import action_catalogue


def evader_actions(config):
    """Nine acceleration/curvature controls, with neutral-first tie ordering."""
    a, k = config.evader.acceleration, config.evader.max_curvature
    return np.array(
        [[0, 0], [-a, 0], [a, 0], [0, -k], [0, k], [-a, -k], [-a, k], [a, -k], [a, k]],
        float,
    )


def successor_matrix(state, config, geometry):
    p, e = action_catalogue(config.pursuer), evader_actions(config)
    records = [[advance(state, u, v, config, geometry) for v in e] for u in p]
    states = np.array([[r.state for r in row] for row in records])
    codes = np.array([[r.event or "continuing" for r in row] for row in records])
    return p, e, records, states, codes


def margin_costs(values, codes):
    """Lexicographic terminal priority; finite sentinels are not capture times.

    The tested maps have bounded fields far inside ±1000. Unknown continuation
    is pessimistic for P. This policy is not intended for unbounded field scales.
    """
    values = np.where(np.isfinite(values), values, np.inf)
    return np.where(
        codes == "pursuer_collision",
        1000.0,
        np.where((codes == "capture") | (codes == "evader_collision"), -1000.0, values),
    )


def select_margin(costs, controls):
    worst = costs.max(axis=1)
    minimum = worst.min()
    if not np.isfinite(minimum):
        raise ValueError("No completely evaluated action row")
    tied = np.flatnonzero(np.isclose(worst, minimum, rtol=0, atol=1e-6))
    return int(min(tied, key=lambda i: (abs(controls[i, 1]), abs(controls[i, 0]), i)))


def select_time_or_margin(controls, records, states, codes, value_source):
    """Minimize worst success-time estimate; use signed field if none is finite."""
    pair = np.array(
        [[r.elapsed for r in row] for row in records]
    ) + value_source.time_value(states)
    elapsed = np.array([[r.elapsed for r in row] for row in records])
    pair = np.where((codes == "capture") | (codes == "evader_collision"), elapsed, pair)
    pair = np.where((codes == "pursuer_collision") | ~np.isfinite(pair), np.inf, pair)
    eligible = np.isfinite(pair).all(axis=1)
    if eligible.any():
        worst = np.where(eligible, pair.max(axis=1), np.inf)
        tied = np.flatnonzero(np.isclose(worst, worst.min(), atol=1e-7, rtol=0))
        index = int(
            min(tied, key=lambda i: (abs(controls[i, 1]), abs(controls[i, 0]), i))
        )
        return index, "time"
    return (
        select_margin(margin_costs(value_source.field(states), codes), controls),
        "margin_fallback",
    )
