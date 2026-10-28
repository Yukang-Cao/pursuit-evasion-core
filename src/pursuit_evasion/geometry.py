"""Small geometric checks with explicit scope; no learned or numerical value."""

import numpy as np


def turning_circle_clearances(poses, speed, max_turn_rate, arena_half_width):
    """Wall clearance of the complete full-left and full-right circles.

    A nonnegative clearance gives one indefinitely repeatable, legal maneuver
    for an exact fixed-speed point-car model in this square. It is sufficient,
    not necessary, for remaining inside the arena forever.
    """
    p = np.asarray(poses)
    radius = speed / max_turn_rate
    offset = radius * np.stack((-np.sin(p[..., 2]), np.cos(p[..., 2])), axis=-1)
    left = p[..., :2] + offset
    right = p[..., :2] - offset
    return (
        arena_half_width - radius - np.max(abs(left), axis=-1),
        arena_half_width - radius - np.max(abs(right), axis=-1),
    )
