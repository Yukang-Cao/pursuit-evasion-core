"""A model-specific analytic check for optimistic free-space capture claims.

This is a diagnostic, not a replacement controller or a certificate for a
numerical DP table. It assumes E can respond to the current P command, as in the
declared P-then-E game. No future P command is disclosed.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CopyingExclusion:
    no_capture_through: float
    signed_payoff_lower_bound: float
    minimum_separation_lower_bound: float
    speed_ratio: float
    transformed_center: tuple
    transformed_radius: float
    turn_circle_center: tuple
    turn_radius: float


def copying_evader_exclusion(relative, config):
    """Return a short-time exclusion bound, or None when its premises do not hold.

    The current states must satisfy vE = gamma*vP > 0. The vehicles must have
    exactly scaled speed/acceleration limits and E enough curvature to mirror
    P's heading rate. E chooses aE=gamma*aP and kE=kP/gamma, preserving the ratio
    and relative heading. This maps capture to P reaching a fixed disk.

    While P must remain forward and can travel at most a quarter-circle arc,
    its path stays outside both initial minimum-turning circles. A capture disk
    strictly contained inside either circle is therefore unreachable.

    The scalar bound is on the signed minimum-separation payoff (distance minus
    physical capture radius) for every deadline up to no_capture_through. It is
    evaluated in floating point; this routine does not model physical uncertainty.

    This is a free-space argument. In a map, the copying E motion may hit a wall
    and cease to be admissible. Do not transfer this bound to an obstacle game
    without separately checking that the copying response remains legal for
    every P motion being considered. No wall-fault outcome is treated as capture
    by this function.
    """
    r = np.asarray(relative, float)
    if r.shape != (5,) or not np.isfinite(r).all():
        raise ValueError("Expected a finite relative state [x,y,psi,vP,vE].")
    p, e = config.pursuer, config.evader
    gamma = e.speed_max / p.speed_max
    if not 0 < gamma < 1:
        return None
    # Exact checks deliberately avoid treating a nearly matching physical model
    # as identical without bounding the extra mismatch.
    if e.speed_min != gamma * p.speed_min or e.acceleration != gamma * p.acceleration:
        return None
    if e.max_curvature * gamma < p.max_curvature:
        return None
    if r[3] <= 0 or r[4] != gamma * r[3]:
        return None
    if r[3] > p.speed_max or r[4] > e.speed_max:
        raise ValueError("State speed exceeds the declared limits.")
    c, s = np.cos(r[2]), np.sin(r[2])
    A = np.eye(2) - gamma * np.array([[c, -s], [s, c]])
    # The expanded squared expression loses significant digits when gamma is
    # close to one and the headings nearly agree. A column norm of A is stable.
    rho = np.hypot(1 - gamma * c, gamma * s)
    q = np.linalg.solve(A, r[:2])
    radius = 1.0 / p.max_curvature
    target_radius = config.capture_radius / rho
    centers = [np.array([0.0, -radius]), np.array([0.0, radius])]
    gaps = [radius - np.linalg.norm(q - center) - target_radius for center in centers]
    which = int(np.argmax(gaps))
    # Retain a substantial floating-point separation from the inclusion boundary.
    slack = 1e-10 * max(1.0, radius, np.linalg.norm(q))
    if gaps[which] <= slack:
        return None
    deadline = min(r[3] / p.acceleration, np.pi * radius / (2 * p.speed_max))
    margin = rho * gaps[which]
    return CopyingExclusion(
        no_capture_through=float(deadline),
        signed_payoff_lower_bound=float(margin),
        minimum_separation_lower_bound=float(config.capture_radius + margin),
        speed_ratio=float(gamma),
        transformed_center=tuple(q),
        transformed_radius=float(target_radius),
        turn_circle_center=tuple(centers[which]),
        turn_radius=float(radius),
    )
