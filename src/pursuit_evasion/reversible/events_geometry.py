"""Held-command event geometry for acceleration-limited reversible point cars.

Wall contacts are found by enumerating exact line/circle intersections and
inverting signed traveled distance, including speed saturation and reversal.
There is no straight-line approximation for nonzero curvature. Floating-point
root tolerances remain explicit; callers retain their declared event priority.
"""

import math
import numpy as np

from .model import integrated_speed, joint_step, vehicle_step

_TAU = 2 * math.pi
_EPS = np.finfo(float).eps


def _quadratic_roots(a, b, c):
    """Real roots, with a stable small root when coefficients differ in scale."""
    if a == 0:
        return [] if b == 0 else [-c / b]
    discriminant = b * b - 4 * a * c
    roundoff = 32 * _EPS * (b * b + abs(4 * a * c))
    if discriminant < -roundoff:
        return []
    discriminant = max(0.0, discriminant)
    root = math.sqrt(discriminant)
    q = -0.5 * (b + math.copysign(root, b))
    if q == 0:
        return [-b / (2 * a)]
    return [q / a, c / q]


def _distance_phases(pose, control, duration, limits):
    speed, acceleration = float(pose[3]), float(control[0])
    if acceleration:
        bound = limits.speed_max if acceleration > 0 else limits.speed_min
        active = min(duration, max(0.0, (bound - speed) / acceleration))
    else:
        active = duration
    final_speed, final_distance = integrated_speed(
        speed, acceleration, duration, limits
    )
    active_distance = speed * active + 0.5 * acceleration * active**2
    distances = [0.0, float(final_distance)]
    if acceleration:
        zero_time = -speed / acceleration
        if 0 < zero_time < active:
            distances.append(speed * zero_time + 0.5 * acceleration * zero_time**2)
    return active, active_distance, float(final_speed), min(distances), max(distances)


def _times_at_distance(distance, speed, acceleration, duration, phases, tolerance):
    active, active_distance, final_speed, _, _ = phases
    candidates = []
    for t in _quadratic_roots(0.5 * acceleration, speed, -distance):
        if -tolerance <= t <= active + tolerance:
            candidates.append(min(active, max(0.0, t)))
    if active < duration and final_speed:
        t = active + (distance - active_distance) / final_speed
        if active - tolerance <= t <= duration + tolerance:
            candidates.append(min(duration, max(active, t)))
    return candidates


def _plane_distances(pose, curvature, axis, plane, low, high, tolerance):
    """Signed distances where the constant-curvature path meets an axis plane.

    tan(turn/2) gives a quadratic without subtracting almost equal headings.
    This remains well-conditioned for tiny nonzero curvature. The turn=pi root
    at infinite tangent is included when the quadratic's leading term vanishes.
    """
    heading = float(pose[2])
    c, s = math.cos(heading), math.sin(heading)
    offset = float(plane - pose[axis])
    if curvature == 0:
        direction = c if axis == 0 else s
        if direction == 0:
            return []
        distance = offset / direction
        return [distance] if low - tolerance <= distance <= high + tolerance else []
    z = curvature * offset
    if axis == 0:
        a, b = z + 2 * s, -2 * c
    else:
        a, b = z - 2 * c, -2 * s
    phases = [2 * math.atan(t) for t in _quadratic_roots(a, b, z)]
    if a == 0:
        phases.append(math.pi)
    angle_low, angle_high = sorted((curvature * low, curvature * high))
    angle_tolerance = abs(curvature) * tolerance + 32 * _EPS * (
        1 + max(abs(angle_low), abs(angle_high))
    )
    distances = []
    for phase in phases:
        first = math.ceil((angle_low - phase - angle_tolerance) / _TAU)
        last = math.floor((angle_high - phase + angle_tolerance) / _TAU)
        for winding in range(first, last + 1):
            distance = (phase + winding * _TAU) / curvature
            if low - tolerance <= distance <= high + tolerance:
                distances.append(distance)
    return distances


def _position_at_distance(pose, curvature, distance):
    half_turn = curvature * distance / 2
    sinc = math.sin(half_turn) / half_turn if half_turn else 1.0
    chord = distance * sinc
    heading = float(pose[2]) + half_turn
    return np.array(
        [
            float(pose[0]) + chord * math.cos(heading),
            float(pose[1]) + chord * math.sin(heading),
        ]
    )


def first_wall_time(pose, control, duration, limits, world, time_tolerance=1e-10):
    """First point-car contact with the square or axis-aligned rectangles.

    Returns None if no contact occurs during the held command. A tangent touch
    counts as contact. Speed can cross zero or hit a bound during the interval.
    This function finds only wall time; it does not choose capture/fault priority.
    """
    pose, control = np.asarray(pose, float), np.asarray(control, float)
    duration = float(duration)
    if pose.shape != (4,) or control.shape != (2,):
        raise ValueError("Expected one pose[4] and control[2].")
    if not np.isfinite(time_tolerance) or time_tolerance <= 0:
        raise ValueError("Time tolerance must be finite and positive.")
    vehicle_step(pose, control, duration, limits)  # shared validation
    if float(world.clearance(pose[:2])) <= 0:
        return 0.0
    if duration == 0 or (world.half_width is None and not world.obstacles):
        return None
    phases = _distance_phases(pose, control, duration, limits)
    low, high = phases[-2:]
    if low == high:
        return None
    distance_tolerance = limits.max_abs_speed * time_tolerance + 32 * _EPS * (
        1 + max(abs(low), abs(high))
    )
    planes = []
    if world.half_width is not None:
        for axis in (0, 1):
            planes.extend(
                (axis, plane) for plane in (-world.half_width, world.half_width)
            )
    for xmin, xmax, ymin, ymax in world.obstacles:
        planes.extend(((0, xmin), (0, xmax), (1, ymin), (1, ymax)))
    curvature = float(control[1])
    contact_times = []
    for axis, plane in planes:
        for distance in _plane_distances(
            pose, curvature, axis, plane, low, high, distance_tolerance
        ):
            point = _position_at_distance(pose, curvature, distance)
            coordinate_tolerance = (
                128
                * _EPS
                * (1 + abs(plane) + np.max(abs(point)) + max(abs(low), abs(high)))
            )
            # Infinite planes also pass outside finite obstacle edges. The
            # actual geometry decides whether the intersection is a contact.
            if abs(point[axis] - plane) > coordinate_tolerance:
                continue
            if float(world.clearance(point)) > coordinate_tolerance:
                continue
            contact_times.extend(
                _times_at_distance(
                    distance,
                    float(pose[3]),
                    float(control[0]),
                    duration,
                    phases,
                    time_tolerance,
                )
            )
    return min(contact_times) if contact_times else None


def _effective_acceleration(speed, command, limits):
    if (speed >= limits.speed_max and command > 0) or (
        speed <= limits.speed_min and command < 0
    ):
        return 0.0
    return command


def _identical_motion(state, p_control, e_control, endpoint, config):
    ap = _effective_acceleration(state[3], p_control[0], config.pursuer)
    ae = _effective_acceleration(state[7], e_control[0], config.evader)
    if state[3] == state[7] == 0 and ap == ae == 0:
        return True
    heading_difference = (float(state[6]) - float(state[2])) % _TAU
    if heading_difference == 0:
        return (
            state[3] == state[7]
            and ap == ae
            and p_control[1] == e_control[1]
            and endpoint[3] == endpoint[7]
        )
    if heading_difference == math.pi:
        return (
            state[3] == -state[7]
            and ap == -ae
            and p_control[1] == -e_control[1]
            and endpoint[3] == -endpoint[7]
        )
    return False


def first_capture_time(
    state, p_control, e_control, duration, config, time_tolerance=1e-10
):
    """Adaptive capture time with an interval-specific relative-velocity bound.

    This uses the same positive-endpoint pruning idea as the existing detector,
    but avoids its loose sum-of-maximum-speeds bound for nearly parallel motion.
    The wall geometry and event-priority rule remain the caller's responsibility.
    """
    state = np.asarray(state, float)
    p_control, e_control = np.asarray(p_control, float), np.asarray(e_control, float)
    duration = float(duration)
    if state.shape != (8,) or p_control.shape != (2,) or e_control.shape != (2,):
        raise ValueError("Expected one state[8] and two controls[2].")
    if not np.isfinite(time_tolerance) or time_tolerance <= 0:
        raise ValueError("Time tolerance must be finite and positive.")
    endpoint = joint_step(state, p_control, e_control, duration, config)

    def gap(z):
        return float(np.linalg.norm(z[:2] - z[4:6]) - config.capture_radius)

    initial_gap = gap(state)
    if initial_gap <= 0:
        return 0.0
    if duration == 0 or _identical_motion(
        state, p_control, e_control, endpoint, config
    ):
        return None
    global_bound = config.pursuer.max_abs_speed + config.evader.max_abs_speed
    stack = [(0.0, duration, state, endpoint, initial_gap, gap(endpoint))]
    while stack:
        left, right, before, after, f_left, f_right = stack.pop()
        if f_left <= 0:
            return left
        width = right - left
        if f_left + f_right > global_bound * width:
            continue
        midpoint = (left + right) / 2
        middle = joint_step(state, p_control, e_control, midpoint, config)
        f_middle = gap(middle)
        p_velocity = middle[3] * np.array([math.cos(middle[2]), math.sin(middle[2])])
        e_velocity = middle[7] * np.array([math.cos(middle[6]), math.sin(middle[6])])
        relative_speed = float(np.linalg.norm(p_velocity - e_velocity))
        max_vp = max(abs(before[3]), abs(after[3]))
        max_ve = max(abs(before[7]), abs(after[7]))
        acceleration_bound = (
            abs(_effective_acceleration(before[3], p_control[0], config.pursuer))
            + max_vp**2 * abs(p_control[1])
            + abs(_effective_acceleration(before[7], e_control[0], config.evader))
            + max_ve**2 * abs(e_control[1])
        )
        local_bound = min(global_bound, relative_speed + acceleration_bound * width / 2)
        if f_left + f_right > local_bound * width:
            continue
        if width <= time_tolerance:
            if min(f_left, f_middle, f_right) <= local_bound * time_tolerance:
                return midpoint
            continue
        stack.append((midpoint, right, middle, after, f_middle, f_right))
        stack.append((left, midpoint, before, middle, f_left, f_middle))
    return None
