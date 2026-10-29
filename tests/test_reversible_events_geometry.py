import math
import numpy as np
import pytest

from pursuit_evasion.game import _first_zero
from pursuit_evasion.reversible.events_geometry import (
    first_wall_time,
    first_capture_time,
)
from pursuit_evasion.reversible.game import WorldMap
from pursuit_evasion.reversible.model import ReversibleConfig, joint_step, vehicle_step


def test_straight_acceleration_and_speed_saturation_wall_time():
    c = ReversibleConfig()
    time = first_wall_time([0, 0, 0, 1], [2, 0], 2, c.pursuer, WorldMap(3))
    assert time == pytest.approx(1.625, abs=1e-12)


def test_reversal_hits_wall_then_returns_inside_before_endpoint():
    c = ReversibleConfig()
    state, action = np.array([2.9, 0, 0, 1]), np.array([-2, 0])
    assert WorldMap(3).clearance(vehicle_step(state, action, 1.5, c.pursuer)[:2]) > 0
    time = first_wall_time(state, action, 1.5, c.pursuer, WorldMap(3))
    assert time == pytest.approx((1 - math.sqrt(0.6)) / 2, abs=1e-12)


def test_negative_speed_limit_and_signed_distance_inversion():
    c = ReversibleConfig()
    time = first_wall_time([0, 0, 0, -0.5], [-2, 0], 2, c.pursuer, WorldMap(1))
    assert time == pytest.approx(1.0625, abs=1e-12)


def test_curved_tangent_contact_between_clear_endpoints():
    c = ReversibleConfig()
    world = WorldMap(None, ((1, 2, 0, 2),))
    state, action = np.array([0, 0, 0, 1]), np.array([0, 1])
    assert world.clearance(state[:2]) > 0
    assert world.clearance(vehicle_step(state, action, 2 * np.pi, c.pursuer)[:2]) > 0
    time = first_wall_time(state, action, 2 * np.pi, c.pursuer, world)
    assert time == pytest.approx(np.pi / 2, abs=1e-11)


def test_plane_intersection_outside_finite_obstacle_is_not_contact():
    c = ReversibleConfig()
    world = WorldMap(None, ((1, 2, -1, 1),))
    assert first_wall_time([0, 3, 0, 1], [0, 0], 3, c.pursuer, world) is None
    # Moving along the extension of an obstacle edge reaches the finite edge.
    time = first_wall_time([1, -2, np.pi / 2, 1], [0, 0], 2, c.pursuer, world)
    assert time == pytest.approx(1, abs=1e-11)


def test_tiny_curvature_does_not_use_a_straight_line_approximation():
    c = ReversibleConfig()
    time = first_wall_time([0, 0, 0, 1], [2, 1e-12], 2, c.pursuer, WorldMap(3))
    assert time == pytest.approx(1.625, abs=1e-10)
    assert (
        first_wall_time(
            [2.999, 0, np.pi / 2, 0.5], [0, 1e-12], 1, c.evader, WorldMap(3)
        )
        is None
    )


def test_negative_curvature_and_reverse_arc():
    c = ReversibleConfig()
    time = first_wall_time([0, 0, 0, -1], [0, -1], 2, c.pursuer, WorldMap(1))
    assert time == pytest.approx(np.pi / 2, abs=1e-11)


def test_initial_contact_zero_motion_and_validation():
    c = ReversibleConfig()
    assert first_wall_time([3, 0, 0, 0], [0, 1], 1, c.pursuer, WorldMap(3)) == 0
    assert first_wall_time([0, 0, 0, 0], [0, 1], 1, c.pursuer, WorldMap(3)) is None
    with pytest.raises(ValueError):
        first_wall_time([0, 0, 0, 0], [3, 0], 1, c.pursuer, WorldMap(3))


def test_random_wall_times_match_existing_adaptive_detector():
    rng = np.random.default_rng(4421)
    c = ReversibleConfig()
    world = WorldMap(3, ((-0.4, 0.5, -0.8, 0.7),))
    for _ in range(120):
        state = np.array(
            [*rng.uniform(-2.9, 2.9, 2), rng.uniform(-np.pi, np.pi), rng.uniform(-1, 2)]
        )
        action = rng.uniform([-2, -1], [2, 1])
        duration = rng.uniform(0.05, 2)
        reference = _first_zero(
            lambda t: float(
                world.clearance(vehicle_step(state, action, t, c.pursuer)[:2])
            ),
            c.pursuer.max_abs_speed,
            duration,
            time_tolerance=1e-10,
        )
        actual = first_wall_time(state, action, duration, c.pursuer, world)
        assert (actual is None) == (reference is None), (
            state,
            action,
            duration,
            actual,
            reference,
        )
        if actual is not None:
            assert actual == pytest.approx(reference, abs=2e-8)


def test_capture_crossing_and_reversal_have_expected_first_times():
    c = ReversibleConfig()
    assert first_capture_time(
        [0, 0, 0, 2, 0.6, 0, 0, 0], [0, 0], [0, 0], 1, c
    ) == pytest.approx(0.15, abs=1e-8)
    assert first_capture_time(
        [0, 0, 0, 1, -0.8, 0, 0, 0], [-2, 0], [0, 0], 2, c
    ) == pytest.approx(1.5, abs=1e-8)


def test_identical_curved_or_opposite_gear_motion_stays_separated():
    c = ReversibleConfig()
    assert (
        first_capture_time([0, 0, 0, 0.5, 0, 0.3001, 0, 0.5], [0, 0.5], [0, 0.5], 2, c)
        is None
    )
    assert (
        first_capture_time(
            [0, 0, 0, -0.5, 0, 0.3001, np.pi, 0.5], [0, 0.5], [0, -0.5], 2, c
        )
        is None
    )


def test_nearly_identical_motion_is_not_treated_as_identical():
    c = ReversibleConfig()
    time = first_capture_time([0, 0, 0, 1, 0.3001, 0, 0, 0.999], [0, 0], [0, 0], 1, c)
    assert time == pytest.approx(0.1, abs=1e-8)


def test_saturated_acceleration_can_still_describe_identical_motion():
    c = ReversibleConfig()
    assert (
        first_capture_time([0, 0, 0, 1, 0.30001, 0, 0, 1], [0, 0], [1, 0], 1, c) is None
    )


def test_random_capture_times_match_existing_detector():
    rng = np.random.default_rng(4214)
    c = ReversibleConfig()
    for _ in range(60):
        state = np.array(
            [
                *rng.uniform(-1, 1, 2),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-1, 2),
                *rng.uniform(-1, 1, 2),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-0.5, 1),
            ]
        )
        p = rng.uniform([-2, -1], [2, 1])
        e = rng.uniform([-1, -2], [1, 2])
        duration = rng.uniform(0.05, 0.5)

        def measure(t):
            z = joint_step(state, p, e, t, c)
            return float(np.linalg.norm(z[:2] - z[4:6]) - c.capture_radius)

        reference = _first_zero(measure, 3, duration, time_tolerance=1e-10)
        actual = first_capture_time(state, p, e, duration, c)
        assert (actual is None) == (reference is None)
        if actual is not None:
            assert actual == pytest.approx(reference, abs=2e-8)
