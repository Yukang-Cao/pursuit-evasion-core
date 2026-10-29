from dataclasses import replace
import numpy as np
import pytest

from pursuit_evasion.reversible.analytic_bounds import copying_evader_exclusion
from pursuit_evasion.reversible.model import (
    ReversibleConfig,
    joint_step,
    relative_state,
)

STATE = np.array(
    [0.389652987115758, -0.19589087194815868, -1.133830961195553, 2.0, 1.0]
)


def test_actual_counterexample_bound_and_tight_copying_trajectory():
    c = ReversibleConfig()
    b = copying_evader_exclusion(STATE, c)
    assert b.no_capture_through == pytest.approx(np.pi / 4)
    assert b.signed_payoff_lower_bound == pytest.approx(0.0133979095, abs=1e-10)
    q = np.asarray(b.transformed_center)
    closest_arc = np.arctan2(q[0], q[1] + b.turn_radius)
    t = closest_arc / 2
    world = np.array([0, 0, 0, 2, *STATE[:2], STATE[2], 1])
    z = joint_step(world, [0, -1], [0, -2], t, c)
    separation = np.linalg.norm(z[:2] - z[4:6])
    assert separation == pytest.approx(b.minimum_separation_lower_bound, abs=1e-12)


def test_bound_also_excludes_an_exact_grid_node():
    b = copying_evader_exclusion([0.3, -0.2, -3 * np.pi / 8, 2, 1], ReversibleConfig())
    assert b.signed_payoff_lower_bound == pytest.approx(0.0014654932, abs=1e-10)
    assert b.no_capture_through > 0.4


def test_current_control_copy_preserves_invariants_and_exclusion():
    c = ReversibleConfig()
    b = copying_evader_exclusion(STATE, c)
    rng = np.random.default_rng(2901)
    for _ in range(20):
        z = np.array([0, 0, 0, 2, *STATE[:2], STATE[2], 1])
        t = 0.0
        while t < b.no_capture_through - 1e-12:
            dt = min(0.05, b.no_capture_through - t)
            u = rng.uniform([-2, -1], [2, 1])
            v = np.array([u[0] / 2, 2 * u[1]])
            z = joint_step(z, u, v, dt, c)
            t += dt
            r = relative_state(z)
            assert r[4] == pytest.approx(r[3] / 2, abs=2e-15)
            assert r[2] == pytest.approx(STATE[2], abs=3e-15)
            assert np.linalg.norm(r[:2]) >= b.minimum_separation_lower_bound - 1e-12


def test_mismatched_model_and_nonmatching_speed_state_are_not_certified():
    c = ReversibleConfig()
    assert copying_evader_exclusion([*STATE[:3], 1.2, 0.9], c) is None
    changed = replace(c, evader=replace(c.evader, acceleration=0.8))
    assert copying_evader_exclusion(STATE, changed) is None
    assert copying_evader_exclusion([0.1, 0, 0, 2, 1], c) is None


def test_reflection_selects_left_circle_and_preserves_bound():
    c = ReversibleConfig()
    original = copying_evader_exclusion(STATE, c)
    reflected = STATE.copy()
    reflected[[1, 2]] *= -1
    bound = copying_evader_exclusion(reflected, c)
    assert bound.turn_circle_center == pytest.approx((0.0, 1.0))
    assert bound.no_capture_through == pytest.approx(original.no_capture_through)
    assert bound.signed_payoff_lower_bound == pytest.approx(
        original.signed_payoff_lower_bound, abs=1e-14
    )
    np.testing.assert_allclose(
        bound.transformed_center,
        np.asarray(original.transformed_center) * [1, -1],
        atol=1e-14,
    )
    q = np.asarray(bound.transformed_center)
    t = np.arctan2(q[0], bound.turn_radius - q[1]) / c.pursuer.speed_max
    world = np.array([0, 0, 0, 2, *reflected[:2], reflected[2], 1])
    z = joint_step(world, [0, 1], [0, 2], t, c)
    assert np.linalg.norm(z[:2] - z[4:6]) == pytest.approx(
        bound.minimum_separation_lower_bound, abs=1e-12
    )


@pytest.mark.parametrize("speed", [0.4, 1.0])
def test_slower_speed_limits_deadline_by_braking_time(speed):
    c = ReversibleConfig()
    state = STATE.copy()
    state[3:] = [speed, speed / 2]
    bound = copying_evader_exclusion(state, c)
    assert bound.no_capture_through == pytest.approx(speed / c.pursuer.acceleration)
    assert bound.no_capture_through < np.pi / 4
    assert bound.signed_payoff_lower_bound == pytest.approx(
        copying_evader_exclusion(STATE, c).signed_payoff_lower_bound
    )
    # Maximum braking reaches zero exactly at the reported endpoint. The copy
    # remains proportional throughout; the exclusion does not extend past the
    # time at which P could begin reversing.
    world = np.array([0, 0, 0, speed, *state[:2], state[2], speed / 2])
    for t in np.linspace(0, bound.no_capture_through, 11):
        z = joint_step(world, [-2, -1], [-1, -2], t, c)
        assert z[3] >= -1e-14
        assert z[7] == pytest.approx(z[3] / 2, abs=1e-14)
        assert np.linalg.norm(z[:2] - z[4:6]) >= (
            bound.minimum_separation_lower_bound - 1e-12
        )
    assert z[3] == pytest.approx(0.0, abs=1e-14)


def test_nearly_equal_speeds_use_stable_transform_scale():
    base = ReversibleConfig()
    gamma = 1.0 - 1e-10
    scale = 1.0 - gamma
    e = replace(
        base.evader,
        speed_min=gamma * base.pursuer.speed_min,
        speed_max=gamma * base.pursuer.speed_max,
        acceleration=gamma * base.pursuer.acceleration,
        max_curvature=base.pursuer.max_curvature / gamma,
    )
    config = replace(base, evader=e, capture_radius=0.1 * scale)
    # Here A=(1-gamma)I. The virtual target disk is ordinary-sized even though
    # the physical distances are tiny. The expanded rho² expression rounds to
    # zero, so this is a regression for cancellation, not a robotics benchmark.
    r = np.array([0.2 * scale, -0.5 * scale, 0.0, 2.0, 2.0 * gamma])
    bound = copying_evader_exclusion(r, config)
    assert bound is not None
    assert bound.transformed_center == pytest.approx((0.2, -0.5))
    assert bound.transformed_radius == pytest.approx(0.1)
    assert bound.signed_payoff_lower_bound > 0
