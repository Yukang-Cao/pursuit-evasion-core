import numpy as np
import pytest
from pursuit_evasion.reversible.model import (
    ReversibleConfig,
    VehicleLimits,
    integrated_speed,
    joint_step,
    relative_state,
    relative_rhs,
    vehicle_step,
    world_rhs,
)


def test_speed_saturation_and_reversal_integrate_distance_not_final_speed():
    p = ReversibleConfig().pursuer
    v, d = integrated_speed(1.5, 2.0, 1.0, p)
    assert v == 2
    assert d == pytest.approx(1.9375)
    state = vehicle_step([0, 0, 0, 1.0], [-2, 0], 1.0, p)
    np.testing.assert_allclose(state, [0, 0, 0, -1], atol=1e-14)
    halfway = vehicle_step([0, 0, 0, 1.0], [-2, 0], 0.5, p)
    np.testing.assert_allclose(halfway, [0.25, 0, 0, 0], atol=1e-14)


def test_held_input_semigroup_including_speed_limit_and_reverse():
    c = ReversibleConfig()
    rng = np.random.default_rng(11)
    for _ in range(70):
        state = np.array(
            [*rng.normal(size=2), rng.uniform(-np.pi, np.pi), rng.uniform(-1, 2)]
        )
        u = np.array([rng.uniform(-2, 2), rng.uniform(-1, 1)])
        whole = vehicle_step(state, u, 1.3, c.pursuer)
        pieces = state
        for _ in range(26):
            pieces = vehicle_step(pieces, u, 0.05, c.pursuer)
        np.testing.assert_allclose(whole, pieces, atol=1e-12)


def test_relative_derivative_matches_world_transform_at_signed_speeds():
    c = ReversibleConfig()
    rng = np.random.default_rng(9)
    for _ in range(60):
        s = np.array(
            [
                *rng.normal(size=2),
                rng.uniform(-1, 1),
                rng.uniform(-0.8, 1.8),
                *rng.normal(size=2),
                rng.uniform(-1, 1),
                rng.uniform(-0.4, 0.9),
            ]
        )
        p = np.array([rng.uniform(-2, 2), rng.uniform(-1, 1)])
        e = np.array([rng.uniform(-1, 1), rng.uniform(-2, 2)])
        f = world_rhs(s, p, e, c)
        h = 1e-6
        numerical = (relative_state(s + h * f) - relative_state(s - h * f)) / (2 * h)
        np.testing.assert_allclose(
            numerical, relative_rhs(relative_state(s), p, e, c), atol=2e-8
        )


def test_zero_speed_cannot_rotate_and_reverse_changes_turn_direction():
    p = ReversibleConfig().pursuer
    np.testing.assert_allclose(vehicle_step([0, 0, 0, 0], [0, 1], 0.4, p), [0, 0, 0, 0])
    forward = vehicle_step([0, 0, 0, 1], [0, 1], 0.2, p)
    reverse = vehicle_step([0, 0, 0, -1], [0, 1], 0.2, p)
    assert forward[2] > 0 and reverse[2] < 0


def test_batch_pair_broadcast_and_serial_agree():
    c = ReversibleConfig()
    s = [0, 0, 0, 0.2, 1, 1, 0.5, -0.1]
    P = np.array([[-2, -1], [0, 0], [2, 1]])
    E = np.array([[-1, -2], [1, 2]])
    result = joint_step(s, P[:, None, :], E[None, :, :], 0.05, c)
    assert result.shape == (3, 2, 8)
    for i in range(3):
        for j in range(2):
            np.testing.assert_allclose(result[i, j], joint_step(s, P[i], E[j], 0.05, c))


def test_configuration_roundtrip_and_invalid_commands():
    c = ReversibleConfig()
    assert ReversibleConfig.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError):
        vehicle_step([0, 0, 0, 2.1], [0, 0], 0.05, c.pursuer)
    with pytest.raises(ValueError):
        vehicle_step([0, 0, 0, 1], [0, 2], 0.05, c.pursuer)
    with pytest.raises(ValueError):
        VehicleLimits(0, 2, 2, 1)
