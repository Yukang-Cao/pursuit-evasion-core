import numpy as np
import pytest
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.game import WorldMap, advance, predict_pairs


def test_reversal_can_capture_between_identical_end_positions():
    c = ReversibleConfig(capture_radius=0.05)
    state = [0, 0, 0, 1, 0.25, 0, 0, 0]
    result = advance(state, [-2, 0], [0, 0], c, 1, WorldMap(None))
    assert result.event == "capture"
    assert result.elapsed == pytest.approx((1 - np.sqrt(0.2)) / 2, abs=2e-8)


def test_wall_crossing_during_reversal_is_detected():
    c = ReversibleConfig()
    state = [2.9, 0, 0, 1, 0, 2, 0, 0]
    result = advance(state, [-2, 0], [0, 0], c, 1)
    assert result.event == "pursuer_boundary_fault"
    assert result.elapsed == pytest.approx((1 - np.sqrt(0.6)) / 2, abs=2e-8)


def test_rectangles_have_signed_lipschitz_clearance():
    world = WorldMap(None, ((0, 1, 0, 1),))
    np.testing.assert_allclose(
        world.clearance([[0.5, 0.5], [2, 0.5], [2, 2]]), [-0.5, 1, np.sqrt(2)]
    )


def test_pair_events_match_scalar_with_negative_speeds():
    c = ReversibleConfig()
    P = np.array([[-2, -1], [0, 0], [2, 1]])
    E = np.array([[-1, -2], [0, 0], [1, 2]])
    states = [
        [0, 0, 0, -0.5, -0.32, 0, 0, 0],
        [2.99, 0, 0, 1, 0, 1, 0, 0.3],
        [0, 0, 0, 0, -2.99, 0, 0, -0.5],
        [0, 0, 0, 0.3, 0.301, 0, np.pi, 0.3],
    ]
    names = {
        None: 0,
        "capture": 1,
        "pursuer_boundary_fault": 2,
        "evader_boundary_fault": 3,
    }
    for state in states:
        successors, times, codes = predict_pairs(state, P, E, c)
        for i in range(len(P)):
            for j in range(len(E)):
                scalar = advance(state, P[i], E[j], c)
                assert names[scalar.event] == codes[i, j]
                assert times[i, j] == pytest.approx(scalar.elapsed, abs=3e-8)
                np.testing.assert_allclose(successors[i, j], scalar.state, atol=7e-8)


def test_fault_is_not_capture_and_tie_priority_is_explicit():
    c = ReversibleConfig()
    e = advance([0, 0, 0, 0, 3, 1, 0, 0], [0, 0], [0, 0], c)
    assert e.pursuer_success and not e.physical_capture
    both = advance([3, 0, 0, 0, 3, 0, 0, 0], [0, 0], [0, 0], c)
    assert both.event == "pursuer_boundary_fault"
