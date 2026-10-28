import numpy as np
import pytest

from pursuit_evasion.model import GameConfig, Vehicle, vehicle_step, joint_step, vector_field, hamiltonian
from pursuit_evasion.game import advance


def test_exact_straight_and_quarter_turn():
    np.testing.assert_allclose(vehicle_step([0, 0, 0], 0, 2, 1), [2, 0, 0], atol=1e-14)
    np.testing.assert_allclose(vehicle_step([0, 0, 0], 1, np.pi/2, 1), [1, 1, np.pi/2], atol=1e-14)


def test_composition_and_ode_derivative():
    c = GameConfig();s = np.array([.3, -.4, .7, 1.2, .2, -1.3])
    direct = joint_step(s, .8, -1.4, .1, c)
    split = joint_step(joint_step(s, .8, -1.4, .04, c), .8, -1.4, .06, c)
    np.testing.assert_allclose(direct, split, atol=1e-13)
    derivative = (joint_step(s, .8, -1.4, 1e-7, c) - s)/1e-7
    np.testing.assert_allclose(derivative, vector_field(s, .8, -1.4, c), atol=1e-7)


def test_hamiltonian_matches_all_corner_controls():
    c = GameConfig();rng = np.random.default_rng(4)
    for _ in range(40):
        s, g = rng.normal(size=6), rng.normal(size=6)
        costs = np.array([[np.dot(g, vector_field(s, u, v, c))
                           for v in [-c.evader.max_turn_rate, c.evader.max_turn_rate]]
                          for u in [-c.pursuer.max_turn_rate, c.pursuer.max_turn_rate]])
        assert hamiltonian(s, g, c) == pytest.approx(costs.max(axis=1).min())


def test_capture_between_endpoints():
    c = GameConfig(arena_half_width=10)
    s = [0, 0, 0, .7, 0, np.pi]
    event = advance(s, 0, 0, c, duration=1)
    assert event.event == 'capture'
    assert event.elapsed == pytest.approx((.7-.3)/1.3, abs=2e-8)
    assert np.linalg.norm(event.state[:2]-event.state[3:5]) == pytest.approx(.3, abs=3e-8)


def test_faults_are_not_counted_as_capture():
    c = GameConfig()
    p_fault = advance([2.98,0,0,0,1,0], 0, 0, c)
    assert p_fault.event == 'pursuer_boundary_fault'
    assert not p_fault.pursuer_success and not p_fault.physical_capture
    e_fault = advance([0,0,0,2.99,1,0], 0, 0, c)
    assert e_fault.event == 'evader_boundary_fault'
    assert e_fault.pursuer_success and not e_fault.physical_capture


def test_initial_contact_and_invalid_control():
    c = GameConfig()
    assert advance([0,0,0,.1,0,0],0,0,c).elapsed == 0
    with pytest.raises(ValueError):advance([0,0,0,1,0,0],2,0,c)


def test_rotating_away_does_not_create_false_capture():
    c = GameConfig(arena_half_width=10)
    s = [0, 0, 0, .31, 0, 0]
    # E faster than P here; separation grows and there must be no event.
    c = GameConfig(arena_half_width=10, pursuer=Vehicle(.1,1),evader=Vehicle(1,2))
    assert advance(s,0,0,c,duration=.5).event is None

