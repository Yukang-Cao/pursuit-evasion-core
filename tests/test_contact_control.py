import numpy as np
from dataclasses import replace
from pursuit_evasion.local_game.control import choose_p, choose_e, admissible_controls
from pursuit_evasion.local_game.native import GeometryEngine
from pursuit_evasion.local_game.geometry import CircleArena
from pursuit_evasion.reversible.model import ReversibleConfig


def test_both_players_use_capture_time_before_margin():
    t = np.array([[1.0, 2.0], [1.4, 1.5]])
    margin = np.array([[10.0, -10.0], [100.0, -100.0]])
    commands = np.array([[0.0, 0.0], [3.0, 1.0]])
    p, replies, worst, _ = choose_p(t, margin, commands, commands)
    assert p == 1
    assert choose_e(t[p], margin[p], commands) == 1
    np.testing.assert_allclose(worst, [2, 1.5])


def test_exact_stop_is_available_when_extreme_accelerations_are_unsafe():
    c = ReversibleConfig()
    limits = replace(c.pursuer, acceleration=3)
    arena = CircleArena([], 0.40055)
    engine = GeometryEngine(arena, limits, "pursuer", threads=1)
    controls, states = admissible_controls(np.array([0, 0, 0, 0.02]), engine, 0.05)
    np.testing.assert_allclose(controls, [[-0.4, 0]], atol=1e-10)
    np.testing.assert_allclose(states[:, 3], 0, atol=1e-14)
