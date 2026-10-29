import numpy as np
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.game import WorldMap
from pursuit_evasion.reversible.controllers import braking_safe_mask, dp_decision


class UnknownTable:
    times = np.array([0.0, 2.0])

    def value(self, s):
        return np.full(np.asarray(s).shape[:-1], np.nan)

    def field(self, s, t):
        return np.linalg.norm(np.asarray(s)[..., :2], axis=-1) - 0.3


def test_braking_guard_distinguishes_inward_and_outward_acceleration():
    c = ReversibleConfig()
    w = WorldMap(3)
    pose = np.array([2.9, 0, 0, 0.5])
    commands = np.array([[-2, 0], [2, 0]])
    np.testing.assert_array_equal(
        braking_safe_mask(pose, commands, c, w), [True, False]
    )


def test_capture_before_wall_does_not_require_postgame_braking():
    c = ReversibleConfig()
    w = WorldMap(3)
    state = np.array([2.96, 0, 0, 2, 2.95, 0.31, -np.pi / 2, 1])
    command, info = dp_decision(
        state, UnknownTable(), c, w, p_actions=np.array([[0, 0]])
    )
    assert info["resolved"] and info["terminal_success"] and info["guard_accepts"]
    assert not info["braking_feasible"]
    assert info["estimated_cost"] < 0.02


def test_unknown_time_remains_signed_guidance_not_fake_capture_time():
    c = ReversibleConfig()
    w = WorldMap(None)
    state = np.array([0, 0, 0, 1, 2, 0, 0, 0.5])
    u, info = dp_decision(state, UnknownTable(), c, w)
    assert not info["resolved"] and info["kind"] == "signed_field"
    assert np.isinf(info["estimated_cost"]) and info["guidance_available"]
