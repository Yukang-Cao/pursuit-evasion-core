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


def test_successor_rule_can_enter_known_domain_from_unknown_current_state():
    from pursuit_evasion.reversible.controllers import Baseline1
    from pursuit_evasion.reversible.mppi import MPPISettings

    class LimitedTable:
        times = np.array([0.0, 4.0])

        def value(self, states):
            s = np.asarray(states)
            return np.where(np.abs(s[..., 0]) <= 4.0, 1.0, np.nan)

        def field(self, states, budget):
            s = np.asarray(states)
            return np.where(np.abs(s[..., 0]) <= 4.0, -1.0, np.nan)

    c = ReversibleConfig()
    s = np.array([0, 0, 0, 2, 4.03, 0, 0, 0])
    new = Baseline1(
        c,
        WorldMap(None),
        LimitedTable(),
        mppi_settings=MPPISettings(samples=16, horizon=0.1),
    )
    new.command(s)
    assert new.last["control_owner"] == "dp"
    assert np.isnan(new.last["dp_value"])
    old = Baseline1(
        c,
        WorldMap(None),
        LimitedTable(),
        takeover="current_and_successor",
        mppi_settings=MPPISettings(samples=16, horizon=0.1),
    )
    old.command(s)
    assert old.last["control_owner"] == "lion_mppi"


def test_one_unknown_e_response_prevents_successor_time_takeover():
    from pursuit_evasion.reversible.controllers import Baseline1
    from pursuit_evasion.reversible.mppi import MPPISettings

    class MissingReply:
        times = np.array([0.0, 4.0])

        def value(self, s):
            s = np.asarray(s)
            return np.where(s[..., 4] > 0.0, np.nan, 1.0)

        def field(self, s, t):
            return np.ones(np.asarray(s).shape[:-1])

    c = ReversibleConfig()
    ctrl = Baseline1(
        c,
        WorldMap(None),
        MissingReply(),
        mppi_settings=MPPISettings(samples=16, horizon=0.1),
    )
    ctrl.command([0, 0, 0, 1, 2, 0, 0, 0])
    assert ctrl.last["control_owner"] == "lion_mppi"
    assert not ctrl.last["dp"]["resolved"]
