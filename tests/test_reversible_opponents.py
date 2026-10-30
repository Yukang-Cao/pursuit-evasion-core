import numpy as np
import pytest

from pursuit_evasion.reversible.game import WorldMap, advance
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.opponents import ReactiveEvader


class TestValue:
    horizon = 20.0

    def value(self, r):
        # The signs make maximum positive E acceleration/curvature unambiguous.
        return 10 + r[..., 4] + r[..., 2]

    def field(self, r, horizon):
        return -np.ones(r.shape[:-1])


def test_dp_response_maximizes_time_instead_of_minimizing():
    c = ReversibleConfig()
    e = ReactiveEvader(c, WorldMap(None), TestValue(), mode="dp")
    state = np.array([0, 0, 0, 0, 2, 0, 0, 0.5])
    action = e.command(state, [0, 0])
    np.testing.assert_allclose(action, [c.evader.acceleration, c.evader.max_curvature])
    assert e.last["selection_basis"] == "one_step_DP_time"


def test_corner_evader_can_brake_and_reverse_legally():
    c = ReversibleConfig()
    world = WorldMap(3)
    e = ReactiveEvader(c, world)
    state = np.array([0, 0, 0, 0, 2.95, 2.95, np.pi / 4, 0.1])
    action = e.command(state, [0, 0])
    assert action[0] < 0
    assert e.last["valid_candidates"] > 0
    step = advance(state, [0, 0], action, c, world=world)
    assert step.event is None


def test_fallback_is_disclosed_when_wall_contact_is_unavoidable():
    c = ReversibleConfig()
    e = ReactiveEvader(c, WorldMap(3))
    state = np.array([0, 0, 0, 0, 2.99, 0, 0, 0.9])
    action = e.command(state, [0, 0])
    assert e.last["fallback"]
    assert e.last["valid_candidates"] == 0
    assert action[0] == -c.evader.acceleration


def test_unresolved_time_uses_signed_field_and_labels_it():
    class Unresolved(TestValue):
        def value(self, r):
            return np.full(r.shape[:-1], np.nan)

        def field(self, r, horizon):
            return 2 + r[..., 4] + r[..., 2]

    c = ReversibleConfig()
    e = ReactiveEvader(c, WorldMap(None), Unresolved(), mode="dp")
    action = e.command([0, 0, 0, 0, 2, 0, 0, 0.5], [0, 0])
    np.testing.assert_allclose(action, [1, 2])
    assert e.last["selection_basis"] == "unresolved_signed_field"
    assert e.last["fallback"]


def test_outside_table_is_not_silently_assigned_infinite_capture_time():
    class Outside(TestValue):
        def value(self, r):
            return np.full(r.shape[:-1], np.nan)

        def field(self, r, horizon):
            return np.full(r.shape[:-1], np.nan)

    c = ReversibleConfig()
    e = ReactiveEvader(c, WorldMap(None), Outside(), mode="dp")
    e.command([0, 0, 0, 0, 2, 0, 0, 0.5], [0, 0])
    assert e.last["selection_basis"] == "outside_table_separation"
    assert e.last["fallback"]


def test_capture_event_overrides_large_terminal_value():
    c = ReversibleConfig()
    e = ReactiveEvader(c, WorldMap(None), TestValue(), mode="dp")
    state = np.array([0, 0, 0, 0, 0.35, 0, np.pi, 1.0])
    action = e.command(state, [0, 0])
    assert advance(state, [0, 0], action, c, world=WorldMap(None)).event is None


def test_evader_wall_fault_overrides_large_terminal_value():
    c = ReversibleConfig()
    world = WorldMap(3)
    e = ReactiveEvader(c, world, TestValue(), mode="dp")
    state = np.array([0, 0, 0, 0, 2.98, 0, 0, 0.4])
    action = e.command(state, [0, 0])
    assert action[0] < 0
    assert advance(state, [0, 0], action, c, world=world).event is None


def test_first_interval_pursuer_fault_does_not_query_terminal_table():
    class NoQuery(TestValue):
        def value(self, r):
            raise AssertionError("Terminal P fault must override the value query.")

    c = ReversibleConfig()
    e = ReactiveEvader(c, WorldMap(3), NoQuery(), mode="dp")
    e.command([2.99, 0, 0, 1, 0, 1, 0, 0], [2, 0])
    assert e.last["selection_basis"] == "first_interval_P_fault"


def test_post_capture_separation_cannot_make_earlier_capture_preferred():
    c = ReversibleConfig()
    world = WorldMap(None)
    e = ReactiveEvader(c, world)
    state = np.array([0, 0, 0, 2, 0.4, 0, np.pi, 0.3])
    action = e.command(state, [0, 0])
    selected = advance(state, [0, 0], action, c, world=world)
    accelerate_into_p = advance(state, [0, 0], [1, 0], c, world=world)
    assert selected.event == accelerate_into_p.event == "capture"
    assert selected.elapsed > accelerate_into_p.elapsed
    assert action[0] < 0


def test_wall_parallel_motion_is_not_rejected_for_small_clearance():
    c = ReversibleConfig()
    e = ReactiveEvader(c, WorldMap(3))
    state = np.array([0, 0, 0, 0, 2.999, 0, np.pi / 2, 0.5])
    times = e._wall_times(state)
    # Neutral input is a legal straight motion parallel to the wall. The
    # exact refinement must retain it even when a coarse distance bound fails.
    assert np.isinf(times[0])


def test_dp_mode_requires_a_table():
    with pytest.raises(ValueError):
        ReactiveEvader(ReversibleConfig(), WorldMap(None), mode="dp")


def test_free_space_shortcut_preserves_the_declared_opponent_policy():
    from dataclasses import replace

    base = ReversibleConfig()
    c = replace(
        base,
        pursuer=replace(base.pursuer, acceleration=3),
        evader=replace(base.evader, acceleration=3),
    )
    fast = ReactiveEvader(c, WorldMap(None))
    reference = ReactiveEvader(c, WorldMap(100))
    rng = np.random.default_rng(3030)
    for _ in range(30):
        s = np.array(
            [
                0,
                0,
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-1, 2),
                rng.uniform(5, 15),
                rng.uniform(-10, 10),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-0.5, 1),
            ]
        )
        u = rng.uniform([-3, -1], [3, 1])
        np.testing.assert_array_equal(fast.command(s, u), reference.command(s, u))
        np.testing.assert_allclose(
            fast.last["candidate_utilities"],
            reference.last["candidate_utilities"],
            rtol=0,
            atol=1e-12,
        )
        assert fast.last["exact_free_space_shortcut"]
        a = advance(s, u, fast.actions[0], c, world=WorldMap(None))
        b = advance(s, u, fast.actions[0], c, world=WorldMap(100))
        np.testing.assert_array_equal(a.state, b.state)
        assert a.event == b.event is None and a.elapsed == b.elapsed
