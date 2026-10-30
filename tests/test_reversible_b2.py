import copy
import numpy as np
import pytest
from pursuit_evasion.reversible.b2 import B2Controller, B2Settings
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.game import WorldMap
from pursuit_evasion.reversible.mppi import MPPISettings


class Known:
    times = np.array([0.0, 8.0])

    def value(self, s):
        return np.ones(np.asarray(s).shape[:-1])

    def field(self, s, t):
        return -np.ones(np.asarray(s).shape[:-1])


def test_b2_shares_the_DP_takeover_and_skips_search_when_available():
    c = ReversibleConfig()
    b = B2Controller(c, WorldMap(None), Known())
    u = b.command([0, 0, 0, 1, 1, 0, 0, 0.5])
    assert b.last["control_owner"] == "dp"
    assert b.last["search"]["skipped_for_DP"]
    assert len(u) == 2


def test_b2_performs_actual_B1_rollouts_and_preserves_policy_state():
    c = ReversibleConfig()
    settings = MPPISettings(
        samples=16, horizon=0.1, sampler="log_mppi", noise_std=(2, 1)
    )
    b = B2Controller(
        c,
        WorldMap(None),
        None,
        B2Settings(depth=2, simulations=12, rollout_steps=1),
        seed=4,
        mppi_settings=settings,
    )
    state = np.array([0, 0, 0, 0, 6, 1, 0, 1])
    u = b.command(state)
    assert b.last["control_owner"] == "uct_b1"
    assert b.last["search"]["rollout_b1_calls"] > 0
    assert b.last["search"]["rollout_motion_steps"] > 0
    assert abs(u[0]) <= 2 and abs(u[1]) <= 1
    before = b.baseline.tracker.nominal.copy()
    rng = copy.deepcopy(b.baseline.tracker.rng.bit_generator.state)
    a = b._continuation(state)
    z = b._continuation(state)
    assert a == z
    np.testing.assert_array_equal(before, b.baseline.tracker.nominal)
    assert rng == b.baseline.tracker.rng.bit_generator.state
    fork = b.fork_b1(state)
    assert fork.table is b.table
    fork.lion.anchor[:] = 99
    assert not np.array_equal(fork.lion.anchor, b.baseline.lion.anchor)


def test_tree_stops_expanding_when_the_actual_hierarchy_would_use_dp():
    class Local(Known):
        def value(self, s):
            a = np.asarray(s)
            return np.where(np.linalg.norm(a[..., :2], axis=-1) <= 0.5, 1.0, np.nan)

    c = ReversibleConfig()
    b = B2Controller(
        c,
        WorldMap(None),
        Local(),
        B2Settings(depth=4, simulations=80, rollout_steps=1),
        seed=2,
        mppi_settings=MPPISettings(samples=16, horizon=0.1),
    )
    b.command([0, 0, 0, 2, 0.6, 0, 0, 0])
    assert b.last["control_owner"] == "uct_b1"
    terminals = [
        n
        for n in b.tree.nodes
        if n.player == "P"
        and b.dp_frontiers.get(np.asarray(n.state, float).tobytes(), False)
    ]
    assert terminals
    assert all(not n.edges for n in terminals)


def test_coarse_tree_interval_does_not_change_physics_or_execution_interval():
    from pursuit_evasion.reversible.model import joint_step

    c = ReversibleConfig()
    b = B2Controller(c, WorldMap(None), None, B2Settings(tree_dt=0.5))
    s = np.array([0, 0, 0, 1, 10, 1, 0, 0.5])
    b.adapter.P = np.array([[0.0, 0.0]])
    b.adapter.E = np.array([[0.0, 0.0]])
    step = b._transition(s, 0, 0)
    np.testing.assert_allclose(step.state, joint_step(s, [0, 0], [0, 0], 0.5, c))
    assert step.cost == 0.5 and not step.terminal
    assert b.config.control_dt == 0.05


def test_optional_rollout_can_execute_dp_instead_of_bootstrapping_immediately():
    class Local(Known):
        def value(self, s):
            a = np.asarray(s)
            return np.where(np.linalg.norm(a[..., :2], axis=-1) < 2, 1.0, np.nan)

    c = ReversibleConfig()
    b = B2Controller(
        c,
        WorldMap(None),
        Local(),
        B2Settings(simulations=4, rollout_steps=1, bootstrap_at_dp=False),
        seed=0,
        mppi_settings=MPPISettings(samples=16, horizon=0.1),
    )
    b.command([0, 0, 0, 0, 6, 0, 0, 1])
    before = b.rollout_motion_steps
    value = b._continuation(np.array([0, 0, 0, 1, 1, 0, 0, 0.5]))
    assert np.isfinite(value)
    assert b.rollout_motion_steps == before + 1


def test_adaptive_probe_restores_root_planner_and_rng_before_long_search():
    from pursuit_evasion.reversible.b2 import AdaptiveB2Controller

    class Local(Known):
        def value(self, s):
            a = np.asarray(s)
            return np.where(np.linalg.norm(a[..., :2], axis=-1) <= 0.5, 1.0, np.nan)

    c = ReversibleConfig()
    mp = MPPISettings(samples=16, horizon=0.1)
    short = B2Settings(depth=3, simulations=128, rollout_steps=1)
    long = B2Settings(depth=3, simulations=128, rollout_steps=3)
    adaptive = AdaptiveB2Controller(
        c,
        WorldMap(None),
        Local(),
        short,
        seed=3,
        mppi_settings=mp,
        long_rollout_steps=3,
    )
    reference = B2Controller(c, WorldMap(None), Local(), long, seed=3, mppi_settings=mp)
    s = np.array([0, 0, 0, 2, 0.6, 0, 0, 0])
    u = adaptive.command(s)
    v = reference.command(s)
    assert adaptive.last["search"]["adaptive_long_rollout"]
    np.testing.assert_array_equal(u, v)
    np.testing.assert_array_equal(
        adaptive.baseline.tracker.nominal, reference.baseline.tracker.nominal
    )
    assert (
        adaptive.baseline.tracker.rng.bit_generator.state
        == reference.baseline.tracker.rng.bit_generator.state
    )
    assert adaptive.rng.bit_generator.state == reference.rng.bit_generator.state
    assert (
        adaptive.last["search"]["total_simulations"]
        >= adaptive.last["search"]["simulations"]
    )


@pytest.mark.parametrize("table", [None, Known()])
def test_search_context_can_be_reconstructed_from_fixed_observations(table):
    from pursuit_evasion.reversible.opponents import ReactiveEvader
    from pursuit_evasion.reversible.game import advance

    c = ReversibleConfig()
    w = WorldMap(None)
    settings = B2Settings(depth=2, simulations=12, rollout_steps=1)
    mp = MPPISettings(samples=16, horizon=0.1)
    actual = B2Controller(c, w, table, settings, seed=5, mppi_settings=mp)
    shadow = B2Controller(c, w, table, settings, seed=5, mppi_settings=mp)
    s = np.array([0, 0, 0, 0, 6, 1, 0, 1])
    actual.reset(s)
    shadow.reset(s)
    e = ReactiveEvader(c, w)
    for k in range(10):
        u = actual.command(s, k * 0.05)
        shadow.baseline.command(s, k * 0.05, diagnostics=False)
        if actual.last["control_owner"] == "uct_b1":
            shadow.rng.integers(2**32)
        np.testing.assert_array_equal(
            actual.baseline.tracker.nominal, shadow.baseline.tracker.nominal
        )
        assert (
            actual.baseline.tracker.rng.bit_generator.state
            == shadow.baseline.tracker.rng.bit_generator.state
        )
        assert actual.rng.bit_generator.state == shadow.rng.bit_generator.state
        np.testing.assert_array_equal(
            actual.baseline.lion.anchor, shadow.baseline.lion.anchor
        )
        s = advance(s, u, e.command(s, u), c, world=w).state


def test_visit_count_root_option_uses_the_common_UCT_recommendation():
    c = ReversibleConfig()
    b = B2Controller(
        c,
        WorldMap(None),
        None,
        B2Settings(
            depth=2, simulations=12, rollout_steps=1, root_selection="most_visited"
        ),
        seed=2,
        mppi_settings=MPPISettings(samples=16, horizon=0.1),
    )
    u = b.command([0, 0, 0, 0, 6, 0, 0, 1])
    s = b.last["search"]
    rows = s["root_actions"]
    n = max(r["visits"] for r in rows)
    selected = min((r for r in rows if r["visits"] == n), key=lambda r: r["mean_cost"])
    np.testing.assert_array_equal(u, selected["command"])
    assert s["root_selection_mode"] == "most_visited"
