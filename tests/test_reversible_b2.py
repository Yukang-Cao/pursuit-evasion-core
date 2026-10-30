import copy
import numpy as np
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
