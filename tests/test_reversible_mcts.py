import numpy as np
import pytest

from pursuit_evasion.reversible.mcts import (
    AdversarialUCT,
    SearchStep,
    UCTController,
    UCTSettings,
)
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.game import WorldMap


def exhaustive(state, depth, p_actions, e_actions, transition, leaf):
    if depth == 0:
        return float(leaf(state))
    return min(
        max(
            step.cost
            + (
                0
                if step.terminal
                else exhaustive(
                    step.state, depth - 1, p_actions, e_actions, transition, leaf
                )
            )
            for step in (transition(state, p, e) for e in e_actions(state, p))
        )
        for p in p_actions(state)
    )


def test_one_step_tree_matches_exhaustive_minimax_and_maximizes_e():
    matrix = np.array([[1.0, 6.0], [4.0, 5.0]])
    P = lambda state: (0, 1)
    E = lambda state, p: (0, 1)
    step = lambda state, p, e: SearchStep(0, matrix[p, e], True)
    leaf = lambda state: 0.0
    tree = AdversarialUCT(0, P, E, step, leaf, depth=1, exploration=2, seed=4)
    result = tree.search(2000)
    assert result["finite_tree_exhausted"]
    assert result["action"] == 1
    assert result["exact_truncated_value"] == exhaustive(0, 1, P, E, step, leaf) == 5.0
    assert all(r["responses_tested"] == 2 for r in result["root_actions"])


def test_future_p_action_can_depend_on_observed_e_successor():
    P = lambda state: (0, 1)
    E = lambda state, p: (0, 1)

    def step(state, p, e):
        if state[0] == 0:
            if p == 1:
                return SearchStep(("terminal",), 2.0, True)
            return SearchStep((1, e), 0.0)
        return SearchStep(("terminal",), 0.0 if p == state[1] else 10.0, True)

    leaf = lambda state: 5.0
    tree = AdversarialUCT((0,), P, E, step, leaf, depth=2, exploration=4, seed=2)
    result = tree.search(20000)
    assert result["finite_tree_exhausted"]
    assert result["action"] == 0
    assert (
        result["exact_truncated_value"] == exhaustive((0,), 2, P, E, step, leaf) == 0.0
    )
    # A fixed future P action loses against one E choice; this tree learns
    # different subsequent actions in the two observed successor states.
    for e in (0, 1):
        node = tree.p_cache[(1, (1, e))]
        assert node.exact_value == 0.0
        best = min(node.actions, key=lambda p: node.edges[p].child.exact_value)
        assert best == e


def test_identical_observed_successors_share_the_future_p_node():
    P = lambda state: (0,)
    E = lambda state, p: (0, 1)
    step = lambda state, p, e: SearchStep(1, 0.0)
    tree = AdversarialUCT(0, P, E, step, lambda state: 3.0, depth=1, seed=7)
    result = tree.search(20)
    responses = tree.root.edges[0].child.edges
    assert responses[0].child is responses[1].child
    assert result["exact_truncated_value"] == 3.0


def test_zero_budget_returns_declared_legal_fallback():
    tree = AdversarialUCT(
        0,
        lambda s: (2, 3),
        lambda s, p: (0,),
        lambda s, p, e: SearchStep(0, 0, True),
        lambda s: 1.0,
    )
    result = tree.search(100, max_seconds=0)
    assert result["simulations"] == 0
    assert result["action"] == 2
    assert result["selection_rule"] == "no_simulation_first_legal_action"


def test_elapsed_cost_is_added_once_per_joint_interval():
    tree = AdversarialUCT(
        0,
        lambda s: (0,),
        lambda s, p: (0,),
        lambda s, p, e: SearchStep(s + 1, 0.05),
        lambda s: 2.0,
        depth=3,
        seed=0,
    )
    result = tree.search(20)
    assert result["finite_tree_exhausted"]
    assert result["exact_truncated_value"] == pytest.approx(2.15)


class SimpleTable:
    times = np.array([0.0, 4.0])

    def value(self, state):
        return np.linalg.norm(np.asarray(state)[..., :2], axis=-1)

    def field(self, state, budget):
        return np.asarray(state)[..., 0] - 1


def test_one_step_vehicle_tree_matches_exhaustive_with_shared_physics():
    c = ReversibleConfig()
    P = np.array([[0, 0], [2, 0]], float)
    E = np.array([[0, 0], [1, 0]], float)
    ctrl = UCTController(
        c,
        WorldMap(None),
        SimpleTable(),
        UCTSettings(depth=1, simulations=500, exploration=2),
        seed=11,
        p_actions=P,
        e_actions=E,
    )
    state = np.array([0, 0, 0, 0, 1.5, 0, 0, 0.5])
    action = ctrl.command(state)
    exact = exhaustive(
        state,
        1,
        ctrl._legal_p,
        lambda s, p: tuple(range(len(E))),
        ctrl._transition,
        ctrl._leaf,
    )
    assert ctrl.last["search"]["finite_tree_exhausted"]
    assert ctrl.last["search"]["exact_truncated_value"] == pytest.approx(exact)
    np.testing.assert_allclose(action, [2, 0])


def test_wall_fault_and_capture_costs_follow_game_events():
    c = ReversibleConfig()
    ctrl = UCTController(
        c, WorldMap(3), SimpleTable(), p_actions=[[0, 0]], e_actions=[[0, 0]]
    )
    capture = ctrl._transition(np.array([0, 0, 0, 2, 0.35, 0, 0, 0]), 0, 0)
    assert capture.terminal and capture.event == "capture"
    assert capture.cost == pytest.approx(0.025, abs=1e-8)
    fault = ctrl._transition(np.array([2.99, 0, 0, 1, 0, 1, 0, 0]), 0, 0)
    assert fault.terminal and fault.event == "pursuer_boundary_fault"
    assert fault.cost > ctrl.settings.pursuer_fault_cost


def test_unknown_leaf_guidance_is_explicit_and_finite():
    class Unknown(SimpleTable):
        def value(self, state):
            return np.nan

        def field(self, state, budget):
            return 0.7

    c = ReversibleConfig()
    ctrl = UCTController(
        c, WorldMap(None), Unknown(), UCTSettings(depth=1, simulations=10), seed=0
    )
    ctrl.command([0, 0, 0, 0.5, 1.5, 0, 0, 0.5])
    counts = ctrl.last["search"]["leaf_modes"]
    assert counts["signed_field_heuristic"] > 0 and counts["DP_time"] == 0
    assert np.isfinite(ctrl.last["search"]["initial_leaf_estimate"])


def test_invalid_unbounded_leaf_is_rejected():
    with pytest.raises(ValueError):
        AdversarialUCT(
            0,
            lambda s: (0,),
            lambda s, p: (0,),
            lambda s, p, e: SearchStep(1, 0),
            lambda s: np.nan,
        )


def test_actual_root_guard_failure_is_reported_separately_from_search_nodes():
    c = ReversibleConfig()
    ctrl = UCTController(
        c, WorldMap(3), SimpleTable(), UCTSettings(simulations=0), seed=0
    )
    ctrl.command([2.99, 0, 0, 1, 0, 1, 0, 0])
    assert ctrl.last["guard_failure"]
    assert ctrl.last["search"]["simulations"] == 0
