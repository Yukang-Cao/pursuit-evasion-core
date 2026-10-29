import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pursuit_evasion.reversible._cfree_mppi import navigation_costs, raw_noise_update
from pursuit_evasion.reversible.game import WorldMap
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.mppi import MPPISettings, ReversibleMPPI


def path_at_x(xs):
    path = np.zeros((len(xs), 1, 4))
    path[:, 0, 0] = xs
    return path


def test_goal_arrival_removes_overshoot_and_terminal_penalty():
    before = path_at_x([0.0, 0.5, 1.0, 2.0, 3.0])
    changed_tail = before.copy()
    changed_tail[3:, 0, :2] = [[-20, 30], [200, -10]]
    mask = np.zeros((5, 1), bool)
    a = navigation_costs(before, [1, 0], mask)
    b = navigation_costs(changed_tail, [1, 0], mask)
    np.testing.assert_allclose(a.costs, [1.25])
    np.testing.assert_array_equal(a.costs, b.costs)
    assert a.goal_step[0] == 2 and a.goal_reached[0]


def test_later_goal_cannot_erase_collision_and_tie_is_blocked():
    path = path_at_x([0.0, 0.5, 1.0, 1.1])
    earlier = np.array([[False], [True], [False], [False]])
    result = navigation_costs(path, [1, 0], earlier, obstacle_penalty=100)
    assert result.blocked[0] and not result.goal_reached[0]
    np.testing.assert_allclose(result.costs, [301])
    tie = np.array([[False], [False], [True], [False]])
    result = navigation_costs(path, [1, 0], tie)
    assert result.blocked[0] and not result.goal_reached[0]


def test_collision_after_navigation_goal_is_outside_that_local_task():
    path = path_at_x([0.0, 1.0, 2.0])
    mask = np.array([[False], [False], [True]])
    result = navigation_costs(path, [1, 0], mask)
    np.testing.assert_allclose(result.costs, [1.0])
    assert result.goal_reached[0] and not result.blocked[0]


def test_controller_can_choose_forward_and_reverse_and_reset():
    c = ReversibleConfig()
    settings = MPPISettings(samples=64)
    forward = ReversibleMPPI(c, settings, seed=4, clearance=WorldMap(None).clearance)
    reverse = ReversibleMPPI(c, settings, seed=4, clearance=WorldMap(None).clearance)
    f = forward.command([0, 0, 0, 0], [1, 0])
    r = reverse.command([0, 0, 0, 0], [-0.8, 0])
    assert f[0] > 0 and r[0] < 0
    assert abs(f[1]) <= c.pursuer.max_curvature
    forward.reset()
    np.testing.assert_array_equal(forward.command([0, 0, 0, 0], [1, 0]), f)
    assert forward.last["selected_valid"]


def test_safe_paths_can_turn_without_terminal_full_circle():
    c = ReversibleConfig()
    planner = ReversibleMPPI(c, MPPISettings(samples=16, horizon=0.5))
    # The stationary initial car is near a corner; no complete R=1 circle fits,
    # but reversing away from that corner is physically feasible.
    pose = np.array([2.8, 2.8, np.pi / 4, 0.0])
    commands = np.zeros((1, planner.steps, 2))
    commands[..., 0] = -c.pursuer.acceleration
    _, valid, _, paths = planner._rollout(pose, commands, [1.5, 1.5])
    assert valid[0]
    assert paths[0, -1, 0] < pose[0] and paths[0, -1, 1] < pose[1]


def test_waypoint_arrival_does_not_hide_wall_hit_in_executed_first_command():
    c = ReversibleConfig()
    planner = ReversibleMPPI(c, MPPISettings(samples=16, horizon=0.2))
    # Local goal already reached, but braking cannot stop the car before this
    # wall. The navigator must not label the upcoming physical command valid.
    planner.command([2.99, 0, 0, 1.0], [2.99, 0])
    assert planner.last["valid_samples"] == 0
    assert not planner.last["selected_valid"]
    assert planner.last["fallback"] == "longest_valid_prefix"


def test_blocked_samples_receive_zero_update_weight():
    nominal = np.zeros((2, 2))
    noise = np.array([[[1, 0], [-1, 0]], [[1, 0], [-1, 0]]], float)
    updated, weights = raw_noise_update(
        nominal, noise, [0, np.inf], [-2, -1], [2, 1], 0.5
    )
    np.testing.assert_allclose(weights, [1, 0])
    np.testing.assert_allclose(updated, [[1, 0], [1, 0]])


def load_reference():
    torch = pytest.importorskip("torch")
    path = Path(__file__).parent / "references/cfree_mppi.py"
    spec = importlib.util.spec_from_file_location("cfree_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return torch, module


def test_cost_matches_unmodified_upstream_on_valid_navigation_events():
    torch, ref = load_reference()
    obj = ref.ReferenceCost()
    obj.config = {}
    obj.dist_weight = 1.0
    obj.terminal_weight = 5.0
    obj.goal_tolerance = 0.15
    obj.robot_footprint_area = 1.0
    obj.collision_occupancy_ratio = 0.35
    obj.obs_penalty = 2000.0
    paths = np.zeros((5, 3, 4))
    paths[:, 0, 0] = [0, 0.5, 1, 1.5, 2]  # goal, then overshoot
    paths[:, 1, 0] = [0, 0.2, 0.4, 0.5, 0.6]  # no event
    paths[:, 2, 0] = [0, -0.2, -0.4, -0.6, -0.8]  # collision, no later goal
    blocked = np.zeros((5, 3), bool)
    blocked[3, 2] = True
    obj._calculate_robot_frame_costmap_cost = lambda points: torch.from_numpy(
        blocked.astype(float)
    )
    expected = obj._calculate_trajectory_costs(
        torch.from_numpy(paths), torch.tensor([1.0, 0.0])
    ).numpy()
    actual = navigation_costs(paths, [1, 0], blocked).costs
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_update_matches_unmodified_upstream_even_with_clamped_samples():
    torch, ref = load_reference()
    rng = np.random.default_rng(71)
    nominal = rng.uniform(-0.5, 0.5, (5, 2))
    noise = rng.normal(size=(5, 13, 2)) * [2, 1]
    costs = rng.uniform(0, 10, 13)
    lower, upper = np.array([-2, -1]), np.array([2, 1])
    obj = ref.ReferenceUpdate()
    obj.device = torch.device("cpu")
    obj.vrange, obj.wrange = [-2, 2], [-1, 1]
    obj.mppi_iterations = 1
    obj.lambda_weight = 0.5
    obj._transform_goal_to_robot_frame = lambda goal, pose: np.asarray(goal)
    obj._generate_noise_torch = lambda: torch.from_numpy(noise)
    obj._rollout_full_controls_torch = lambda controls, state: controls
    obj._calculate_trajectory_costs = lambda trajectories, goal: torch.from_numpy(costs)
    expected, _ = obj._run_mppi_optimization(
        torch.from_numpy(nominal.copy()), np.zeros(3), np.array([1.0, 0.0])
    )
    actual, _ = raw_noise_update(nominal, noise, costs, lower, upper, 0.5)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=0, atol=1e-12)
