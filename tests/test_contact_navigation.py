from dataclasses import replace
import numpy as np
from pursuit_evasion.local_game.geometry import CircleArena
from pursuit_evasion.local_game.navigation import FootprintMPPI, RouteGuide
from pursuit_evasion.local_game.perception import LidarMemory
from pursuit_evasion.reversible.model import ReversibleConfig


def setup():
    c = ReversibleConfig()
    return replace(
        c,
        pursuer=replace(c.pursuer, acceleration=3),
        evader=replace(c.evader, acceleration=3),
        point_vehicles=False,
        capture_rule="rectangle_contact",
        boundary_rule="collision_constrained",
    )


def test_waypoint_arrival_ends_navigation_cost_but_keeps_safe_stop():
    c = setup()
    a = CircleArena([[3, 0, 0.15]], 5)
    tracker = FootprintMPPI(c, a, seed=1)
    tracker.goal_heading = 0
    pose = np.array([0, 0, 0, 2.0])
    commands = np.zeros((1, tracker.steps, 2))
    cost, valid, _, path = tracker._rollout(pose, commands, np.array([1.0, 0.0]))
    assert valid[
        0
    ]  # A later collision after the safely reachable waypoint does not invalidate arrival.
    assert np.isfinite(cost[0])
    assert path[0, -1, 0] > 3


def test_navigation_uses_observed_surfaces_and_keeps_reverse_available():
    c = setup()
    a = CircleArena([[1, 0, 0.2]], 3)
    memory = LidarMemory(a)
    memory.observe([0, 0, 0, 0])
    observed = memory.observed_arena()
    assert len(observed.circles) > 0
    tracker = FootprintMPPI(c, observed, seed=2)
    assert tracker.limits.speed_min < 0
    guide = RouteGuide(a, "pursuer")
    guide.update(memory)
    target, heading, path = guide.pursuit(
        np.array([0, 0, 0, 0]), np.array([2, 0, 0, 0])
    )
    assert np.isfinite(target).all() and np.isfinite(heading)
    assert len(path) > 1
