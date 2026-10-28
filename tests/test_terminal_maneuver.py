import numpy as np

from pursuit_evasion.geometry import turning_circle_clearances
from pursuit_evasion.model import GameConfig, vehicle_step
from pursuit_evasion.controllers.mppi import MPPITracker, MPPISettings


def test_turning_circle_check_detects_the_corner_dead_end():
    c = GameConfig()
    pose = np.array([1.3, 1.4, np.pi / 2])
    left, right = turning_circle_clearances(pose, 1.0, 1.0, 3.0)
    assert left > 0 and right < 0
    endpoint = vehicle_step(pose, -1.0, 1.5, 1.0)
    assert np.max(abs(endpoint[:2])) < 3  # short rollout stays inside
    a, b = turning_circle_clearances(endpoint, 1.0, 1.0, 3.0)
    assert max(a, b) < 0  # neither full turning-circle continuation fits


def test_feasible_warm_start_survives_replanning():
    c = GameConfig()
    tracker = MPPITracker(c, MPPISettings(samples=32), seed=0)
    pose = np.array([1.3, 1.4, np.pi / 2])
    goal = np.array([2.1, 1.7])
    for _ in range(70):
        u = tracker.command(pose, goal, diagnostics=False)
        assert tracker.last["valid_samples"] >= 1
        assert tracker.last["terminal_circle_clearance"] >= 0.045 - 1e-10
        for t in np.linspace(0, c.control_dt, 9):
            assert np.max(abs(vehicle_step(pose, u, t, 1.0)[:2])) < 3.0
        pose = vehicle_step(pose, u, c.control_dt, 1.0)
