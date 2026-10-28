import numpy as np

from pursuit_evasion.model import GameConfig
from pursuit_evasion.controllers.lion import LionTarget
from pursuit_evasion.controllers.mppi import MPPITracker, MPPISettings


def test_lion_ideal_target_is_on_anchor_ray_and_within_step():
    teacher = LionTarget(step_length=1.0)
    s = np.array([-1.0, 0, 0, 1.0, 1.0, 0])
    teacher.reset(s)
    g = teacher.target_at(s, 0)
    assert np.linalg.norm(g - s[:2]) <= 1 + 1e-12
    ray = s[3:5] - teacher.anchor
    delta = g - teacher.anchor
    assert abs(ray[0] * delta[1] - ray[1] * delta[0]) < 1e-12
    np.testing.assert_array_equal(teacher.target_at(s, 0.1), g)


def test_mppi_reproducible_bounded_and_resettable():
    c = GameConfig()
    settings = MPPISettings(samples=32, horizon=0.3)
    a = MPPITracker(c, settings, seed=11)
    b = MPPITracker(c, settings, seed=11)
    pose = [0, 0, 0]
    goal = [1, 0]
    u = a.command(pose, goal)
    v = b.command(pose, goal)
    assert u == v and abs(u) <= c.pursuer.max_turn_rate
    assert a.last["valid_samples"] > 0
    assert len(a.last["nominal_rollout"]) == a.steps + 1
    a.reset()
    assert a.command(pose, goal) == u
