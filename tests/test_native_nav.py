from dataclasses import replace
import shutil
import numpy as np
import pytest
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.game import WorldMap
from pursuit_evasion.reversible.mppi import ReversibleMPPI, MPPISettings

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None, reason="native cost needs g++"
)


def test_native_cost_matches_existing_numpy_early_goal_objective():
    c = ReversibleConfig()
    settings = MPPISettings(samples=128, horizon=1.5)
    a = ReversibleMPPI(c, settings, 0, clearance=WorldMap(None).clearance)
    b = ReversibleMPPI(
        c,
        replace(settings, cost_backend="native_free"),
        0,
        clearance=WorldMap(None).clearance,
    )
    rng = np.random.default_rng(30)
    for speed in [-1.0, 0.0, 2.0]:
        for goal in [[0, 0], [0.4, 0.2], [-2, 1], [3, -2]]:
            controls = rng.uniform([-2, -1], [2, 1], size=(128, 30, 2))
            pose = [0, 0, 0.3, speed]
            expected = a._rollout(pose, controls, goal)[0]
            actual = b._rollout(pose, controls, goal, return_paths=False)[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_native_backend_preserves_weighted_update_and_rejects_maps():
    c = ReversibleConfig()
    settings = MPPISettings(samples=128, sampler="log_mppi", noise_std=(2, 1))
    a = ReversibleMPPI(c, settings, 2, clearance=WorldMap(None).clearance)
    b = ReversibleMPPI(
        c,
        replace(settings, cost_backend="native_free"),
        2,
        clearance=WorldMap(None).clearance,
    )
    for _ in range(5):
        ua = a.command([0, 0, 0.5, -0.3], [-1, 2], diagnostics=False)
        ub = b.command([0, 0, 0.5, -0.3], [-1, 2], diagnostics=False)
        np.testing.assert_allclose(ua, ub, atol=1e-10, rtol=1e-10)
    with pytest.raises(ValueError):
        ReversibleMPPI(
            c,
            replace(settings, cost_backend="native_free"),
            clearance=WorldMap(3).clearance,
        )
