import numpy as np
import pytest

from pursuit_evasion.game import advance, predict_pairs
from pursuit_evasion.model import GameConfig


def test_batched_events_match_scalar_adaptive_events():
    c = GameConfig()
    rng = np.random.default_rng(9)
    cases = [
        [0, 0, 0, 0.32, 0, np.pi],
        [2.99, 0, 0, 0, 1, 0],
        [0, 0, 0, 2.99, 1, 0],
        [0, 0, 0, 0, 0.301, -np.pi / 2],
        [3.0, 0, 0, 3.0, 0, 0],
    ]
    for _ in range(15):
        p = rng.uniform(-2.8, 2.8, 2)
        a = rng.uniform(-np.pi, np.pi)
        e = p + rng.uniform(0.29, 0.37) * np.array([np.cos(a), np.sin(a)])
        cases.append([*p, rng.uniform(-np.pi, np.pi), *e, rng.uniform(-np.pi, np.pi)])
    P = np.array([-1.0, 0.0, 1.0])
    E = np.array([-2.0, 0.0, 2.0])
    names = {
        None: 0,
        "capture": 1,
        "pursuer_boundary_fault": 2,
        "evader_boundary_fault": 3,
    }
    for state in cases:
        _, times, codes = predict_pairs(state, P, E, c)
        for i, u in enumerate(P):
            for j, v in enumerate(E):
                event = advance(state, u, v, c)
                assert codes[i, j] == names[event.event]
                assert times[i, j] == pytest.approx(event.elapsed, abs=3e-8)
