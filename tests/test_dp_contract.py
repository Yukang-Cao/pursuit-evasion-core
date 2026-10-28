import numpy as np

from pursuit_evasion.model import GameConfig
from pursuit_evasion.dp.grid import GridSpec, target_and_avoid, output_times


def test_grid_order_and_periodic_angles():
    g = GridSpec(9, 8, 3.5)
    assert g.shape == (9, 9, 8, 9, 9, 8)
    assert g.states == 419904
    assert g.axes[2][0] == -np.pi
    assert g.axes[2][-1] < np.pi
    np.testing.assert_allclose(output_times(0.23, 0.1), [0, 0.1, 0.2, 0.23])


def test_target_avoid_and_wall_priority():
    c = GameConfig()
    g = GridSpec(9, 8, 3.5)
    target, avoid = target_and_avoid(g, c)
    v = np.maximum(target, avoid)
    assert v.dtype == np.float32
    assert v[4, 4, 0, 4, 4, 0] < 0  # coincident interior cars
    assert v[8, 4, 0, 8, 4, 0] > 0  # P boundary fault overrides success
    assert v[4, 4, 0, 8, 4, 0] < 0  # E boundary fault, not physical capture
    assert v[4, 4, 0, 6, 4, 0] > 0  # initially separated interior states
