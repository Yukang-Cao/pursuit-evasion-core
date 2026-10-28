import shutil
import numpy as np
import pytest

from pursuit_evasion.dp.grid import GridSpec, target_and_avoid
from pursuit_evasion.dp.native import solve
from pursuit_evasion.dp.table import LocalTable
from pursuit_evasion.model import GameConfig


@pytest.mark.skipif(shutil.which("g++") is None, reason="native backend requires g++")
def test_native_initialization_and_monotone_budget(tmp_path):
    c = GameConfig()
    g = GridSpec(5, 4, 3.5)
    meta = solve(c, g, tmp_path, horizon=0.2, output_dt=0.1, threads=2)
    table = LocalTable(tmp_path)
    target, avoid = target_and_avoid(g, c)
    np.testing.assert_allclose(table.fields[0], np.maximum(target, avoid), atol=1e-6)
    assert np.diff(table.fields, axis=0).max() <= 1e-6
    assert meta["internal_steps"] >= 2
    # A distant interior pair cannot capture or touch a wall during 0.2 s.
    # Even the maximum closing speed can reduce separation by only 0.26 m.
    far = [-1.75, 0, 0, 1.75, 0, np.pi]
    assert np.isnan(table.value(far))
    assert table.value([0, 0, 0, 0.1, 0, 0]) == 0
