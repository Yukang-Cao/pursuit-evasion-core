import json
import numpy as np
import pytest

from pursuit_evasion.model import GameConfig
from pursuit_evasion.dp.grid import GridSpec
from pursuit_evasion.dp.table import LocalTable


@pytest.fixture
def table(tmp_path):
    grid = GridSpec(5, 4, 3.5)
    c = GameConfig()
    times = [0.0, 1.0, 2.0]
    # A synthetic monotone field with a known crossing, for query correctness.
    states = np.meshgrid(*grid.axes, indexing="ij")
    base = 1.0 + 0.05 * states[0] + 0.05 * states[4]
    values = np.stack([base - t for t in times]).astype(np.float32)
    values.tofile(tmp_path / "fields.f32")
    meta = dict(
        grid=grid.to_dict(),
        config=c.to_dict(),
        times=times,
        status="complete",
        history_saved=True,
        tube_projection="min over every internal update, then max with avoid",
    )
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return LocalTable(tmp_path)


def test_interpolated_front_crossing_and_periodic_heading(table):
    state = np.array([-1.0, 0.1, 0.7, 1.0, 0.4, -1.2])
    expected = 1 + 0.05 * state[0] + 0.05 * state[4]
    assert table.value(state) == pytest.approx(expected, abs=2e-7)
    shifted = state.copy()
    shifted[[2, 5]] += 2 * np.pi
    assert table.value(shifted) == pytest.approx(expected, abs=2e-7)
    assert table.field(state, 0.3) == pytest.approx(expected - 0.3, abs=2e-7)


def test_invalid_queries_remain_unknown(table):
    state = np.array([0, 0, 0, 1, 0.4, 0.0])
    for bad in [np.nan, np.inf, 10.0]:
        s = state.copy()
        s[0] = bad
        assert np.isnan(table.value(s))
    assert np.isnan(table.field(state, 3.0))
    assert np.isnan(table.field(state, np.nan))


def test_terminal_events_override_numerical_interpolation(table):
    assert table.value([0, 0, 0, 0.1, 0, 0]) == 0
    assert table.value([0, 0, 0, 3.0, 1.0, 0]) == 0  # E fault, not capture
    assert np.isnan(table.value([3.0, 0, 0, 3.0, 0, 0]))  # P fault has priority
    assert np.isnan(table.value([0, 0, np.nan, 0.1, 0, 0]))
    assert np.isnan(table.value([0, 0, 0, np.inf, 0, 0]))


def test_batch_shape_is_preserved(table):
    x = np.array([[-1, 0, 0, 1, 0.4, 0], [0.3, 0, np.pi, -1, -0.5, 0]])
    q = table.field(np.broadcast_to(x, (3, 2, 6)), 0.5)
    assert q.shape == (3, 2)
    np.testing.assert_allclose(q[0], table.field(x, 0.5))
