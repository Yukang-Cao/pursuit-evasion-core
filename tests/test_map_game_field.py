from dataclasses import replace
import json
import numpy as np
import pytest

from pursuit_evasion.reversible.footprint_map import FootprintMap
from pursuit_evasion.reversible.map_game_field import MapGameField, first_crossing
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.world_dp import WorldGrid


def test_first_crossing_interpolates_time_and_preserves_unresolved():
    values = np.array([[1, -1, 1, 1], [0.5, -2, 0.5, 0.5], [-0.5, -3, 0.1, -0.5]])
    result = first_crossing([0, 1, 2], values, [True, True, True, False])
    np.testing.assert_allclose(result[:2], [1.5, 0])
    assert np.isnan(result[2:]).all()
    with pytest.raises(ValueError):
        first_crossing([0, 0, 2], values)


def test_retained_history_queries_budget_and_exact_initial_geometry(tmp_path):
    c = replace(ReversibleConfig(), capture_radius=0.5, point_vehicles=False)
    g = WorldGrid(5, 4, 3, 3, 2)
    m = FootprintMap(5)
    meta = dict(
        schema="footprint_map_game8d_v1",
        status="complete",
        config=c.to_dict(),
        geometry=m.to_dict(),
        grid=dict(positions=5, headings=4, p_speeds=3, e_speeds=3, extent=2),
        times=[0, 0.5, 1],
        history_saved=True,
    )
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    fields = np.stack([np.full(g.shape, x, np.float32) for x in [0.9, 0.3, -0.3]])
    fields.tofile(tmp_path / "fields.f32")
    table = MapGameField(tmp_path)
    state = np.array([0, 0, 0, 0, 1.1, 0, 0, 0])
    np.testing.assert_allclose(table.field(state), -0.3, atol=1e-7)
    np.testing.assert_allclose(table.field(state, budget=0.75), 0, atol=1e-7)
    np.testing.assert_allclose(table.time_value(state), 0.75, atol=1e-7)
    captured = state.copy()
    captured[4] = 0.4
    # The physical zero-time target overrides inaccurate/interpolated initial labels.
    assert table.time_value(captured) == 0
    outside = state.copy()
    outside[0] = 3
    result = table.time_value(np.stack([state, captured, outside]).reshape(1, 3, 8))
    assert result.shape == (1, 3)
    assert np.isnan(result[0, 2])
    with pytest.raises(ValueError, match="not available"):
        table.field(state, budget=2)
