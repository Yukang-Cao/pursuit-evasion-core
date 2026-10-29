import json
import numpy as np
import pytest
from pursuit_evasion.reversible.free_dp import RelativeGrid, physics_signature
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.relative_table import RelativeTable
from pursuit_evasion.reversible.viewer import interpolate_slice, arrival_slice, Explorer


def fixture(tmp_path):
    c = ReversibleConfig()
    g = RelativeGrid(9, 8, 7, 7, 2)
    axes = g.axes(c)
    vp, ve, psi, y, x = np.meshgrid(*axes, indexing="ij")
    times = np.array([0, 0.5, 1, 1.5, 2])
    fields = np.asarray(
        [x + 0.2 * y + 0.3 * vp - 0.4 * ve + 0.2 * np.cos(psi) - t for t in times],
        np.float32,
    )
    folder = tmp_path / "field"
    folder.mkdir()
    fields.tofile(folder / "fields.f32")
    (folder / "meta.json").write_text(
        json.dumps(
            dict(
                schema="reversible_relative5d_v1",
                status="complete",
                history_saved=True,
                tube_projection="min over every internal update",
                times=times.tolist(),
                axes=[a.tolist() for a in axes],
                physics=physics_signature(c),
            )
        )
    )
    return RelativeTable(folder), c


def test_native_slice_matches_general_queries_and_periodic_heading(tmp_path):
    table, c = fixture(tmp_path)
    for psi, vp, ve in [(3.12, -0.37, 0.31), (-3.4, 1.1, -0.1), (0, 2, 1)]:
        field = interpolate_slice(table, psi, vp, ve)
        xx, yy = np.meshgrid(table.axes[4], table.axes[3])
        states = np.stack(
            [xx, yy, np.full_like(xx, psi), np.full_like(xx, vp), np.full_like(xx, ve)],
            axis=-1,
        )
        for i, t in enumerate(table.times):
            np.testing.assert_allclose(field[i], table.field(states, t), atol=1e-12)
        values = arrival_slice(
            field, table.times, table.axes[4], table.axes[3], c.capture_radius
        )
        np.testing.assert_allclose(
            values, table.value(states), atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            field, interpolate_slice(table, psi + 2 * np.pi, vp, ve), atol=1e-12
        )
    with pytest.raises(ValueError):
        interpolate_slice(table, 0, 2.1, 0)


def test_explorer_keeps_world_pose_separate_from_free_space_value(tmp_path):
    table, c = fixture(tmp_path)
    manifest = tmp_path / "viewer.json"
    manifest.write_text(
        json.dumps(
            {"tables": [{"id": "a", "label": "test", "path": str(table.folder)}]}
        )
    )
    e = Explorer(manifest)
    s = [0, 0, 0.3, 1, 1, 0.5, 1.2, 0.5]
    a = e.query({"table": "a", "budget": 2, "state": s, "free_world": True})
    shifted = s.copy()
    shifted[0] += 0.4
    shifted[1] -= 0.2
    shifted[4] += 0.4
    shifted[5] -= 0.2
    b = e.query({"table": "a", "budget": 2, "state": shifted, "free_world": True})
    np.testing.assert_allclose(a["relative"], b["relative"], atol=1e-14)
    assert a["state_value"] == pytest.approx(b["state_value"])
    assert a["values"] == b["values"]
    with pytest.raises(ValueError):
        e.query({"table": "a", "budget": 0.7, "state": s})
    assert e.metadata()["tables"][0]["max_budget"] == 2


def test_wall_invalid_state_is_not_mislabeled_as_capture(tmp_path):
    table, c = fixture(tmp_path)
    manifest = tmp_path / "viewer.json"
    manifest.write_text(
        json.dumps(
            {"tables": [{"id": "a", "label": "test", "path": str(table.folder)}]}
        )
    )
    explorer = Explorer(manifest)
    data = explorer.query(
        {"table": "a", "budget": 2, "state": [4, 0, 0, 0, 4, 0, 0, 0]}
    )
    assert not data["physical_state_valid"]
    assert data["predicted_B1_mode"].startswith("invalid")
    assert explorer.metadata()["presets"]
