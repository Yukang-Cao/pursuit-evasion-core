from dataclasses import replace
import json
import numpy as np
import pytest
from pursuit_evasion.reversible.footprint_map import Footprint, FootprintMap
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.world_dp import WorldGrid


def test_footprint_heading_and_player_size_change_collision():
    m = FootprintMap(2.5)
    assert m.clearance([2.2, 0, 0], "pursuer") < 0
    assert m.clearance([2.2, 0, np.pi / 2], "pursuer") > 0
    assert m.clearance([2.2, 0, 0], "evader") > 0
    corridor = FootprintMap(2.5, ((-2.5, 2.5, 0.3, 2.5), (-2.5, 2.5, -2.5, -0.3)))
    assert corridor.clearance([0, 0, 0], "pursuer") > 0
    assert corridor.clearance([0, 0, np.pi / 2], "pursuer") < 0
    assert corridor.clearance([0, 0, np.pi / 2], "evader") > 0


def test_joint_quarter_rotation_preserves_geometry_and_event_order():
    m = FootprintMap(2.5, ((-0.7, -0.4, -1, 0.5), (0.6, 1.1, 0.8, 1.7)))
    boxes = tuple((-b[3], -b[2], b[0], b[1]) for b in m.obstacles)
    r = FootprintMap(2.5, boxes)
    rng = np.random.default_rng(5)
    s = rng.uniform(-2, 2, (100, 4))
    s[:, 2] = rng.uniform(-np.pi, np.pi, 100)
    rotated = s.copy()
    rotated[:, 0] = -s[:, 1]
    rotated[:, 1] = s[:, 0]
    rotated[:, 2] += np.pi / 2
    for role in ("pursuer", "evader"):
        np.testing.assert_allclose(
            m.clearance(s, role), r.clearance(rotated, role), atol=2e-15
        )
    s = np.array([2.2, 0, 0, 0, 2.25, 0, 0, 0])
    assert (
        m.event(s, 0.5) == "pursuer_collision"
    )  # P fault takes priority over capture.
    s = np.array([0, 0, 0, 0, 0.4, 0, 0, 0])
    assert m.event(s, 0.5) == "capture"


@pytest.mark.parametrize("order", [1, 2])
def test_gpu_map_update_matches_independent_numpy(order, tmp_path):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No CUDA device")
    from pursuit_evasion.reversible.gpu_map_game import solve
    from test_reversible_world_dp import reference_rhs

    c = ReversibleConfig()
    c = replace(
        c,
        pursuer=replace(c.pursuer, acceleration=3),
        evader=replace(c.evader, acceleration=3),
        capture_radius=0.5,
        point_vehicles=False,
        arena_half_width=2.5,
    )
    g = WorldGrid(5, 4, 3, 3, 3)
    m = FootprintMap(2.5, ((-0.25, 0.25, -1.4, 0.1),))
    states = np.stack(
        np.broadcast_arrays(*np.meshgrid(*g.axes(c), indexing="ij", sparse=True)),
        axis=-1,
    )
    physical_initial, avoid = m.surfaces(states, 0.5)
    dt = 0.01
    folder = tmp_path / "gpu"
    meta = solve(
        c,
        g,
        m,
        folder,
        horizon=dt,
        output_dt=dt,
        order=order,
        save_final=True,
        save_history=True,
        query_indices=np.arange(g.states),
    )
    labels = np.load(folder / "labels.npz")
    initial = labels["values"][0].reshape(g.shape)
    # Compare the same float32 initial field and stored RK stage. ENO stencil
    # ties can differ if the CPU instead starts from an unrounded float64 field.
    np.testing.assert_allclose(initial, physical_initial, atol=3e-7, rtol=3e-7)
    stage = (initial + dt * reference_rhs(initial, g, c, order)).astype(np.float32)
    expected = (
        stage
        if order == 1
        else 0.5 * (initial + stage + dt * reference_rhs(stage, g, c, order))
    )
    expected = np.maximum(np.minimum(expected, initial), avoid)
    actual = np.fromfile(folder / "final.f32", np.float32).reshape(g.shape)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
    history = np.fromfile(folder / "fields.f32", np.float32).reshape(2, *g.shape)
    np.testing.assert_array_equal(history[0], initial)
    np.testing.assert_array_equal(history[1], actual)
    assert meta["steps"] == 1
    assert meta["target"] == "capture_OR_evader_collision"
    labels = np.load(folder / "labels.npz")
    np.testing.assert_array_equal(
        labels["values"][-1], actual.ravel()[labels["indices"]]
    )


def test_gpu_resource_guard_does_not_allocate_large_field(tmp_path):
    pytest.importorskip("cupy")
    from pursuit_evasion.reversible.gpu_map_game import solve

    with pytest.raises(MemoryError):
        solve(
            replace(ReversibleConfig(), point_vehicles=False),
            WorldGrid(21, 16, 7, 7, 3.5),
            FootprintMap(3),
            tmp_path / "huge",
            max_working_gb=0.001,
        )


def test_continuous_footprint_event_catches_crossing_and_reversal():
    from pursuit_evasion.reversible.footprint_map import advance

    c = ReversibleConfig()
    c = replace(
        c,
        pursuer=replace(c.pursuer, acceleration=3),
        evader=replace(c.evader, acceleration=3),
        capture_radius=0.5,
        point_vehicles=False,
    )
    m = FootprintMap(3, ((0, 0.05, -0.5, 0.5),))
    s = np.array([-1, 0, 0, 2, 0, 2, 0, 0], dtype=float)
    r = advance(s, [0, 0], [0, 0], c, m, duration=1)
    assert r.event == "pursuer_collision"
    assert r.elapsed == pytest.approx(0.3, abs=1e-7)
    s[0] = -0.6
    r = advance(s, [-3, 0], [0, 0], c, m, duration=1.5)
    assert r.event == "pursuer_collision"
    assert r.elapsed == pytest.approx((2 - np.sqrt(2.8)) / 3, abs=1e-7)


def test_collision_contract_cannot_silently_fall_back_to_points():
    from pursuit_evasion.reversible.game import advance as point_advance, predict_pairs
    from pursuit_evasion.reversible.footprint_map import advance as body_advance
    from pursuit_evasion.reversible.gpu_map_game import solve

    c = replace(ReversibleConfig(), point_vehicles=False)
    s = np.array([0, 0, 0, 0, 1, 0, 0, 0], float)
    with pytest.raises(ValueError, match="Point-game"):
        point_advance(s, [0, 0], [0, 0], c)
    with pytest.raises(ValueError, match="Point-game"):
        predict_pairs(s, [[0, 0]], [[0, 0]], c)
    with pytest.raises(ValueError, match="point_vehicles=False"):
        body_advance(s, [0, 0], [0, 0], ReversibleConfig(), FootprintMap())
    with pytest.raises(ValueError, match="point_vehicles=False"):
        solve(ReversibleConfig(), WorldGrid(), FootprintMap(), "/unused")
