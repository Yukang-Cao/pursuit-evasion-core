from dataclasses import replace
import numpy as np
import pytest
from pursuit_evasion.local_game.geometry import CircleArena, rectangle_separation
from pursuit_evasion.local_game.native import GeometryEngine
from pursuit_evasion.reversible.model import ReversibleConfig, vehicle_step, wrap_angle


def config():
    c = ReversibleConfig()
    return replace(
        c,
        pursuer=replace(c.pursuer, acceleration=3),
        evader=replace(c.evader, acceleration=3),
        point_vehicles=False,
        capture_rule="rectangle_contact",
        boundary_rule="collision_constrained",
    )


def test_rectangle_contact_depends_on_heading_not_extra_state():
    p = np.array([0.0, 0, 0, 0])
    e = np.array([0.6, 0, 0, 0])
    assert rectangle_separation(p, e) == pytest.approx(0, abs=1e-15)
    e[0] = 0
    e[1] = 0.375
    assert rectangle_separation(p, e) == pytest.approx(0, abs=1e-15)
    e[1] = 0.5
    assert rectangle_separation(p, e) > 0
    e[2] = np.pi / 2
    assert rectangle_separation(p, e) > 0
    e[1] = 0.45
    assert rectangle_separation(p, e) == pytest.approx(0, abs=1e-15)
    theta = 0.73
    for s in (p, e):
        x, y = s[:2]
        s[:2] = [
            np.cos(theta) * x - np.sin(theta) * y,
            np.sin(theta) * x + np.cos(theta) * y,
        ]
        s[2] += theta
    assert rectangle_separation(p, e) == pytest.approx(0, abs=1e-15)


def test_native_kinematics_and_clearance_match_numpy():
    c = config()
    arena = CircleArena(np.array([[0.4, 0.7, 0.2], [-1, 0.2, 0.15]]), 3)
    rng = np.random.default_rng(57)
    s = rng.uniform(-2, 2, (300, 4))
    s[:, 2] = rng.uniform(-np.pi, np.pi, len(s))
    s[:, 3] = rng.uniform(-1, 2, len(s))
    engine = GeometryEngine(arena, c.pursuer, "pursuer", threads=2)
    np.testing.assert_allclose(engine.clearance(s), arena.clearance(s), atol=1e-12)
    controls = np.array([[-3, -1], [-3, 0], [0, 0], [3, 1]], float)
    safe, nxt = engine.actions(s, controls, 0.05)
    actual = vehicle_step(s[:, None], controls[None], 0.05, c.pursuer)
    error = nxt - actual
    error[..., 2] = wrap_angle(error[..., 2])
    assert abs(error).max() < 1e-12
    assert (engine.clearance(nxt[safe]) > 0).all()
    assert engine.backup_safe(nxt[safe]).all()


def test_safety_rejects_late_braking_but_preserves_stopping():
    c = config()
    a = CircleArena([], 3)
    e = GeometryEngine(a, c.pursuer, "pursuer", threads=1)
    assert not e.backup_safe([2.3, 0, 0, 2])
    assert e.backup_safe([2.3, 0, 0, 0])
    assert e.backup_safe([1.8, 0, 0, 2])


def test_continuous_rectangle_capture_time_and_grazing():
    from pursuit_evasion.local_game.native import capture_times

    c = config()
    arena = CircleArena([], 5)
    p = GeometryEngine(arena, c.pursuer, "pursuer")
    e = GeometryEngine(arena, c.evader, "evader")
    state = np.array([0, 0, 0, 1, 1, 0, 0, 0], float)
    assert capture_times(state, [[0, 0]], [[0, 0]], 1, p, e)[0, 0] == pytest.approx(
        0.4, abs=2e-8
    )
    state = np.array([0, 0, 0, 2, 0.5, 0.4, 0, 0], float)
    assert np.isinf(capture_times(state, [[0, 0]], [[0, 0]], 1, p, e)[0, 0])


def test_lidar_returns_nearest_surface_and_memory_does_not_reveal_shadow():
    from pursuit_evasion.local_game.native import scan_ranges
    from pursuit_evasion.local_game.perception import LidarMemory, SensorSettings

    a = CircleArena([[2, 0, 0.5]], 5)
    ranges = scan_ranges(a, [0, 0, 0], beams=360, max_range=6)
    assert ranges[0] == pytest.approx(1.5)
    assert ranges[180] == pytest.approx(5.0)
    memory = LidarMemory(
        a, SensorSettings(range=6, beams=720, extent=4, image_size=129)
    )
    memory.observe([0, 0, 0])
    image, frame = memory.image([0, 0, 0])
    ideal, _ = memory.image([0, 0, 0], ideal=True)
    # x=3 directly behind the obstacle is unknown in the raw observation.
    assert image[1, 64, 112] == 0
    assert ideal[1, 64, 112] == 1
    assert image[1, 64, 72] == 1
    assert image.shape == (2, 129, 129)


def test_braking_respects_command_period_and_exact_stop_control():
    c = config()
    arena = CircleArena([], 0.40055)
    engine = GeometryEngine(arena, c.pursuer, "pursuer", threads=1)
    state = np.array([0, 0, 0, 0.02])
    assert engine.backup_safe(state)
    legal, _ = engine.actions(state, [[-3, 0], [0, 0], [3, 0], [-0.4, 0]], 0.05)
    np.testing.assert_array_equal(legal[0], [False, False, False, True])
    a = CircleArena([], 3)
    engine = GeometryEngine(a, c.evader, "evader", threads=1)
    # Continuous max braking would fit, but a held-command final stop needs
    # slightly more distance. Do not falsely label that state stop-feasible.
    assert not engine.backup_safe([2.7892, 0, 0, 0.25])
