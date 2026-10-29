"""Independent update, signed-speed minimax and signed-history query checks."""

import json
from dataclasses import replace
from itertools import product
import shutil

import numpy as np
import pytest

from pursuit_evasion.reversible.free_dp import RelativeGrid, solve, physics_signature
from pursuit_evasion.reversible.model import ReversibleConfig, relative_rhs
from pursuit_evasion.reversible.relative_table import RelativeTable


def reference_euler(field, grid, config, dt, cast=True, order=1):
    axes = grid.axes(config)
    arrays = np.meshgrid(*axes, indexing="ij")
    states = np.stack([arrays[4], arrays[3], arrays[2], arrays[0], arrays[1]], axis=-1)
    gradients = []
    jumps = []
    f = field.astype(np.float64)
    for axis, coords in enumerate(axes):
        h = coords[1] - coords[0]
        minus = (f - np.roll(f, 1, axis=axis)) / h
        plus = (np.roll(f, -1, axis=axis) - f) / h
        raw_minus = minus.copy()
        raw_plus = plus.copy()
        if order == 2:
            center = (np.roll(f, -1, axis=axis) - 2 * f + np.roll(f, 1, axis=axis)) / (
                h * h
            )
            second_left = (
                f - 2 * np.roll(f, 1, axis=axis) + np.roll(f, 2, axis=axis)
            ) / (h * h)
            second_right = (
                np.roll(f, -2, axis=axis) - 2 * np.roll(f, -1, axis=axis) + f
            ) / (h * h)
            if axis != 2:
                boundary = [slice(None)] * 5
                boundary[axis] = slice(0, 2)
                second_left[tuple(boundary)] = center[tuple(boundary)]
                boundary[axis] = slice(-2, None)
                second_right[tuple(boundary)] = center[tuple(boundary)]
            minus += (
                0.5 * h * np.where(abs(second_left) < abs(center), second_left, center)
            )
            plus -= (
                0.5
                * h
                * np.where(abs(center) < abs(second_right), center, second_right)
            )
        if axis != 2:
            lower = [slice(None)] * 5
            lower[axis] = 0
            upper = [slice(None)] * 5
            upper[axis] = -1
            minus[tuple(lower)] = plus[tuple(lower)] = raw_plus[tuple(lower)]
            plus[tuple(upper)] = minus[tuple(upper)] = raw_minus[tuple(upper)]
        gradients.append((minus + plus) / 2)
        jumps.append(plus - minus)
    grad = np.stack(
        [gradients[4], gradients[3], gradients[2], gradients[0], gradients[1]], axis=-1
    )
    # Explicit four-by-four control enumeration, independent of the kernel's
    # closed-form absolute-value Hamiltonian. The shared vector field clamps
    # acceleration outward at a speed bound, matching the physical contract.
    p, e = config.pursuer, config.evader
    worst = []
    for a, k in product(
        [-p.acceleration, p.acceleration], [-p.max_curvature, p.max_curvature]
    ):
        responses = []
        for b, l in product(
            [-e.acceleration, e.acceleration], [-e.max_curvature, e.max_curvature]
        ):
            responses.append(
                np.sum(grad * relative_rhs(states, [a, k], [b, l], config), axis=-1)
            )
        worst.append(np.max(responses, axis=0))
    H = np.min(worst, axis=0)
    x, y, psi, vp, ve = np.moveaxis(states, -1, 0)
    wp = abs(vp) * p.max_curvature
    we = abs(ve) * e.max_curvature
    bounds = [
        p.acceleration,
        e.acceleration,
        wp + we,
        abs(ve * np.sin(psi)) + wp * abs(x),
        abs(ve * np.cos(psi) - vp) + wp * abs(y),
    ]
    H += sum(0.5 * a * j for a, j in zip(bounds, jumps))
    updated = f + dt * np.minimum(H, 0)
    return updated.astype(np.float32) if cast else updated


@pytest.mark.skipif(shutil.which("g++") is None, reason="native test requires g++")
@pytest.mark.parametrize("order", [1, 2])
def test_native_matches_explicit_control_backup_across_signed_speed_bounds(
    tmp_path, order
):
    c = ReversibleConfig()
    g = RelativeGrid(9, 8, 7, 7, 1.0)
    meta = solve(
        c, g, tmp_path / "field", horizon=0.06, output_dt=0.02, order=order, threads=2
    )
    table = RelativeTable(tmp_path / "field", c)
    f = np.array(table.fields[0])
    t = 0
    for index, goal in enumerate(table.times[1:], start=1):
        while t < goal - 1e-12:
            dt = min(meta["cfl_dt_limit"], goal - t)
            if order == 1:
                f = reference_euler(f, g, c, dt)
            else:
                stage = reference_euler(f, g, c, dt, order=2)
                f = (
                    0.5
                    * (
                        f.astype(np.float64)
                        + reference_euler(stage, g, c, dt, cast=False, order=2)
                    )
                ).astype(np.float32)
            t += dt
        np.testing.assert_allclose(table.fields[index], f, atol=2e-6, rtol=2e-6)
    assert np.max(np.diff(table.fields, axis=0)) <= 0
    assert meta["target"] == "physical_capture_only"
    with pytest.raises(FileExistsError):
        solve(c, g, tmp_path / "field", horizon=0.1)


def synthetic_table(tmp_path):
    c = ReversibleConfig()
    g = RelativeGrid(9, 8, 7, 7, 2)
    axes = g.axes(c)
    vp, ve, psi, y, x = np.meshgrid(*axes, indexing="ij")
    times = np.array([0, 0.5, 1, 1.5, 2])
    field = np.array([x + 0.2 * y + 0.3 * vp - 0.4 * ve - t for t in times], np.float32)
    folder = tmp_path / "synthetic"
    folder.mkdir()
    field.tofile(folder / "fields.f32")
    # An intentionally bogus arrival field: querying must ignore it.
    np.full(g.shape, 123, dtype=np.float32).tofile(folder / "arrival.f32")
    meta = dict(
        schema="reversible_relative5d_v1",
        status="complete",
        history_saved=True,
        tube_projection="min over every internal update",
        times=times.tolist(),
        axes=[a.tolist() for a in axes],
        physics=physics_signature(c),
        config=c.to_dict(),
        grid=g.to_dict(),
    )
    (folder / "meta.json").write_text(json.dumps(meta))
    return RelativeTable(folder, c), c


def test_signed_history_query_order_periodicity_and_unknown(tmp_path):
    table, c = synthetic_table(tmp_path)
    states = np.array([[1, 0.7, 0.4, -0.5, 0.4], [0.8, -0.4, 3.2, 0.5, 0.1]])
    expected = (
        states[:, 0] + 0.2 * states[:, 1] + 0.3 * states[:, 3] - 0.4 * states[:, 4]
    )
    np.testing.assert_allclose(table.value(states), expected, atol=2e-7)
    shifted = states.copy()
    shifted[:, 2] += 2 * np.pi
    np.testing.assert_allclose(table.value(shifted), expected, atol=2e-7)
    assert np.isnan(table.value([2.1, 0, 0, 0, 0]))
    assert np.isnan(table.value([1, 0, 0, 2.1, 0]))
    assert np.isnan(table.value([np.nan, 0, 0, 0, 0]))
    assert np.isnan(table.field(states, -0.01)).all()
    assert np.isnan(table.field(states, 2.01)).all()
    assert table.value([0.1, 0.1, 0, -1, 1]) == 0
    table.assert_compatible(replace(c, arena_half_width=10))
    with pytest.raises(ValueError):
        table.assert_compatible(replace(c, capture_radius=0.4))


def test_hamiltonian_controls_flip_curvature_with_signed_speed(tmp_path):
    table, c = synthetic_table(tmp_path)
    s = np.array(
        [[1, 0.7, 0.4, -0.5, 0.4], [1, 0.7, 0.4, 0.5, 0.4], [1, 0.7, 0.4, -1, -0.5]]
    )
    p, e = table.hamiltonian_controls(s, 1)
    np.testing.assert_allclose(p[:2, 1], [1, -1])
    np.testing.assert_allclose(p[:2, 0], [-2, -2])
    assert p[2, 0] == 0 and e[2, 0] == 0
    np.testing.assert_allclose(
        table.gradient(s, 1), np.broadcast_to([1, 0.2, 0, 0.3, -0.4], (3, 5)), atol=1e-6
    )


def test_budget_preflight_does_not_allocate_large_field(tmp_path):
    with pytest.raises(MemoryError):
        solve(
            ReversibleConfig(),
            RelativeGrid(101, 48, 13, 13, 4),
            tmp_path / "large",
            max_working_gb=0.001,
        )
    assert not (tmp_path / "large").exists()


def test_budget_view_limits_queries_without_copying_history(tmp_path):
    full, c = synthetic_table(tmp_path)
    short = RelativeTable(full.folder, c, max_budget=1.0)
    state = [1.2, 0.3, 0, 1, 0.2]
    assert np.isfinite(full.value(state)) and np.isnan(short.value(state))
    np.testing.assert_array_equal(short.fields, full.fields[:3])
    assert np.isnan(short.field(state, 1.1))
    assert full.horizon == 2 and short.horizon == 1
    with pytest.raises(ValueError):
        RelativeTable(full.folder, c, max_budget=0.7)
