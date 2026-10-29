from itertools import product
import json

import numpy as np
import pytest

from pursuit_evasion.reversible.model import ReversibleConfig, world_rhs
from pursuit_evasion.reversible.world_dp import (
    WorldGrid,
    global_rate_bound,
    hamiltonian,
    solve,
    surfaces,
)
from pursuit_evasion.reversible.world_table import WorldTable, interpolation_stencil


def test_hamiltonian_matches_enumerated_control_pairs_at_signed_and_boundary_speeds():
    config = ReversibleConfig()
    rng = np.random.default_rng(77)
    p_actions = np.array(list(product([-2, 2], [-1, 1])))
    e_actions = np.array(list(product([-1, 1], [-2, 2])))
    for vp, ve in product([-1, -0.4, 0, 0.7, 2], [-0.5, -0.1, 0, 0.4, 1]):
        state = np.array(
            [
                0.1,
                0.2,
                rng.uniform(-np.pi, np.pi),
                vp,
                0.8,
                -0.4,
                rng.uniform(-np.pi, np.pi),
                ve,
            ]
        )
        gradient = rng.normal(size=8)
        field = world_rhs(state, p_actions[:, None, :], e_actions[None, :, :], config)
        exact = np.min(np.max(np.sum(field * gradient, axis=-1), axis=1))
        assert hamiltonian(state, gradient, config) == pytest.approx(exact, abs=1e-12)


def reference_rhs(value, grid, config, order, speed_boundary="linear"):
    """Independent NumPy full-array ENO/LF reference for tiny-grid validation."""
    axes = grid.axes(config)

    def shift(axis, offset):
        n = len(axes[axis])
        q = np.arange(n) + offset
        if axis in (2, 6):
            return np.take(value, q % n, axis=axis)
        result = np.take(value, np.clip(q, 0, n - 1), axis=axis).copy()
        for edge in (0, n - 1):
            mask = q < 0 if edge == 0 else q >= n
            if not mask.any():
                continue
            a = np.take(value, edge, axis=axis)
            b = np.take(value, 1 if edge == 0 else n - 2, axis=axis)
            for index in np.flatnonzero(mask):
                sl = [slice(None)] * 8
                sl[axis] = index
                result[tuple(sl)] = a + abs(q[index] - edge) * abs(b - a) * np.sign(a)
        return result

    derivatives = []
    diss = 0.0
    states = np.stack(
        np.broadcast_arrays(*np.meshgrid(*axes, indexing="ij", sparse=True)), axis=-1
    )
    p, e = config.pursuer, config.evader
    alphas = [
        abs(states[..., 3] * np.cos(states[..., 2])),
        abs(states[..., 3] * np.sin(states[..., 2])),
        abs(states[..., 3]) * p.max_curvature,
        p.acceleration,
        abs(states[..., 7] * np.cos(states[..., 6])),
        abs(states[..., 7] * np.sin(states[..., 6])),
        abs(states[..., 7]) * e.max_curvature,
        e.acceleration,
    ]
    for k, h in enumerate(grid.spacings(config)):
        minus, plus = shift(k, -1), shift(k, 1)
        left = (value - minus) / h
        right = (plus - value) / h
        if order == 2:
            dl = (value - 2 * minus + shift(k, -2)) / h**2
            dc = (plus - 2 * value + minus) / h**2
            dr = (shift(k, 2) - 2 * plus + value) / h**2
            if k in (3, 7) and speed_boundary == "linear":
                low = [slice(None)] * 8
                low[k] = 1
                high = [slice(None)] * 8
                high[k] = -2
                dl[tuple(low)] = dc[tuple(low)]
                dr[tuple(high)] = dc[tuple(high)]
            left += 0.5 * h * np.where(abs(dl) <= abs(dc), dl, dc)
            right -= 0.5 * h * np.where(abs(dc) <= abs(dr), dc, dr)
        if k in (3, 7) and speed_boundary == "linear":
            low = [slice(None)] * 8
            low[k] = 0
            high = [slice(None)] * 8
            high[k] = -1
            left[tuple(low)] = right[tuple(low)] = (
                np.take(value, 1, axis=k) - np.take(value, 0, axis=k)
            ) / h
            left[tuple(high)] = right[tuple(high)] = (
                np.take(value, -1, axis=k) - np.take(value, -2, axis=k)
            ) / h
        derivatives.append(0.5 * (left + right))
        diss += 0.5 * alphas[k] * (right - left)
    return hamiltonian(states, np.stack(derivatives, axis=-1), config) + diss


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("speed_boundary", ["linear", "legacy_signed"])
def test_native_tiny_update_matches_independent_numpy_and_probe_query(
    tmp_path, order, speed_boundary
):
    config = ReversibleConfig()
    grid = WorldGrid(5, 4, 3, 3, 3.5)
    dt = 0.01
    assert dt < 0.8 / global_rate_bound(config, grid)
    probe = np.array(
        [[-1, 0, 0, 0.5, 0, 1, np.pi, 0.25], [0, 0, 0, 0.5, 0.1, 0.1, 0, 0.25]]
    )
    out = tmp_path / "solve"
    meta = solve(
        config,
        grid,
        out,
        horizon=dt,
        output_dt=dt,
        order=order,
        threads=2,
        history=True,
        probes=probe,
        speed_boundary=speed_boundary,
    )
    target, avoid = surfaces(config, grid, dtype=np.float64)
    initial = np.maximum(target, avoid)
    stage = initial + dt * reference_rhs(initial, grid, config, order, speed_boundary)
    expected = (
        stage
        if order == 1
        else 0.5
        * (
            initial
            + stage
            + dt * reference_rhs(stage, grid, config, order, speed_boundary)
        )
    )
    expected = np.maximum(np.minimum(expected, initial), avoid)
    actual = np.memmap(out / "final.f32", dtype=np.float32, shape=grid.shape, mode="r")
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert meta["internal_steps"] == 1
    assert np.max(actual - initial) < 2e-6
    assert np.min(actual - avoid) > -2e-6
    table = WorldTable(
        out, allow_legacy_speed_boundary=(speed_boundary == "legacy_signed")
    )
    saved = np.fromfile(out / "probe_fields.f64", dtype=np.float64).reshape(2, 2)
    np.testing.assert_allclose(table.field(probe, 0), saved[0], atol=1e-12)
    np.testing.assert_allclose(table.field(probe, dt), saved[1], atol=1e-12)
    assert table.value(probe[1]) == 0
    outside = probe[0].copy()
    outside[0] = 20
    assert np.isnan(table.field(outside, dt))
    assert np.isnan(table.value(outside))


def test_stencil_periodic_wrap_and_multilinear_exactness():
    config = ReversibleConfig()
    grid = WorldGrid(5, 8, 3, 3)
    s = np.array(
        [
            [0.2, -0.3, np.pi, 0.1, 0.7, 0.8, -np.pi, 0.2],
            [0.2, -0.3, -np.pi, 0.1, 0.7, 0.8, np.pi, 0.2],
        ]
    )
    ii, ww, valid, shape = interpolation_stencil(s, grid.axes(config), (2, 6))
    np.testing.assert_array_equal(ii[0], ii[1])
    np.testing.assert_allclose(ww.sum(1), 1)
    assert valid.all() and shape == (2,)
    fields = np.broadcast_arrays(
        *np.meshgrid(*grid.axes(config), indexing="ij", sparse=True)
    )
    linear = sum(fields[k] * (k + 1) for k in (0, 1, 3, 4, 5, 7))
    expected = sum(s[:, k] * (k + 1) for k in (0, 1, 3, 4, 5, 7))
    np.testing.assert_allclose(
        (linear.reshape(-1)[ii] * ww).sum(1), expected, atol=1e-12
    )


def test_resource_guard_happens_before_allocation(tmp_path):
    with pytest.raises(MemoryError):
        solve(
            ReversibleConfig(),
            WorldGrid(17, 16, 7, 7),
            tmp_path / "big",
            max_core_gb=0.001,
        )
    assert not (tmp_path / "big").exists()


def test_speed_limit_generator_matches_inward_semilagrangian_derivative():
    from pursuit_evasion.reversible.model import joint_step

    c = ReversibleConfig()
    g = WorldGrid(5, 4, 4, 4)
    axes = g.axes(c)
    mesh = np.broadcast_arrays(*np.meshgrid(*axes, indexing="ij", sparse=True))
    phi = 10 - mesh[0] - mesh[3]
    index = (2, 2, 2, 3, 2, 2, 2, 3)
    corrected = reference_rhs(phi, g, c, 1, "linear")[index]
    legacy = reference_rhs(phi, g, c, 1, "legacy_signed")[index]
    state = np.array([0, 0, 0, 2, 0, 0, 0, 1.0])
    P = np.array(list(product([-2, 0, 2], [-1, 0, 1])))
    E = np.array(list(product([-1, 0, 1], [-2, 0, 2])))
    h = 1e-5
    successor = joint_step(state, P[:, None, :], E[None, :, :], h, c)
    difference = (
        (10 - successor[..., 0] - successor[..., 3]) - (10 - state[0] - state[3])
    ) / h
    finite_hold = np.min(np.max(difference, axis=1))
    assert finite_hold == pytest.approx(-2, abs=1e-8)
    assert corrected == pytest.approx(finite_hold, abs=1e-8)
    assert legacy == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("order", [1, 2])
def test_resume_reuses_parent_history_and_matches_monolithic_solve(tmp_path, order):
    config = ReversibleConfig()
    grid = WorldGrid(5, 4, 4, 4)
    parent, extension, monolithic = [
        tmp_path / name for name in ("parent", "extension", "monolithic")
    ]
    solve(
        config,
        grid,
        parent,
        horizon=0.1,
        output_dt=0.05,
        order=order,
        history=True,
        threads=1,
    )
    meta = solve(
        config,
        grid,
        extension,
        horizon=0.2,
        output_dt=0.05,
        order=order,
        history=True,
        threads=1,
        resume_from=parent,
        write_final=False,
        write_arrival=False,
    )
    solve(
        config,
        grid,
        monolithic,
        horizon=0.2,
        output_dt=0.05,
        order=order,
        history=True,
        threads=1,
    )
    assert meta["start_time"] == 0.1
    assert len(meta["times"]) == 2
    assert not (extension / "final.f32").exists()
    assert not (extension / "arrival.f32").exists()
    assert (extension / "fields.f32").stat().st_size == 2 * grid.states * 4
    chained, full = WorldTable(extension), WorldTable(monolithic)
    np.testing.assert_allclose(chained.times, full.times, atol=1e-15)
    np.testing.assert_array_equal(chained.final, full.final)
    state = np.array([0, 0, 0, 1, 0.7, 0.2, 0, 0.5])
    states = np.broadcast_to(state, (5, 8))
    budgets = np.array([0, 0.025, 0.1, 0.125, 0.2])
    np.testing.assert_array_equal(
        chained.field(states, budgets), full.field(states, budgets)
    )
    np.testing.assert_equal(chained.value(states), full.value(states))
    assert (
        meta["cumulative_internal_steps"]
        == json.loads((monolithic / "meta.json").read_text())["internal_steps"]
    )
