"""Adapter to StanfordASL/hj_reachability; no upstream numerical code is modified.

The tiny counted integration loop mirrors hj.step so internal update counts can
be reported. Its ENO derivatives and Runge-Kutta stages are upstream functions.
"""

import importlib.metadata
import json
from pathlib import Path
import resource
import time

import numpy as np

from .grid import GridSpec, global_rate_bound, output_times, target_and_avoid
from ..model import GameConfig


def make_dynamics(config):
    import hj_reachability as hj
    import jax.numpy as jnp

    class SquareCars(hj.ControlAndDisturbanceAffineDynamics):
        def __init__(self):
            super().__init__(
                "min",
                "max",
                hj.sets.Box(
                    jnp.array([-config.pursuer.max_turn_rate]),
                    jnp.array([config.pursuer.max_turn_rate]),
                ),
                hj.sets.Box(
                    jnp.array([-config.evader.max_turn_rate]),
                    jnp.array([config.evader.max_turn_rate]),
                ),
            )

        def open_loop_dynamics(self, state, time):
            return jnp.array(
                [
                    config.pursuer.speed * jnp.cos(state[2]),
                    config.pursuer.speed * jnp.sin(state[2]),
                    0.0,
                    config.evader.speed * jnp.cos(state[5]),
                    config.evader.speed * jnp.sin(state[5]),
                    0.0,
                ]
            )

        def control_jacobian(self, state, time):
            return jnp.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])

        def disturbance_jacobian(self, state, time):
            return jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])

        def partial_max_magnitudes(self, state, time, value, grad_value_box):
            return jnp.abs(self.open_loop_dynamics(state, time)) + jnp.array(
                [
                    0.0,
                    0.0,
                    config.pursuer.max_turn_rate,
                    0.0,
                    0.0,
                    config.evader.max_turn_rate,
                ]
            )

    return SquareCars()


def solve(
    config: GameConfig,
    grid: GridSpec,
    output,
    *,
    horizon=2.0,
    output_dt=0.1,
    order=2,
    cfl=0.8,
    threads=8,
    history=True,
):
    import hj_reachability as hj
    import jax
    import jax.numpy as jnp

    output = Path(output).resolve()
    if (output / "meta.json").exists():
        raise FileExistsError(f"Preserve completed solve: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    times = output_times(horizon, output_dt)
    meta = dict(
        backend="hj_reachability_jax",
        config=config.to_dict(),
        grid=grid.to_dict(),
        horizon=horizon,
        output_dt=output_dt,
        times=times.tolist(),
        order=order,
        cfl=cfl,
        history_saved=history,
        dtype="float32",
        status="running",
        target="capture_or_evader_boundary_fault",
        avoid="pursuer_boundary_fault",
        tube_projection="min over every internal update, then max with avoid",
        device=str(jax.devices()[0]),
        jax_version=jax.__version__,
        hj_reachability_version=importlib.metadata.version("hj-reachability"),
        loop="counted wrapper around upstream time_integrator; same update as hj.step",
    )
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    bounds = hj.sets.Box(
        jnp.array(
            [-grid.extent, -grid.extent, -np.pi, -grid.extent, -grid.extent, -np.pi]
        ),
        jnp.array([grid.extent, grid.extent, np.pi, grid.extent, grid.extent, np.pi]),
    )
    g = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        bounds, grid.shape, periodic_dims=(2, 5)
    )
    target_np, avoid_np = target_and_avoid(grid, config)
    initial = np.maximum(target_np, avoid_np)
    target, avoid = jax.device_put(target_np), jax.device_put(avoid_np)
    values = jax.device_put(initial)
    values.block_until_ready()
    # Same pointwise LF coefficients as the native/ODP kernels. Adjust CFL to
    # their sum-of-axis-maxima step bound instead of max-of-pointwise-sums.
    theta = grid.axes[2]
    local_rate = (config.pursuer.speed + config.evader.speed) * np.max(
        abs(np.cos(theta)) + abs(np.sin(theta))
    ) / grid.spacings[0] + (
        config.pursuer.max_turn_rate + config.evader.max_turn_rate
    ) / grid.spacings[
        2
    ]
    adjusted_cfl = cfl * local_rate / global_rate_bound(grid, config)
    settings = hj.SolverSettings.with_accuracy(
        "low" if order == 1 else "medium", CFL_number=adjusted_cfl
    )
    dynamics = make_dynamics(config)

    @jax.jit
    def step_counted(t, v, end):
        def body(carry):
            tt, vv, n = carry
            tt, vv = settings.time_integrator(settings, dynamics, g, tt, vv, end)
            vv = jnp.maximum(jnp.minimum(vv, carry[1]), avoid)
            return tt, vv, n + 1

        return jax.lax.while_loop(
            lambda carry: carry[0] > end, body, (t, v, jnp.int32(0))
        )

    meta["initialization_seconds"] = time.perf_counter() - started
    t = time.perf_counter()
    executable = step_counted.lower(
        jnp.float32(0), values, jnp.float32(-times[1])
    ).compile()
    meta["compile_seconds"] = time.perf_counter() - t
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    del target_np, avoid_np
    arrival = np.where(initial <= 0, 0.0, np.nan).astype(np.float32)
    previous = initial
    fields = (output / "fields.f32").open("wb") if history else None
    write_seconds = 0.0
    transfer_seconds = 0.0
    update_seconds = 0.0
    counts = []
    frame_times = []
    if fields:
        t = time.perf_counter()
        previous.tofile(fields)
        write_seconds += time.perf_counter() - t
    for i in range(1, len(times)):
        t = time.perf_counter()
        tt, values, n = executable(
            jnp.float32(-times[i - 1]), values, jnp.float32(-times[i])
        )
        values.block_until_ready()
        elapsed = time.perf_counter() - t
        update_seconds += elapsed
        counts.append(int(n))
        frame_times.append(elapsed)
        t = time.perf_counter()
        host = np.asarray(jax.device_get(values))
        transfer_seconds += time.perf_counter() - t
        fresh = (previous > 0) & (host <= 0) & np.isnan(arrival)
        arrival[fresh] = times[i - 1] + (times[i] - times[i - 1]) * previous[fresh] / (
            previous[fresh] - host[fresh]
        )
        if fields:
            t = time.perf_counter()
            host.tofile(fields)
            write_seconds += time.perf_counter() - t
        previous = host
        print(
            f"budget {times[i]:.3f} steps {sum(counts)} updates_s {update_seconds:.3f}",
            flush=True,
        )
    if fields:
        fields.close()
    t = time.perf_counter()
    previous.tofile(output / "final.f32")
    arrival.tofile(output / "arrival.f32")
    write_seconds += time.perf_counter() - t
    stats = jax.devices()[0].memory_stats() or {}
    meta.update(
        status="complete",
        update_seconds=update_seconds,
        transfer_seconds=transfer_seconds,
        write_seconds=write_seconds,
        process_seconds=time.perf_counter() - started,
        internal_steps=sum(counts),
        steps_per_output=counts,
        update_seconds_per_output=frame_times,
        frames=len(times),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        device_memory_stats={
            k: int(v) for k, v in stats.items() if isinstance(v, (int, np.integer))
        },
        adjusted_upstream_cfl=adjusted_cfl,
        arrival_record="interpolated output-frame crossing; online queries use fields",
        arithmetic="float32 fields and float32 local arithmetic",
    )
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
