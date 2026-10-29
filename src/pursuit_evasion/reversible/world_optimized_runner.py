"""Isolated Python-3.8-compatible OptimizedDP runner for the reversible 8D game.

Invoke in the isolated legacy environment so HeteroCL does not enter the online environment.
The upstream HJSolver and graph_8D are retained. Model-specific inward speed
derivatives are injected explicitly; generic signed-distance boundaries remain
a diagnostic option. hcl.build is wrapped for separate timing.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import importlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

# Support the isolated legacy environment without installing this package there.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.world_dp import WorldGrid, output_times, surfaces


def install_speed_boundaries(hcl):
    """Physical speed-state boundaries use only inward stencil values.

    A bounded-speed Bellman successor never queries beyond its speed interval.
    This differs from signed-distance extrapolation at spatial truncation edges.
    """
    module = importlib.import_module("odp.computeGraphs.graph_8D")
    from odp.computeGraphs.CustomGraphFunctions import my_abs

    saved = {}

    def make_derivative(axis, order):
        def derivative(*args):
            indices, value, grid = list(args[:8]), args[8], args[9]
            pos, n, spacing = indices[axis], value.shape[axis], float(grid.dx[axis])

            def at(offset):
                q = list(indices)
                q[axis] = pos + offset
                return value[tuple(q)]

            left = hcl.scalar(0.0, "speed_left_" + str(axis))
            right = hcl.scalar(0.0, "speed_right_" + str(axis))
            with hcl.if_(pos == 0):
                left[0] = (at(1) - at(0)) / spacing
                right[0] = left[0]
            with hcl.elif_(pos == n - 1):
                left[0] = (at(0) - at(-1)) / spacing
                right[0] = left[0]
            with hcl.else_():
                left[0] = (at(0) - at(-1)) / spacing
                right[0] = (at(1) - at(0)) / spacing
                if order == 2:
                    center = hcl.scalar(0.0, "speed_center_" + str(axis))
                    low = hcl.scalar(0.0, "speed_low_" + str(axis))
                    high = hcl.scalar(0.0, "speed_high_" + str(axis))
                    center[0] = (at(1) - 2 * at(0) + at(-1)) / (spacing * spacing)
                    low[0], high[0] = center[0], center[0]
                    with hcl.if_(pos >= 2):
                        low[0] = (at(0) - 2 * at(-1) + at(-2)) / (spacing * spacing)
                    with hcl.if_(pos + 2 < n):
                        high[0] = (at(2) - 2 * at(1) + at(0)) / (spacing * spacing)
                    with hcl.if_(my_abs(low[0]) <= my_abs(center[0])):
                        left[0] = left[0] + 0.5 * spacing * low[0]
                    with hcl.else_():
                        left[0] = left[0] + 0.5 * spacing * center[0]
                    with hcl.if_(my_abs(center[0]) <= my_abs(high[0])):
                        right[0] = right[0] - 0.5 * spacing * center[0]
                    with hcl.else_():
                        right[0] = right[0] - 0.5 * spacing * high[0]
            return left[0], right[0]

        return derivative

    for axis in (3, 7):
        for name, order in [
            ("spa_derivX%d_8d" % axis, 1),
            ("secondOrder_ENO8D_X%d" % axis, 2),
        ]:
            saved[name] = getattr(module, name)
            setattr(module, name, make_derivative(axis, order))
    return module, saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--positions", type=int, required=True)
    p.add_argument("--headings", type=int, required=True)
    p.add_argument("--extent", type=float, required=True)
    p.add_argument("--speeds", type=int, default=4)
    p.add_argument("--horizon", type=float, required=True)
    p.add_argument("--output-dt", type=float, required=True)
    p.add_argument("--order", type=int, required=True)
    p.add_argument("--history", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--speed-boundary", choices=["linear", "legacy_signed"], default="linear"
    )
    a = p.parse_args()
    config = ReversibleConfig.from_dict(json.loads(Path(a.config).read_text()))
    grid = WorldGrid(a.positions, a.headings, a.speeds, a.speeds, a.extent)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "meta.json").exists():
        raise FileExistsError(output)
    import heterocl as hcl
    import odp
    from odp.Grid import Grid
    from odp.solver import HJSolver

    boundary_module, original_boundaries = None, {}
    if a.speed_boundary == "linear":
        boundary_module, original_boundaries = install_speed_boundaries(hcl)

    class SquareCars:
        def opt_ctrl(self, t, state, grad):
            p = config.pursuer
            accel = hcl.scalar(p.acceleration, "p_accel")
            curve = hcl.scalar(p.max_curvature, "p_curve")
            zero1, zero2 = hcl.scalar(0.0, "p_zero1"), hcl.scalar(0.0, "p_zero2")
            with hcl.if_(grad[3] > 0):
                accel[0] = -p.acceleration
            with hcl.if_(state[3] * grad[2] > 0):
                curve[0] = -p.max_curvature
            with hcl.if_(hcl.and_(state[3] <= p.speed_min, accel[0] < 0)):
                accel[0] = 0
            with hcl.if_(hcl.and_(state[3] >= p.speed_max, accel[0] > 0)):
                accel[0] = 0
            return (
                accel[0],
                curve[0],
                zero1[0],
                zero2[0],
                zero1[0],
                zero2[0],
                zero1[0],
                zero2[0],
            )

        def opt_dstb(self, t, state, grad):
            e = config.evader
            accel = hcl.scalar(e.acceleration, "e_accel")
            curve = hcl.scalar(e.max_curvature, "e_curve")
            zero1, zero2 = hcl.scalar(0.0, "e_zero1"), hcl.scalar(0.0, "e_zero2")
            with hcl.if_(grad[7] < 0):
                accel[0] = -e.acceleration
            with hcl.if_(state[7] * grad[6] < 0):
                curve[0] = -e.max_curvature
            with hcl.if_(hcl.and_(state[7] <= e.speed_min, accel[0] < 0)):
                accel[0] = 0
            with hcl.if_(hcl.and_(state[7] >= e.speed_max, accel[0] > 0)):
                accel[0] = 0
            return (
                accel[0],
                curve[0],
                zero1[0],
                zero2[0],
                zero1[0],
                zero2[0],
                zero1[0],
                zero2[0],
            )

        def dynamics(self, t, state, u, d):
            return (
                state[3] * hcl.cos(state[2]),
                state[3] * hcl.sin(state[2]),
                state[3] * u[1],
                u[0],
                state[7] * hcl.cos(state[6]),
                state[7] * hcl.sin(state[6]),
                state[7] * d[1],
                d[0],
            )

    source_root = Path(inspect.getfile(HJSolver)).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    times = output_times(a.horizon, a.output_dt)
    meta = dict(
        backend="optimized_dp_world8d",
        config=config.to_dict(),
        grid=grid.to_dict(config),
        horizon=a.horizon,
        output_dt=a.output_dt,
        times=times.tolist(),
        order=a.order,
        cfl=0.8,
        threads=a.threads,
        history_saved=bool(a.history),
        dtype="float32",
        status="running",
        upstream_revision=revision,
        upstream_solver_sha256=hashlib.sha256(
            (source_root / "odp/solver.py").read_bytes()
        ).hexdigest(),
        target="capture_or_evader_boundary_fault",
        avoid="pursuer_boundary_fault",
        tube_projection="min over completed numerical updates, then max with avoid",
        python=sys.version,
        arithmetic="upstream float32 fields; upstream NumPy host integration",
        speed_boundary=a.speed_boundary,
        boundary_adapter=(
            "inward one-sided speed derivatives on axes3,7"
            if a.speed_boundary == "linear"
            else "none; generic signed-distance extrapolation"
        ),
    )
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    start = time.perf_counter()
    target, avoid = surfaces(config, grid)
    minimum = np.array(
        [
            -grid.extent,
            -grid.extent,
            -np.pi,
            config.pursuer.speed_min,
            -grid.extent,
            -grid.extent,
            -np.pi,
            config.evader.speed_min,
        ]
    )
    maximum = np.array(
        [
            grid.extent,
            grid.extent,
            np.pi,
            config.pursuer.speed_max,
            grid.extent,
            grid.extent,
            np.pi,
            config.evader.speed_max,
        ]
    )
    g = Grid(minimum, maximum, 8, list(grid.shape), periodicDims=[2, 6])

    timing = {
        "compile_seconds": 0.0,
        "compiled_kernel_seconds": 0.0,
        "compiled_kernel_calls": 0,
    }
    original_build = hcl.build

    def measured_build(*args, **kwargs):
        t = time.perf_counter()
        fn = original_build(*args, **kwargs)
        timing["compile_seconds"] += time.perf_counter() - t

        def call(*xs, **kw):
            t = time.perf_counter()
            out = fn(*xs, **kw)
            timing["compiled_kernel_seconds"] += time.perf_counter() - t
            timing["compiled_kernel_calls"] += 1
            return out

        return call

    hcl.build = measured_build
    solver_log = output / "solver.log"
    try:
        with solver_log.open("w") as log, contextlib.redirect_stdout(log):
            t = time.perf_counter()
            value, arrival = HJSolver(
                SquareCars(),
                g,
                [target, -avoid],
                times,
                {
                    "TargetSetMode": "minVOverTime",
                    "ObstacleSetMode": "maxVWithObstacle",
                },
                saveAllTimeSteps=bool(a.history),
                accuracy="low" if a.order == 1 else "medium",
                computeTimeToReach=True,
            )
            total = time.perf_counter() - t
    finally:
        hcl.build = original_build
        if boundary_module is not None:
            for name, function in original_boundaries.items():
                setattr(boundary_module, name, function)
    write_start = time.perf_counter()
    if a.history:
        # Upstream history is reversed time on its LAST axis and often float64.
        with (output / "fields.f32").open("wb") as f:
            for k in range(len(times)):
                np.asarray(value[..., len(times) - 1 - k], np.float32).tofile(f)
        final = np.asarray(value[..., 0], np.float32)
    else:
        final = np.asarray(value, np.float32)
    final.tofile(output / "final.f32")
    arrival = np.where(arrival >= 10000, np.nan, arrival).astype(np.float32)
    arrival.tofile(output / "arrival.f32")
    text = solver_log.read_text()
    match = re.search(r"Total kernel time \(s\): ([0-9.]+)", text)
    meta.update(
        timing,
        status="complete",
        hjsolver_seconds=total,
        integration_seconds_reported=float(match.group(1)) if match else None,
        internal_steps=timing["compiled_kernel_calls"] // a.order,
        write_seconds=time.perf_counter() - write_start,
        process_seconds=time.perf_counter() - start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        arrival_record="upstream node entry at first internal nonpositive step; online queries use fields",
        history_allocation_note="upstream saveAllTimeSteps may allocate float64 history; recorded output converted to float32",
    )
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            path = output / "meta.json"
            if path.exists():
                meta = json.loads(path.read_text())
                if meta.get("status") == "running":
                    meta.update(status="failed", error=repr(error))
                    path.write_text(json.dumps(meta, indent=2) + "\n")
        raise
