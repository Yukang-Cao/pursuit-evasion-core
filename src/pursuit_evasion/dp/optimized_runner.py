"""Isolated Python-3.8-compatible runner for the published OptimizedDP API.

Run through dp.optimized so HeteroCL does not enter the online environment.
The upstream HJSolver and graph_6D are unmodified; hcl.build is wrapped only to
measure compilation and compiled-kernel time separately from host integration.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
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
from pursuit_evasion.model import GameConfig
from pursuit_evasion.dp.grid import GridSpec, output_times, target_and_avoid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--positions", type=int, required=True)
    p.add_argument("--headings", type=int, required=True)
    p.add_argument("--extent", type=float, required=True)
    p.add_argument("--horizon", type=float, required=True)
    p.add_argument("--output-dt", type=float, required=True)
    p.add_argument("--order", type=int, required=True)
    p.add_argument("--history", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    config = GameConfig.load(a.config)
    grid = GridSpec(a.positions, a.headings, a.extent)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "meta.json").exists():
        raise FileExistsError(output)
    import heterocl as hcl
    import odp
    from odp.Grid import Grid
    from odp.solver import HJSolver

    class SquareCars:
        def opt_ctrl(self, t, state, spat_deriv):
            w = hcl.scalar(config.pursuer.max_turn_rate, "p_turn")
            z1, z2, z3 = (
                hcl.scalar(0.0, "p_z1"),
                hcl.scalar(0.0, "p_z2"),
                hcl.scalar(0.0, "p_z3"),
            )
            with hcl.if_(spat_deriv[2] > 0):
                w[0] = -config.pursuer.max_turn_rate
            return w[0], z1[0], z2[0], z3[0]

        def opt_dstb(self, t, state, spat_deriv):
            w = hcl.scalar(config.evader.max_turn_rate, "e_turn")
            z1, z2, z3 = (
                hcl.scalar(0.0, "e_z1"),
                hcl.scalar(0.0, "e_z2"),
                hcl.scalar(0.0, "e_z3"),
            )
            with hcl.if_(spat_deriv[5] < 0):
                w[0] = -config.evader.max_turn_rate
            return w[0], z1[0], z2[0], z3[0]

        def dynamics(self, t, state, uOpt, dOpt):
            return (
                config.pursuer.speed * hcl.cos(state[2]),
                config.pursuer.speed * hcl.sin(state[2]),
                uOpt[0],
                config.evader.speed * hcl.cos(state[5]),
                config.evader.speed * hcl.sin(state[5]),
                dOpt[0],
            )

    source_root = Path(inspect.getfile(HJSolver)).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    times = output_times(a.horizon, a.output_dt)
    meta = dict(
        backend="optimized_dp",
        config=config.to_dict(),
        grid=grid.to_dict(),
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
        tube_projection="min over every internal update, then max with avoid",
        python=sys.version,
        arithmetic="upstream float32 fields; upstream NumPy host integration",
    )
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    start = time.perf_counter()
    target, avoid = target_and_avoid(grid, config)
    minimum = np.array(
        [-grid.extent, -grid.extent, -np.pi, -grid.extent, -grid.extent, -np.pi]
    )
    maximum = np.array(
        [grid.extent, grid.extent, np.pi, grid.extent, grid.extent, np.pi]
    )
    g = Grid(minimum, maximum, 6, list(grid.shape), periodicDims=[2, 5])
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
    main()
