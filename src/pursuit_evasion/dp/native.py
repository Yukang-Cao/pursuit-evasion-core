"""Compile and run the inspectable native square solver, with explicit output paths."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .grid import GridSpec, output_times
from ..model import GameConfig
from .options import validate_options


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
    validate_options(horizon, output_dt, order, cfl, threads)
    output = Path(output).resolve()
    if (output / "meta.json").exists():
        raise FileExistsError(f"Preserve completed solve: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("native_square.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = (
        Path(
            os.environ.get(
                "PE_BUILD_CACHE", Path.home() / ".cache" / "pursuit-evasion-core"
            )
        )
        / digest
    )
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "square6d"
    t = time.perf_counter()
    compile_seconds = 0.0
    if not binary.exists():
        subprocess.run(
            ["g++", "-O3", "-std=c++17", "-fopenmp", str(source), "-o", str(binary)],
            check=True,
        )
        compile_seconds = time.perf_counter() - t
    options = dict(
        positions=grid.positions,
        headings=grid.headings,
        extent=grid.extent,
        arena=config.arena_half_width,
        radius=config.capture_radius,
        vp=config.pursuer.speed,
        ve=config.evader.speed,
        wp=config.pursuer.max_turn_rate,
        we=config.evader.max_turn_rate,
        horizon=horizon,
        **{"output-dt": output_dt},
        order=order,
        cfl=cfl,
        threads=threads,
        history=int(history),
        output=str(output),
    )
    command = [str(binary)] + [
        str(v) for key, value in options.items() for v in ("--" + key, value)
    ]
    meta = dict(
        backend="native_cpp",
        config=config.to_dict(),
        grid=grid.to_dict(),
        horizon=horizon,
        output_dt=output_dt,
        times=output_times(horizon, output_dt).tolist(),
        order=order,
        cfl=cfl,
        threads=threads,
        history_saved=history,
        dtype="float32",
        source_sha256=digest,
        compile_seconds=compile_seconds,
        command=command,
        target="capture_or_evader_boundary_fault",
        avoid="pursuer_boundary_fault",
        tube_projection="min over every internal update, then max with avoid",
        status="running",
    )
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (output / "solver.log").open("w") as log:
        t = time.perf_counter()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    meta.update(
        subprocess_seconds=time.perf_counter() - t,
        status="complete" if result.returncode == 0 else "failed",
    )
    if result.returncode == 0:
        meta.update(json.loads((output / "native_timing.json").read_text()))
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"Native solve failed; see {output/'solver.log'}")
    return meta
