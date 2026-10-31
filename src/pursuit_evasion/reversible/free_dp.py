"""Reversible 5D free-space game: an inspectable native HJI adapter.

This migrates the existing relative-grid scheme, with signed speeds, separate
curvature limits and explicit acceleration bounds. It is a numerical reference,
not a certificate. No world map or wall outcome enters the free-space target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


@dataclass(frozen=True)
class RelativeGrid:
    positions: int = 41
    headings: int = 24
    p_speeds: int = 7
    e_speeds: int = 7
    extent: float = 4.0

    def __post_init__(self):
        for name in ("positions", "headings", "p_speeds", "e_speeds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or int(value) != value
                or value < (4 if name == "headings" else 3)
            ):
                raise ValueError(f"Invalid {name}: {value}")
        if not np.isfinite(self.extent) or self.extent <= 0:
            raise ValueError("extent must be finite and positive")

    @property
    def shape(self):
        """C-contiguous storage order, with x fastest."""
        return (
            self.p_speeds,
            self.e_speeds,
            self.headings,
            self.positions,
            self.positions,
        )

    @property
    def states(self):
        return math.prod(self.shape)

    def axes(self, config):
        return (
            np.linspace(
                config.pursuer.speed_min, config.pursuer.speed_max, self.p_speeds
            ),
            np.linspace(
                config.evader.speed_min, config.evader.speed_max, self.e_speeds
            ),
            np.linspace(-np.pi, np.pi, self.headings, endpoint=False),
            np.linspace(-self.extent, self.extent, self.positions),
            np.linspace(-self.extent, self.extent, self.positions),
        )

    def to_dict(self):
        return asdict(self)


def output_times(horizon, output_dt):
    count = math.ceil(horizon / output_dt - 1e-10)
    return np.minimum(np.arange(count + 1) * output_dt, horizon)


def physics_signature(config):
    """Map placement and control cadence are not free-space PDE parameters."""
    result = {
        "capture_radius": float(config.capture_radius),
        "pursuer": {
            k: float(getattr(config.pursuer, k))
            for k in ("speed_min", "speed_max", "acceleration", "max_curvature")
        },
        "evader": {
            k: float(getattr(config.evader, k))
            for k in ("speed_min", "speed_max", "acceleration", "max_curvature")
        },
    }

    if config.capture_rule != "center_distance":
        result["capture_rule"] = config.capture_rule
    return result


def _validate(config, grid, horizon, output_dt, order, cfl, threads):
    values = [horizon, output_dt, cfl, config.capture_radius, grid.extent]
    for vehicle in (config.pursuer, config.evader):
        values.extend(
            [
                vehicle.speed_min,
                vehicle.speed_max,
                vehicle.acceleration,
                vehicle.max_curvature,
            ]
        )
        if (
            vehicle.speed_min >= vehicle.speed_max
            or vehicle.acceleration <= 0
            or vehicle.max_curvature <= 0
        ):
            raise ValueError("Invalid vehicle limits")
    if (
        not np.isfinite(values).all()
        or horizon <= 0
        or output_dt <= 0
        or not 0 < cfl <= 1
    ):
        raise ValueError("Invalid numeric solve options")
    if (
        not 0 < config.capture_radius < grid.extent
        or order not in (1, 2)
        or threads < 1
    ):
        raise ValueError("Invalid radius, order or thread count")


def solve(
    config,
    grid,
    output,
    *,
    horizon=2.0,
    output_dt=0.1,
    order=2,
    cfl=0.8,
    threads=4,
    history=True,
    max_working_gb=8.0,
    disk_reserve_gb=10.0,
):
    """Compute a new field directory, refusing overwrite and excessive allocation.

    max_working_gb is a per-run guard, not an accuracy or feasibility claim.
    History is optional for resource probes; online time queries require it.
    """
    solve_started = time.perf_counter()
    if config.capture_rule != "center_distance":
        raise ValueError("This legacy free-space solver uses circular capture")
    _validate(config, grid, horizon, output_dt, order, cfl, threads)
    if (
        not np.isfinite([max_working_gb, disk_reserve_gb]).all()
        or max_working_gb <= 0
        or disk_reserve_gb < 0
    ):
        raise ValueError("Invalid memory/disk guards")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve existing output: {output}")
    times = output_times(horizon, output_dt)
    working = grid.states * 12
    storage = grid.states * 4 * (2 + (len(times) if history else 0))
    if working > max_working_gb * 1e9:
        raise MemoryError(
            f"Core arrays need {working / 1e9:.3f} GB, above guard {max_working_gb}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    if storage + disk_reserve_gb * 1e9 > free:
        raise OSError(
            f"Output needs {storage / 1e9:.3f} GB while retaining {disk_reserve_gb} GB disk reserve"
        )
    output.mkdir(exist_ok=True)
    source = Path(__file__).with_name("native_relative.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "source.cpp").write_bytes(source.read_bytes())
    cache = (
        Path(
            os.environ.get(
                "PE_BUILD_CACHE", Path.home() / ".cache/pursuit-evasion-core"
            )
        )
        / digest
    )
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "relative5d"
    compile_seconds = 0.0
    if not binary.exists():
        start = time.perf_counter()
        subprocess.run(
            ["g++", "-O3", "-std=c++17", "-fopenmp", str(source), "-o", str(binary)],
            check=True,
        )
        compile_seconds = time.perf_counter() - start
    p, e = config.pursuer, config.evader
    options = {
        "positions": grid.positions,
        "headings": grid.headings,
        "p-speeds": grid.p_speeds,
        "e-speeds": grid.e_speeds,
        "extent": grid.extent,
        "radius": config.capture_radius,
        "p-min": p.speed_min,
        "p-max": p.speed_max,
        "e-min": e.speed_min,
        "e-max": e.speed_max,
        "p-acceleration": p.acceleration,
        "e-acceleration": e.acceleration,
        "p-curvature": p.max_curvature,
        "e-curvature": e.max_curvature,
        "horizon": horizon,
        "output-dt": output_dt,
        "order": order,
        "cfl": cfl,
        "threads": threads,
        "history": int(history),
        "output": str(output),
    }
    command = [str(binary)] + [
        str(x) for k, v in options.items() for x in ("--" + k, v)
    ]
    meta = {
        "schema": "reversible_relative5d_v1",
        "backend": "native_cpp",
        "local_precision": "float64",
        "config": config.to_dict(),
        "physics": physics_signature(config),
        "grid": grid.to_dict(),
        "shape": grid.shape,
        "state_order": ["x", "y", "psi", "vP", "vE"],
        "axis_order": ["vP", "vE", "psi", "y", "x"],
        "axes": [a.tolist() for a in grid.axes(config)],
        "horizon": horizon,
        "output_dt": output_dt,
        "times": times.tolist(),
        "order": order,
        "cfl": cfl,
        "threads": threads,
        "history_saved": bool(history),
        "dtype": "float32",
        "target": "physical_capture_only",
        "avoid": None,
        "tube_projection": "min over every internal update",
        "position_boundary": "one-sided linear extrapolation; outside queries unresolved",
        "speed_boundary": "acceleration restricted to inward or zero at speed limits",
        "heading_boundary": "periodic",
        "source_sha256": digest,
        "source_file": "source.cpp",
        "compile_seconds": compile_seconds,
        "command": command,
        "predicted_working_bytes": working,
        "predicted_output_bytes": storage,
        "status": "running",
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (output / "solver.log").open("w") as log:
        start = time.perf_counter()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    meta.update(
        status="complete" if result.returncode == 0 else "failed",
        subprocess_seconds=time.perf_counter() - start,
        returncode=result.returncode,
    )
    if result.returncode == 0:
        meta.update(json.loads((output / "timing.json").read_text()))
        if meta["frames"] != len(times):
            meta["status"] = "failed"
            meta["failure"] = "Output time count mismatch"
    meta["total_wall_seconds"] = time.perf_counter() - solve_started
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if meta["status"] != "complete":
        raise RuntimeError(f"Relative solve failed: {output / 'solver.log'}")
    return meta


def main():
    import argparse
    from .model import ReversibleConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--config")
    parser.add_argument("--positions", type=int, default=41)
    parser.add_argument("--headings", type=int, default=24)
    parser.add_argument("--p-speeds", type=int, default=7)
    parser.add_argument("--e-speeds", type=int, default=7)
    parser.add_argument("--extent", type=float, default=4)
    parser.add_argument("--horizon", type=float, default=2)
    parser.add_argument("--output-dt", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--max-working-gb", type=float, default=8)
    args = parser.parse_args()
    config = (
        ReversibleConfig.from_dict(json.loads(Path(args.config).read_text()))
        if args.config
        else ReversibleConfig()
    )
    grid = RelativeGrid(
        args.positions, args.headings, args.p_speeds, args.e_speeds, args.extent
    )
    result = solve(
        config,
        grid,
        args.output,
        horizon=args.horizon,
        output_dt=args.output_dt,
        threads=args.threads,
        order=args.order,
        history=not args.no_history,
        max_working_gb=args.max_working_gb,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "states", "update_seconds", "peak_rss_kib")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
