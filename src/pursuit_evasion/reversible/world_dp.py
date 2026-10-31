"""Dense 8D square-game HJI solve and explicit resource accounting.

This extends the inspectable native 6D backend to signed-speed states. It is
not a claim that dense grids are the only representation of an 8D game.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .model import ReversibleConfig

STATE_ORDER = ["xP", "yP", "thetaP", "vP", "xE", "yE", "thetaE", "vE"]
PERIODIC_AXES = (2, 6)


@dataclass(frozen=True)
class WorldGrid:
    positions: int = 9
    headings: int = 8
    p_speeds: int = 7
    e_speeds: int = 7
    extent: float = 3.5

    def __post_init__(self):
        for name, minimum in [
            ("positions", 5),
            ("headings", 4),
            ("p_speeds", 3),
            ("e_speeds", 3),
        ]:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not np.isfinite(self.extent) or self.extent <= 0:
            raise ValueError("extent must be finite and positive.")

    @property
    def shape(self):
        n, h = self.positions, self.headings
        return (n, n, h, self.p_speeds, n, n, h, self.e_speeds)

    @property
    def states(self):
        return math.prod(self.shape)

    def axes(self, config):
        x = np.linspace(-self.extent, self.extent, self.positions)
        theta = -np.pi + np.arange(self.headings) * 2 * np.pi / self.headings
        vp = np.linspace(
            config.pursuer.speed_min, config.pursuer.speed_max, self.p_speeds
        )
        ve = np.linspace(
            config.evader.speed_min, config.evader.speed_max, self.e_speeds
        )
        return (x, x, theta, vp, x, x, theta, ve)

    def spacings(self, config):
        return tuple(float(a[1] - a[0]) for a in self.axes(config))

    def to_dict(self, config):
        axis = self.axes(config)[0]
        wall_offsets = [
            float(np.min(abs(axis - wall)))
            for wall in (-config.arena_half_width, config.arena_half_width)
        ]
        return dict(
            **asdict(self),
            shape=list(self.shape),
            states=self.states,
            spacings=list(self.spacings(config)),
            periodic_axes=list(PERIODIC_AXES),
            state_order=STATE_ORDER,
            axis_bounds=[[float(a[0]), float(a[-1])] for a in self.axes(config)],
            physical_wall_grid_offsets=wall_offsets,
            physical_walls_on_nodes=max(wall_offsets) < 1e-8,
        )


def output_times(horizon, output_dt):
    if not np.isfinite([horizon, output_dt]).all() or min(horizon, output_dt) <= 0:
        raise ValueError("Positive finite horizon and output_dt are required.")
    return np.unique(
        np.minimum(
            np.arange(int(np.ceil(horizon / output_dt)) + 1) * output_dt, horizon
        )
    )


def global_rate_bound(config, grid):
    dx, _, dh, dvp, _, _, _, dve = grid.spacings(config)
    p, e = config.pursuer, config.evader
    vp, ve = max(abs(p.speed_min), abs(p.speed_max)), max(
        abs(e.speed_min), abs(e.speed_max)
    )
    return (
        2 * (vp + ve) / dx
        + (vp * p.max_curvature + ve * e.max_curvature) / dh
        + p.acceleration / dvp
        + e.acceleration / dve
    )


def surfaces(config, grid, dtype=np.float32):
    """Capture-or-E-fault target, P-fault avoid; testing-sized grids only."""
    if grid.extent <= config.arena_half_width:
        raise ValueError("World grid needs an exterior padding band.")
    axes = grid.axes(config)
    shaped = []
    for k in (0, 1, 4, 5):
        dims = [1] * 8
        dims[k] = len(axes[k])
        shaped.append(np.asarray(axes[k], dtype=dtype).reshape(dims))
    xp, yp, xe, ye = shaped
    cap = np.hypot(xp - xe, yp - ye) - config.capture_radius
    e_fault = config.arena_half_width - np.maximum(abs(xe), abs(ye))
    p_fault = np.maximum(abs(xp), abs(yp)) - config.arena_half_width + 1e-6
    return (
        np.broadcast_to(np.minimum(cap, e_fault), grid.shape).copy(),
        np.broadcast_to(p_fault, grid.shape).copy(),
    )


def hamiltonian(states, gradients, config):
    """Analytic min-P/max-E Hamiltonian; speed bounds restrict acceleration."""
    s, g = np.asarray(states), np.asarray(gradients)
    p, e = config.pursuer, config.evader
    result = s[..., 3] * (np.cos(s[..., 2]) * g[..., 0] + np.sin(s[..., 2]) * g[..., 1])
    result += s[..., 7] * (
        np.cos(s[..., 6]) * g[..., 4] + np.sin(s[..., 6]) * g[..., 5]
    )
    result -= abs(s[..., 3]) * p.max_curvature * abs(g[..., 2])
    result += abs(s[..., 7]) * e.max_curvature * abs(g[..., 6])
    for k, v, limits, minimize in ((3, s[..., 3], p, True), (7, s[..., 7], e, False)):
        lo = np.where(v <= limits.speed_min + 1e-10, 0, -limits.acceleration)
        hi = np.where(v >= limits.speed_max - 1e-10, 0, limits.acceleration)
        result += (np.minimum if minimize else np.maximum)(
            lo * g[..., k], hi * g[..., k]
        )
    return result


def _compile():
    source = Path(__file__).with_name("native_world.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    root = Path(
        os.environ.get(
            "PE_BUILD_CACHE", Path.home() / ".cache" / "pursuit-evasion-core"
        )
    )
    folder = root / digest
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "reversible_world8d"
    begin = time.perf_counter()
    if not binary.exists():
        subprocess.run(
            ["g++", "-O3", "-std=c++17", "-fopenmp", str(source), "-o", str(binary)],
            check=True,
        )
        duration = time.perf_counter() - begin
    else:
        duration = 0.0
    return binary, digest, duration


def solve(
    config,
    grid,
    output,
    *,
    horizon=0.5,
    output_dt=0.25,
    order=2,
    cfl=0.8,
    threads=8,
    history=False,
    max_core_gb=12.0,
    max_output_gb=30.0,
    probes=None,
    speed_boundary="linear",
    resume_from=None,
    write_final=True,
    write_arrival=True,
):
    """Solve the fixed square, retaining history only when explicitly requested.

    probes may be an (N,8) array: signed values are stored at output times using
    multilinear spatial interpolation. They do not influence the computation.
    """
    if not isinstance(config, ReversibleConfig):
        raise TypeError("Use the reversible physical model.")
    if not config.point_vehicles:
        raise ValueError(
            "This square backend uses points; use the footprint-map backend."
        )
    if getattr(config, "obstacles", ()):
        raise ValueError(
            "This native 8D backend currently supports the square boundary only."
        )
    if grid.extent <= config.arena_half_width:
        raise ValueError("Numerical domain must include exterior padding.")
    if speed_boundary not in ("linear", "legacy_signed"):
        raise ValueError("Speed boundary must be linear or legacy_signed.")
    if (
        order not in (1, 2)
        or not 0 < cfl <= 1
        or not isinstance(threads, int)
        or threads < 1
    ):
        raise ValueError("Require order1/2, CFL in(0,1], positive integer threads.")
    parent = None
    parent_path = None
    start_time = 0.0
    if resume_from is not None:
        parent_path = Path(resume_from).resolve()
        parent = json.loads((parent_path / "meta.json").read_text())
        if (
            parent.get("status") != "complete"
            or parent.get("backend") != "native_world8d"
        ):
            raise ValueError("Continuation requires a completed native world table.")
        if parent["config"] != config.to_dict():
            raise ValueError("Continuation physics/map must match the parent.")
        signature = ("positions", "headings", "p_speeds", "e_speeds", "extent")
        if any(parent["grid"][k] != getattr(grid, k) for k in signature):
            raise ValueError("Continuation grid must match the parent.")
        if (
            parent.get("speed_boundary", "legacy_signed") != speed_boundary
            or parent["order"] != order
            or parent["cfl"] != cfl
        ):
            raise ValueError("Continuation numerical scheme must match the parent.")
        if history and not parent["history_saved"]:
            raise ValueError(
                "A continued query history requires the retained parent history."
            )
        start_time = float(parent["horizon"])
        if horizon <= start_time:
            raise ValueError("Continuation horizon must exceed the parent's horizon.")
    if not history and not write_final:
        raise ValueError("Retain history or a final field.")
    times = output_times(horizon - start_time, output_dt) + start_time
    if parent is not None:
        times = times[1:]  # Parent retains the shared boundary field.
    core_bytes = grid.states * 12
    output_bytes = (
        grid.states
        * 4
        * (int(write_final) + int(write_arrival) + (len(times) if history else 0))
    )
    if (
        not np.isfinite([max_core_gb, max_output_gb]).all()
        or min(max_core_gb, max_output_gb) <= 0
    ):
        raise ValueError("Positive explicit resource limits are required.")
    if core_bytes > max_core_gb * 1e9 or output_bytes > max_output_gb * 1e9:
        raise MemoryError(
            f"Requested core {core_bytes/1e9:.3f}GB, outputs {output_bytes/1e9:.3f}GB exceed limits."
        )
    output = Path(output).resolve()
    if (output / "meta.json").exists():
        raise FileExistsError(f"Preserve existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    binary, digest, compile_seconds = _compile()
    p, e = config.pursuer, config.evader
    options = dict(
        positions=grid.positions,
        headings=grid.headings,
        **{"p-speeds": grid.p_speeds, "e-speeds": grid.e_speeds},
        extent=grid.extent,
        arena=config.arena_half_width,
        radius=config.capture_radius,
        **{
            "p-min": p.speed_min,
            "p-max": p.speed_max,
            "e-min": e.speed_min,
            "e-max": e.speed_max,
            "p-accel": p.acceleration,
            "e-accel": e.acceleration,
            "p-curvature": p.max_curvature,
            "e-curvature": e.max_curvature,
        },
        horizon=horizon,
        **{"output-dt": output_dt},
        order=order,
        cfl=cfl,
        threads=threads,
        history=int(history),
        output=str(output),
        **{"linear-speed-boundary": int(speed_boundary == "linear")},
        **{
            "start-time": start_time,
            "write-final": int(write_final),
            "write-arrival": int(write_arrival),
        },
    )
    if parent is not None:
        if parent.get("final_source", "final.f32") == "history_last":
            initial_path = parent_path / "fields.f32"
            initial_offset = (len(parent["times"]) - 1) * grid.states * 4
        else:
            initial_path = parent_path / "final.f32"
            initial_offset = 0
        if initial_path.stat().st_size < initial_offset + grid.states * 4:
            raise ValueError("Parent continuation field is incomplete.")
        options.update(
            {"initial-field": str(initial_path), "initial-offset": initial_offset}
        )
        if (parent_path / "arrival.f32").exists():
            options["initial-arrival"] = str(parent_path / "arrival.f32")
    probe_count = 0
    if probes is not None:
        from .world_table import interpolation_stencil

        queries = np.asarray(probes, dtype=float).reshape(-1, 8)
        ii, ww, valid, _ = interpolation_stencil(
            queries, grid.axes(config), PERIODIC_AXES
        )
        if not valid.all():
            raise ValueError("All probe states must lie in the numerical domain.")
        probe_count = len(queries)
        ii.astype(np.uint64).tofile(output / "probe_indices.u64")
        ww.astype(np.float64).tofile(output / "probe_weights.f64")
        queries.tofile(output / "probe_states.f64")
        options.update(
            {
                "probe-count": probe_count,
                "probe-indices": str(output / "probe_indices.u64"),
                "probe-weights": str(output / "probe_weights.f64"),
            }
        )
    command = [str(binary)] + [
        str(x) for k, v in options.items() for x in ("--" + k, v)
    ]
    meta = dict(
        backend="native_world8d",
        status="running",
        config=config.to_dict(),
        grid=grid.to_dict(config),
        horizon=horizon,
        output_dt=output_dt,
        times=times.tolist(),
        order=order,
        cfl=cfl,
        threads=threads,
        history_saved=history,
        dtype="float32",
        layout="C",
        target="capture_or_evader_boundary_fault",
        avoid="pursuer_boundary_fault",
        value_meaning="numerical_time_to_P_success_not_unconditionally_physical_capture",
        tube_projection="min over completed numerical updates, then max with avoid",
        source_sha256=digest,
        compile_seconds=compile_seconds,
        command=command,
        predicted_core_bytes=core_bytes,
        predicted_output_bytes=output_bytes,
        probe_count=probe_count,
        peak_rss_scope="native solver subprocess; compiler process measured separately",
        speed_boundary=speed_boundary,
        start_time=start_time,
        final_source="final.f32" if write_final else "history_last",
        arrival_retained=bool(write_arrival),
    )
    if parent is not None:
        meta["parent_table"] = os.path.relpath(parent_path, output)
        meta["parent_source_sha256"] = parent["source_sha256"]
        meta["continuation_note"] = (
            "Parent signed history is referenced, not copied; output times contain only new frames."
        )
    (output / "native_source.cpp").write_bytes(
        Path(__file__).with_name("native_world.cpp").read_bytes()
    )
    if parent is not None:
        required = ["meta.json", "fields.f32"]
        if parent.get("final_source", "final.f32") != "history_last":
            required.append("final.f32")
        (output / "dependencies.json").write_text(
            json.dumps(
                {
                    "parent_table": meta["parent_table"],
                    "required_parent_files_for_queries": required,
                    "recursive_parent_dependencies": bool(parent.get("parent_table")),
                    "instruction": "Retain the parent history while this continuation is in use.",
                },
                indent=2,
            )
            + "\n"
        )
    meta["source_snapshot"] = "native_source.cpp"
    path = output / "meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    start = time.perf_counter()
    with (output / "solver.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    meta.update(
        status="complete" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        subprocess_seconds=time.perf_counter() - start,
    )
    if result.returncode == 0:
        meta.update(json.loads((output / "native_timing.json").read_text()))
        if parent is not None:
            meta["cumulative_update_seconds"] = (
                parent.get("cumulative_update_seconds", parent["update_seconds"])
                + meta["update_seconds"]
            )
            meta["cumulative_internal_steps"] = (
                parent.get("cumulative_internal_steps", parent["internal_steps"])
                + meta["internal_steps"]
            )
    path.write_text(json.dumps(meta, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"8D solve failed; inspect internal log at {output}")
    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--positions", type=int, default=9)
    parser.add_argument("--headings", type=int, default=8)
    parser.add_argument("--speeds", type=int, default=7)
    parser.add_argument("--extent", type=float, default=3.5)
    parser.add_argument("--horizon", type=float, default=0.5)
    parser.add_argument("--output-dt", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--max-core-gb", type=float, default=12)
    parser.add_argument("--max-output-gb", type=float, default=30)
    a = parser.parse_args()
    m = solve(
        ReversibleConfig(),
        WorldGrid(a.positions, a.headings, a.speeds, a.speeds, a.extent),
        a.output,
        horizon=a.horizon,
        output_dt=a.output_dt,
        threads=a.threads,
        history=a.history,
        order=a.order,
        max_core_gb=a.max_core_gb,
        max_output_gb=a.max_output_gb,
    )
    print(
        json.dumps(
            {k: m[k] for k in ["status", "states", "update_seconds", "peak_rss_bytes"]},
            indent=2,
        )
    )
