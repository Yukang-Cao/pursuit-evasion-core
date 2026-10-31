"""Optional OpenMP geometry acceleration, using the shared held-command model."""

from pathlib import Path
import ctypes, hashlib, subprocess, tempfile, os
import numpy as np

_LIB = None


def library():
    global _LIB
    if _LIB is not None:
        return _LIB
    src = Path(__file__).with_name("native_geometry.cpp")
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    output = Path(tempfile.gettempdir()) / ("pe-local-geometry-" + digest + ".so")
    if not output.exists():
        fd, temp = tempfile.mkstemp(suffix=".so")
        os.close(fd)
        try:
            subprocess.run(
                [
                    "g++",
                    "-O3",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    "-fopenmp",
                    str(src),
                    "-o",
                    temp,
                ],
                check=True,
            )
            os.replace(temp, output)
        finally:
            if Path(temp).exists():
                Path(temp).unlink()
    lib = ctypes.CDLL(str(output))
    D = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    U = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    lib.batch_clearance.argtypes = [
        D,
        ctypes.c_long,
        D,
        ctypes.c_int,
        ctypes.c_double,
        D,
        D,
        ctypes.c_int,
    ]
    lib.batch_backups.argtypes = [
        D,
        ctypes.c_long,
        D,
        ctypes.c_int,
        ctypes.c_double,
        D,
        U,
        ctypes.c_int,
    ]
    lib.batch_actions.argtypes = [
        D,
        ctypes.c_long,
        D,
        ctypes.c_int,
        ctypes.c_double,
        D,
        ctypes.c_int,
        ctypes.c_double,
        D,
        U,
        D,
        ctypes.c_int,
    ]
    lib.pair_capture_times.argtypes = [
        D,
        D,
        ctypes.c_int,
        D,
        ctypes.c_int,
        ctypes.c_double,
        D,
        D,
        D,
        ctypes.c_int,
    ]
    lib.lidar_ranges.argtypes = [
        D,
        D,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        D,
    ]
    lib.configuration_connections.argtypes = [
        D,
        D,
        ctypes.c_long,
        D,
        ctypes.c_int,
        ctypes.c_double,
        D,
        U,
        ctypes.c_int,
    ]
    for name in [
        "configuration_connections",
        "batch_clearance",
        "batch_backups",
        "batch_actions",
        "pair_capture_times",
        "lidar_ranges",
    ]:
        getattr(lib, name).restype = None
    _LIB = lib
    return lib


class GeometryEngine:
    def __init__(self, arena, limits, role, threads=6, control_dt=0.05):
        self.arena = arena
        self.limits = limits
        self.role = role
        self.threads = threads
        b = getattr(arena, role)
        self.params = np.array(
            [
                limits.speed_min,
                limits.speed_max,
                limits.acceleration,
                limits.max_curvature,
                b.length,
                b.width,
                control_dt,
            ],
            float,
        )
        self.obstacles = np.ascontiguousarray(arena.circles)
        self.lib = library()

    def _states(self, s):
        s = np.asarray(s, float)
        if s.shape[-1] != 4 or not np.isfinite(s).all():
            raise ValueError("Finite vehicle states[...,4] required")
        return np.ascontiguousarray(s.reshape(-1, 4))

    def clearance(self, s):
        x = self._states(s)
        out = np.empty(len(x))
        self.lib.batch_clearance(
            x,
            len(x),
            self.obstacles,
            len(self.obstacles),
            self.arena.half_width,
            self.params,
            out,
            self.threads,
        )
        return out.reshape(np.shape(s)[:-1])

    def backup_safe(self, s):
        x = self._states(s)
        out = np.empty(len(x), np.uint8)
        self.lib.batch_backups(
            x,
            len(x),
            self.obstacles,
            len(self.obstacles),
            self.arena.half_width,
            self.params,
            out,
            self.threads,
        )
        return out.astype(bool).reshape(np.shape(s)[:-1])

    def connections(self, start, target):
        a = self._states(start)
        b = self._states(target)
        if a.shape != b.shape:
            raise ValueError("Matched configuration arrays required")
        out = np.empty(len(a), np.uint8)
        self.lib.configuration_connections(
            a,
            b,
            len(a),
            self.obstacles,
            len(self.obstacles),
            self.arena.half_width,
            self.params,
            out,
            self.threads,
        )
        return out.astype(bool)

    def actions(self, s, controls, dt):
        x = self._states(s)
        u = np.ascontiguousarray(controls, float).reshape(-1, 2)
        ok = np.empty((len(x), len(u)), np.uint8)
        nxt = np.empty((len(x), len(u), 4))
        self.lib.batch_actions(
            x,
            len(x),
            u,
            len(u),
            dt,
            self.obstacles,
            len(self.obstacles),
            self.arena.half_width,
            self.params,
            ok,
            nxt,
            self.threads,
        )
        return ok.astype(bool), nxt


def capture_times(state, p_controls, e_controls, dt, p_engine, e_engine):
    s = np.ascontiguousarray(state, float)
    p = np.ascontiguousarray(p_controls, float)
    e = np.ascontiguousarray(e_controls, float)
    out = np.empty((len(p), len(e)))
    library().pair_capture_times(
        s, p, len(p), e, len(e), dt, p_engine.params, e_engine.params, out, 1
    )
    return out


def scan_ranges(arena, pose, beams=720, max_range=9.0):
    pose = np.ascontiguousarray(pose, float)
    out = np.empty(beams)
    library().lidar_ranges(
        pose,
        np.ascontiguousarray(arena.circles),
        len(arena.circles),
        arena.half_width,
        beams,
        max_range,
        out,
    )
    return out
