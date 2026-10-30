"""Optional native evaluator for the unchanged free-space MPPI objective."""

from functools import lru_cache
from pathlib import Path
import ctypes
import hashlib
import os
import subprocess
import tempfile
import numpy as np


@lru_cache(maxsize=1)
def _kernel():
    source = Path(__file__).with_name("native_nav_cost.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    folder = (
        Path(
            os.environ.get(
                "PE_BUILD_CACHE", Path.home() / ".cache/pursuit-evasion-core"
            )
        )
        / digest
    )
    folder.mkdir(parents=True, exist_ok=True)
    library = folder / "navigation_cost.so"
    if not library.exists():
        # Atomic publication permits independent worker processes to initialize.
        fd, temp = tempfile.mkstemp(suffix=".so", dir=folder)
        os.close(fd)
        try:
            subprocess.run(
                [
                    "g++",
                    "-O3",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    temp,
                ],
                check=True,
                capture_output=True,
            )
            os.replace(temp, library)
        finally:
            if Path(temp).exists():
                Path(temp).unlink()
    lib = ctypes.CDLL(str(library))
    fn = lib.pe_free_costs
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    fn.argtypes = [array, array, array, array, ctypes.c_int, ctypes.c_int] + [
        ctypes.c_double
    ] * 6
    fn.restype = None
    return lib, fn


def costs(pose, commands, target, limits, settings, dt):
    pose = np.ascontiguousarray(pose, dtype=np.float64)
    commands = np.ascontiguousarray(commands, dtype=np.float64)
    target = np.ascontiguousarray(target, dtype=np.float64)
    if (
        pose.shape != (4,)
        or commands.ndim != 3
        or commands.shape[-1] != 2
        or target.shape != (2,)
    ):
        raise ValueError("Invalid free-space cost shapes")
    if (
        not np.isfinite(pose).all()
        or not np.isfinite(commands).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Inputs must be finite")
    if pose[3] < limits.speed_min - 1e-9 or pose[3] > limits.speed_max + 1e-9:
        raise ValueError("Invalid initial speed")
    if np.any(abs(commands[..., 0]) > limits.acceleration + 1e-9) or np.any(
        abs(commands[..., 1]) > limits.max_curvature + 1e-9
    ):
        raise ValueError("Invalid control")
    result = np.empty(len(commands))
    _, fn = _kernel()
    fn(
        pose,
        commands,
        target,
        result,
        len(commands),
        commands.shape[1],
        dt,
        limits.speed_min,
        limits.speed_max,
        settings.goal_tolerance**2,
        settings.distance_weight,
        settings.terminal_weight,
    )
    return result
