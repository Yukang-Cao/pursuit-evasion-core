"""Build local sensor observations for sampled contact-game reference queries.

The numerical reference supplies labels. This module supplies only observations
that could have been obtained from scans, plus a separately named ideal-map input.
"""

from pathlib import Path
import json
import time
import numpy as np

from .geometry import CircleArena
from .native import GeometryEngine
from .perception import LidarMemory, SensorSettings, observed_clearances
from ..reversible.model import ReversibleConfig


def build_observations(folder, settings=SensorSettings()):
    """Write single-scan, accumulated-scan and ideal-map inputs beside labels.

    The accumulated input uses two collision-free excursions from a stopped
    observation pose, then returns to the original pose. It is a small training
    approximation to a sensor history; it does not reveal hidden map cells.
    """
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    arena = CircleArena.from_dict(meta["spec"]["arena"])
    config = ReversibleConfig.from_dict(meta["spec"]["config"])
    queries = np.load(folder / "queries.npz")
    frames = queries["frames"]
    partial, history, ideal = [], [], []
    local = np.empty_like(queries["states"])
    geometries = {
        name: np.empty((len(local), 3), np.float32)
        for name in ["partial", "memory", "ideal"]
    }
    started = time.perf_counter()
    for k, observation_pose in enumerate(frames):
        memory = LidarMemory(arena, settings)
        memory.observe(observation_pose[:3])
        image, frame = memory.image(observation_pose[:3])
        complete, _ = memory.image(observation_pose[:3], ideal=True)
        role = "pursuer" if int(observation_pose[3]) == 0 else "evader"
        limits = getattr(config, role)
        engine = GeometryEngine(
            arena, limits, role, threads=1, control_dt=config.control_dt
        )
        commands = np.array(
            [
                [a, kappa]
                for a in [-limits.acceleration, limits.acceleration]
                for kappa in [-limits.max_curvature, 0, limits.max_curvature]
            ]
        )
        valid, poses = engine.actions(np.r_[observation_pose[:3], 0.0], commands, 0.6)
        candidates = np.flatnonzero(valid[0])
        rng = np.random.default_rng(1031 + k)
        for j in rng.permutation(candidates)[:2]:
            memory.observe(poses[0, j])
        memory.observe(observation_pose[:3])
        accumulated, _ = memory.image(observation_pose[:3])
        ids = np.flatnonzero(queries["frame_ids"] == k)
        states = queries["states"][ids].astype(float)
        z = frame.to_local(states)
        local[ids] = z
        phi = arena.capture_value(states)
        for name, observation in [
            ("partial", image),
            ("memory", accumulated),
            ("ideal", complete),
        ]:
            pc, ec = observed_clearances(
                observation, z, settings.extent, arena.pursuer, arena.evader
            )
            geometries[name][ids] = np.stack([phi, pc, ec], -1)
        partial.append(image.astype(np.float16))
        history.append(accumulated.astype(np.float16))
        ideal.append(complete[0].astype(np.float16))
        if k % 32 == 0:
            print(
                f"observation {k}/{len(frames)}; {time.perf_counter()-started:.1f} s",
                flush=True,
            )
    np.savez_compressed(
        folder / "observations.npz",
        partial=np.stack(partial),
        memory=np.stack(history),
        ideal_sdf=np.stack(ideal),
        local_states=local,
        geometry_partial=geometries["partial"],
        geometry_memory=geometries["memory"],
        geometry_ideal=geometries["ideal"],
        frame_ids=queries["frame_ids"],
        frames=frames,
    )
    summary = dict(
        version=2,
        sensor=settings.__dict__,
        seconds=time.perf_counter() - started,
        input_scope="Single noiseless scan and accumulated scans from two collision-free reversible excursions; no full-map geometry copied into observed inputs; ideal SDF separate",
        scalar_clearance="15-point rectangular stencil in the observed SDF, not privileged physical clearance",
    )
    (folder / "observations_meta.json").write_text(json.dumps(summary, indent=2))
    return summary
