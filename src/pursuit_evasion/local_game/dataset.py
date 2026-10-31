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

    The accumulated input combines the root scan with two scans at collision-free
    successor poses from a stopped root. This is synthetic short-memory
    augmentation, not a recorded continuous round trip. Every added cell is
    visible from one of those scan poses; runtime memory uses the actual chase.
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
        input_scope="Single noiseless scan and synthetic short memory from two collision-free successor scan poses; not a recorded round trip; no full-map geometry copied into observed inputs; ideal SDF separate",
        scalar_clearance="15-point rectangular stencil in the observed SDF, not privileged physical clearance",
    )
    (folder / "observations_meta.json").write_text(json.dumps(summary, indent=2))
    return summary


def sample_control_successors(
    original_folder,
    reference_folder,
    output_folder,
    *,
    seed=1031,
    group_size=8,
    max_frame_attempts=3000,
):
    """Label legal one-command successors in unchanged P-centered observations.

    The original dataset supplies observation poses and candidate E poses. Root
    speeds are sampled continuously, subject to the shared braking constraint.
    Each accepted root contributes up to group_size admissible P/E successor
    pairs. Query count, images, reference physics and time budgets stay fixed.
    This changes the supervision distribution; it does not train a policy.

    A retained reference history is required. Output must be a different folder
    so that the original experiment remains reproducible. Labels use the same
    continuous history-query API as the numerical controller.
    """
    from .reference import ContactReference
    from .control import admissible_controls
    from ..reversible.model import joint_step

    original_folder = Path(original_folder).resolve()
    reference_folder = Path(reference_folder).resolve()
    output_folder = Path(output_folder).resolve()
    if output_folder in {original_folder, reference_folder}:
        raise ValueError(
            "Successor data must not overwrite the original dataset/reference"
        )
    if group_size < 1 or max_frame_attempts < 1:
        raise ValueError("Sampling limits must be positive")
    output_folder.mkdir(parents=True, exist_ok=True)
    if any(output_folder.iterdir()):
        raise ValueError("Use an empty output directory for a new sampling run")
    reference = ContactReference(reference_folder)
    original_meta = json.loads((original_folder / "meta.json").read_text())
    original_queries = np.load(original_folder / "queries.npz")
    frames = original_queries["frames"]
    if not np.all(frames[:, 3] == 0):
        raise ValueError("This sampler requires P-centered observation frames")
    counts = np.bincount(original_queries["frame_ids"], minlength=len(frames))
    if len(np.unique(counts)) != 1 or counts[0] < 1:
        raise ValueError("Expected an equal positive query count per frame")
    queries_per_frame = int(counts[0])
    if original_meta["spec"]["config"] != reference.config.to_dict():
        raise ValueError("Original dataset and reference physics differ")
    if original_meta["spec"]["arena"] != reference.arena.to_dict():
        raise ValueError("Original dataset and reference maps/footprints differ")
    reference_meta = json.loads((reference_folder / "meta.json").read_text())
    numerical_keys = [
        "headings",
        "position_step",
        "horizon",
        "output_dt",
        "exit_radius",
    ]
    if original_meta["times"] != reference_meta["times"] or any(
        original_meta["spec"][key] != reference_meta["spec"][key]
        for key in numerical_keys
    ):
        raise ValueError("Original dataset and reference grids/budgets differ")
    rng = np.random.default_rng(seed)
    dt = reference.config.control_dt
    engines = [
        GeometryEngine(
            reference.arena,
            getattr(reference.config, role),
            role,
            threads=1,
            control_dt=dt,
        )
        for role in ["pursuer", "evader"]
    ]
    states, roots, p_commands, e_commands, frame_ids, attempts = [], [], [], [], [], []
    started = time.perf_counter()
    for frame_id, observation_pose in enumerate(frames):
        pool = original_queries["states"][original_queries["frame_ids"] == frame_id]
        collected = 0
        tries = 0
        while collected < queries_per_frame:
            tries += 1
            if tries > max_frame_attempts:
                raise RuntimeError(
                    f"Insufficient legal roots at frame {frame_id}: {collected} queries"
                )
            root = pool[rng.integers(len(pool))].copy()
            root[:3] = observation_pose[:3]
            for offset, limits in [
                (0, reference.config.pursuer),
                (4, reference.config.evader),
            ]:
                root[offset + 3] = rng.uniform(limits.speed_min, limits.speed_max)
            if (
                reference.arena.capture_value(root) <= 0
                or not engines[0].backup_safe(root[:4])
                or not engines[1].backup_safe(root[4:])
            ):
                continue
            P, _ = admissible_controls(root[:4], engines[0], dt)
            E, _ = admissible_controls(root[4:], engines[1], dt)
            pi, ei = np.meshgrid(np.arange(len(P)), np.arange(len(E)), indexing="ij")
            pi, ei = pi.ravel(), ei.ravel()
            children = joint_step(
                np.broadcast_to(root, (len(pi), 8)), P[pi], E[ei], dt, reference.config
            )
            inside = (
                np.linalg.norm(children[:, :2] - children[:, 4:6], axis=1)
                <= reference.domain.exit_radius
            )
            selected = rng.permutation(np.flatnonzero(inside))[
                : min(group_size, queries_per_frame - collected)
            ]
            if not len(selected):
                continue
            _, supported, _ = reference.fields(children[selected])
            selected = selected[supported]
            if not len(selected):
                continue
            states.extend(children[selected])
            roots.extend(np.tile(root, (len(selected), 1)))
            p_commands.extend(P[pi[selected]])
            e_commands.extend(E[ei[selected]])
            frame_ids.extend([frame_id] * len(selected))
            collected += len(selected)
        attempts.append(tries)
    states = np.asarray(states)
    values = []
    for first in range(0, len(states), 2048):
        field, supported, _ = reference.fields(states[first : first + 2048])
        if not supported.all() or not np.isfinite(field).all():
            raise RuntimeError(
                "A selected training query lacks a finite supported field"
            )
        values.append(field)
    np.savez_compressed(
        output_folder / "queries.npz",
        frames=frames,
        frame_ids=np.asarray(frame_ids),
        states=states,
        continuous=True,
    )
    np.savez_compressed(
        output_folder / "labels.npz", values=np.concatenate(values, axis=1)
    )
    np.savez_compressed(
        output_folder / "generation_controls.npz",
        roots=np.asarray(roots),
        P_controls=np.asarray(p_commands),
        E_controls=np.asarray(e_commands),
    )
    original_meta["sampling_override"] = {
        "method": "legal control successors with continuous initial speeds",
        "seed": seed,
        "group_size": group_size,
    }
    (output_folder / "meta.json").write_text(json.dumps(original_meta, indent=2))
    build_observations(output_folder)
    predicted = joint_step(
        np.asarray(roots),
        np.asarray(p_commands),
        np.asarray(e_commands),
        dt,
        reference.config,
    )
    error = float(abs(predicted - states).max())
    if error >= 1e-10:
        raise RuntimeError(
            "Generated successors disagree with held-command propagation"
        )
    before = np.load(original_folder / "observations.npz")
    after = np.load(output_folder / "observations.npz")
    same_images = all(
        np.array_equal(before[key], after[key])
        for key in ["partial", "memory", "ideal_sdf", "frames"]
    )
    if not same_images:
        raise RuntimeError(
            "Observation images changed during a query-only data experiment"
        )
    summary = {
        "status": "complete",
        "queries": len(states),
        "frames": len(frames),
        "generation_seconds": time.perf_counter() - started,
        "max_transition_error": error,
        "images_identical": same_images,
        "max_root_attempts": max(attempts),
        "seed": seed,
        "label_source": "unchanged numerical reference through continuous history reader",
    }
    (output_folder / "generation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary
