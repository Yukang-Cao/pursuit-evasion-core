"""Streaming reference solves and compact signed-history queries for contact games."""

from pathlib import Path
import json, time, os, hashlib, shutil
import numpy as np
from .grid_game import (
    make_grid,
    PairDomain,
    VehicleGrid,
    packed_gpu_arrays,
    initialize_packed,
    backup_packed,
)
from .geometry import CircleArena
from ..reversible.model import ReversibleConfig
from ..reversible.map_game_field import first_crossing


def write_half_preserving_sign(values, file):
    for start in range(0, len(values), 2_000_000):
        x = values[start : start + 2_000_000]
        y = x.astype(np.float16)
        y[(x > 0) & (y == 0)] = np.nextafter(np.float16(0), np.float16(1))
        y[(x < 0) & (y == 0)] = np.nextafter(np.float16(0), np.float16(-1))
        y.tofile(file)


def sample_queries(
    p, e, domain, arena, config, *, frames=192, per_frame=256, seed=1031
):
    from .native import GeometryEngine

    rng = np.random.default_rng(seed)
    observation = []
    pi = []
    ei = []
    frame_ids = []
    for f in range(frames):
        observer = 0
        own = p if observer == 0 else e
        other = e if observer == 0 else p
        for attempt in range(1000):
            i = int(rng.integers(len(own.states)))
            pose = own.states[i, :3].copy()
            nearby = np.flatnonzero(
                np.linalg.norm(other.states[:, :2] - pose[:2], axis=1)
                <= domain.exit_radius - 0.1
            )
            if len(nearby) < 30:
                continue
            j = rng.choice(nearby, per_frame * 3, replace=True)
            same = np.flatnonzero(np.all(own.states[:, :3] == pose, axis=1))
            ii = rng.choice(same, len(j), replace=True)
            pids, eids = (ii, j) if observer == 0 else (j, ii)
            q = np.concatenate([p.states[pids], e.states[eids]], 1)
            ok = arena.capture_value(q) > 0.03
            pids, eids = pids[ok][:per_frame], eids[ok][:per_frame]
            if len(pids) < per_frame:
                continue
            sensor = pose.copy()
            for _ in range(20):
                jitter = 0.11 if observer == 0 else 0.06
                sensor = pose + np.array(
                    [
                        rng.uniform(-jitter, jitter),
                        rng.uniform(-jitter, jitter),
                        rng.uniform(-0.11, 0.11),
                    ]
                )
                if (
                    arena.clearance(
                        np.r_[sensor, 0], "pursuer" if observer == 0 else "evader"
                    )
                    > 1e-4
                ):
                    break
            else:
                sensor = pose.copy()
            observation.append([*sensor, observer])
            pi.extend(pids)
            ei.extend(eids)
            frame_ids.extend([f] * per_frame)
            break
        else:
            raise ValueError("Could not sample a legal observation/query group")
    pi = np.array(pi, np.int32)
    ei = np.array(ei, np.int32)
    local = domain.lookup[domain.p_position[pi], ei]
    assert (local >= 0).all()
    compact = domain.offsets[pi] + local
    states = np.concatenate([p.states[pi], e.states[ei]], 1)
    from ..reversible.model import wrap_angle

    proposal = states.copy()
    continuous = rng.random(len(states)) < 0.5
    for k, limits in [(0, config.pursuer), (4, config.evader)]:
        proposal[:, k : k + 2] += rng.uniform(-0.12, 0.12, (len(states), 2))
        proposal[:, k + 2] = wrap_angle(
            proposal[:, k + 2] + rng.uniform(-0.15, 0.15, len(states))
        )
        proposal[:, k + 3] = np.clip(
            proposal[:, k + 3] + rng.uniform(-0.1, 0.1, len(states)),
            limits.speed_min,
            limits.speed_max,
        )
    pe = GeometryEngine(arena, config.pursuer, "pursuer", control_dt=config.control_dt)
    ee = GeometryEngine(arena, config.evader, "evader", control_dt=config.control_dt)
    legal = pe.backup_safe(proposal[:, :4]) & ee.backup_safe(proposal[:, 4:])
    legal &= (arena.capture_value(proposal) > 0.03) & (
        np.linalg.norm(proposal[:, :2] - proposal[:, 4:6], axis=1)
        < domain.exit_radius - 0.02
    )
    legal &= p.stencil(proposal[:, :4])[2] & e.stencil(proposal[:, 4:])[2]
    continuous &= legal
    states[continuous] = proposal[continuous]
    return dict(
        frames=np.array(observation),
        frame_ids=np.array(frame_ids, np.int32),
        p_indices=pi,
        e_indices=ei,
        original_compact_indices=compact,
        states=states,
        continuous=continuous,
    )


QUERY_CUDA = r"""
extern "C" __global__ void query_field(const float*f,const int*pi,const float*pw,const int*ei,const float*ew,const int*ppos,const int*lookup,const long long*offset,int n,int ne,float exit_value,float*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float total=0;
 for(int a=0;a<16;a++)if(pw[16*i+a]>0){int p=pi[16*i+a];float value=0;
  for(int b=0;b<16;b++)if(ew[16*i+b]>0){int e=ei[16*i+b],rank=lookup[(long long)ppos[p]*ne+e];value+=ew[16*i+b]*(rank<0?exit_value:f[offset[p]+rank]);}
  total+=pw[16*i+a]*value;
 }out[i]=total;
}
"""


def solve(
    config,
    arena,
    output,
    *,
    headings=16,
    position_step=0.3,
    horizon=2.0,
    output_dt=0.25,
    exit_radius=3.0,
    retain_history=False,
    frames=192,
    per_frame=256,
    seed=1031,
    max_gb=7.6,
):
    if (
        config.capture_rule != "rectangle_contact"
        or config.boundary_rule != "collision_constrained"
        or config.point_vehicles
    ):
        raise ValueError(
            "Use the explicit finite-body collision-constrained contact game"
        )
    import cupy as cp

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dt = config.control_dt
    steps = round(horizon / dt)
    every = round(output_dt / dt)
    if (
        min(steps, every) < 1
        or not np.isclose(steps * dt, horizon)
        or not np.isclose(every * dt, output_dt)
    ):
        raise ValueError("Budgets must be integer control periods")
    spec = dict(
        config=config.to_dict(),
        arena=arena.to_dict(),
        headings=headings,
        position_step=position_step,
        horizon=horizon,
        output_dt=output_dt,
        exit_radius=exit_radius,
        retain_history=retain_history,
        frames=frames,
        per_frame=per_frame,
        seed=seed,
        query_version=3,
    )
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    mp = output / "meta.json"
    started = time.perf_counter()
    resume = mp.exists()
    if resume:
        meta = json.loads(mp.read_text())
        if meta["signature"] != signature:
            raise ValueError("Existing output belongs to another model/setup")
        if meta["status"] == "complete":
            return meta
        p = VehicleGrid.load(output / "p_grid.npz")
        e = VehicleGrid.load(output / "e_grid.npz")
        domain = PairDomain.load(output / "domain.npz")
        queries = dict(np.load(output / "queries.npz"))
    else:
        p = make_grid(
            arena,
            config.pursuer,
            "pursuer",
            headings=headings,
            position_step=position_step,
            speeds=np.linspace(config.pursuer.speed_min, config.pursuer.speed_max, 7),
            dt=dt,
        )
        e = make_grid(
            arena,
            config.evader,
            "evader",
            headings=headings,
            position_step=position_step,
            speeds=np.linspace(config.evader.speed_min, config.evader.speed_max, 7),
            dt=dt,
        )
        domain = PairDomain.build(p, e, exit_radius)
        p.save(output / "p_grid.npz")
        e.save(output / "e_grid.npz")
        domain.save(output / "domain.npz")
        queries = sample_queries(
            p, e, domain, arena, config, frames=frames, per_frame=per_frame, seed=seed
        )
        np.savez_compressed(output / "queries.npz", **queries)
        meta = dict(
            schema="local_contact_compact8d_v1",
            status="running",
            signature=signature,
            spec=spec,
            p_states=len(p.states),
            e_states=len(e.states),
            pair_states=domain.states,
            unpruned_pair_states=len(p.states) * len(e.states),
            history_dtype="float16_sign_preserved",
            completed_step=0,
            completed_outputs=0,
            times=[],
            update_seconds=0.0,
            interpolation="multilinear, omit invalid single-vehicle vertices and renormalize",
            action_constraint="continuous obstacle-free command plus one collision-free constant-curvature braking backup",
            objective="capture by budget before center separation exceeds local radius; range exit is local failure, not a physical capture",
            numerical_game="finite-grid semi-Lagrangian minP maxE, exact factorization of the product interpolation; no dimension reduction",
        )
        mp.write_text(json.dumps(meta, indent=2))
    p.bind_geometry(arena, config.pursuer, "pursuer", dt)
    e.bind_geometry(arena, config.evader, "evader", dt)
    # One retained history plus the old/new atomic checkpoints must fit together.
    expected = (
        len(range(0, steps + 1, every)) * 2 if retain_history else 0
    ) * domain.states + 8 * domain.states
    already = sum(f.stat().st_size for f in output.iterdir() if f.is_file())
    if shutil.disk_usage(output).free < max(0, expected - already) + 8 * 10**9:
        raise OSError(
            "Insufficient disk space for history/checkpoint transaction plus8GB reserve"
        )
    cp.get_default_memory_pool().free_all_blocks()
    arrays = packed_gpu_arrays(p, e, domain, max_gb=max_gb, tiled_output=True)
    pi, pw, pvalid, _, _ = p.stencil(queries["states"][:, :4])
    ei, ew, evalid, _, _ = e.stencil(queries["states"][:, 4:])
    if not (pvalid & evalid).all():
        raise ValueError("Generated query has no numerical support")
    query_arrays = [cp.asarray(x) for x in [pi, pw, ei, ew]]
    query_kernel = cp.RawKernel(QUERY_CUDA, "query_field")
    query_result = cp.empty(len(pi), cp.float32)

    def sample_current():
        query_kernel(
            ((len(pi) + 255) // 256,),
            (256,),
            (
                arrays["current"],
                *query_arrays,
                arrays["p_position"],
                arrays["lookup"],
                arrays["offsets"],
                np.int32(len(pi)),
                np.int32(len(e.states)),
                np.float32(exit_radius),
                query_result,
            ),
        )
        return cp.asnumpy(query_result)

    checkpoint = output / meta.get("checkpoint", "missing")
    if resume and meta["completed_outputs"] and checkpoint.exists():
        field = np.memmap(checkpoint, np.float32, mode="r", shape=(domain.states,))
        arrays["current"].set(field)
        prior = np.load(output / "labels.npz")
        labels = list(prior["values"][: meta["completed_outputs"]])
        times = list(prior["times"][: meta["completed_outputs"]])
        start = meta["completed_step"]
        del field
    else:
        initialize_packed(arrays, p, e, domain, arena)
        cp.cuda.Stream.null.synchronize()
        labels = []
        times = []
        start = 0
    history = output / "fields.f16"
    if retain_history:
        with history.open("ab" if history.exists() else "wb") as f:
            f.truncate(meta["completed_outputs"] * domain.states * 2)

    def save_output(step):
        nonlocal labels, times
        cp.cuda.Stream.null.synchronize()
        labels.append(
            arena.capture_value(queries["states"]).astype(np.float32)
            if step == 0
            else sample_current()
        )
        times.append(step * dt)
        host = arrays["next_cpu"]
        if step == 0:
            arrays["current"].get(out=host)
        if retain_history:
            with history.open("ab") as file:
                write_half_preserving_sign(host, file)
        np.savez_compressed(
            output / "labels.npz", times=np.array(times), values=np.stack(labels)
        )
        new_name = f"checkpoint_{step:04d}.f32"
        tmp = output / (new_name + ".partial")
        host.tofile(tmp)
        os.replace(tmp, output / new_name)
        old = meta.get("checkpoint")
        meta.update(
            completed_step=step,
            completed_outputs=len(times),
            times=times,
            checkpoint=new_name,
            gpu_working_bytes=arrays["memory_bytes"],
            host_working_bytes=domain.states * 4,
        )
        temporary = mp.with_suffix(".tmp")
        temporary.write_text(json.dumps(meta, indent=2))
        os.replace(temporary, mp)
        if old and old != new_name and (output / old).exists():
            (output / old).unlink()
        print(
            f'budget {step*dt:.2f}s; states {domain.states:,}; updates {meta["update_seconds"]:.1f}s',
            flush=True,
        )

    if not labels:
        save_output(0)
    for step in range(start + 1, steps + 1):
        then = time.perf_counter()
        backup_packed(arrays, p, e, domain)
        cp.cuda.Stream.null.synchronize()
        meta["update_seconds"] += time.perf_counter() - then
        if step % every == 0 or step == steps:
            save_output(step)
    meta.update(
        status="complete",
        wall_seconds_this_process=time.perf_counter() - started,
        saved_bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
    )
    mp.write_text(json.dumps(meta, indent=2))
    (output / meta["checkpoint"]).unlink()
    meta.pop("checkpoint")
    meta["saved_bytes"] = sum(f.stat().st_size for f in output.iterdir() if f.is_file())
    mp.write_text(json.dumps(meta, indent=2))
    return meta


class ContactReference:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text())
        if self.meta["status"] != "complete" or not self.meta["spec"]["retain_history"]:
            raise ValueError("A complete signed history is required")
        if not (self.folder / "fields.f16").exists():
            raise ValueError(
                "History was not retained; regenerate into a new output folder"
            )
        self.config = ReversibleConfig.from_dict(self.meta["spec"]["config"])
        self.arena = CircleArena.from_dict(self.meta["spec"]["arena"])
        self.p = VehicleGrid.load(self.folder / "p_grid.npz")
        self.e = VehicleGrid.load(self.folder / "e_grid.npz")
        self.domain = PairDomain.load(self.folder / "domain.npz")
        self.p.bind_geometry(
            self.arena, self.config.pursuer, "pursuer", self.config.control_dt
        )
        self.e.bind_geometry(
            self.arena, self.config.evader, "evader", self.config.control_dt
        )
        self.times = np.array(self.meta["times"])
        self.horizon = self.times[-1]
        expected = len(self.times) * self.domain.states * 2
        if (self.folder / "fields.f16").stat().st_size != expected:
            raise ValueError("Incomplete history length")
        self.values = np.memmap(
            self.folder / "fields.f16",
            np.float16,
            mode="r",
            shape=(len(self.times), self.domain.states),
        )

    def stencil(self, s):
        s = np.asarray(s, float)
        shape = s.shape[:-1]
        flat = s.reshape(-1, 8)
        pi, pw, pv, _, _ = self.p.stencil(flat[:, :4])
        ei, ew, ev, _, _ = self.e.stencil(flat[:, 4:])
        rank = self.domain.lookup[
            self.domain.p_position[pi][:, :, None], ei[:, None, :]
        ]
        ids = self.domain.offsets[pi][:, :, None] + np.maximum(rank, 0)
        weights = pw[:, :, None] * ew[:, None, :]
        valid = pv & ev
        inside = (
            np.linalg.norm(flat[:, :2] - flat[:, 4:6], axis=1)
            <= self.domain.exit_radius
        )
        return ids, weights, rank >= 0, valid, inside, shape

    def fields(self, s):
        ids, w, kept, valid, inside, shape = self.stencil(s)
        values = np.sum(
            np.where(
                kept[None],
                self.values[:, ids].astype(np.float32),
                self.domain.exit_radius,
            )
            * w[None],
            axis=(-1, -2),
        )
        phi = self.arena.capture_value(np.asarray(s).reshape(-1, 8))
        values = np.minimum(values, phi[None])
        values = np.where(inside[None], values, self.domain.exit_radius)
        values[:, ~valid] = np.nan
        values[0, valid & inside] = phi[valid & inside]
        return values, valid & inside, shape

    def field(self, s, budget=None):
        values, valid, shape = self.fields(s)
        budget = self.horizon if budget is None else budget
        if not 0 <= budget <= self.horizon:
            raise ValueError("Budget outside reference")
        i = min(
            np.searchsorted(self.times, budget, side="right") - 1, len(self.times) - 2
        )
        f = (budget - self.times[i]) / (self.times[i + 1] - self.times[i])
        return ((1 - f) * values[i] + f * values[i + 1]).reshape(shape)

    def time_value(self, s):
        values, valid, shape = self.fields(s)
        return first_crossing(self.times, values, valid).reshape(shape)
