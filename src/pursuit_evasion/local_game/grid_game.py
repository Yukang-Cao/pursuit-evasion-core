"""Compact eight-state semi-Lagrangian game reference.

Each axis of the stored matrix indexes a legal FOUR-dimensional vehicle state.
The product therefore still represents eight dynamic state variables. Invalid
obstacle/braking configurations are omitted, not reduced by symmetry.

A held-control successor uses multilinear interpolation on the physical grid.
Weights on omitted configurations are removed and the remaining weights are
normalized. This boundary interpolation is an explicit numerical approximation.
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import json, time
import numpy as np
from .native import GeometryEngine

BITS = np.asarray(list(product([0, 1], repeat=4)), np.int64)


@dataclass
class VehicleGrid:
    axes: tuple
    states: np.ndarray
    dense_to_compact: np.ndarray
    controls: np.ndarray
    successors: np.ndarray
    weights: np.ndarray
    allowed: np.ndarray
    retained_mass: np.ndarray

    def bind_geometry(self, arena, limits, role, dt=0.05):
        self._engine = GeometryEngine(arena, limits, role, threads=1, control_dt=dt)
        self._tree = None
        return self

    def _nearest_supported(self, states):
        from scipy.spatial import cKDTree

        dx = self.axes[0][1] - self.axes[0][0]
        dh = 2 * np.sin(np.pi / len(self.axes[2]))
        dv = np.min(np.diff(self.axes[3]))

        def features(s):
            return np.stack(
                [
                    s[:, 0] / dx,
                    s[:, 1] / dx,
                    np.cos(s[:, 2]) / dh,
                    np.sin(s[:, 2]) / dh,
                    s[:, 3] / dv,
                ],
                -1,
            )

        if self._tree is None:
            self._tree = cKDTree(features(self.states))
        distance, index = self._tree.query(
            features(states), k=min(64, len(self.states))
        )
        target = self.states[index]
        connected = self._engine.connections(
            np.broadcast_to(states[:, None], target.shape).reshape(-1, 4),
            target.reshape(-1, 4),
        ).reshape(index.shape)
        out_i = np.zeros((len(states), 16), np.int32)
        out_w = np.zeros((len(states), 16), np.float32)
        valid = np.zeros(len(states), bool)
        for row in range(len(states)):
            selected = np.flatnonzero(connected[row])[:16]
            if not len(selected):
                continue
            w = 1 / np.maximum(distance[row, selected], 1e-8) ** 2
            w /= w.sum()
            out_i[row, : len(selected)] = index[row, selected]
            out_w[row, : len(selected)] = w
            valid[row] = True
        return out_i, out_w, valid

    def stencil(self, states):
        s = np.asarray(states, float)
        shape = s.shape[:-1]
        flat = s.reshape(-1, 4)
        inds = np.zeros((len(flat), 16), np.int64)
        w = np.ones((len(flat), 16))
        valid = np.isfinite(flat).all(1)
        strides = np.cumprod((1,) + tuple(len(a) for a in self.axes)[:0:-1])[::-1]
        for k, (axis, stride) in enumerate(zip(self.axes, strides)):
            x = flat[:, k]
            if k == 2:
                f = ((x + np.pi) % (2 * np.pi)) / (2 * np.pi / len(axis))
                j = np.floor(f).astype(np.int64) % len(axis)
                frac = f - np.floor(f)
            else:
                valid &= (x >= axis[0] - 1e-9) & (x <= axis[-1] + 1e-9)
                z = np.clip(x, axis[0], axis[-1])
                j = np.clip(
                    np.searchsorted(axis, z, side="right") - 1, 0, len(axis) - 2
                )
                frac = (z - axis[j]) / (axis[j + 1] - axis[j])
            coord = j[:, None] + BITS[None, :, k]
            if k == 2:
                coord %= len(axis)
            inds += coord * stride
            w *= np.where(BITS[None, :, k], frac[:, None], 1 - frac[:, None])
        ids = self.dense_to_compact[inds]
        w = np.where(ids >= 0, w, 0)
        mass = w.sum(1)
        missing = valid & (mass <= 1e-9)
        valid &= mass > 1e-9
        w = np.divide(
            w, mass[:, None], out=np.zeros_like(w), where=mass[:, None] > 1e-9
        )
        ids = np.maximum(ids, 0).astype(np.int32)
        w = w.astype(np.float32)
        if missing.any() and hasattr(self, "_engine"):
            unique, inverse = np.unique(flat[missing], axis=0, return_inverse=True)
            ni, nw, nv = self._nearest_supported(unique)
            ids[missing] = ni[inverse]
            w[missing] = nw[inverse]
            valid[missing] = nv[inverse]
        return ids, w, valid, shape, mass

    def save(self, path):
        np.savez_compressed(
            path,
            states=self.states,
            dense_to_compact=self.dense_to_compact,
            controls=self.controls,
            successors=self.successors,
            weights=self.weights,
            allowed=self.allowed,
            retained_mass=self.retained_mass,
            **{"axis" + str(i): a for i, a in enumerate(self.axes)},
        )

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(
            tuple(z["axis" + str(i)] for i in range(4)),
            *[
                z[k]
                for k in [
                    "states",
                    "dense_to_compact",
                    "controls",
                    "successors",
                    "weights",
                    "allowed",
                    "retained_mass",
                ]
            ],
        )


def make_grid(
    arena, limits, role, *, position_step=0.3, headings=8, speeds=None, dt=0.05
):
    if speeds is None:
        speeds = [-1, 0, 0.5, 1, 2] if role == "pursuer" else [-0.5, 0, 0.25, 0.5, 1]
    n = int(round(2 * arena.half_width / position_step))
    x = (np.arange(n + 2) - (n + 1) / 2) * position_step
    axes = (
        x,
        x,
        np.arange(headings) * 2 * np.pi / headings - np.pi,
        np.array(speeds, float),
    )
    states = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 4)
    engine = GeometryEngine(arena, limits, role, control_dt=dt)
    legal = engine.backup_safe(states)
    mapping = np.full(len(states), -1, np.int32)
    mapping[legal] = np.arange(legal.sum())
    kept = states[legal]
    controls = np.array(
        [
            [a, k]
            for a in [-limits.acceleration, 0, limits.acceleration]
            for k in [-limits.max_curvature, 0, limits.max_curvature]
        ],
        float,
    )
    grid = VehicleGrid(
        axes, kept, mapping, controls, None, None, None, None
    ).bind_geometry(arena, limits, role, dt)
    allowed, nxt = engine.actions(kept, controls, dt)
    idx, w, valid, _, mass = grid.stencil(nxt)
    grid.successors = idx.reshape(len(kept), len(controls), 16)
    grid.weights = w.reshape(len(kept), len(controls), 16)
    grid.allowed = allowed & valid.reshape(allowed.shape)
    grid.retained_mass = mass.reshape(allowed.shape)
    if not grid.allowed.any(axis=1).all():
        raise ValueError("A retained state has no legal interpolable action")
    return grid


CUDA = r"""
extern "C" __global__ void initialize(float*f,const float*p,const float*e,long long N,int ne,float pl,float pw,float el,float ew,float exit_radius){
 long long j=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(j>=N)return;
 int ip=j/ne,ie=j%ne;const float*a=p+4*ip,*b=e+4*ie;float dx=b[0]-a[0],dy=b[1]-a[1];
 if(dx*dx+dy*dy>exit_radius*exit_radius){f[j]=exit_radius;return;}
 float c=cosf(b[2]-a[2]),s=sinf(b[2]-a[2]),ca=fabsf(c),sa=fabsf(s),cp=cosf(a[2]),sp=sinf(a[2]),ce=cosf(b[2]),se=sinf(b[2]);
 float x=fabsf(dx*cp+dy*sp)-.5f*(pl+el*ca+ew*sa),y=fabsf(-dx*sp+dy*cp)-.5f*(pw+el*sa+ew*ca);
 float u=fabsf(dx*ce+dy*se)-.5f*(el+pl*ca+pw*sa),v=fabsf(-dx*se+dy*ce)-.5f*(ew+pl*sa+pw*ca);
 f[j]=fmaxf(fmaxf(x,y),fmaxf(u,v));
}
extern "C" __global__ void interpolate_p(const float*f,float*g,const int*ids,const float*w,const unsigned char*allowed,int begin,int count,int ne,int na,int action){
 long long j=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(j>=(long long)count*ne)return;
 int local=j/ne,ie=j%ne,ip=begin+local;long long base=((long long)ip*na+action)*16;
 if(!allowed[(long long)ip*na+action]){g[j]=0;return;}
 float value=0;
 #pragma unroll
 for(int k=0;k<16;k++)if(w[base+k]>0)value+=w[base+k]*f[(long long)ids[base+k]*ne+ie];
 g[j]=value;
}
extern "C" __global__ void reduce_e(const float*g,float*out,const int*ids,const float*w,const unsigned char*ea,const unsigned char*pa,int begin,int count,int ne,int npa,int nea,int action,const float*p,const float*e,float exit_radius){
 long long j=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(j>=(long long)count*ne)return;
 int local=j/ne,ie=j%ne,ip=begin+local;if(!pa[(long long)ip*npa+action])return;
 float dx=p[4*ip]-e[4*ie],dy=p[4*ip+1]-e[4*ie+1];if(dx*dx+dy*dy>exit_radius*exit_radius)return;
 long long dest=(long long)ip*ne+ie;float best=out[dest],worst=-1e30f;
 for(int a=0;a<nea;a++)if(ea[(long long)ie*nea+a]){
  long long base=((long long)ie*nea+a)*16;float value=0;
  #pragma unroll
  for(int k=0;k<16;k++)if(w[base+k]>0)value+=w[base+k]*g[(long long)local*ne+ids[base+k]];
  worst=fmaxf(worst,value);if(worst>=best)return;
 }
 out[dest]=fminf(best,worst);
}
"""


def gpu_arrays(p, e, *, max_gb=7.6, chunk=256):
    import cupy as cp

    n = len(p.states) * len(e.states)
    need = n * 8 + min(chunk, len(p.states)) * len(e.states) * 4 + 64 * 2**20
    free, _ = cp.cuda.runtime.memGetInfo()
    if need > max_gb * 1e9 or need > free * 0.95:
        raise MemoryError(f"{need/1e9:.2f}GB required; {free/1e9:.2f}GB free")
    arrays = {}
    for role, grid in [("p", p), ("e", e)]:
        arrays[role + "states"] = cp.asarray(grid.states, dtype=cp.float32)
        arrays[role + "ids"] = cp.asarray(grid.successors)
        arrays[role + "weights"] = cp.asarray(grid.weights)
        arrays[role + "allowed"] = cp.asarray(grid.allowed, dtype=cp.uint8)
    arrays["current"] = cp.empty(n, cp.float32)
    arrays["next"] = cp.empty(n, cp.float32)
    arrays["temp"] = cp.empty(min(chunk, len(p.states)) * len(e.states), cp.float32)
    arrays["kernels"] = {
        name: cp.RawKernel(CUDA, name)
        for name in ["initialize", "interpolate_p", "reduce_e"]
    }
    arrays["memory_bytes"] = need
    arrays["chunk"] = chunk
    return arrays


def initialize_gpu(a, p, e, arena, exit_radius=3.0):
    a["exit_radius"] = float(exit_radius)
    n = len(p.states) * len(e.states)
    a["kernels"]["initialize"](
        ((n + 255) // 256,),
        (256,),
        (
            a["current"],
            a["pstates"],
            a["estates"],
            np.int64(n),
            np.int32(len(e.states)),
            *map(
                np.float32,
                [
                    arena.pursuer.length,
                    arena.pursuer.width,
                    arena.evader.length,
                    arena.evader.width,
                    exit_radius,
                ],
            ),
        ),
    )


def backup_gpu(a, p, e):
    import cupy as cp

    a["next"][...] = a["current"]
    ne = len(e.states)
    chunk = a["chunk"]
    npa = len(p.controls)
    nea = len(e.controls)
    for begin in range(0, len(p.states), chunk):
        count = min(chunk, len(p.states) - begin)
        launch = (((count * ne + 255) // 256,), (256,))
        for u in range(npa):
            a["kernels"]["interpolate_p"](
                *launch,
                (
                    a["current"],
                    a["temp"],
                    a["pids"],
                    a["pweights"],
                    a["pallowed"],
                    np.int32(begin),
                    np.int32(count),
                    np.int32(ne),
                    np.int32(npa),
                    np.int32(u),
                ),
            )
            a["kernels"]["reduce_e"](
                *launch,
                (
                    a["temp"],
                    a["next"],
                    a["eids"],
                    a["eweights"],
                    a["eallowed"],
                    a["pallowed"],
                    np.int32(begin),
                    np.int32(count),
                    np.int32(ne),
                    np.int32(npa),
                    np.int32(nea),
                    np.int32(u),
                    a["pstates"],
                    a["estates"],
                    np.float32(a["exit_radius"]),
                ),
            )
    a["current"], a["next"] = a["next"], a["current"]


@dataclass
class PairDomain:
    p_position: np.ndarray
    lookup: np.ndarray
    e_indices: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray
    exit_radius: float

    @property
    def states(self):
        return int(self.offsets[-1])

    @classmethod
    def build(cls, p, e, exit_radius=3.0):
        positions, which = np.unique(p.states[:, :2], axis=0, return_inverse=True)
        lookup = np.full((len(positions), len(e.states)), -1, np.int32)
        lists = []
        for i, xy in enumerate(positions):
            ids = np.flatnonzero(
                np.linalg.norm(e.states[:, :2] - xy, axis=1) <= exit_radius + 1e-8
            )
            lists.append(ids)
            lookup[i, ids] = np.arange(len(ids), dtype=np.int32)
        counts = np.array([len(x) for x in lists], np.int32)
        inverse = np.zeros((len(positions), int(counts.max())), np.int32)
        for i, x in enumerate(lists):
            inverse[i, : len(x)] = x
        offsets = np.concatenate([[0], np.cumsum(counts[which], dtype=np.int64)])
        return cls(
            which.astype(np.int32), lookup, inverse, counts, offsets, float(exit_radius)
        )

    def save(self, path):
        np.savez_compressed(
            path,
            p_position=self.p_position,
            lookup=self.lookup,
            e_indices=self.e_indices,
            counts=self.counts,
            offsets=self.offsets,
            exit_radius=self.exit_radius,
        )

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(
            *[z[k] for k in ["p_position", "lookup", "e_indices", "counts", "offsets"]],
            float(z["exit_radius"]),
        )


PACKED_CUDA = r"""
extern "C" __global__ void init_packed(float*f,const float*p,const float*e,const int*ppos,const int*ei,const int*count,const long long*offset,int np,int maxcount,float pl,float pw,float el,float ew){
 int ip=blockIdx.y+blockIdx.z*gridDim.y,k=blockIdx.x*blockDim.x+threadIdx.x;if(ip>=np||k>=count[ppos[ip]])return;
 int ie=ei[(long long)ppos[ip]*maxcount+k];long long j=offset[ip]+k;
 const float*a=p+4*ip,*b=e+4*ie;float dx=b[0]-a[0],dy=b[1]-a[1];
 float c=cosf(b[2]-a[2]),s=sinf(b[2]-a[2]),ca=fabsf(c),sa=fabsf(s),cp=cosf(a[2]),sp=sinf(a[2]),ce=cosf(b[2]),se=sinf(b[2]);
 float x=fabsf(dx*cp+dy*sp)-.5f*(pl+el*ca+ew*sa),y=fabsf(-dx*sp+dy*cp)-.5f*(pw+el*sa+ew*ca);
 float u=fabsf(dx*ce+dy*se)-.5f*(el+pl*ca+pw*sa),v=fabsf(-dx*se+dy*ce)-.5f*(ew+pl*sa+pw*ca);
 f[j]=fmaxf(fmaxf(x,y),fmaxf(u,v));
}
extern "C" __global__ void p_packed(const float*f,float*g,const int*ids,const float*w,const unsigned char*allowed,int begin,int count,int ne,int na,int action,const int*ppos,const int*lookup,const long long*offset,const float*p,const float*e,float reach,float exit_value){
 long long j=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(j>=(long long)count*ne)return;
 int local=j/ne,ie=j%ne,ip=begin+local;
 if(!allowed[(long long)ip*na+action]){g[j]=0;return;}
 float dx=p[4*ip]-e[4*ie],dy=p[4*ip+1]-e[4*ie+1];
 if(dx*dx+dy*dy>reach*reach){g[j]=exit_value;return;}
 long long base=((long long)ip*na+action)*16;float value=0;
 #pragma unroll
 for(int k=0;k<16;k++)if(w[base+k]>0){int next=ids[base+k],rank=lookup[(long long)ppos[next]*ne+ie];value+=w[base+k]*(rank<0?exit_value:f[offset[next]+rank]);}
 g[j]=value;
}
extern "C" __global__ void e_packed(const float*g,float*out,const int*ids,const float*w,const unsigned char*ea,const unsigned char*pa,int begin,int countp,int ne,int npa,int nea,int action,const int*ppos,const int*ei,const int*counts,const long long*offset,int maxcount,long long output_base){
 int local=blockIdx.y,k=blockIdx.x*blockDim.x+threadIdx.x,ip=begin+local;
 if(local>=countp||k>=counts[ppos[ip]]||!pa[(long long)ip*npa+action])return;
 int ie=ei[(long long)ppos[ip]*maxcount+k];long long dest=offset[ip]+k-output_base;float best=out[dest],worst=-1e30f;
 for(int a=0;a<nea;a++)if(ea[(long long)ie*nea+a]){
  long long base=((long long)ie*nea+a)*16;float value=0;
  #pragma unroll
  for(int q=0;q<16;q++)if(w[base+q]>0)value+=w[base+q]*g[(long long)local*ne+ids[base+q]];
  worst=fmaxf(worst,value);if(worst>=best)return;
 }
 out[dest]=fminf(best,worst);
}
"""


def packed_gpu_arrays(p, e, domain, *, max_gb=7.6, chunk=256, tiled_output=False):
    import cupy as cp

    need = (
        domain.states * (4 if tiled_output else 8)
        + min(chunk, len(p.states)) * len(e.states) * 4
        + domain.lookup.nbytes
        + domain.e_indices.nbytes
        + 128 * 2**20
    )
    if tiled_output:
        need += min(chunk, len(p.states)) * domain.e_indices.shape[1] * 4
    free, _ = cp.cuda.runtime.memGetInfo()
    if need > max_gb * 1e9 or need > free * 0.95:
        raise MemoryError(f"{need/1e9:.2f}GB required; {free/1e9:.2f}GB free")
    a = {}
    for role, grid in [("p", p), ("e", e)]:
        a[role + "states"] = cp.asarray(grid.states, dtype=cp.float32)
        a[role + "ids"] = cp.asarray(grid.successors)
        a[role + "weights"] = cp.asarray(grid.weights)
        a[role + "allowed"] = cp.asarray(grid.allowed, dtype=cp.uint8)
    for key in ["p_position", "lookup", "e_indices", "counts", "offsets"]:
        a[key] = cp.asarray(getattr(domain, key))
    a["current"] = cp.empty(domain.states, cp.float32)
    a["next"] = cp.empty(
        (
            min(chunk, len(p.states)) * domain.e_indices.shape[1]
            if tiled_output
            else domain.states
        ),
        cp.float32,
    )
    if tiled_output:
        a["next_cpu"] = np.empty(domain.states, np.float32)
    a["temp"] = cp.empty(min(chunk, len(p.states)) * len(e.states), cp.float32)
    a["kernels"] = {
        name: cp.RawKernel(PACKED_CUDA, name)
        for name in ["init_packed", "p_packed", "e_packed"]
    }
    reach = 0.0
    for start in range(0, len(e.states), 1024):
        ids = e.successors[start : start + 1024]
        dist = np.linalg.norm(
            e.states[ids, :2] - e.states[start : start + 1024, None, None, :2], axis=-1
        )
        reach = max(
            reach, float(np.where(e.weights[start : start + 1024] > 0, dist, 0).max())
        )
    a.update(
        memory_bytes=need,
        chunk=chunk,
        exit_radius=domain.exit_radius,
        e_reach=reach + 1e-4,
        tiled_output=tiled_output,
    )
    return a


def initialize_packed(a, p, e, domain, arena):
    maxcount = domain.e_indices.shape[1]
    a["kernels"]["init_packed"](
        ((maxcount + 255) // 256, min(len(p.states), 65535), (len(p.states) + 65534) // 65535),
        (256,),
        (
            a["current"],
            a["pstates"],
            a["estates"],
            a["p_position"],
            a["e_indices"],
            a["counts"],
            a["offsets"],
            np.int32(len(p.states)),
            np.int32(maxcount),
            *map(
                np.float32,
                [
                    arena.pursuer.length,
                    arena.pursuer.width,
                    arena.evader.length,
                    arena.evader.width,
                ],
            ),
        ),
    )


def backup_packed(a, p, e, domain):
    import cupy as cp

    tiled = a["tiled_output"]
    ne = len(e.states)
    chunk = a["chunk"]
    npa = len(p.controls)
    nea = len(e.controls)
    maxcount = domain.e_indices.shape[1]
    if not tiled:
        a["next"][...] = a["current"]
    for begin in range(0, len(p.states), chunk):
        count = min(chunk, len(p.states) - begin)
        first, last = int(domain.offsets[begin]), int(domain.offsets[begin + count])
        length = last - first
        if tiled:
            a["next"][:length] = a["current"][first:last]
        for u in range(npa):
            a["kernels"]["p_packed"](
                ((count * ne + 255) // 256,),
                (256,),
                (
                    a["current"],
                    a["temp"],
                    a["pids"],
                    a["pweights"],
                    a["pallowed"],
                    np.int32(begin),
                    np.int32(count),
                    np.int32(ne),
                    np.int32(npa),
                    np.int32(u),
                    a["p_position"],
                    a["lookup"],
                    a["offsets"],
                    a["pstates"],
                    a["estates"],
                    np.float32(domain.exit_radius + a["e_reach"]),
                    np.float32(domain.exit_radius),
                ),
            )
            a["kernels"]["e_packed"](
                ((maxcount + 255) // 256, count),
                (256,),
                (
                    a["temp"],
                    a["next"],
                    a["eids"],
                    a["eweights"],
                    a["eallowed"],
                    a["pallowed"],
                    np.int32(begin),
                    np.int32(count),
                    np.int32(ne),
                    np.int32(npa),
                    np.int32(nea),
                    np.int32(u),
                    a["p_position"],
                    a["e_indices"],
                    a["counts"],
                    a["offsets"],
                    np.int32(maxcount),
                    np.int64(first if tiled else 0),
                ),
            )
        if tiled:
            a["next_cpu"][first:last] = cp.asnumpy(a["next"][:length])
    if tiled:
        a["current"].set(a["next_cpu"])
    else:
        a["current"], a["next"] = a["next"], a["current"]
