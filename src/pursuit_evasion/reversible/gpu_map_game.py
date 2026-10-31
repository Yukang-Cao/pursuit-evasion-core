"""Optional 8D CuPy HJI solver with explicit oriented footprints and obstacles.

This pilot retains the existing first-fault-loses game. Its signed field is a
P-success reach-avoid value, NOT automatically a physical capture-time value.
Labels can be sampled at grid nodes without retaining a full field history.
"""

from pathlib import Path
import json, time, hashlib, math
import numpy as np
from .world_dp import WorldGrid, global_rate_bound, output_times
from .footprint_map import FootprintMap

CUDA = r"""
#define NX @NX@
#define NH @NH@
#define NV @NV@
#define NE @NE@
__device__ float at(const float*s,long long j,long long stride,int pos,int n,int off,bool periodic){
 int q=pos+off;
 if(periodic){q=(q%n+n)%n;return s[j+(q-pos)*stride];}
 if(q<0){long long e=j-(long long)pos*stride;float a=s[e],b=s[e+stride];return a+(-q)*fabsf(b-a)*((a>0)-(a<0));}
 if(q>=n){long long e=j+(long long)(n-1-pos)*stride;float a=s[e],b=s[e-stride];return a+(q-n+1)*fabsf(a-b)*((a>0)-(a<0));}
 return s[j+(long long)off*stride];
}
__device__ void slope(const float*s,long long j,long long stride,int pos,int n,float h,bool periodic,bool speed,int order,float&l,float&r){
 float f=s[j];
 if(speed&&pos==0){l=r=(s[j+stride]-f)/h;return;}
 if(speed&&pos==n-1){l=r=(f-s[j-stride])/h;return;}
 float fm=at(s,j,stride,pos,n,-1,periodic),fp=at(s,j,stride,pos,n,1,periodic);
 l=(f-fm)/h;r=(fp-f)/h;
 if(order==2){
  float dc=(fp-2*f+fm)/(h*h);
  float dl=speed&&pos<2?dc:(f-2*fm+at(s,j,stride,pos,n,-2,periodic))/(h*h);
  float dr=speed&&pos+2>=n?dc:(at(s,j,stride,pos,n,2,periodic)-2*fp+f)/(h*h);
  l+=.5f*h*(fabsf(dl)<=fabsf(dc)?dl:dc);r-=.5f*h*(fabsf(dc)<=fabsf(dr)?dc:dr);
 }
}
__device__ void coords(long long j,int*q){
 q[7]=j%NE;j/=NE;q[6]=j%NH;j/=NH;q[5]=j%NX;j/=NX;q[4]=j%NX;j/=NX;
 q[3]=j%NV;j/=NV;q[2]=j%NH;j/=NH;q[1]=j%NX;j/=NX;q[0]=j;
}
extern "C" __global__ void init(float*a,float*arrival,const float*pc,const float*ec,long long N,float extent,float radius){
 long long j=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(j>=N)return;
 int q[8];coords(j,q);float dx=2*extent/(NX-1);
 float cap=hypotf((q[0]-q[4])*dx,(q[1]-q[5])*dx)-radius;
 float p=pc[(q[0]*NX+q[1])*NH+q[2]],e=ec[(q[4]*NX+q[5])*NH+q[6]];
 float v=fmaxf(fminf(cap,e),-p+1e-6f);a[j]=v;arrival[j]=v<=0?0:nanf("");
}
extern "C" __global__ void advance(const float*src,float*dst,float*arrival,const float*pc,
 const float*sn,const float*cs,long long N,float extent,float plo,float phi,float elo,float ehi,
 float ap,float ae,float kp,float ke,float dt,float tau,int order,int blend){
 long long j=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(j>=N)return;
 constexpr int dims[8]={NX,NX,NH,NV,NX,NX,NH,NE};
 constexpr long long strides[8]={(long long)NX*NH*NV*NX*NX*NH*NE,(long long)NH*NV*NX*NX*NH*NE,(long long)NV*NX*NX*NH*NE,(long long)NX*NX*NH*NE,(long long)NX*NH*NE,(long long)NH*NE,NE,1};
 int q[8];coords(j,q);
 float dx=2*extent/(NX-1),dh=6.2831853071795864769f/NH,dvp=(phi-plo)/(NV-1),dve=(ehi-elo)/(NE-1);
 float spacing[8]={dx,dx,dh,dvp,dx,dx,dh,dve};
 float vp=plo+q[3]*dvp,ve=elo+q[7]*dve,wp=fabsf(vp)*kp,we=fabsf(ve)*ke;
 float flow[8]={vp*cs[q[2]],vp*sn[q[2]],0,0,ve*cs[q[6]],ve*sn[q[6]],0,0};
 float alpha[8]={fabsf(flow[0]),fabsf(flow[1]),wp,ap,fabsf(flow[4]),fabsf(flow[5]),we,ae};
 float g[8],H=0,diss=0;
 #pragma unroll
 for(int k=0;k<8;k++){float l,r;slope(src,j,strides[k],q[k],dims[k],spacing[k],k==2||k==6,k==3||k==7,order,l,r);g[k]=.5f*(l+r);H+=flow[k]*g[k];diss+=.5f*alpha[k]*(r-l);}
 float plow=q[3]==0?0:-ap,phigh=q[3]==NV-1?0:ap,elow=q[7]==0?0:-ae,ehigh=q[7]==NE-1?0:ae;
 H+=-wp*fabsf(g[2])+we*fabsf(g[6])+fminf(plow*g[3],phigh*g[3])+fmaxf(elow*g[7],ehigh*g[7]);
 float old=src[j],previous=blend?dst[j]:old,value=old+dt*(H+diss);
 if(blend)value=.5f*(previous+value);
 if(order==1||blend){
  float avoid=-pc[(q[0]*NX+q[1])*NH+q[2]]+1e-6f;
  value=fmaxf(fminf(value,previous),avoid);
  if(previous>0&&value<=0&&isnan(arrival[j]))arrival[j]=tau+dt*previous/(previous-value);
 }
 dst[j]=value;
}
"""


def clearance_grids(geometry, grid, config):
    axes = grid.axes(config)
    x, y, h = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    poses = np.stack([x, y, h], axis=-1)
    return tuple(
        np.asarray(geometry.clearance(poses, role), np.float32)
        for role in ("pursuer", "evader")
    )


def node_states(indices, grid, config):
    q = np.unravel_index(np.asarray(indices, np.int64), grid.shape)
    return np.stack([axis[i] for axis, i in zip(grid.axes(config), q)], axis=-1)


def solve(
    config,
    grid,
    geometry,
    output,
    *,
    horizon=1.5,
    output_dt=0.5,
    order=2,
    cfl=0.7,
    query_indices=None,
    query_states=None,
    save_final=False,
    save_history=False,
    max_working_gb=7.6,
):
    if config.capture_rule != "center_distance" or config.boundary_rule != "first_fault_loses":
        raise ValueError("Legacy map HJI uses center-distance capture and first-fault-loses")
    if config.point_vehicles:
        raise ValueError("Footprint HJI requires point_vehicles=False.")
    if not isinstance(geometry, FootprintMap) or not isinstance(grid, WorldGrid):
        raise TypeError("Use explicit FootprintMap and WorldGrid")
    if order not in (1, 2) or not 0 < cfl <= 1 or grid.extent <= geometry.half_width:
        raise ValueError("Require order1/2, CFL<=1 and spatial exterior padding")
    import cupy as cp

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "meta.json").exists():
        raise FileExistsError("Preserve existing solve")
    n = grid.states
    working = n * 12
    cp.get_default_memory_pool().free_all_blocks()
    free, total = cp.cuda.runtime.memGetInfo()
    if working > max_working_gb * 1e9 or working + 128 * 2**20 > free:
        raise MemoryError(
            f"Need {working/1e9:.3f} GB core GPU arrays; free {free/1e9:.3f}"
        )
    idx = np.asarray([] if query_indices is None else query_indices, np.int64)
    if idx.ndim != 1 or ((idx < 0) | (idx >= n)).any():
        raise ValueError("Invalid query node indices")
    stencil = weights = None
    states = node_states(idx, grid, config)
    if query_states is not None:
        if query_indices is not None:
            raise ValueError("Choose node indices OR continuous query states")
        from .world_table import interpolation_stencil

        states = np.asarray(query_states, float)
        if states.ndim != 2 or states.shape[1] != 8 or len(states) > 100000:
            raise ValueError("Expected at most100000 world states[n,8]")
        stencil, weights, valid, _ = interpolation_stencil(
            states, grid.axes(config), (2, 6)
        )
        if not valid.all():
            raise ValueError("Query outside numerical domain")
        if working + len(states) * 256 * 20 + 128 * 2**20 > free:
            raise MemoryError("Insufficient GPU room for query interpolation workspace")
    source = (
        CUDA.replace("@NX@", str(grid.positions))
        .replace("@NH@", str(grid.headings))
        .replace("@NV@", str(grid.p_speeds))
        .replace("@NE@", str(grid.e_speeds))
    )
    (output / "kernel.cu").write_text(source)
    times = output_times(horizon, output_dt)
    import shutil

    expected_bytes = 4 * n * (int(save_final) + len(times) * int(save_history))
    if shutil.disk_usage(output).free < expected_bytes + 5 * 10**9:
        raise OSError("Insufficient disk space for requested retained fields")
    started = time.perf_counter()
    meta = dict(
        schema="footprint_map_game8d_v1",
        status="running",
        config=config.to_dict(),
        geometry=geometry.to_dict(),
        grid=grid.to_dict(config),
        times=times.tolist(),
        target="capture_OR_evader_collision",
        avoid="pursuer_collision",
        game_rule="first_fault_loses; P collision wins numerical event ties",
        output_meaning="signed finite-budget P-success reach-avoid value, NOT capture time",
        core_array_bytes=working,
        order=order,
        cfl=cfl,
        local_precision="float32",
        state_order=["xP", "yP", "headingP", "vP", "xE", "yE", "headingE", "vE"],
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        query_count=len(states),
        history_saved=bool(save_history),
    )
    (output / "meta.json").write_text(json.dumps(meta, indent=2))
    init = cp.RawKernel(source, "init", options=("--std=c++17",))
    advance = cp.RawKernel(source, "advance", options=("--std=c++17",))
    t = time.perf_counter()
    init.compile()
    advance.compile()
    meta["compile_seconds"] = time.perf_counter() - t
    pc, ec = clearance_grids(geometry, grid, config)
    pc, ec = cp.asarray(pc), cp.asarray(ec)
    angles = grid.axes(config)[2]
    sn = cp.asarray(np.sin(angles), dtype=cp.float32)
    cs = cp.asarray(np.cos(angles), dtype=cp.float32)
    a = cp.empty(n, np.float32)
    b = cp.empty_like(a)
    arrival = cp.empty_like(a)
    qi = cp.asarray(idx if stencil is None else stencil)
    qw = None if weights is None else cp.asarray(weights, dtype=cp.float32)

    def sample():
        result = a[qi] if qw is None else (a[qi] * qw).sum(axis=1)
        return cp.asnumpy(result)

    block = (256,)
    launch = ((n + 255) // 256,)
    f = np.float32
    init(
        launch,
        block,
        (a, arrival, pc, ec, np.int64(n), f(grid.extent), f(config.capture_radius)),
    )
    cp.cuda.Stream.null.synchronize()
    history_file = (output / "fields.f32").open("wb") if save_history else None
    if history_file is not None:
        cp.asnumpy(a).tofile(history_file)
    samples = [sample()]
    p, e = config.pursuer, config.evader
    base = (
        arrival,
        pc,
        sn,
        cs,
        np.int64(n),
        f(grid.extent),
        f(p.speed_min),
        f(p.speed_max),
        f(e.speed_min),
        f(e.speed_max),
        f(p.acceleration),
        f(e.acceleration),
        f(p.max_curvature),
        f(e.max_curvature),
    )
    maxdt = cfl / global_rate_bound(config, grid)
    tau = 0.0
    steps = 0
    updates = 0.0
    for goal in times[1:]:
        t = time.perf_counter()
        while tau < goal - 1e-10:
            dt = min(maxdt, float(goal) - tau)
            advance(
                launch,
                block,
                (a, b, *base, f(dt), f(tau), np.int32(order), np.int32(0)),
            )
            if order == 2:
                advance(
                    launch,
                    block,
                    (b, a, *base, f(dt), f(tau), np.int32(order), np.int32(1)),
                )
            else:
                a, b = b, a
            tau += dt
            steps += 1
        cp.cuda.Stream.null.synchronize()
        updates += time.perf_counter() - t
        samples.append(sample())
        if history_file is not None:
            cp.asnumpy(a).tofile(history_file)
        with (output / "progress.jsonl").open("a") as file:
            file.write(
                json.dumps({"budget": tau, "steps": steps, "update_seconds": updates})
                + "\n"
            )
        print(f"budget={tau:.3f}s updates={updates:.2f}s states={n:,}", flush=True)
    if history_file is not None:
        history_file.close()
    for start in range(0, n, 4_000_000):
        if not bool(cp.isfinite(a[start : start + 4_000_000]).all()):
            raise RuntimeError("Non-finite field")
    np.savez_compressed(
        output / "labels.npz",
        indices=idx,
        states=states,
        times=times,
        values=np.stack(samples),
        arrival=cp.asnumpy(arrival[qi]) if qw is None else np.full(len(states), np.nan),
    )
    if save_final:
        cp.asnumpy(a).tofile(output / "final.f32")
    meta.update(
        status="complete",
        update_seconds=updates,
        steps=steps,
        max_internal_dt=maxdt,
        wall_seconds=time.perf_counter() - started,
        gpu_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
        saved_final=bool(save_final),
        output_bytes=sum(x.stat().st_size for x in output.iterdir() if x.is_file()),
    )
    (output / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
