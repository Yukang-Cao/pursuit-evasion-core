"""Optional CuPy implementation of the same relative ENO/LF/RK update.

Imports CuPy only when solve() is called. Float32 local arithmetic is a separately
identified backend; comparisons against the native double-local update are
required before using a new numerical configuration as a reference.
"""

from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import time
import numpy as np

from .free_dp import RelativeGrid, _validate, output_times, physics_signature

_SOURCE = r"""
__device__ double val(const float*s,long long j,long long stride,int pos,int n,int off,int periodic){int q=pos+off;if(periodic)q=(q%n+n)%n;return s[j+(q-pos)*stride];}
__device__ void slopes(const float*s,long long j,long long stride,int pos,int n,double h,int periodic,int order,double&dm,double&dp){
 double v=s[j];if(!periodic&&pos==0){dm=dp=(val(s,j,stride,pos,n,1,periodic)-v)/h;return;}if(!periodic&&pos==n-1){dm=dp=(v-val(s,j,stride,pos,n,-1,periodic))/h;return;}
 double vm=val(s,j,stride,pos,n,-1,periodic),vp=val(s,j,stride,pos,n,1,periodic);dm=(v-vm)/h;dp=(vp-v)/h;if(order==1)return;
 double sc=(vp-2*v+vm)/(h*h),sl=(periodic||pos>=2)?(v-2*vm+val(s,j,stride,pos,n,-2,periodic))/(h*h):sc,sr=(periodic||pos+2<n)?(val(s,j,stride,pos,n,2,periodic)-2*vp+v)/(h*h):sc;dm+=.5*h*(fabs(sl)<fabs(sc)?sl:sc);dp-=.5*h*(fabs(sc)<fabs(sr)?sc:sr);
}
extern "C" __global__ void advance(const float*src,float*dst,float*arrival,const double*sn,const double*cs,long long N,int nx,int np,int nv,int ne,double W,double vlo,double vhi,double elo,double ehi,double ap,double ae,double kp,double ke,double dt,double time,int order,int blend){
 long long j=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(j>=N)return;
 int ix=j%nx,iy=j/nx%nx,ip=j/((long long)nx*nx)%np,ie=j/((long long)nx*nx*np)%ne,iv=j/((long long)nx*nx*np*ne);
 long long sy=nx,sp=(long long)nx*nx,se=sp*np,sv=se*ne;
 double dx=2*W/(nx-1),dp=6.2831853071795864769/np,dvp=(vhi-vlo)/(nv-1),dve=(ehi-elo)/(ne-1),x=-W+ix*dx,y=-W+iy*dx,vp=vlo+iv*dvp,ve=elo+ie*dve;
 double wp=fabs(vp)*kp,we=fabs(ve)*ke,bx=ve*cs[ip]-vp,by=ve*sn[ip];
 double xm,xp,ym,yp,pm,pp,vm,vplus,em,ep;
 slopes(src,j,1,ix,nx,dx,0,order,xm,xp);slopes(src,j,sy,iy,nx,dx,0,order,ym,yp);slopes(src,j,sp,ip,np,dp,1,order,pm,pp);slopes(src,j,sv,iv,nv,dvp,0,order,vm,vplus);slopes(src,j,se,ie,ne,dve,0,order,em,ep);
 double gx=.5*(xm+xp),gy=.5*(ym+yp),gp=.5*(pm+pp),gv=.5*(vm+vplus),ge=.5*(em+ep);
 double plow=iv==0?0:-ap,phigh=iv==nv-1?0:ap,elow=ie==0?0:-ae,ehigh=ie==ne-1?0:ae;
 double H=bx*gx+by*gy-wp*fabs(y*gx-x*gy-gp)+we*fabs(gp)+fmin(plow*gv,phigh*gv)+fmax(elow*ge,ehigh*ge);
 H+=.5*((fabs(bx)+wp*fabs(y))*(xp-xm)+(fabs(by)+wp*fabs(x))*(yp-ym)+(wp+we)*(pp-pm)+ap*(vplus-vm)+ae*(ep-em));
 double old=src[j],previous=blend?dst[j]:old,value=old+dt*fmin(0.,H);if(blend)value=.5*(previous+value);dst[j]=(float)value;
 if((blend||order==1)&&previous>0&&dst[j]<=0&&isnan(arrival[j]))arrival[j]=(float)(time+dt*previous/(previous-(double)dst[j]));
}
extern "C" __global__ void initialize(const float*plane,float*field,float*arrival,long long N,long long plane_size){
 long long j=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(j>=N)return;
 float value=plane[j%plane_size];field[j]=value;arrival[j]=value<=0?0:nanf("");
}
"""


def source_code(local_precision):
    if local_precision == "float64":
        return _SOURCE
    if local_precision != "float32":
        raise ValueError("local_precision must be float32 or float64")
    text = (
        _SOURCE.replace("double", "float")
        .replace("fabs(", "fabsf(")
        .replace("fmin(", "fminf(")
        .replace("fmax(", "fmaxf(")
    )
    return re.sub(
        r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?)(?![\w.])",
        lambda m: m.group(1) + "f",
        text,
    )


def solve(
    config,
    grid,
    output,
    *,
    horizon=2.0,
    output_dt=0.1,
    order=2,
    cfl=0.8,
    history=True,
    local_precision="float64",
    max_working_gb=6.0,
    disk_reserve_gb=10.0,
    device=0,
    probes=None,
    save_final=True,
):
    solve_started = time.perf_counter()
    if config.capture_rule != "center_distance":
        raise ValueError("This legacy free-space solver uses circular capture")
    _validate(config, grid, horizon, output_dt, order, cfl, 1)
    if (
        not np.isfinite([max_working_gb, disk_reserve_gb]).all()
        or max_working_gb <= 0
        or disk_reserve_gb < 0
    ):
        raise ValueError("Invalid memory/disk guards")
    import cupy as cp

    cp.cuda.Device(device).use()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve existing output: {output}")
    times = output_times(horizon, output_dt)
    N = grid.states
    working = 12 * N
    storage = 4 * N * ((2 if save_final else 0) + (len(times) if history else 0)) + (
        0 if probes is None else 8 * len(probes) * len(times)
    )
    if working > max_working_gb * 1e9:
        raise MemoryError(
            f"GPU core arrays need {working/1e9:.3f} GB, above guard {max_working_gb}"
        )
    free_device, _ = cp.cuda.runtime.memGetInfo()
    if working + max(256 * 1024**2, N) > free_device:
        raise MemoryError(
            "Insufficient free GPU memory including reduction-workspace reserve"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if storage + disk_reserve_gb * 1e9 > shutil.disk_usage(output.parent).free:
        raise OSError("Insufficient disk space after reserve")
    output.mkdir(exist_ok=True)
    code = source_code(local_precision)
    digest = hashlib.sha256(code.encode()).hexdigest()
    (output / "kernel.cu").write_text(code)
    kernel = cp.RawKernel(code, "advance", options=("--std=c++17", "--fmad=false"))
    initialize = cp.RawKernel(
        code, "initialize", options=("--std=c++17", "--fmad=false")
    )
    started = time.perf_counter()
    kernel.compile()
    initialize.compile()
    compile_seconds = time.perf_counter() - started
    p, e = config.pursuer, config.evader
    dx = 2 * grid.extent / (grid.positions - 1)
    dphi = 2 * np.pi / grid.headings
    dvp = (p.speed_max - p.speed_min) / (grid.p_speeds - 1)
    dve = (e.speed_max - e.speed_min) / (grid.e_speeds - 1)
    VP = max(abs(p.speed_min), abs(p.speed_max))
    VE = max(abs(e.speed_min), abs(e.speed_max))
    cfl_dt = cfl / (
        (VP + VE + VP * p.max_curvature * grid.extent) / dx
        + (VE + VP * p.max_curvature * grid.extent) / dx
        + (VP * p.max_curvature + VE * e.max_curvature) / dphi
        + p.acceleration / dvp
        + e.acceleration / dve
    )
    axes = grid.axes(config)
    coord = -grid.extent + np.arange(grid.positions) * dx
    initial2 = (
        np.hypot(coord[:, None], coord[None, :]) - config.capture_radius
    ).astype(np.float32)
    start = time.perf_counter()
    plane = cp.asarray(initial2.ravel())
    a = cp.empty(N, dtype=cp.float32)
    b = cp.empty_like(a)
    arrival = cp.empty_like(a)
    initialize(
        ((N + 255) // 256,),
        (256,),
        (plane, a, arrival, np.int64(N), np.int64(initial2.size)),
    )
    real = np.float32 if local_precision == "float32" else np.float64
    angles = [-math.pi + i * dphi for i in range(grid.headings)]
    sn = cp.asarray([math.sin(q) for q in angles], dtype=real)
    cs = cp.asarray([math.cos(q) for q in angles], dtype=real)
    probe_indices = probe_weights = probe_states = None
    if probes is not None:
        from itertools import product

        probe_states = np.asarray(probes, dtype=float)
        if (
            probe_states.ndim != 2
            or probe_states.shape[1] != 5
            or len(probe_states) > 100000
            or not np.isfinite(probe_states).all()
        ):
            raise ValueError("Probes must be finite relative states[n,5]")
        bits = np.array(list(product([0, 1], repeat=5)), np.int64)
        strides = np.cumprod((1,) + grid.shape[:0:-1])[::-1]
        indices = np.zeros((len(probe_states), 32), np.int64)
        weights = np.ones((len(probe_states), 32))
        reordered = probe_states[:, [3, 4, 2, 1, 0]]
        for k, (axis, n) in enumerate(zip(axes, grid.shape)):
            h = axis[1] - axis[0]
            q = reordered[:, k]
            if k == 2:
                f = ((q + np.pi) % (2 * np.pi)) / h
                j = np.floor(f).astype(np.int64) % n
                w = f - np.floor(f)
            else:
                if np.any((q < axis[0]) | (q > axis[-1])):
                    raise ValueError("A probe lies outside the grid")
                f = (q - axis[0]) / h
                j = np.minimum(np.floor(f).astype(np.int64), n - 2)
                w = f - j
            coord = j[:, None] + bits[None, :, k]
            if k == 2:
                coord %= n
            indices += coord * strides[k]
            weights *= np.where(bits[None, :, k], w[:, None], 1 - w[:, None])
        probe_indices = cp.asarray(indices)
        probe_weights = cp.asarray(weights)
    cp.cuda.Stream.null.synchronize()
    allocation_upload_seconds = time.perf_counter() - start
    props = cp.cuda.runtime.getDeviceProperties(device)
    name = props["name"]
    name = name.decode() if isinstance(name, bytes) else str(name)
    meta = {
        "schema": "reversible_relative5d_v1",
        "backend": "cupy_rawkernel",
        "local_precision": local_precision,
        "config": config.to_dict(),
        "physics": physics_signature(config),
        "grid": grid.to_dict(),
        "shape": grid.shape,
        "state_order": ["x", "y", "psi", "vP", "vE"],
        "axis_order": ["vP", "vE", "psi", "y", "x"],
        "axes": [q.tolist() for q in axes],
        "horizon": horizon,
        "output_dt": output_dt,
        "times": times.tolist(),
        "order": order,
        "cfl": cfl,
        "cfl_dt_limit": cfl_dt,
        "history_saved": bool(history),
        "dtype": "float32",
        "target": "physical_capture_only",
        "avoid": None,
        "tube_projection": "min over every internal update",
        "position_boundary": "one-sided linear extrapolation; outside queries unresolved",
        "speed_boundary": "acceleration restricted to inward or zero at speed limits",
        "heading_boundary": "periodic",
        "kernel_sha256": digest,
        "kernel_file": "kernel.cu",
        "compile_seconds": compile_seconds,
        "allocation_upload_seconds": allocation_upload_seconds,
        "predicted_working_bytes": working,
        "predicted_output_bytes": storage,
        "device": name,
        "cupy_version": cp.__version__,
        "status": "running",
        "final_saved": bool(save_final),
        "probe_states": None if probe_states is None else probe_states.tolist(),
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    scalar_args = (
        np.int64(N),
        np.int32(grid.positions),
        np.int32(grid.headings),
        np.int32(grid.p_speeds),
        np.int32(grid.e_speeds),
        *[
            real(v)
            for v in (
                grid.extent,
                p.speed_min,
                p.speed_max,
                e.speed_min,
                e.speed_max,
                p.acceleration,
                e.acceleration,
                p.max_curvature,
                e.max_curvature,
            )
        ],
    )
    t = 0.0
    steps = 0
    update_seconds = 0.0
    transfer_seconds = 0.0
    write_seconds = 0.0
    frames = 1
    probe_seconds = 0.0
    probe_file = (
        (output / "probes.f64").open("wb") if probe_states is not None else None
    )

    def save_probes():
        nonlocal probe_seconds
        if probe_file is not None:
            before = time.perf_counter()
            cp.asnumpy((a[probe_indices] * probe_weights).sum(axis=1)).astype(
                np.float64
            ).tofile(probe_file)
            probe_file.flush()
            probe_seconds += time.perf_counter() - before

    save_probes()
    history_file = (output / "fields.f32").open("wb") if history else None
    host = np.empty(N, np.float32) if history or save_final else None
    if history_file:
        start = time.perf_counter()
        a.get(out=host)
        transfer_seconds += time.perf_counter() - start
        start = time.perf_counter()
        host.tofile(history_file)
        write_seconds += time.perf_counter() - start
    start_all = time.perf_counter()
    try:
        with (output / "progress.jsonl").open("w") as progress:
            for layer, goal in enumerate(times[1:], start=1):
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                begin.record()
                while t < goal - 1e-12:
                    dt = min(cfl_dt, goal - t)
                    args = (*scalar_args, real(dt), real(t), np.int32(order))
                    kernel(
                        ((N + 127) // 128,),
                        (128,),
                        (a, b, arrival, sn, cs, *args, np.int32(0)),
                    )
                    if order == 2:
                        kernel(
                            ((N + 127) // 128,),
                            (128,),
                            (b, a, arrival, sn, cs, *args, np.int32(1)),
                        )
                    else:
                        a, b = b, a
                    t += dt
                    steps += 1
                end.record()
                end.synchronize()
                update_seconds += cp.cuda.get_elapsed_time(begin, end) / 1000
                if history_file:
                    start = time.perf_counter()
                    a.get(out=host)
                    transfer_seconds += time.perf_counter() - start
                    if not np.isfinite(host).all():
                        raise FloatingPointError("Nonfinite field")
                    start = time.perf_counter()
                    host.tofile(history_file)
                    write_seconds += time.perf_counter() - start
                save_probes()
                frames += 1
                if layer % 5 == 0 or layer == len(times) - 1:
                    resolved = int(cp.count_nonzero(cp.isfinite(arrival)).item())
                    row = dict(
                        time=float(goal),
                        steps=steps,
                        resolved=resolved,
                        elapsed_seconds=time.perf_counter() - start_all,
                        total_wall_seconds=time.perf_counter() - solve_started,
                    )
                    progress.write(json.dumps(row) + "\n")
                    progress.flush()
        if not bool(cp.isfinite(a).all().item()):
            raise FloatingPointError("Nonfinite final field")
        if save_final:
            for device_array, filename in ((a, "final.f32"), (arrival, "arrival.f32")):
                start = time.perf_counter()
                device_array.get(out=host)
                transfer_seconds += time.perf_counter() - start
                start = time.perf_counter()
                host.tofile(output / filename)
                write_seconds += time.perf_counter() - start
        meta.update(
            status="complete",
            states=N,
            working_array_bytes=working,
            internal_steps=steps,
            frames=frames,
            update_seconds=update_seconds,
            device_to_host_seconds=transfer_seconds,
            history_and_final_write_seconds=write_seconds,
            probe_seconds=probe_seconds,
            elapsed_seconds=time.perf_counter() - start_all,
            total_wall_seconds=time.perf_counter() - solve_started,
            gpu_pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
            gpu_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
        )
    except Exception as exc:
        meta.update(status="failed", failure=str(exc))
        raise
    finally:
        if history_file:
            history_file.close()
        if probe_file:
            probe_file.close()
        (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    import argparse
    from .model import ReversibleConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--config")
    parser.add_argument("--positions", type=int, default=61)
    parser.add_argument("--headings", type=int, default=32)
    parser.add_argument("--p-speeds", type=int, default=13)
    parser.add_argument("--e-speeds", type=int, default=7)
    parser.add_argument("--extent", type=float, default=4)
    parser.add_argument("--horizon", type=float, default=4)
    parser.add_argument("--output-dt", type=float, default=0.1)
    parser.add_argument("--order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--local-precision", choices=["float32", "float64"], default="float64"
    )
    parser.add_argument("--max-working-gb", type=float, default=6)
    args = parser.parse_args()
    config = (
        ReversibleConfig.from_dict(json.loads(Path(args.config).read_text()))
        if args.config
        else ReversibleConfig()
    )
    g = RelativeGrid(
        args.positions, args.headings, args.p_speeds, args.e_speeds, args.extent
    )
    m = solve(
        config,
        g,
        args.output,
        horizon=args.horizon,
        output_dt=args.output_dt,
        order=args.order,
        history=not args.no_history,
        local_precision=args.local_precision,
        max_working_gb=args.max_working_gb,
    )
    print(
        json.dumps(
            {
                k: m[k]
                for k in [
                    "status",
                    "states",
                    "update_seconds",
                    "device_to_host_seconds",
                    "working_array_bytes",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
