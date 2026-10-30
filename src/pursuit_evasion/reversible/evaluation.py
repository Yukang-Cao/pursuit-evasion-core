"""Recorded closed-loop games, with physical time separate from controller time."""

from pathlib import Path
import json
import hashlib
import time
import numpy as np

from .game import advance
from .model import relative_state


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def compact_decision(data):
    """Keep meaningful decisions, not a full sampled-bank dump at every tick."""
    out = {
        k: data[k]
        for k in (
            "control_owner",
            "takeover_rule",
            "dp_value",
            "shielded",
            "guard_failure",
            "seconds",
        )
        if k in data
    }
    if data.get("dp") is not None:
        out["dp"] = data["dp"]
    if "search" in data:
        # Keep per-decision search work/timing, not a full tree at every tick.
        out["search"] = {
            k: v
            for k, v in data["search"].items()
            if isinstance(v, (str, int, float, bool)) or v is None
        }
    if "lion" in data:
        out["lion"] = data["lion"]
    if "mppi" in data:
        out["mppi"] = {
            k: v
            for k, v in data["mppi"].items()
            if np.ndim(v) == 0 or isinstance(v, str)
        }
    return out


def run_episode(
    pursuer,
    evader,
    initial,
    config,
    world,
    *,
    max_time=20.0,
    case="case",
    seed=0,
    output=None,
):
    state = np.asarray(initial, float).copy()
    pursuer.reset(state)
    if hasattr(evader, "reset"):
        evader.reset()
    t = 0.0
    trace = []
    p_times = []
    e_times = []
    minimum = np.inf
    event = None
    owners = []
    started = time.perf_counter()
    for _ in range(int(np.ceil(max_time / config.control_dt)) + 1):
        if t >= max_time - 1e-10:
            break
        begin = time.perf_counter()
        u = np.asarray(pursuer.command(state, t, diagnostics=False))
        p_times.append(time.perf_counter() - begin)
        begin = time.perf_counter()
        v = np.asarray(evader.command(state, u))
        e_times.append(time.perf_counter() - begin)
        decision = compact_decision(pursuer.last)
        owners.append(decision.get("control_owner", "unknown"))
        trace.append(
            dict(
                t=t,
                state=state.tolist(),
                p_control=u.tolist(),
                e_control=v.tolist(),
                pursuer=decision,
                evader={
                    k: z
                    for k, z in getattr(evader, "last", {}).items()
                    if np.ndim(z) == 0 or isinstance(z, str)
                },
                p_seconds=p_times[-1],
                e_seconds=e_times[-1],
            )
        )
        result = advance(
            state, u, v, config, min(config.control_dt, max_time - t), world
        )
        state = result.state
        t += result.elapsed
        minimum = min(minimum, float(np.linalg.norm(state[:2] - state[4:6])))
        if result.terminal:
            event = result.event
            break
    trace.append(
        dict(
            t=t,
            state=state.tolist(),
            p_control=None,
            e_control=None,
            pursuer={},
            evader={},
        )
    )
    admissions = sum(
        x.startswith("dp") and (i == 0 or not owners[i - 1].startswith("dp"))
        for i, x in enumerate(owners)
    )
    sources = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in [
            "model.py",
            "game.py",
            "events_geometry.py",
            "controllers.py",
            "mppi.py",
            "_cfree_mppi.py",
            "opponents.py",
        ]
    }
    result = dict(
        source_sha256=sources,
        schema="reversible_episode_v1",
        case=case,
        seed=int(seed),
        config=config.to_dict(),
        world=world.to_dict(),
        initial=np.asarray(initial).tolist(),
        initial_relative=relative_state(initial).tolist(),
        controller=getattr(pursuer, "mode", type(pursuer).__name__),
        evader_policy=getattr(evader, "mode", type(evader).__name__),
        outcome=event or "timeout",
        physical_capture=event == "capture",
        end_time=t,
        minimum_separation=minimum,
        handoffs=int(admissions),
        dp_steps=sum(x.startswith("dp") for x in owners),
        steps=len(owners),
        guard_failures=sum(bool(x["pursuer"].get("guard_failure")) for x in trace[:-1]),
        shielded_steps=sum(bool(x["pursuer"].get("shielded")) for x in trace[:-1]),
        controller_latency=dict(
            p_median_ms=float(np.median(p_times)) * 1000,
            p_p95_ms=float(np.percentile(p_times, 95)) * 1000,
            p_max_ms=float(max(p_times)) * 1000,
            e_median_ms=float(np.median(e_times)) * 1000,
            e_p95_ms=float(np.percentile(e_times, 95)) * 1000,
        ),
        process_seconds=time.perf_counter() - started,
        trace=trace,
    )
    result = json_safe(result)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(
            json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
        )
        tmp.replace(output)
    return result
