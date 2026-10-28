"""Reproducible episode recording, without plotting or research-report dependencies."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .game import advance


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_episode(
    config,
    pursuer,
    evader,
    initial,
    *,
    max_time=20.0,
    seed=0,
    case_name="case",
    diagnostic_stride=10,
):
    state = np.asarray(initial, dtype=float).copy()
    pursuer.reset(state)
    elapsed = 0.0
    trace = []
    event = None
    started = time.perf_counter()
    p_times = []
    e_times = []
    initial_event = advance(state, 0.0, 0.0, config, duration=0.0)
    if initial_event.terminal:
        event = initial_event.event
    while elapsed < max_time - 1e-10:
        if event is not None:
            break
        t = time.perf_counter()
        u = pursuer.command(
            state,
            elapsed,
            diagnostics=diagnostic_stride > 0 and len(trace) % diagnostic_stride == 0,
        )
        p_ms = 1000 * (time.perf_counter() - t)
        t = time.perf_counter()
        v = evader.command(state, u)
        e_ms = 1000 * (time.perf_counter() - t)
        p_times.append(p_ms)
        e_times.append(e_ms)
        record = dict(
            time=elapsed,
            state=state.tolist(),
            p_turn=u,
            e_turn=v,
            pursuer=pursuer.last,
            evader=evader.last,
            p_controller_ms=p_ms,
            e_controller_ms=e_ms,
        )
        # Copy nested diagnostics now; controller-owned dictionaries change later.
        trace.append(json_value(record))
        transition = advance(
            state, u, v, config, min(config.control_dt, max_time - elapsed)
        )
        state = transition.state
        elapsed += transition.elapsed
        if transition.terminal:
            event = transition.event
            break
    trace.append(dict(time=elapsed, state=state.tolist(), event=event or "timeout"))

    def stats(values):
        a = np.asarray(values)
        return (
            dict(
                mean=float(a.mean()),
                p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)),
                maximum=float(a.max()),
                above_control_period=int((a > config.control_dt * 1000).sum()),
                calls=len(a),
            )
            if len(a)
            else {}
        )

    return json_value(
        dict(
            case=case_name,
            seed=seed,
            config=config.to_dict(),
            controller_settings={
                "mppi": asdict(pursuer.tracker.settings),
                "lion": asdict(pursuer.lion),
                "DP_enabled": pursuer.use_dp,
                "DP_P_actions": len(pursuer.dp.P) if pursuer.dp else None,
                "DP_E_actions": len(pursuer.dp.E) if pursuer.dp else None,
            },
            initial=list(initial),
            outcome=event or "timeout",
            physical_capture=event == "capture",
            pursuer_success=event in ("capture", "evader_boundary_fault"),
            capture_time=elapsed if event == "capture" else None,
            observed_game_seconds=elapsed,
            episode_wall_seconds=time.perf_counter() - started,
            p_controller_ms=stats(p_times),
            e_controller_ms=stats(e_times),
            handoffs=pursuer.handoffs,
            abandoned_finishes=pursuer.abandoned_finishes,
            trace=trace,
            timing_scope="laptop P/E controller calls; logical simulation time; latency not injected; no Jetson claim",
        )
    )


def save_episode(path, episode):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    if path.exists():
        raise FileExistsError(f"Preserve completed episode: {path}")
    path.write_text(json.dumps(json_value(episode), indent=2, allow_nan=False) + "\n")
