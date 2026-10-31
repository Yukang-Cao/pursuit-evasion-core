"""Continuous contact replays with explicit local-domain and collision outcomes."""

import time
import numpy as np
from .native import GeometryEngine
from .control import pair_predictions, costs, choose_p, choose_e
from .perception import LidarMemory, SensorSettings
from ..reversible.model import joint_step


def local_episode(
    initial,
    reference,
    p_source=None,
    e_source=None,
    *,
    max_time=15.0,
    sensor_settings=SensorSettings(),
    ideal_input=False,
):
    config = reference.config
    arena = reference.arena
    p_source = reference if p_source is None else p_source
    e_source = reference if e_source is None else e_source
    s = np.asarray(initial, float).copy()
    p_engine = GeometryEngine(
        arena, config.pursuer, "pursuer", threads=1, control_dt=config.control_dt
    )
    e_engine = GeometryEngine(
        arena, config.evader, "evader", threads=1, control_dt=config.control_dt
    )
    if not p_engine.backup_safe(s[:4]) or not e_engine.backup_safe(s[4:]):
        raise ValueError("Initial state lacks the declared braking continuation")
    memories = [
        LidarMemory(arena, sensor_settings),
        LidarMemory(arena, sensor_settings),
    ]
    trace = []
    t = 0.0
    event = None
    began = time.perf_counter()
    while t < max_time - 1e-10:
        if arena.capture_value(s) <= 1e-9:
            event = "capture"
            break
        if np.linalg.norm(s[:2] - s[4:6]) > reference.domain.exit_radius + 1e-8:
            event = "local_range_exit"
            break
        started = time.perf_counter()
        sensor_time = 0.0
        tick = time.perf_counter()
        memories[0].observe(s[:4])
        if hasattr(p_source, "prepare") or hasattr(e_source, "prepare"):
            image, frame = memories[0].image(s[:4], ideal=ideal_input)
            prepared = set()
            for source in [p_source, e_source]:
                if hasattr(source, "prepare") and id(source) not in prepared:
                    source.prepare(image, frame)
                    prepared.add(id(source))
        sensor_time = time.perf_counter() - tick
        P, E, states, contact = pair_predictions(s, config, arena, p_engine, e_engine)
        pt, pm, pvalid = costs(
            p_source, states, contact, config.control_dt, reference.domain.exit_radius
        )
        pi, replies, wt, wm = choose_p(pt, pm, P, E)
        if p_source is e_source:
            et, em, evalid = pt, pm, pvalid
        else:
            et, em, evalid = costs(
                e_source,
                states[pi : pi + 1],
                contact[pi : pi + 1],
                config.control_dt,
                reference.domain.exit_radius,
            )
            et, em = et[0], em[0]
        ei = choose_e(
            et[pi] if p_source is e_source else et,
            em[pi] if p_source is e_source else em,
            E,
        )
        elapsed = min(config.control_dt, float(contact[pi, ei]))
        trace.append(
            dict(
                t=t,
                state=s.tolist(),
                p_control=P[pi].tolist(),
                e_control=E[ei].tolist(),
                p_mode=getattr(p_source, "name", "numerical DP"),
                e_mode=getattr(e_source, "name", "numerical DP"),
                p_predicted_worst_time=float(wt[pi]) if np.isfinite(wt[pi]) else None,
                p_predicted_worst_margin=float(wm[pi]),
                p_controls=len(P),
                e_controls=len(E),
                p_supported_fraction=float(pvalid.mean()),
                e_supported_fraction=float(evalid.mean()),
                sensing_seconds=sensor_time,
                decision_seconds=time.perf_counter() - started,
            )
        )
        s = joint_step(s, P[pi], E[ei], elapsed, config)
        t += elapsed
        if (
            arena.clearance(s[:4], "pursuer") < -1e-8
            or arena.clearance(s[4:], "evader") < -1e-8
        ):
            event = "obstacle_contact"
            break
        if np.isfinite(contact[pi, ei]):
            event = "capture"
            break
    trace.append(dict(t=t, state=s.tolist()))
    return dict(
        initial=np.asarray(initial).tolist(),
        event=event or "timeout",
        end_time=t,
        trace=trace,
        config=config.to_dict(),
        arena=arena.to_dict(),
        local_radius=reference.domain.exit_radius,
        objective="capture before local-range exit; local diagnostic, not complete global chase",
        p_source=getattr(p_source, "name", "numerical DP"),
        e_source=getattr(e_source, "name", "numerical DP"),
        perception="fresh P-body-frame LiDAR with accumulated static observations; shared value frame for P and E",
        safety="common privileged full-map braking-feasibility constraint; not sensor-only navigation",
        process_seconds=time.perf_counter() - began,
    )
