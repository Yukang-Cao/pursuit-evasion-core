"""BARN chase wrapper with an explicit approach/local-finish handoff.

P normally requires finite continuation times against every admissible E reply.
The optional use_contact_cost mode also executes supported finite-budget contact
cost guidance when time is unresolved. It is an experimental approach heuristic,
not a capture guarantee. E uses the same time-first objective locally; outside
the local domain both players share route-guided navigation.
"""

import time
import numpy as np
from .native import GeometryEngine, capture_times
from .control import admissible_controls, pair_predictions, costs, choose_p, choose_e
from .navigation import RouteGuide, FootprintMPPI
from .perception import LidarMemory, SensorSettings
from ..reversible.model import joint_step, vehicle_step


def safe_projection(pose, proposed, engine, dt, controls=None):
    legal, _ = engine.actions(pose, np.asarray(proposed)[None], dt)
    if legal[0, 0]:
        return np.asarray(proposed), False
    if controls is None:
        controls, _ = admissible_controls(pose, engine, dt)
    scale = np.array([engine.limits.acceleration, engine.limits.max_curvature])
    index = np.argmin(np.sum(((controls - proposed) / scale) ** 2, axis=1))
    return controls[index], True


def chase(
    initial,
    reference,
    p_source=None,
    e_source=None,
    *,
    max_time=20.0,
    seed=0,
    value_observation="memory",
    oracle_navigation=False,
    approach_only=False,
    audit_reference=False,
    use_contact_cost=False,
):
    config = reference.config
    arena = reference.arena
    for source in [p_source, e_source]:
        if source is not None and hasattr(source, "p_body"):
            if source.p_body != arena.pursuer or source.e_body != arena.evader:
                raise ValueError(
                    "Value-source footprints differ from the physical game"
                )
    dt = config.control_dt
    radius = reference.domain.exit_radius
    p_source = reference if p_source is None else p_source
    e_source = reference if e_source is None else e_source
    p_engine = GeometryEngine(
        arena, config.pursuer, "pursuer", threads=1, control_dt=dt
    )
    e_engine = GeometryEngine(arena, config.evader, "evader", threads=1, control_dt=dt)
    s = np.asarray(initial, float).copy()
    if not p_engine.backup_safe(s[:4]) or not e_engine.backup_safe(s[4:]):
        raise ValueError("Initial state violates braking constraint")
    memories = [LidarMemory(arena), LidarMemory(arena)]
    guides = [RouteGuide(arena, "pursuer"), RouteGuide(arena, "evader")]
    trackers = [
        FootprintMPPI(config, arena, "pursuer", seed),
        FootprintMPPI(config, arena, "evader", seed + 10000),
    ]
    previous = ["none", "none"]
    trace = []
    t = 0.0
    event = None
    started = time.perf_counter()
    handoffs = []
    while t < max_time - 1e-10:
        if arena.capture_value(s) <= 1e-9:
            event = "capture"
            break
        tick = time.perf_counter()
        for j in [0, 1]:
            memories[j].observe(s[4 * j : 4 * j + 4])
        if hasattr(p_source, "prepare") or hasattr(e_source, "prepare"):
            observation = memories[0]
            if value_observation == "single":
                observation = LidarMemory(arena)
                observation.observe(s[:4])
            image, frame = observation.image(s[:4], ideal=value_observation == "ideal")
            prepared = set()
            for source in [p_source, e_source]:
                if hasattr(source, "prepare") and id(source) not in prepared:
                    source.prepare(image, frame)
                    prepared.add(id(source))
        sensing = time.perf_counter() - tick
        P, E, z, contact = pair_predictions(s, config, arena, p_engine, e_engine)
        near = (
            np.linalg.norm(s[:2] - s[4:6])
            <= radius
            + (config.pursuer.max_abs_speed + config.evader.max_abs_speed) * dt
        )
        pt = pm = pvalid = None
        p_time = None
        waypoint = None
        p_override = False
        e_override = False
        if near and not approach_only:
            pt, pm, pvalid = costs(p_source, z, contact, dt, radius)
            pi, replies, wt, wm = choose_p(pt, pm, P, E)
            finish = np.isfinite(wt[pi])
        else:
            finish = False
        local_progress = bool(
            use_contact_cost
            and not finish
            and near
            and not approach_only
            and pvalid is not None
            and pvalid[pi].all()
            and np.isfinite(wm[pi])
            and wm[pi] < radius - 1e-7
        )
        if finish or local_progress:
            pu = P[pi]
            p_mode = (
                "learned finish"
                if hasattr(p_source, "prepare")
                else (
                    "free-space finish"
                    if getattr(p_source, "name", "") == "free-space DP"
                    else "numerical finish"
                )
            )
            if local_progress:
                p_mode = (
                    "learned local progress"
                    if hasattr(p_source, "prepare")
                    else "DP local progress"
                )
            p_time = float(wt[pi]) if np.isfinite(wt[pi]) else None
            if previous[0] == "approach":
                handoffs.append(dict(t=t, state=s.tolist(), predicted_time=p_time))
        else:
            p_mode = "approach"
            guides[0].update(memories[0], ideal=oracle_navigation)
            waypoint, heading, _ = guides[0].pursuit(s[:4], s[4:])
            if previous[0] != "approach":
                trackers[0].reset()
            trackers[0].update_geometry(
                arena if oracle_navigation else memories[0].observed_arena()
            )
            proposal = trackers[0].command_pose(s[:4], waypoint, heading)
            pu, p_override = safe_projection(s[:4], proposal, p_engine, dt, P)
        pn = vehicle_step(s[:4], pu, dt, config.pursuer)
        en = vehicle_step(s[4:][None], E, dt, config.evader)
        children = np.concatenate([np.broadcast_to(pn, (len(E), 4)), en], -1)[None]
        hits = capture_times(s, pu[None], E, dt, p_engine, e_engine)
        local_e = (
            np.any(
                np.linalg.norm(children[..., :2] - children[..., 4:6], axis=-1)
                <= radius
            )
            | np.isfinite(hits).any()
        )
        e_predicted_time = None
        e_time_unresolved = None
        if local_e:
            if (finish or local_progress) and p_source is e_source:
                et, em = pt[pi], pm[pi]
            else:
                et, em, _ = costs(e_source, children, hits, dt, radius)
                et, em = et[0], em[0]
            ei = choose_e(et, em, E)
            eu = E[ei]
            e_predicted_time = float(et[ei]) if np.isfinite(et[ei]) else None
            e_time_unresolved = bool(not np.isfinite(et[ei]))
            e_mode = (
                "learned response"
                if hasattr(e_source, "prepare")
                else "numerical response"
            )
        else:
            e_mode = "escape navigation"
            guides[1].update(memories[1], ideal=oracle_navigation)
            target, heading, _ = guides[1].escape(s[4:], s[:4])
            if previous[1] != "escape navigation":
                trackers[1].reset()
            trackers[1].update_geometry(
                arena if oracle_navigation else memories[1].observed_arena()
            )
            proposal = trackers[1].command_pose(s[4:], target, heading)
            eu, e_override = safe_projection(s[4:], proposal, e_engine, dt, E)
        core_seconds = time.perf_counter() - tick
        audit = {}
        audit_start = time.perf_counter()
        if audit_reference and near:
            if p_source is reference and pt is not None:
                rt, rm = pt, pm
            else:
                rt, rm, _ = costs(reference, z, contact, dt, radius)
            rpi, _, rwt, _ = choose_p(rt, rm, P, E)
            chosen_t, chosen_m, chosen_valid = costs(
                reference, children, hits, dt, radius
            )
            inside_replies = (
                np.linalg.norm(children[..., :2] - children[..., 4:6], axis=-1)
                <= radius
            )
            rei = choose_e(chosen_t[0], chosen_m[0], E)
            worst = chosen_t[0, rei]
            actual_indices = np.flatnonzero(
                np.all(np.isclose(E, eu, atol=1e-8, rtol=0), axis=1)
            )
            actual_time = (
                chosen_t[0, actual_indices[0]] if len(actual_indices) else np.nan
            )
            audit = dict(
                reference_has_finish=bool(np.isfinite(rwt[rpi])),
                reference_inside_reply_count=int(inside_replies.sum()),
                reference_unsupported_inside_replies=int(
                    (inside_replies & ~chosen_valid).sum()
                ),
                reference_best_time=float(rwt[rpi]) if np.isfinite(rwt[rpi]) else None,
                reference_best_command=P[rpi].tolist(),
                reference_time_after_chosen_P=(
                    float(worst) if np.isfinite(worst) else None
                ),
                reference_E_has_unresolved_reply=bool(np.isinf(worst)),
                reference_time_after_chosen_E=(
                    float(actual_time) if np.isfinite(actual_time) else None
                ),
            )
        audit_seconds = time.perf_counter() - audit_start
        actual = capture_times(s, pu[None], eu[None], dt, p_engine, e_engine)[0, 0]
        elapsed = min(dt, float(actual), max_time - t)
        trace.append(
            dict(
                t=t,
                state=s.tolist(),
                p_control=pu.tolist(),
                e_control=eu.tolist(),
                p_mode=p_mode,
                e_mode=e_mode,
                e_predicted_capture_time=e_predicted_time,
                e_capture_time_unresolved=e_time_unresolved,
                waypoint=None if waypoint is None else waypoint.tolist(),
                predicted_finish_time=p_time,
                local_best_worst_contact_measure=(
                    float(wm[pi]) if pt is not None else None
                ),
                local_value_supported=(
                    bool(pvalid[pi].all()) if pvalid is not None else None
                ),
                p_safety_projection=p_override,
                e_safety_projection=e_override,
                p_supported_fraction=None if pvalid is None else float(pvalid.mean()),
                sensing_seconds=sensing,
                decision_seconds=core_seconds,
                audit_seconds=audit_seconds,
                **audit,
            )
        )
        s = joint_step(s, pu, eu, elapsed, config)
        t += elapsed
        previous = [p_mode, e_mode]
        if (
            arena.clearance(s[:4], "pursuer") < -1e-8
            or arena.clearance(s[4:], "evader") < -1e-8
        ):
            event = "obstacle_contact"
            break
        if np.isfinite(actual) and actual <= elapsed + 1e-10:
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
        handoffs=handoffs,
        p_source=getattr(p_source, "name", "numerical DP"),
        e_source=getattr(e_source, "name", "numerical DP"),
        value_observation=value_observation,
        use_contact_cost=bool(use_contact_cost),
        navigation=(
            "observed-surface route-guided log-MPPI"
            if not oracle_navigation
            else "full-map navigation diagnostic"
        ),
        safety="shared privileged physical braking feasibility; obstacle contacts are failures, never captures",
        model_scope="local expert2s/range3m, common approach otherwise; exact joint state; static-map noiseless LiDAR",
        process_seconds=time.perf_counter() - started,
    )
