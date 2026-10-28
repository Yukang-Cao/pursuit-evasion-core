"""Convert auditable teacher episodes into direct-action and waypoint labels.

This exports demonstrations; it does not train a policy. DP-owned actions remain
in the raw record but are masked out of the approach-policy imitation target.
"""

import json
from pathlib import Path
import numpy as np

OBSERVATION_NAMES = [
    "p_x_scaled",
    "p_y_scaled",
    "sin_p_heading",
    "cos_p_heading",
    "e_x_scaled",
    "e_y_scaled",
    "sin_e_heading",
    "cos_e_heading",
    "anchor_x_scaled",
    "anchor_y_scaled",
    "previous_p_turn_scaled",
]


def export_demonstrations(episode_paths, output, test_cases=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "dataset.npz").exists():
        raise FileExistsError(output / "dataset.npz")
    observations = []
    commands = []
    targets = []
    actor_masks = []
    episode_ids = []
    splits = []
    teacher_update_states = []
    teacher_context_times = []
    manifest = []
    for eid, path in enumerate(episode_paths):
        path = Path(path)
        d = json.loads(path.read_text())
        L = d["config"]["arena_half_width"]
        bound = d["config"]["pursuer"]["max_turn_rate"]
        anchor = np.asarray(d["initial"][:2])
        previous = 0.0
        split = int(d["case"] in test_cases)
        begin = len(observations)
        for row in d["trace"][:-1]:
            s = row["state"]
            u = row["p_turn"]
            owner = row["pursuer"]["control_owner"]
            observations.append(
                [
                    s[0] / L,
                    s[1] / L,
                    np.sin(s[2]),
                    np.cos(s[2]),
                    s[3] / L,
                    s[4] / L,
                    np.sin(s[5]),
                    np.cos(s[5]),
                    anchor[0] / L,
                    anchor[1] / L,
                    previous / bound,
                ]
            )
            commands.append(u / bound)
            target = row["pursuer"].get("lion", {}).get("target", [np.nan, np.nan])
            targets.append(np.asarray(target) / L)
            # Accept historical episode records without relabeling their evidence.
            actor_masks.append(owner in ("lion_mppi", "teacher"))
            context = row["pursuer"].get("lion", {})
            teacher_update_states.append(
                context.get("observation_at_update", [np.nan] * 6)
            )
            teacher_context_times.append(
                [
                    row["time"] - context.get("target_updated_at", row["time"]),
                    row["pursuer"].get("cooldown_remaining", 0.0),
                ]
            )
            episode_ids.append(eid)
            splits.append(split)
            previous = u
        manifest.append(
            dict(
                episode=eid,
                file=str(path.resolve()),
                case=d["case"],
                seed=d["seed"],
                outcome=d["outcome"],
                physical_capture=d["physical_capture"],
                split="test" if split else "train",
                row_start=begin,
                row_stop=len(observations),
            )
        )
    np.savez_compressed(
        output / "dataset.npz",
        observations=np.asarray(observations, np.float32),
        command_targets=np.asarray(commands, np.float32),
        waypoint_targets=np.asarray(targets, np.float32),
        approach_mask=np.asarray(actor_masks, bool),
        episode_ids=np.asarray(episode_ids, np.int32),
        test_mask=np.asarray(splits, bool),
        teacher_observation_at_update=np.asarray(teacher_update_states, np.float32),
        teacher_goal_age_and_cooldown=np.asarray(teacher_context_times, np.float32),
    )
    meta = dict(
        status="demonstration export only; no model trained",
        observation_names=OBSERVATION_NAMES,
        command_units="normalized P turn rate; multiply by configured max_turn_rate",
        waypoint_units="world xy divided by arena_half_width",
        total_rows=len(observations),
        approach_rows=int(sum(actor_masks)),
        episodes=manifest,
        test_cases=list(test_cases),
        leakage_rule="whole named starting configurations assigned to one split; no teacher target in observations",
        retained_failures=True,
        DP_actions_masked_for_approach_imitation=True,
        observation_scope="Compact observations are not asserted to reproduce a stateful MPPI teacher exactly. Raw episodes and explicit target-update history are retained for recurrent/augmented students.",
    )
    (output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
