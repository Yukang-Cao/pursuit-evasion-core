"""Replay one local footprint-game chase using a numerical or learned P value.

The actual E uses the declared one-step numerical-reference margin response.
This is a diagnostic controller comparison, not an optimal-opponent benchmark.
Requires a retained reference history and optionally a trained checkpoint.
"""

from pathlib import Path
import argparse, json
import numpy as np
from pursuit_evasion.reversible.map_game_field import MapGameField
from pursuit_evasion.reversible.map_game_policy import (
    successor_matrix,
    margin_costs,
    select_time_or_margin,
)
from pursuit_evasion.reversible.evaluation import json_safe

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--checkpoint", type=Path)
p.add_argument("--device", default="cpu")
p.add_argument("--state", nargs=8, type=float, required=True)
p.add_argument("--max-time", type=float, default=6)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if not np.isfinite(a.max_time) or a.max_time <= 0:
    raise ValueError("max-time must be positive and finite")
if a.output.exists():
    raise FileExistsError("Choose a new output file")
ref = MapGameField(a.reference)
source = ref
if a.checkpoint:
    from pursuit_evasion.reversible.neural_map_field import NeuralMapField

    source = NeuralMapField(a.checkpoint, ref.config, ref.geometry, device=a.device)
    if a.device.startswith("cuda"):
        import torch

        torch.set_float32_matmul_precision("high")
s = np.asarray(a.state)
t = 0.0
trace = []
while t < a.max_time - 1e-10:
    P, E, records, states, codes = successor_matrix(s, ref.config, ref.geometry)
    pi, mode = select_time_or_margin(P, records, states, codes, source)
    ei = int(np.argmax(margin_costs(ref.field(states), codes)[pi]))
    r = records[pi][ei]
    trace.append(
        dict(
            t=t,
            state=s.tolist(),
            p_control=P[pi].tolist(),
            e_control=E[ei].tolist(),
            p_selection=mode,
        )
    )
    s = r.state
    t += r.elapsed
    if r.terminal:
        break
trace.append(dict(t=t, state=s.tolist()))
result = dict(
    event=r.event or "timeout",
    end_time=t,
    trace=trace,
    initial=a.state,
    config=ref.config.to_dict(),
    geometry=ref.geometry.to_dict(),
    pursuer="learned value" if a.checkpoint else "numerical reference",
    evader="one-step numerical-reference signed-margin response after observing P command",
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(json_safe(result), indent=2))
print(f"{result['event']} at {t:.3f} s; {len(trace)-1} decisions")
