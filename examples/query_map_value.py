"""Query the learned value using a map and state, without opening a DP field."""

from pathlib import Path
import argparse, json
import numpy as np
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.footprint_map import FootprintMap
from pursuit_evasion.reversible.planning_frame import PlanningFrame

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument(
    "--suite",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "configs/map_value_pilot_suite.json",
)
p.add_argument("--map", default="test_17")
p.add_argument("--state", nargs=8, type=float, required=True, metavar="STATE")
p.add_argument("--budget", type=float, default=2)
p.add_argument("--frame", nargs=3, type=float, default=[0, 0, 0], metavar="FRAME")
p.add_argument("--device", default="cpu")
args = p.parse_args()
data = json.loads(args.suite.read_text())
config = ReversibleConfig.from_dict(data["config"])
entry = next(e for e in data["maps"] if e["id"] == args.map)
geometry = FootprintMap.from_dict(entry["geometry"])
from pursuit_evasion.reversible.neural_map_field import NeuralMapField

source = NeuralMapField(
    args.checkpoint,
    config,
    geometry,
    device=args.device,
    frame=PlanningFrame(*args.frame),
)
value = float(source.field(np.asarray(args.state), args.budget))
if not np.isfinite(value):
    raise ValueError("Query is outside the supported spatial/speed range")
print(f"Signed P-success value at budget {args.budget:g} s: {value:+.6f}")
print("This is a learned game-field estimate, not capture time or a guarantee.")
