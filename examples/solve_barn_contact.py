"""Generate an8D local contact-game reference on an original BARN occupancy grid."""

from pathlib import Path
from dataclasses import replace
import argparse, json
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.local_game.geometry import CircleArena
from pursuit_evasion.local_game.reference import solve

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--grid", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--scale", type=float, default=2)
p.add_argument("--horizon", type=float, default=2)
p.add_argument("--position-step", type=float, default=0.3)
p.add_argument("--headings", type=int, default=16)
p.add_argument("--retain-history", action="store_true")
p.add_argument("--seed", type=int, default=1031)
a = p.parse_args()
c = ReversibleConfig()
arena = CircleArena.from_barn(a.grid, scale=a.scale)
c = replace(
    c,
    name="barn_contact_a3",
    pursuer=replace(c.pursuer, acceleration=3),
    evader=replace(c.evader, acceleration=3),
    arena_half_width=arena.half_width,
    point_vehicles=False,
    capture_rule="rectangle_contact",
    boundary_rule="collision_constrained",
)
result = solve(
    c,
    arena,
    a.output,
    headings=a.headings,
    position_step=a.position_step,
    horizon=a.horizon,
    retain_history=a.retain_history,
    seed=a.seed,
)
print(
    f"{result['status']}: {result['pair_states']:,} eight-state configurations; GPU working storage {result['gpu_working_bytes']/1e9:.2f} GB"
)
