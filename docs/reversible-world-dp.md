# Reversible fixed-map DP

The physical model is shared with the relative solver in
`reversible/model.py`. The world field has axes
`[xP, yP, thetaP, vP, xE, yE, thetaE, vE]`. Speed is a state; acceleration
and curvature are controls. A command is held for the simulator's control
interval, while the HJI reference permits continuously updated controls.

## What the field means

The current map is a square. Its boundary convention is **first fault loses**:
the target is physical capture or an evader boundary fault; a pursuer boundary
fault is excluded. Thus the extracted time is a numerical **P-success time**,
not always a physical capture time. Report those outcomes separately.

`world_dp.hamiltonian` gives the analytic P-minimizing/E-maximizing expression.
The tests also enumerate control pairs and compare them with that expression,
including negative, zero, and limiting speeds. Acceleration pointing out of
the speed interval has zero physical effect at the corresponding speed bound.

`native_world.cpp` applies ENO1 or ENO2 derivatives, local Lax–Friedrichs
dissipation, and RK1 or RK2 time integration. Its three principal arrays are
the current field, next/stage field, and diagnostic node-entry time. The
array-size calculation is not a claim about every backend's working memory.

Speed-axis boundaries use inward one-sided slopes. They are physical state
limits, unlike a spatial domain truncation. For example, at maximum P speed,
all admissible Bellman successors have the same or lower speed. Generic
signed-distance extrapolation can reflect a negative speed derivative when
the field is positive, introducing an artificial LF contribution. A direct
finite-hold generator test checks the corrected boundary independently.
Preliminary signed-speed-boundary tables are rejected by `WorldTable` unless
explicitly opened for diagnostics with `allow_legacy_speed_boundary=True`.

## Small reproducible solve

From an environment with the base package installed and `g++` available:

```python
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.world_dp import WorldGrid, solve
from pursuit_evasion.reversible.world_table import WorldTable

config = ReversibleConfig()
grid = WorldGrid(positions=15, headings=4, p_speeds=4, e_speeds=4)
solve(config, grid, "/tmp/world8d-smoke", horizon=0.1, output_dt=0.1,
      history=True, max_core_gb=1, max_output_gb=1)
table = WorldTable("/tmp/world8d-smoke")
```

This is an implementation smoke test, not a useful-resolution controller
benchmark. `solve` refuses to overwrite an existing run and checks explicit
memory/output limits before allocating. It records the source hash and source
snapshot, physical spacings, compile/update/write times, and actual process
peak RSS. Compiler memory is outside the native subprocess RSS measurement.

History is disabled by default. Without it, `WorldTable.field` can query the
final budget and `contains` can classify the numerical final sublevel set.
`value` requires signed histories: it spatially interpolates fields and finds
their first zero crossing. It does **not** interpolate the diagnostic
`arrival.f32` array. Unresolved/out-of-domain queries remain NaN. Exact terminal
events override the numerical time query.

## Coarse-grid checks that matter

Two numerical issues are distinct:

- A wall lying between nodes can cause interpolation of the initial
  capture-or-E-fault field to have the wrong sign near a corner. Metadata
  records the distance between physical walls and their nearest grid nodes.
  Aligning the square boundaries helps; it is not a general accuracy proof.
- Even with aligned walls, the joint capture set is a thin diagonal in four
  position coordinates. A coarse grid can poorly represent capture between
  grid nodes. An exact terminal check fixes detection during execution but
  cannot repair an inaccurate propagated field.

Keep both field checks and physical replays. Successful allocation or agreement
between two implementations does not establish useful capture decisions.
The capture-diagonal effect is not unique to eight dimensions: it also occurs
in lower-dimensional joint world grids. Extra speed axes consume memory that
might otherwise permit finer position sampling.

## Independent public backend

`world_optimized_runner.py` adapts the same game to the upstream OptimizedDP
8D graph and HJSolver, with explicit inward derivative functions for the two
physical speed axes. No upstream files are edited; the boundary adapter is
recorded in metadata. It runs in the isolated legacy environment
described in `setup.md`; it is not imported by online controllers. The graph's
API expects eight padded control slots even though each player has only two
physical inputs. The adapter documents those inputs through its dynamics.

The command accepts `--config`, `--output`, `--positions`, `--headings`,
`--speeds`, `--extent`, `--horizon`, `--output-dt`, `--order`, `--history` and
`--threads`, and `--speed-boundary`. Start with a tiny grid. The earlier generic
signed-distance configuration is retained only as `legacy_signed` for audits.
Second-order fields under matched physical speed boundaries have small numerical
differences, including differences near nonsmooth data; compare signs and
refinement rather than expecting bitwise equality. The published implementation
uses additional full-grid intermediates and may allocate float64 history.
