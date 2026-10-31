# pursuit-evasion-core

Inspectable models, numerical game solvers and controllers for vehicle
pursuit-evasion. This repository holds reusable code and short technical
documentation. Research diaries, HTML reports, large fields and videos live
outside the repository.

## Start here

The current [map-conditioned value pilot](docs/map-conditioned-value.md) adds
oriented footprints, an eight-dimensional GPU HJI reference, fixed-frame
coordinate transforms, and a compact map encoder/value-query network.
It is a controlled feasibility study; the output is a signed local game value,
and a reliable control advantage is not yet established.

The Oct 30 free-space experiment uses `configs/reversible_free_a3.json`: both
acceleration limits are 3 m/s². Pass that explicit configuration when recomputing
tables; the older default model and its records remain available.

The current reversible-car work is in [`reversible/`](src/pursuit_evasion/reversible/).
Start with [its physical model](docs/reversible-model.md) and
[the actual B1 switching rule](docs/reversible-b1.md), then the
[reused MPPI tracker](docs/reversible-mppi.md), [free-space DP](docs/reversible-free-dp.md),
[fixed-map DP](docs/reversible-world-dp.md), and
[B2 with B1 rollouts](docs/reversible-b2.md). The earlier
[DP-leaf UCT prototype](docs/reversible-uct.md) is retained separately.
A small [analytic capture check](docs/reversible-analytic-check.md) documents a
known numerical limitation and its reproducible test.
The [interactive value viewer](docs/value-viewer.md) links world configurations,
relative table slices, and recorded controller/waypoint transitions.
Run the current model through [the free-space benchmark guide](docs/free-space-benchmark.md)
and [`examples/chase_free.py`](examples/chase_free.py). The fixed-speed model below
is retained for earlier reproducible experiments.

A tiny reversible CPU solve (requires `g++` with OpenMP):

```python
import json
from pathlib import Path
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.free_dp import RelativeGrid, solve

config = ReversibleConfig.from_dict(json.loads(Path("configs/reversible_free_a3.json").read_text()))
solve(config, RelativeGrid(17, 12, 7, 7, 4.0),
      "/tmp/pe-reversible-smoke", horizon=0.2)
```

Use a fresh output path. This tiny grid checks the implementation; it is not a
control-quality reference. The linked solver documents explain larger runs,
model compatibility, and retained-history dependencies.

1. Read [the physical game and boundary rules](docs/model.md).
2. Follow [the code map](docs/architecture.md).
3. Check [what is implemented](docs/status.md) before using an example.

Follow [setup](docs/setup.md) for a tested lightweight CPU installation and
separate public-solver environments. The [migration guide](docs/migration.md)
explains which code belongs here and which research artifacts remain local.

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Run a small native solve (requires `g++` with OpenMP):

```bash
python -m pursuit_evasion solve --config configs/square6d.json \
  --backend native --positions 9 --headings 8 --horizon 0.2 \
  --output /tmp/pe-square-smoke
```

Output paths are explicit and completed runs are not overwritten. This tiny
grid is an implementation check, not an accurate control solution.
See [numerical conventions](docs/numerics.md) before comparing backends.

Run baseline 1 after computing a sufficiently resolved local table:

```bash
python -m pursuit_evasion chase --config configs/square6d.json \
  --cases configs/development_cases.json --case approach \
  --table /path/to/completed/local-table --seed 0 \
  --output /tmp/pe-approach.json
```

Use `--without-dp` to run the same lion-guided strategy and MPPI tracker without the DP
finish. The development cases are diagnostics, not a final benchmark.
Read [the controller walkthrough](docs/controllers.md) and
[profiling scope](docs/profiling.md) before interpreting outcomes or latency.

The initial model is two fixed-speed, bounded-turn-rate cars in a square. It
uses a six-dimensional **world** state. It is not the older relative free-space
game. Capture and boundary faults are reported separately.

Offline solvers may use a workstation. The intended online deployment target
is Jetson Orin; no onboard performance claim is made before profiling there.
