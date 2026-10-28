# pursuit-evasion-core

Inspectable models, numerical game solvers and controllers for vehicle
pursuit-evasion. This repository holds reusable code and short technical
documentation. Research diaries, HTML reports, large fields and videos live
outside the repository.

## Start here

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
