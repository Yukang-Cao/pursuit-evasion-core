# pursuit-evasion-core

Inspectable models, numerical game solvers and controllers for vehicle
pursuit-evasion. This repository holds reusable code and short technical
documentation. Research diaries, HTML reports, large fields and videos live
outside the repository.

## Start here

1. Read [the physical game and boundary rules](docs/model.md).
2. Follow [the code map](docs/architecture.md).
3. Check [what is implemented](docs/status.md) before using an example.

```bash
python -m pip install -e '.[test]'
python -m pytest
```

The initial model is two fixed-speed, bounded-turn-rate cars in a square. It
uses a six-dimensional **world** state. It is not the older relative free-space
game. Capture and boundary faults are reported separately.

Offline solvers may use a workstation. The intended online deployment target
is Jetson Orin; no onboard performance claim is made before profiling there.
