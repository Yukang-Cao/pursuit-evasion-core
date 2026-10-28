# Working in pursuit-evasion-core

This repository contains reusable research code, not the private research diary.
Read `docs/model.md` before changing dynamics, costs, controls, or boundary rules.
Read `docs/architecture.md` for entrypoints and module responsibilities.

- Reserve “teacher” for imitation/RL contexts. Elsewhere name the actual
  strategy or controller, such as lion-guided strategy or MPPI tracker.
- Keep a single explicit state order and unit convention across every backend.
- Distinguish physical capture, boundary faults, timeouts, and unresolved queries.
- Preserve negative results; a successful solve is not a correctness proof.
- Label numerical approximations and model changes. Never claim a certificate
  based only on sampled replays or agreement between implementations.
- Separate offline compile/solve/write timing from online query/controller timing.
- No online workstation or cloud dependency: the deployment target is Jetson Orin.
- Keep numerical backends optional. Importing a controller must not import JAX,
  HeteroCL, a compiler, or an offline dataset.
- Keep configuration explicit and paths independent of the shell's current folder.
- Store generated fields, checkpoints, reports and videos outside Git. Keep tiny
  deterministic test fixtures and commands needed to regenerate larger artifacts.
- Prefer clear functions and modules to plugin registries or deep inheritance.
- Verify shared physics, event order, solver signs, query domains and reproducible
  controller decisions with meaningful tests before performance sweeps.
- Use focused commits, e.g. `feat(game): add fixed-speed square dynamics` or
  `fix(dp): preserve unresolved queries outside the grid`.
- Document what is implemented versus planned. Update `docs/status.md` at a
  milestone; do not present future CLI commands as working examples.

