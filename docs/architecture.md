# Code map

## Current reversible free-space experiment

Start with [the reversible model](reversible-model.md), then
[the reproducible benchmark](free-space-benchmark.md). The current entrypoint is
`examples/chase_free.py`. The original fixed-speed CLI below is separate.

1. `reversible/model.py` and `game.py`: signed-speed dynamics and capture events.
2. `reversible/free_dp.py`, `gpu_free_dp.py`, `relative_table.py`: offline fields
   and model-checked online queries. The GPU implementation is optional.
3. `reversible/controllers.py`: successor-based B1 switch; `lion.py` and
   `mppi.py`: geometric target and navigation tracking.
4. `reversible/b2.py`: UCT with isolated B1 continuation simulations; `mcts.py`:
   generic adversarial tree and the retained earlier DP-leaf prototype.
5. `reversible/evaluation.py`: physical outcomes, controller phases and timings.
6. `reversible/viewer.py` / `viewer.html`: interactive slices and recorded chases.

The optional `native_nav.py`/`native_nav_cost.cpp` backend speeds up the unchanged
free-space MPPI objective inside B2. It does not implement a different planner.

## Earlier fixed-speed implementation

Start with [the model](model.md), then use [setup](setup.md) for a small CPU
run. Package paths below are relative to `src/pursuit_evasion/`.

1. `model.py`: state order, physical limits, exact held-command arcs, and the
   continuous Hamiltonian. `configs/square6d.json` supplies the actual parameters.
2. `game.py`: stop at the first physical event and distinguish capture from a
   boundary fault. `tests/test_model.py` and `tests/test_pair_events.py` check
   scalar and batched event handling.
3. `dp/grid.py`: grid axes, target/avoid sets, and output budgets.
   `dp/table.py`: memory-mapped signed fields and numerical time-to-success
   queries. An unresolved query is NaN, not a proven escape.
4. `controllers/hybrid.py`: decide whether to approach or attempt a local DP
   finish. Follow `dp_finish.py` for one-step minimax extraction, `lion.py` for
   the geometric target, and `mppi.py` for P-only target tracking.
5. `geometry.py`: the complete-turning-circle continuation used by MPPI.
   `controllers/opponents.py`: the declared reactive development evader.
6. `evaluation.py`: record controls, states, terminal events, handoffs and call
   timings. `demonstrations.py`: export commands and targets with an approach
   mask; it does not train a policy.
7. `dp/native.py` and `dp/native_square.cpp`: one complete offline solve path.
   Then read `dp/jax_backend.py` and `dp/optimized_runner.py` for the external
   numerical adapters. Their shared conventions are in [numerics](numerics.md).

## Dependency boundary

The installed base package needs NumPy. Importing a controller or loading a
completed local table does not import JAX, HeteroCL, or a compiler. The `solve`
CLI imports only its selected backend. OptimizedDP runs in a separate legacy
Python process; it never becomes an online dependency.

`python -m pursuit_evasion` is the current entrypoint. It provides `solve` and
`chase`; experiment orchestration and report rendering remain outside this
repository. Paths supplied on the command line are ordinary filesystem paths.
Use absolute paths when invoking the package from another working directory.

## What belongs here

- `dp/`: offline adapters, field metadata and online queries.
- `controllers/`: geometric targets, MPPI, the DP finish and the handoff rule.
- `configs/`: explicit physical parameters and small development start states.
- `tests/`: small deterministic checks of shared contracts and behavior.
- `docs/`: setup, numerical scope, code navigation and implementation status.

UCT is implemented in the reversible package. Cloning and PPO remain planned;
no empty learning package is provided. Add a module when it has an implemented,
tested responsibility.
Generated fields, replays, datasets and reports belong in an explicit output
directory outside Git. See [the migration boundary](migration.md).

