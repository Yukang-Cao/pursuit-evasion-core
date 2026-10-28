# Read the code in this order

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

UCT, cloning and PPO are planned, so no empty `search/` or `learning/` package is
provided. Add a module when it has an implemented, tested responsibility.
Generated fields, replays, datasets and reports belong in an explicit output
directory outside Git. See [the migration boundary](migration.md).

