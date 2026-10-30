# B2: UCT with B1 continuation simulations

B2 retains B1's successor-based DP takeover. When that decision is available,
both controllers use it. Outside that region, B2 searches over current vehicle
commands instead of directly executing the geometric approach proposal.

## One online decision

1. Evaluate B1 once at the current observed state. If it selects DP, execute the
   same command and skip search.
2. Otherwise keep B1's proposed command as one candidate, alongside the nine
   bounded acceleration/curvature commands (remove exact duplicates).
3. Build a P-then-E tree. P chooses a current command; E chooses a response;
   both cars move for one 0.05 s interval. Future P choices branch on the observed
   joint successor. The tree uses the shared physical simulator and event rules.
4. At a newly encountered frontier, simulate B1 feedback for `rollout_steps`.
   Each step recomputes the lion waypoint and MPPI action, and simulates the
   declared reactive separation E policy. This is actual B1 policy execution,
   not a held future P sequence.
5. A DP takeover in that simulation ends the rollout with B1's estimated DP
   continuation cost. Physical capture ends it with elapsed time. A bounded
   rollout that reaches neither uses the same explicitly heuristic tail as the
   earlier DP-leaf prototype.
6. When the total budget permits, initialize every root P/E response pair once.
   Spend the remaining simulations using UCT: P minimizes mean cost minus an
   exploration bonus; E maximizes mean cost plus the bonus. At the root, choose
   the smallest worst-response mean among P rows whose E responses were all
   evaluated. With no complete row, retain the B1 proposal. Execute only the
   first command, then replan from the next measured state.

A tree frontier where B1 would immediately take over with DP is not expanded
into alternative P commands. This keeps the searched continuation consistent
with the actual hierarchical controller. Root means remain finite-budget estimates,
not exact worst-case continuous-game values.

`B2Settings` controls tree depth, simulation count, B1 rollout length, exploration,
and optional wall-time cutoff. The initial development setting is depth 6,
128 total simulations, and 3 B1 rollout steps (0.15 s). These are computational settings,
not a claim that a short forecast is sufficient for a far-away adversarial game.
Report the actual explored depth and cost. The generic tree may stop early when
its represented finite game is completely resolved.

## What the continuation assumes

The tree explicitly considers sampled E responses. Beyond its frontier, the
rollout uses the deterministic `ReactiveEvader(mode='separation', horizon=1)`.
The resulting estimate is not an exact worst-response value of the full game.
An unresolved terminal state uses the documented DP-field/distance heuristic;
it does not fabricate a known capture time.

Every rollout gets its own B1 instance with copied nominal controls, RNG state,
and geometric anchor from the current B1 proposal. The table is shared read-only.
The warm-start snapshot is a declared initialization at the frontier; it is not
advanced by running extra hidden MPPI optimizers along the searched prefix.
Simulation cannot consume the actual controller's future RNG or change its anchor.

This implementation is in `reversible/b2.py`; the common UCT engine remains in
`reversible/mcts.py`. The older `UCTController` still exists as the DP-leaf
prototype for explicit comparison. Neither implementation is a new UCT algorithm.

## Recording and validation

`control_owner='uct_b1'` marks a searched action; `dp` marks a shared DP action.
The record distinguishes tree simulations, B1 rollout calls, rollout motion
steps, continuation endings, actual search depth and timings. Full tree snapshots
are optional; compact chase records retain numerical work/timing statistics.

Tests cover the common DP takeover, actual B1 feedback rollout calls, admissible
commands, repeatable continuation values, and isolation of mutable policy state.
The common UCT tests independently compare small searched games with exhaustive
minimax. Empirical capture benefit and real-time suitability require evaluation;
code execution alone does not establish either.


## Explicit variants

`root_selection="worst_response_mean"` is the initial B2 rule described above.
`root_selection="most_visited"` instead uses the common UCT engine's root visit
recommendation, breaking visit ties by lower mean cost. Both keep the same root
response initialization and declared simulation budget. Log the rule in every
comparison rather than conflating their results.

`bootstrap_at_dp=False` continues simulating the DP controller inside the bounded
B1 rollout instead of terminating there. It does not disable DP during execution.
`tree_dt` can make hypothetical branch intervals coarser while real execution
remains at `control_dt`; that approximation checks future DP takeover only at
branch points. The primary comparison leaves it unset, using 0.05 s throughout.

`AdaptiveB2Controller` first runs the short search. If a B1 continuation reaches
DP, it repeats from exactly the same root planner/RNG state with longer rollouts.
The default long continuation is 40 updates (2 s). This can use twice the search
simulations; its timings include both passes. It is a development ablation,
not a claim of optimal budget allocation or improved performance.

## Optional exact-cost native backend

`rollout_cost_backend="native_free"` evaluates the same MPPI sampled-control
objective in C++, with the same early-goal termination and weighted update.
The NumPy implementation remains the default and the diagnostic reference.
This optional path supports free space only and rejects map constraints.
It requires `g++`, compiles lazily to a source-hash cache, and shares no mutable
planning state between rollouts. Unit checks compare costs and selected commands;
complete reference episodes also check executed-state parity.

## Current empirical scope

The initial full-grid run reached only two or three joint tree intervals at most
on the inspected cases, despite a six-step depth cap. Root response initialization
uses 81 or 90 of the 128 simulations. A three-step continuation can require up to
387 separate B1 updates per decision. A large fraction of computation therefore
goes into short MPPI continuations, not deep strategic exploration.

See [the benchmark guide](free-space-benchmark.md) for the predefined conditions
and retained negative result. This implementation is not currently a 20 Hz robot
controller. B3/B4 learning is still unimplemented.
