# DP-leaf adversarial UCT prototype

This is a first tree-search implementation, not the completed baseline that uses
B1 to simulate the remaining game. A new frontier is evaluated with the numeric
DP table or an explicit fallback heuristic. It never invokes an MPPI optimizer
inside a simulation.

## One simulation

1. At a P node, choose a currently unvisited action, or minimize mean cost minus
   an exploration bonus. Candidate actions pass the same braking-continuation
   check used by B1, with the same immediate-success exception.
2. At the corresponding E node, E sees that current P command. Choose an
   unvisited response, or maximize mean cost plus the exploration bonus.
3. Execute the pair through the shared game simulator for exactly one control
   interval: normally 0.05 s, shortened only by the first terminal event.
4. If the successor is already in the tree, continue. The next P action is
   selected at that observed state; it is not a previously committed sequence.
5. At a new frontier, evaluate its remaining cost, then back up the cost through
   the selected P and E edges. Each physical interval contributes its elapsed
   time exactly once.

The bonus is `c * sqrt(log(node visits + 1) / action visits)`, with opposite
signs for minimizing P and maximizing E. The prototype uses c = 0.25 in its
cost units. It is an exploration setting, not a validated confidence interval
or an optimized parameter.

The default catalogues contain nine acceleration/curvature pairs per player.
Distinct command labels that give exactly identical observed successors share
the next P node. A hidden E action label must not reveal information beyond the
observed physical state.

## Leaf costs and physical events

- A finite numeric DP value is the estimated remaining capture time.
- An unresolved in-domain time uses `table horizon + positive signed field /
  speed scale`. The scale is the positive difference of maximum forward speeds
  for the current model (1 m/s); it is a heuristic conversion, not a time bound.
- An out-of-domain query uses `table horizon + remaining distance / speed scale`.
  With no table, only the distance estimate is used.
- Capture and E boundary fault terminate with their elapsed time. An E fault
  remains distinct from physical capture in recorded game outcomes.
- P boundary fault adds a finite penalty of 1,000. This is a search loss penalty,
  not a claim that capture takes 1,000 seconds.

The logs count each leaf type separately. Root means are costs observed under
adaptive search, not exact worst-case capture times. The maximum observed E
response mean is also reported, with its number of tested responses; untested
responses remain unknown.

## What the depth means

Depth counts complete P/E command pairs. The requested depth six permits an
explicit map-aware prefix of 0.30 s. It does not guarantee that the available
simulation budget reaches that depth.

The recorded deepest prefix is distinct from the continuation information in
the leaf. A 0.10 s searched prefix followed by a value from a 4 s free-space
table is not merely 0.10 s of total game information. But the table does not
evaluate subsequent map constraints. An unresolved signed-field heuristic also
does not construct an additional 4 s trajectory.

## Root action and finite-tree checks

The usual root decision is the most visited P action, breaking visit ties by
lower mean cost. If all represented branches reach terminal/depth-limited leaves,
the prototype can back up their exact minimax value and select that finite-game
optimum. This exhaustion concerns the supplied action sets, finite depth, and
leaf heuristic only; it does not certify the continuous pursuit-evasion game.

Tests compare the implementation with exhaustive minimax on a small matrix game,
a two-step game where P must choose different future actions after different E
responses, and a one-step game using the actual vehicle simulator. Additional
tests check event costs, identical-state merging, unknown-leaf labels and budgets.

## Runtime and baseline scope

`UCTSettings` controls the maximum simulation count and an optional wall-time
limit. The limit is checked between simulations. One expensive simulation can
overrun it; this is not a hard real-time guarantee. Report total controller time,
not just the tree loop. The tree is rebuilt at each control update in this first
implementation.

`UCTController(config, world, table, settings, seed)` implements the same
`reset(state)` / `command(state, t, diagnostics=False)` interface as B1.
`last['search']` contains root visits, mean costs, response coverage, deepest
explicit prefix, table horizon, leaf types, work counts, and timing. A detailed
tree snapshot is optional for a visual algorithm walkthrough.

The design reuses the P/E node separation, UCT signs, and frontier-value idea of
the earlier local `research/tree_capture/engine.cpp` investigation. The code here
is a smaller Python implementation with the revised shared physics and explicit
remaining-cost accounting. It is not presented as a new search algorithm.
