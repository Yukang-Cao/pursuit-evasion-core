# Baseline 1: reuse a free-space table inside a map

B1 combines a geometric approach controller with short DP-based control decisions.
The stored table has no walls. The online controller checks candidate motion
against the map and uses the approach controller when its DP conditions fail.

## The actual switch, every 0.05 seconds

1. Transform the world state to relative E position, relative heading, and both
   signed speeds. Query the free-space table for an estimated capture time.
2. If the current state is outside the table or has no time estimate within its
   stored coverage, use lion + MPPI.
3. Otherwise, enumerate nine current P commands and nine E responses. Each
   command is an acceleration/curvature pair. Simulate one control interval with
   physical capture and boundary events.
4. Reject P commands that lack a checked collision-free braking continuation.
   For each remaining P command, take its worst E response cost. A continuing
   pair's cost is elapsed time plus the successor's DP time estimate. Unresolved
   successor times make the row unavailable to B1. Immediate terminal outcomes
   follow the declared game rules.
5. If any accepted row has a finite worst-response cost, execute the row with
   the lowest such cost. Otherwise use lion + MPPI.

This check repeats after each executed command. A run can switch from approach
into DP and later back to approach. There is no fixed distance threshold and no
one-way, permanent handoff.

The wall/braking check tests the proposed command, then braking to zero speed
with straight, maximum-left, or maximum-right curvature. A terminal P success
within the first command is handled separately. This checks a short continuation;
it does not inspect the whole eventual capture trajectory.

## Approach branch

`LionTarget` selects a position waypoint. The MPPI tracker samples admissible
acceleration/curvature sequences toward that waypoint, with map collision checks
and early local-goal termination. The physically executed command also goes
through the shared wall/braking check. New B1 instances default to the tested
high-variance log-MPPI sampler. Explicit `mppi_settings=MPPISettings()` reproduces
the older Gaussian approach setting; see [tracker details](reversible-mppi.md).

## What “estimated time” and “field guidance” mean

A completed table can still have states without a capture-time estimate within
its stored budget. “Unresolved” does not mean escape, nor an execution deadline.
The solver retains signed capture-by-budget fields and obtains a time estimate
from their first zero crossing.

`mode="hybrid"` is B1 and requires the time-based conditions above.
`mode="dp"` is a separate diagnostic controller: it can rank actions by the raw
signed field when times are unresolved. That alternative is not silently part of
B1. `mode="approach"` disables DP and supplies the approach-only comparison.

## Current evidence and scope

The corrected recorded B1 comparisons cover six named square-wall starting
configurations, with the same declared reactive E policy across controllers.
B1 captured in all six; approach-only captured in five. B1 was not faster in every
case. These are development checks, not a broad adversarial benchmark.

A separately selected wall stress uses fixed E commands. Its Gaussian approach
gets stuck after a DP-to-approach switch. The tested log-MPPI variant captured in
three seeds on that case. Extending table coverage also repairs that particular
switching failure. Those results do not establish general superiority.

No completed reversible B1 chase in the current experiment collection includes
an interior obstacle. The map geometry API supports rectangles, but that software
capability is not experimental validation. The next application test is an
interior obstacle with a reactive E, using the same reusable free-space table.

Numerical-table accuracy checks are retained as diagnostics; they are not the
B1 deliverable or the current research contribution. The relevant code is
`reversible/controllers.py`: `Baseline1.command`, `dp_decision`, and
`braking_safe_mask`.
