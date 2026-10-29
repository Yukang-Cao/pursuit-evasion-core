# Implementation status

Updated 2026-10-29. This is a runnable research baseline, not a certified
continuous-game solution or an onboard release.

## Current reversible-car implementation

The `reversible/` package now includes acceleration-limited signed-speed dynamics,
continuous event handling, a documented adaptation of the existing C-Free-Uniform
MPPI routines, a lion-guided B1 controller, and declared reactive opponents.
Targets and both players' commands update every 0.05 s. The legacy turning-circle
filter and steering-effort penalty are absent from this tracker.

A reusable 5D relative HJI solver has CPU and GPU implementations. A fixed-square
8D HJI solver supports selected queries, signed histories and continuation runs.
Physical speed boundaries use inward derivatives; generic signed-distance ghost
extrapolation is explicitly rejected by the standard world-table reader. Small
independent numerical checks and analytic capture cases supplement solver parity.

A first adversarial UCT implementation uses DP/signed-field leaf estimates. It is
**not yet the planned B2 using B1 rollout continuations**. Cloning and PPO are not
implemented for the revised model. No robot or Jetson result is claimed.

Current integrated check: 141 tests passed; two optional Torch reference checks
were skipped in the minimal test environment and passed separately in the Torch
CPU environment. Some numerical-table feedback cases still fail their predicted
capture deadline. Refinement and obstacle-transfer diagnostics document both successful captures
and unresolved failures; successful computation is not a capture certificate.
The fixed-map solver completed a four-second budget using a retained two-second
prefix and an extension, with matching resumed/monolithic numerical tests.

A new independent [analytic exclusion check](reversible-analytic-check.md) proves
that some short-time numerical capture labels are false, including at exact
stored grid nodes. It uses a legal current-command-copying E response in free
space. This rules out a query-interpolation-only explanation for those labels.
The check is diagnostic and does not modify solver or controller behavior. A
reliable numerical repair and the original problematic initial state's optimal
capture time remain unresolved.

See [the reversible model](reversible-model.md) and the per-module documents linked
from the README. Generated fields, replays and reports remain outside this repo.

## Earlier fixed-speed implementation (retained)



- The six-dimensional fixed-speed point-car game in one fixed square, exact
  held-command arcs, continuous-event checks, and separate capture/wall outcomes.
- Native C++, JAX/hj_reachability and isolated OptimizedDP numerical adapters;
  signed-history storage, metadata and memory-mapped online queries.
- Baseline 1: a lion-guided geometric target, a NumPy MPPI tracker with a legal
  terminal turning-circle continuation, and a sampled one-step DP finish with
  a countdown, abandonment record and cooldown. `--without-dp` supplies the
  matched approach-only ablation.
- A declared reactive evader, episode/latency records, and demonstration export
  with both turn-command and waypoint labels. DP-owned rows are masked for
  approach-policy imitation.

## Evidence obtained

All three numerical adapters have executed the same six-dimensional game.
Native solves include 29 position samples with 24 headings and 41 position
samples with 32 headings per vehicle, each at a two-second horizon. Retained
fields are local research artifacts, not repository fixtures. Larger execution
does not by itself establish numerical convergence.

A development cohort compared B1 and the approach-only ablation on four named
starts and three seeds each. B1 physically captured in 9/12 episodes; the
ablation captured in 4/12. The other episodes timed out, including every
`side` case for both methods. Some accepted DP finishes were abandoned before
capture. These are diagnostic observations against one heuristic opponent,
not a general performance or worst-case capture claim.

A fresh, noneditable CPU installation passed the repository tests and ran a
tiny native solve plus a short controller episode from outside the checkout.
The packaged native source was present, and importing the hybrid controller
loaded neither public solver. See [the tested setup](setup.md).

## Limits and next work

The first-boundary-fault-loses rule is a provisional modeling choice. The DP
target includes E wall faults, so its value is time-to-P-success; it is not
unconditionally physical capture time. Continuous PDE controls and sampled
executed controls also differ. Field comparisons, refinement and controller
replays supply checks, not a capture certificate.

For this earlier fixed-speed version, the original B2/B3/B4 pipeline was not completed.
The revised-model 8D solver and DP-leaf UCT prototype are described above. Exported demonstrations are not a trained
student. Map-conditioned learning remains deferred.

Jetson Orin at at least 10 Hz is the online target. No Orin performance has been
measured. The exact device, RAM, power mode, table storage and complete-cycle
latency remain deployment questions; see [profiling](profiling.md).

