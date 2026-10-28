# Implementation status

Updated 2026-10-28. This is a runnable research baseline, not a certified
continuous-game solution or an onboard release.

## Implemented

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

Matched UCT (B2), behavior cloning (B3), PPO (B4), and an eight-dimensional
extension are not implemented. Exported demonstrations are not a trained
student. Map-conditioned learning remains deferred.

Jetson Orin at at least 10 Hz is the online target. No Orin performance has been
measured. The exact device, RAM, power mode, table storage and complete-cycle
latency remain deployment questions; see [profiling](profiling.md).

