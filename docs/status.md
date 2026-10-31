# Implementation status

Updated 2026-10-31. This is a runnable research baseline, not a certified
continuous-game solution or an onboard release.

## BARN contact-game run in progress, Oct 31

The new benchmark uses ten original BARN maps at twice their physical scale,
with five predefined starts per map. Rectangular contact replaces circular
capture. A common braking-feasible action constraint excludes obstacle contact;
obstacle contacts are never credited as captures.

Implemented and checked: finite-body geometry, continuous contact events,
LiDAR memory and body-frame views, compact 8D Bellman updates, streamed history
queries, and matched P/E capture-time extraction. Reference generation has completed most training maps. Five whole-chase
validation starts captured without obstacle contact using the common route
approach and numerical finish. These are development checks, not held-out
performance. Neural training and final cross-play remain pending.

The reference retains all eight dynamic variables and omits illegal states
or states outside the local separation domain. Tiling the output through RAM
permits a finer grid on the laptop GPU. Independent small-game tests check the
update. See [the contact-game contract](barn-contact-game.md).

## Map-conditioned value pilot, Oct 30 evening

Implemented explicit rectangular footprints and an 8D GPU HJI reference on a
22-map room/doorway family, with A3 reversible dynamics and 0.5 m capture radius.
P is 0.80 × 0.50 m; E is provisionally 0.40 × 0.25 m. The first-collision-loses
rule is retained and physical captures are counted separately from E collisions.
The coarse 262-million-state, two-second reference took about 26 s and 3.14 GB
core GPU memory per map on the current laptop.

A supervised map encoder + interpolated spatial features + query MLP learns
signed finite-budget P-success values. Eight queried vehicle coordinates remain
in one fixed planning frame. No dense 8D output grid is built online. A
separate translated/rotated-frame augmentation check reused physical labels.

Map features improved held-out value fitting in two seeds. The final 28-start
local controller tests did not establish a capture benefit: reference and map
network each captured 27/28 with one E collision, while the no-image MLP captured
27/28 and 28/28 across seeds. A finer free-space-value baseline with the same
immediate collision checks captured all 28 starts. Earlier fixed-budget signed-margin selection
results are preserved but are not used as a superior-control claim.

The public label, training, table-free query and local-chase examples ran.
The public chase reproduced its recorded trajectory exactly. Reintegrating all
292 development/final episodes matched all 5,240 recorded transitions and events.
The full installed-package suite passed 180 tests with three CUDA tests skipped;
the CUDA footprint/history group separately passed nine tests. These are consistency
checks, not independent physical or optimality proofs.

Still unimplemented: a partial-observation/crop-exit game definition, global
hierarchy integration of the new local network, sensor-noise tests, a broad
obstacle-game benchmark, and Jetson deployment. No toros job
was run. B3 imitation and B4 PPO remain separate unfinished deliverables; this
supervised value pilot is not completion of those policy-learning baselines.
See [map-conditioned-value.md](map-conditioned-value.md).

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

The Oct 30 configuration `reversible_free_a3.json` uses equal acceleration limits
of 3 m/s² and free space. B1 now evaluates successor actions even when its current
state is unresolved; the old gate remains an explicit option. A bounded B2 UCT
implementation now simulates B1 feedback continuations, with documented rollout
initialization and heuristic truncation. The first 64-case comparison did not improve B1: 33 captures versus 36 for B1. The completed visit-count root variant captured 32/64; conditional longer B1 continuations captured 33/64. None improved the 36/64 B1 capture count.
The earlier DP-leaf prototype remains separate. Cloning and PPO are not yet
implemented for the revised model; no robot or Jetson result is claimed.

Current integrated check: 164 tests passed; two optional Torch reference checks
were skipped in the minimal test environment and passed separately in the Torch
CPU environment. Some numerical-table feedback cases still fail their predicted
capture deadline. Refinement and obstacle-transfer diagnostics document both successful captures
and unresolved failures; successful computation is not a capture certificate.
The fixed-map solver completed a four-second budget using a retained two-second
prefix and an extension, with matching resumed/monolithic numerical tests.

Current free-space evidence: approach-only and B1 each capture 36/64 predefined
starts. Mean time with timeouts counted at 45 s is 28.57 s versus 28.39 s.
The selected paired replay improves 30.22 to 27.67 s; both paths are identical
until DP takes over at 22.80 s. All 28 B1 timeouts remain in approach mode.
The fine 604-million-state table took 40.66 minutes to compute and occupies
103.94 GB. See [the reproducible benchmark](free-space-benchmark.md).

The optional native free-space MPPI cost backend preserves the objective and
was checked against nine complete NumPy-reference B2 episodes with identical
executed states. It reduces rollout cost, but B2 still exceeds the 50 ms control
budget. Adaptive longer rollouts near DP did not improve aggregate captures.
Warm serial laptop calls measured median / 95th percentile: one value lookup
0.110 / 0.449 ms; B1 approach 6.278 / 6.358 ms; B1 DP 0.590 / 0.622 ms;
B2 far-state search 906.17 / 916.72 ms. These are repeated fixed-state calls,
not Jetson measurements or an end-to-end latency guarantee.

The updated viewer uses the table's declared dynamics, successor-based switching,
fixed replay axes and lazy episode loading; all completed cases can be inspected.

Earlier model/map evidence follows and is not mixed into the new free-space grid.

Current B1 map evidence covers boundary-wall cases, not interior obstacles.
The corrected six named square-wall starts produced six B1 captures and five
approach-only captures against a declared reactive heuristic E. B1 is not faster
in every case. The separate scripted-E stall and log-MPPI repair are development
diagnostics. See [the B1 rule and validation scope](reversible-b1.md).

The [numerical exclusion diagnostic](reversible-analytic-check.md) is preserved
for later inspection. It is not the main B1 deliverable or the current research
focus; further numerical repair is deferred.

See [the reversible model](reversible-model.md) and the per-module documents linked
from the README. Generated fields, replays and reports remain outside this repo.

## Interactive inspection and optional sampler

A Python-backed local value viewer now exposes 5D slices, world-pose editing,
coverage selection, explicit controller/opponent labels, recorded MPPI waypoints,
and figure export. It queries completed tables without recomputing DP. See
[the viewer guide](value-viewer.md).

The reversible tracker also offers a signed normal-times-lognormal sampler.
The declared high-variance configuration escaped the selected near-wall Gaussian
stall and captured in three tested seeds; the opponent was the same fixed script.
This is a case-specific workaround, not evidence of general adversarial success.
New B1 instances default to the tested log sampler. The explicit Gaussian setting
and previous recordings are preserved.

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
student. Map-conditioned learning was subsequently implemented in the separate pilot above; it is not integrated into these older baselines.

Jetson Orin at at least 10 Hz is the online target. No Orin performance has been
measured. The exact device, RAM, power mode, table storage and complete-cycle
latency remain deployment questions; see [profiling](profiling.md).

