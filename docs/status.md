# Implementation status

Updated 2026-10-31. This is a runnable research baseline, not a certified
continuous-game solution or an onboard release.

## BARN contact-game benchmark completed, Oct 31

Implemented body-centered LiDAR updates at 20 Hz, rectangular contact capture,
compact eight-state map-specific DP supervision, supervised value models, and
numerical/learned/free-space cross-play on ten held-out BARN maps × five starts.
All 25 training/validation references and both training seeds completed on the
laptop. No toros computation was used.

Against the same numerical E, approach-only captured 23/50; numerical P and
map-conditioned learned P captured the same 32/50; a finer free-space value and
a learned state/observed-clearance model each captured 39/50. The second map
training seed captured 31/50. Better field fitting did not establish a chase
advantage for the map encoder. Of the numerical and learned timeouts, 16/18 and
17/18 respectively never entered local feedback.

A separately fixed follow-up on 40 starts permits supported contact-cost
approach when no finite capture estimate exists. Numerical P improves 27→30
captures; learned P drops 27→23. The original switch remains the default; the
alternative is exposed as `--use-contact-cost`. It is not a universally better
controller or a guarantee. Both public flag settings exactly reproduce their
recorded selected-case trajectories (253 and 401 states).

The full experiment includes 650 primary, 200 follow-up and 100 extra cross-play
runs. Reintegration verified all 178,858 recorded transitions and intermediate
obstacle/terminal contact checks. CPU tests: 194 passed, seven optional tests
skipped; four compact GPU-solver checks passed separately. Numerical reference
regeneration matched original sampled labels and state graphs exactly. These
checks establish implementation consistency, not continuous-game optimality.

The production grid uses 0.30 m positions, 16 headings and seven speeds per car,
with a 2 s local budget and 3 m separation limit. GPU core storage was 2.0–7.0 GB;
successful training/validation references took 3.9–16.0 minutes each. The learned
map weights occupy 1.18 MB. Quiet warm local decisions including scan processing
and encoding had state-wise medians 6.9–9.7 ms over 14 checked laptop states.
No Jetson timing or sensor-only deployment is claimed.

All controllers currently share privileged exact-map physical action checks.
The learned value receives scan images and a known-space mask; both learned
players share P's local value observation. Full local geometry is a distinct
ablation. No obstacle collision is credited as capture.

A bounded successor-query data test is complete. It replaced 245,760 queries
on five training maps with legal continuous control successors while preserving
images, total query count, architecture, loss, 80,000 steps and seed 0. Captures
remained 32/50 under the original switch and 23/40 under contact-cost approach.
The latter gained three and lost three captures. Initial-control-query field
MAE improved 0.0876→0.0801 m without a capture-count improvement. The selected
BARN100 local failure still brakes initially and still times out. This is a
baseline-quality diagnostic, not a proposed contribution.

A selected BARN0 numerical-budget check also completed. Against the fixed E2
policy, map P2 captures 3/5 and map P4 2/5. Against E4, both capture 5/5. The
free-space P2 baseline captures 5/5 against either E, while free-space P4 captures
3/5 and 4/5. Thus the diagonal P4/E4 improvement cannot be credited to P.
The longer map solve took 28.3 minutes and 5.88 GB core GPU storage. Its 2 s
prefix matches the original labels/graphs; the free-space prefix is byte-identical.
All ten original map/free P2 trajectories against E2 reproduced exactly.

The complete comparison set now contains 1,090 recordings and 206,619 checked
control intervals. The additional four local extraction checks are separate.
One primary starting configuration reaches an unavailable numerical E query for
one interval; excluding the same case from all primary methods gives learned
31/49 versus free-space 38/49, preserving the conclusion. Updated replay labels
mark unavailable queries rather than silently calling every response optimal.

The next recommended bounded check is to retain every 0.05 s signed field,
rather than every 0.25 s, on one map and repeat the controls against an unchanged
E. This tests time interpolation at the actual control interval without changing
the numerical game. It is proposed, not completed. A numerical local replay
keeps predicting about one second while P oscillates near zero speed; a small
countdown-field test did not repair that selected stall. State resolution,
control extraction and local-exit modelling remain alternative explanations.

Map-aware MCTS, imitation/PPO policies, noisy sensing and onboard validation
remain unfinished. No toros computation was used. See
[the contact-game contract](barn-contact-game.md) and
[the benchmark protocol](barn-benchmark.md).

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

