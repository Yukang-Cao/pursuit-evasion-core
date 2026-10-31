# BARN contact-game experiment

This is the revised finite-body experiment. It is separate from the earlier
circle-capture, first-collision-loses pilots. The fixed BARN comparison is complete;
checked results and limitations are tracked in docs/status.md.

## Physical model and local objective

Both cars retain [x, y, heading, signed speed]. P is a 0.80×0.50 m rectangle;
E is 0.40×0.25 m. Positions are rectangle centers. Both acceleration bounds are
3 m/s². P speed is[-1,2]m/s and curvature[-1,1]m⁻¹; E speed[-0.5,1]m/s and
curvature[-2,2]m⁻¹. Commands update every 0.05 s.

Capture is first oriented-rectangle contact. The inherited capture_radius
configuration field is unused when capture_rule=rectangle_contact.
Obstacle contact is never counted as capture.

The numerical local objective is capture within a finite budget before center
separation exceeds 3 m. Leaving this region means no local finish; it is not a
claim that E escapes forever. The larger pursuit needs an approach policy
outside the local finish. Do not call this an infinite-horizon optimal solution.

## Collision-constrained action game

A retained configuration admits at least one collision-free braking maneuver
with constant curvature from{-maximum,0,+maximum}. Braking uses held 0.05 s
commands and adjusts the last acceleration to stop exactly at an update.
The geometry checker certifies the swept arc using adaptive clearance bounds.

An action is admissible when its complete control interval avoids obstacles
and its endpoint still admits such a braking maneuver. This is a sufficient
safety restriction, narrower than all physically possible collision-free
strategies. Numerical labels and controller action banks use the same rule.
The controller adds a=-v/dt when needed to stop exactly at low speed.

The controlled benchmark currently supplies exact map-based admissibility to
all compared controllers. This is an explicitly privileged feasibility layer,
not an end-to-end sensor-only navigation result. Learned values receive local
observations; the numerical reference has the full fixed map.

## Eight-state reference without a dense eight-axis allocation

local_game/grid_game.py stores each vehicle's legal four-state configurations.
The joint value indexes pairs of these configurations. All eight dynamic
variables remain; omitted states are illegal configurations or outside the
local separation domain. This is sparsity, not a symmetry reduction to 5D.

Each Bellman update minimizes over P's actions and maximizes over E's actions.
Continuous successor states use multilinear interpolation. Invalid grid
vertices are omitted and weights normalized. Empty interpolation cells use
nearby legal configurations connected without crossing an obstacle, with
positive normalized inverse-distance weights. This boundary treatment is a
numerical approximation. On the checked production grid it does not change
node-transition graphs; it extends queries at intermediate continuous states.

The joint interpolation factors exactly: for EACH P action, first interpolate
P's successor, then interpolate and maximize E responses, and finally minimize
P actions. Maximizing E before P interpolation would be a different operation.
An independent brute-force small-game test checks the actual factorization.

Pair values outside the 3 m local radius have a fixed positive exit value. They
need not be stored. The GPU keeps one current float32 field. New values are
computed in tiles, written to a host array, then uploaded after a full backup.
This shifts some storage/transfer work to RAM without changing the update.

Signed histories can be stored in float16 while preserving nonzero signs.
Continuous queries interpolate the signed fields before locating a first zero
crossing. They also enforce the known current geometric payoff as an upper
bound: available extra time cannot make the reachable contact margin worse.
Times beyond the retained budget remain unresolved, not predicted captures.

## Dataset and observations

Use the original BARN grid files and their YAML resolution. An occupied cell
represents the dataset's circular cylinder; scale centers and radii physically.
The current 2×scale produces 9×9 m arenas from the original 30×30,0.15 m grids.
An enclosing square boundary is added for pursuit–evasion. This is a BARN-derived
PE benchmark, not the original navigation challenge or its scoring protocol.

The fixed test IDs are 0,33,66,100,133,166,199,233,266,299, with five starts each.
Starting configurations are selected from geometry/dynamics, never successful
controller outcomes. Entire BARN maps are separated between training,
validation and testing.

local_game/perception.py simulates static-obstacle LiDAR returns and accumulates
observed cells. A fresh P-centered body-frame image is generated at every
control update. Inputs distinguish observed occupancy from unknown space.
The learned local game uses one shared P-centered value frame for both players.
A full local-map image is a separate information ablation, not silently used in
place of LiDAR. Joint vehicle states are exact in this simulation.

The map image covers±6 m. For the 2 s budget,3 m initial separation and current
speed/footprint bounds, this covers the physical motion and stopping context
needed around P. Occluded geometry can still be missing. Full-map DP labels
therefore provide privileged supervision; a learned predictor from partial
observations is not an exact partially observed game solution.

## Code entrypoints

- geometry.py: circular BARN obstacles and exact rectangular contact sign.
- native.py / native_geometry.cpp: held-command propagation, collision/braking
  checks, contact event times, range scans and connected-query checks.
- grid_game.py: compact state grids, interpolation, local pair packing and GPU
  Bellman updates.
- reference.py: streamed solves, checkpoints, sampled labels and history queries.
- perception.py: LiDAR memory, body-frame images and observed-map features.
- control.py: matching P-min/E-max time objectives and deterministic ties.
- simulate.py: local feedback diagnostic replays with separate outcomes.
- learning.py: optional map encoder and monotone remaining-budget value curves.
- navigation.py: shared observed-map route targets and reversible footprint MPPI.
- hierarchy.py: full chases, successor-based handoffs, cross-play and timings.
- free_value.py: obstacle-blind body-contact value comparison with the same physics.

Dependencies remain optional. The base package is NumPy-only; this experiment
also uses SciPy, a C++/OpenMP compiler, optional CuPy for reference computation,
and optional PyTorch for learning. Reference tables, models, images and replays
belong outside the code repository.

## Whole chase and comparison contract

`examples/chase_barn_contact.py` replays one initial joint state. The numerical
reference supplies physical geometry and local signed histories. A learned
source needs only its checkpoint, a fresh local image, and the joint state for
value queries; the benchmark still shares exact physical action feasibility.

At every 0.05 s update, P tests its admissible commands against the admissible
E replies. P enters the local finish if at least one command has a finite
remaining-time estimate against every tested E reply. The current state need
not already have a finite estimate. Otherwise P uses the common approach.
E responds to the selected P command by maximizing the same time estimate.
When all local times are unresolved, the signed contact field breaks ties;
outside the local domain E uses the common escape-navigation heuristic. This
is a P-then-E sampled game, not an independently simultaneous-action solver.

The approach uses a route through currently observed obstacle surfaces. Unknown
space is optimistic for this navigation heuristic, while the value network
receives an explicit known-space mask. P follows the route toward E; E's far
heuristic seeks greater route distance. A 2 m lookahead sets the waypoint.
The existing log-MPPI update tracks a position and forward-heading target with
512 samples over 2 s. Reverse maneuvers remain available. Navigation costs end
at waypoint arrival; arrival does not end the pursuit game. Swept footprints
and a braking continuation filter the samples. Exact-map feasibility projects
an unsafe proposed first command onto the shared safe command set.

This is a route-guided BARN adaptation of the earlier hierarchy, not an unchanged
lion strategy. Its purpose is to keep the approach identical across local-value
comparisons. No new optimality claim is made for the route or tracker.

The free-space control comparison uses a separate rectangular-contact table
with the same acceleration, footprint, local radius and time budget. It does
not reuse earlier circular-capture histories. Its immediate physical action
filter still knows the obstacles, but its future value does not.

Primary comparisons are numerical P/numerical E, learned P/numerical E, and
learned P/learned E. A rise in capture rate in the last pairing can mean E became
weaker; it is not by itself evidence that learned P improved. Cross-play and
shared-state action checks are needed to interpret the result.

## Reproducing supervised value fitting

After numerical reference generation, `examples/build_contact_observations.py`
creates single-scan, accumulated-scan and ideal local-map inputs beside sampled
labels. `examples/fit_contact_values.py --train ... --validation ... --output ...`
trains the image model and a state/observed-clearance MLP on the same queries.
These are supervised value models; no imitation or PPO is involved.

`local_game/dataset.py` owns the observation/label pairing. `training.py` owns
sampling, the weighted field loss, validation and atomic checkpoints. Whole
physical maps must be disjoint between training and validation. Checkpoints
are selected by accumulated-observation field error, not test chase outcomes.
The model predicts a monotone curve over remaining-time budgets, beginning at
the exact current rectangular-contact geometry. A first zero crossing yields
an estimated capture time; no crossing means unresolved within the budget.

Training mixes current scans, synthetic short-memory inputs and ideal local
maps. The synthetic memory combines the root scan with two visible scans from
collision-free successor poses; it is not a recorded continuous round trip.
Executed-controller memory instead accumulates its actual past scans. The ideal input is explicitly tagged in evaluation; it is not silently
supplied at deployment. Both models receive current contact geometry and
clearance features computed from the observed image. Thus the no-image model
is a state-and-clearance baseline, not a fully map-blind predictor.

The chase example also supports `--learned-only --checkpoint MODEL.pt`. In that
mode both players use the network, and the supplied reference directory needs
only `meta.json` describing the physical game. No numerical value arrays or grid
files are loaded. This path exactly reproduced a two-second learned-pair replay
when tested with a metadata-only directory. The common full-map physical action
check remains; this is table-free value inference, not a sensor-only safety claim.

The present observation/query sampler places P close to the observation-frame
origin (the measured training offsets are below about 0.32 m and 0.26 rad).
That covers the one-step candidate queries used here. It does not establish
accuracy for deeper MCTS leaf states far from the encoded root pose. Such use
needs broader query training or a separately validated frame/encoding scheme.
The image-extent check in the query API is a geometric-domain check, not an
accuracy guarantee over every state inside the image.


## Optional contact-cost approach

The default handoff requires a finite worst-response capture time. The optional
`chase(..., use_contact_cost=True)` / `--use-contact-cost` additionally executes
the selected minmax contact-cost command when every continuing query is supported
and its worst contact cost is below the local exit cost. This is an experimental
approach heuristic; it is not a promise of capture. It leaves all dynamics,
opponent choices, control timing, models and physical constraints unchanged.
The returned trace distinguishes local finish, contact-cost approach and MPPI.
The baseline gate remains the default because the broader rule improved the
numerical reference on the tested cohort but reduced captures with learned and
free-space values. See [the benchmark](barn-benchmark.md).

`--observation memory` is the standard scan-memory input. `single` uses only the
current scan; `ideal` supplies complete local geometry as an information
ablation. An ideal input is not a sensor-only deployment result.
