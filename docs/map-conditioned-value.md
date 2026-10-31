# Map-conditioned local game-value pilot

The offline reference keeps both car poses and signed speeds: eight physical
state coordinates. Online, a map encoder and a small query MLP replace a dense
per-map lookup. This is a feasibility study on a controlled map family, not a
claim of a solved general pursuit–evasion problem.

## Model and meaning of the output

P is a centered 0.80 × 0.50 m rectangle. E is provisionally 0.40 × 0.25 m.
Both have the existing A3 reversible dynamics. Set point_vehicles=False in the
shared configuration; older point-collision APIs reject this model. Capture uses center separation
0.5 m, independently of body overlap. The internal state order remains
[xP, yP, headingP, vP, xE, yE, headingE, vE].

The current reference retains **first collision loses**. Its target is capture
OR an E collision; P collision is avoided. Therefore its signed value predicts
P success within a time budget, not unconditionally physical capture time.
Recorded captures and E collisions must be reported separately. This game rule
and the provisional E footprint need review before a final robot benchmark.

The footprint_map module uses separating axes for oriented-rectangle versus
axis-aligned obstacle collision. It also detects events within held-control
intervals with a conservative bound on clearance change. The physical model is
shared across label generation and local policy evaluation.

## Reference computation

The gpu_map_game module implements the eight-dimensional min-P/max-E HJI
update, using ENO derivatives, local Lax–Friedrichs dissipation and first/second
order time updates. Float32 local arithmetic is explicit. Speed boundaries use
inward derivatives; heading axes are periodic. Tiny updates are checked against
the independent NumPy reference from the earlier world solver.

The representative grid has 17 samples on each of the four position axes,
8 on each heading axis and 7 on each speed axis: 261,921,856 states.
Positions span ±3 m around a known 5 × 5 m room, so position spacing is 0.375 m.
The reference runs to 2 s. These are coarse numerical labels.

Instead of retaining every full field, the solver samples 48,000 legal,
initially uncaptured grid states at budgets 0, 0.5, 1, 1.5 and 2 s.
Selected final fields are retained for independent querying and diagnostics.
Continuous query locations are also supported through multilinear interpolation.

MapGameField reads a retained final field or an optional signed-field history.
field(states, budget) returns the signed field and rejects a missing budget.
time_value(states) requires the history and estimates the first zero crossing
of spatially interpolated fields. It uses exact current geometry at time zero,
returns NaN when unresolved, and means time to P success under the collision
rule—not unconditionally capture time. It interpolates signed fields before
finding a crossing, rather than interpolating finite/infinite arrival labels.

## Neural inputs and outputs

The map_value module is an optional PyTorch component. A U-Net-style encoder
creates dense spatial features. Bilinear queries at both P and E positions,
pooled map features, the two full vehicle configurations, time budget and current
footprint clearances feed an MLP.

The network predicts:
initial geometric payoff + time budget × learned correction.

It therefore matches the known zero-time boundary by construction. It does not
enforce all HJI inequalities or guarantee accurate control derivatives.

The comparison MLP receives the SAME state, time, initial payoff and current
clearances, but no map image. It is a state-and-clearance baseline, not a truly
map-blind baseline. It has 341,569 parameters, versus 293,425 for the conditional
model, so the conditional advantage cannot be described simply as more capacity.

The current implementation uses normalization scales appropriate to this fixed
P/E model. Other dynamics/footprints require compatible data and retraining.

## Coordinate frames

PlanningFrame transforms both cars and the map consistently. The dynamics
commute with that coordinate transform. P is at the origin when a frame is
created at its current pose, but a future P query moves relative to that FIXED
frame. Both future poses must therefore remain in the query.

A five-variable relative representation is also possible if the map is
re-expressed at each queried P pose. Five vehicle variables with an unchanged
map that omits P's placement are generally insufficient. The network's eight
query coordinates do not imply allocating an eight-dimensional output grid.

Rigid frame augmentation can reuse the same physical reference labels. It must
transform the map observation and both queried poses together and avoid losing
relevant map context through cropping.

## Reproduction

Install the base package and a working CuPy environment for labels. For neural
training, use a compatible PyTorch installation separately. Tested locally:
CuPy 14.2 with CUDA 13 for references; PyTorch 2.8 + CUDA 12.8 for learning.

    python examples/generate_map_value_data.py --output /tmp/pe-map-smoke --smoke
    python examples/generate_map_value_data.py --output /path/to/map-data \
      --retain-final empty train_01 ood_corner --retain-history test_17 test_18
    python examples/train_map_value.py --dataset /path/to/map-data \
      --output /path/to/map-models --steps 60000 --seed 0 --device cuda

The smoke grid is only an implementation check. Full reference generation uses
the explicit suite in configs/map_value_pilot_suite.json. Training uses
whole-layout train/validation/test splits, and checkpoint selection sees only
validation data. The additional L-shaped scene is a limited out-of-family test.

## Scope of current evidence

On the laptop, a main-grid two-second solve took about 25–26 s with 3.14 GB
of core GPU arrays. A spatially finer 610-million-state solve used 7.32 GB
and about 70 s. Refinement changed approximately 4% of the inspected binary
value signs; it is not a convergence certificate.

Two matched 60,000-step training runs reduced held-out signed-value MAE with
map features (about 0.037 to 0.020). This was an MSE fitting experiment, not RL.
An initial fixed-budget signed-margin action selector showed a large capture
gain in only one seed. Changing P extraction to first-crossing time, with the
same networks and actual E policy, removed that apparent gain.
The final 28-start diagnostic on two held-out layouts produced 27 captures and
one E collision for the reference and map network in each seed. The no-image
MLP produced 27 captures in seed 0 and 28 in seed 1. All methods captured all
12 starts in the separately selected cross-door subset in both seeds.
There is therefore no demonstrated capture-count benefit from map context yet.
These results establish an inspectable pipeline; they do not establish a
robustly superior controller, sensor-noise robustness, or onboard performance.

A coarse lattice initially missed physically passable 0.6 m openings.
Those four training openings were aligned to the lattice before learning;
the original cases were preserved. Arbitrary narrow passages remain an
important reference-quality issue. Footprint inflation does not by itself
eliminate topology changes or discretization sensitivity.


## Query without a reference table

The neural query itself needs no precomputed field. The numerical reference is
used for evaluation and for the diagnostic opponent, not as a neural input.

    python examples/query_map_value.py --checkpoint /path/to/map_value_best.pt \
      --map test_17 --state 1.5 0.375 -2.35619449 1.5 1.125 -1.875 2.35619449 1 \
      --budget 2 --device cpu

The initial payoff and clearances are computed from current geometry. They are
not future DP labels. In this pilot that geometry is analytically specified;
a real perception pipeline would need consistent footprint queries from its
observed occupancy/SDF data.

A local observation edge must not silently become a physical wall. The current
suite uses actual room walls. A larger-world local patch needs an explicit
outside-region convention, such as capture before E leaves the known region.
That boundary-aware reference and the full hierarchy remain future work.

## Local controller reproduction

map_game_policy contains the nine-by-nine successor calculation and the
minimax selector. select_time_or_margin uses the lowest worst-response time
when a complete finite row exists; otherwise it uses the signed budget value.
The actual diagnostic E observes P's command and maximizes the numerical
reference's two-second signed value. It is not an exact time-optimal E.

    python examples/chase_map_value.py --reference /path/to/map-data/results/test_17 \
      --checkpoint /path/to/map-models/map_value_best.pt --device cuda \
      --state -0.75 -1.125 0 1.5 0.75 -0.75 -0.7853981633974483 0 \
      --output /path/to/new-chase.json

Omit --checkpoint for the numerical-reference P. The reference must retain
history (see --retain-history above). NeuralMapField itself does not open a DP
table; this replay loads one to define the common diagnostic opponent.
Both poses, their speeds, the physical footprints, and the collision/capture
rules are checked against checkpoint metadata. The table-free query example
above is useful for inspecting the network separately from this opponent.

The public replay reproduced the selected map-conditioned trajectory exactly,
including its 0.9461658 s capture. This is a reproduction check, not evidence
of superior control. The private experiment recordings are intentionally not
shipped in the code repository.

## Runtime scope

On the 8 GB RTX PRO 2000 Blackwell laptop GPU, cached 81-state neural field
queries took about 0.88 ms median (one budget), while map encoding took 0.53 ms.
A full warm fixed-state P decision including all footprint-aware successors and
five-budget time extraction took about 30 ms. Complete recorded diagnostic
steps had a roughly 231 ms 95th percentile, including the reference-based E.
These measurements have different scopes and are not interchangeable. No Jetson
or complete real-time deployment result is claimed.
