# Fixed BARN pursuit–evasion comparison

Read [the physical and information contract](barn-contact-game.md) first.
This benchmark tests local game decisions in clutter, not global optimality or
long-range pursuit. It uses original BARN cylinder layouts, scaled by two and
surrounded by a square boundary. The resulting arenas are 9×9 m. This is a
BARN-derived pursuit task, not the official navigation score.

## Frozen configurations and data split

`configs/barn_contact_50.json` contains the complete physical configuration and
50 initial joint states in order `[xP,yP,headingP,vP,xE,yE,headingE,vE]`.
Positions are metres, headings radians, speeds m/s. The five starts in each map
cover stopped cars, moving cars, P reversing, center-line occlusion, and a fast
misaligned P. Initial center separation is 1–2.5 m. States were selected using
geometry and braking feasibility before observing controller performance.

- Test maps: 0,33,66,100,133,166,199,233,266,299.
- Training maps: 89,207,161,153,205,225,41,105,36,192,244,156,123,213,7,263,256,267,239,84.
- Validation maps: 26,2,104,145,218.

Each training/validation map uses 192 observation frames and 256 numerical-value
queries per frame. Two training seeds use 80,000 steps and select checkpoints
by validation field error on accumulated scan observations. Test chase outcomes
are not the checkpoint selection criterion. The initial sampler uses grid
states plus small random perturbations; it leaves speed bands that ordinary
controls reach without direct supervision. A separate successor-sampling
follow-up is tracked independently and must not overwrite this original dataset.

## Reproduction entrypoints

Download grid files from the [official BARN dataset](https://www.cs.utexas.edu/~xiao/BARN/BARN.html).
Keep maps, references, observations and checkpoints outside this repository.
With the optional dependencies described in the contact-game guide installed:

```bash
python examples/solve_barn_contact.py \
  --grid /path/to/BARN_dataset/grid_files/grid_0.npy \
  --output /path/to/reference_000 \
  --scale 2 --horizon 2 --position-step 0.3 --headings 16 \
  --retain-history --seed 103100

python examples/build_contact_observations.py /path/to/reference_000
```

For training, repeat generation on the training and validation IDs, preserving
whole-map separation, then run `examples/fit_contact_values.py --train ...
--validation ... --output ... --steps 80000 --seed 0`. This fits value functions;
it does not implement imitation learning or PPO.

For a recorded starting configuration, use `examples/chase_barn_contact.py`
with its eight coordinates from the manifest. A concrete first case is:

```bash
python examples/chase_barn_contact.py /path/to/reference_000 /path/to/chase.json \
  --initial -1.65 -2.25 -2.748893571891069 0 -3.15 -1.95 -1.5707963267948966 0 \
  --seed 103100
```

Add `--checkpoint /path/to/map_value_best.pt` for learned P against numerical E.
Add `--learned-evader` for the learned pair. Use `--approach-only` for route +
MPPI throughout. `--use-contact-cost` enables the separately tested broader use
rule; the original finite-time switch remains the default. `--observation ideal`
is a full-local-map information ablation. `--learned-only` requires only game
metadata and a checkpoint, with no numerical value arrays; exact-map physical
feasibility remains privileged in this prototype.

The chase random seed is `103100 + 10*map_id + case_index`. The reference-query
seed is `103100 + map_id`. These are different purposes. Replay comparisons use
the same initial states, physical rules, seeds and opponent policy.

## What the original 50 cases establish

Against numerical E, approach-only captured 23/50, numerical P and learned P
(seed 0) captured the same 32/50, and both a finer free-space value and the
no-image state/clearance model captured 39/50. The second map-network seed
captured 31/50. A 20 s timeout is not a proof of indefinite escape.

The primary outcome mixes the approach controller, handoff and local control:
16/18 numerical timeouts and 17/18 learned timeouts never entered local feedback.
Do not describe them as failed numerical finishes. Likewise, a higher capture
rate when E also uses a network may indicate a weaker E. The completed 3×3
cross-play compares numerical, learned and free-space policies on the same starts.

The free-space baseline uses a different HJI discretization and finer resolution
(0.1125 m, 11.25° relative heading, P/E speed spacing 0.25/0.125 m/s). It retains
the same bodies, dynamics, two-second budget, three-metre exit and immediate
physical action filter. It is a practical method comparison, not a controlled
map-removal ablation of the same numerical solver.

## Follow-up rule and diagnostics

After development on maps 0/33, a fixed rule also used the contact-cost ranking
when capture time was unresolved, provided continuing queries were supported.
The eight other maps supplied 40 prospective starts. Numerical captures rose
27→30, learned captures fell 27→23, no-image captures fell 31→25 and free-space
captures fell 31→24. This rule is retained as an explicit option, not promoted
to the default.

On a selected learned-controller failure, full local geometry and one- or
three-command numerical prefixes did not restore capture. The first command
alone therefore does not explain the full failure. A verified training-speed
coverage gap and opposite value-versus-speed trends motivate a data test; they
do not yet establish causality. Preserve these failed interventions alongside
successful examples.

## Evidence and scope

All 950 primary/follow-up/cross-play recordings passed reintegration and
intermediate physical checks: 178,858 command intervals. Neither player collided
with an obstacle in these checked runs. Capture was verified from rectangular
contact, not assigned when E touched a wall.

The numerical reference is a coarse local approximation. The learned value
receives P-centered local scans, while every method shares exact full-map
collision/braking admissibility. Both learned players locally share P's value
observation. Joint state estimates are exact; car-to-car LiDAR occlusion and
sensor noise are not modeled. No Jetson result is claimed.

The data and examples support a repeatable research comparison. They do not yet
establish a map-encoder performance advantage or an RSS-level contribution.
