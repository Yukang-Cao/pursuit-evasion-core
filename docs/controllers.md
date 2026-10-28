# Baseline 1, step by step

Read `controllers/hybrid.py`, then `dp_finish.py`, `lion.py`, and `mppi.py`.

At each control update:

1. Test whether one P command has resolved outcomes against every tested E
   command under the partial table. Physical capture stops the game early.
2. If a DP finish is available, execute its one-step minimax command. Keep a
   countdown to the accepted finish deadline; do not renew a fresh budget at
   every step. A failed finish is logged, followed by a short approach cooldown.
3. Otherwise choose a geometric target using the anchor-to-current-E ray.
4. MPPI samples P steering sequences, evaluates target tracking, and applies a
   cost-weighted update. Execute the first command, not the whole rollout.

The target update period and physical control interval are separate parameters.
The default target period is 0.25 s, but P's sampled and executed controls can
change every 0.05 s. The tracker predicts P only; it is not a nested P/E game.

## Lion-guided strategy

The anchor starts at P's initial position. An ideal target lies on the ray from
that anchor to the currently observed E and within the target-selection step disk.
When vehicle tracking breaks that alignment, an explicit recovery target is
used. This is a **lion-guided vehicle adaptation**, not an implementation-level
claim of the original polygon capture theorem. A nearby geometric target can
still be difficult or impossible to reach quickly with a turning-limited car.

## MPPI scope

The running cost is integrated squared target distance plus turn effort; a
terminal squared-distance term is added. Gaussian perturbations and deterministic
nominal/straight/full-turn witnesses form the bank. The update uses normalized
exponential cost weights and effective control perturbations after clipping.
This is an MPPI-style target tracker, not a claim of exact path-integral
optimality for the constrained game.

The complete weighted result is checked again: averaging valid controls does
not guarantee a valid resulting path. A sampled witness is used if necessary.
If no full-horizon sample is valid, the longest-safe-prefix fallback is recorded.
No fallback changes the fixed speed or stops a car at a wall.

The default approach tracker also requires a legal complete turning circle at
the rollout endpoint. This is a sufficient terminal continuation: keep turning
fully left or right and the exact point car stays in the square. The warm start
appends that maneuver, and the nominal rollout is included in the next bank.
This addresses the observed short-horizon corner failure; it is standard
terminal-feasibility reasoning, not a claimed new pursuit strategy. Use
`--no-terminal-circle` to reproduce the simpler tracker as an explicit ablation.
The condition is conservative and does not certify the numerical DP component.

## Development E policy

`ReactiveEvader` compares a small bank of constant-turn predictions over one
second. It observes P's current candidate command, keeps its own trajectory
inside the square when possible, and tries to maintain separation. It is a
declared heuristic opponent, not an optimal evader. The actual game replans both
players every control interval. Use other opponents for later robustness tests.

## Learning labels

`demonstrations.py` saves both teacher targets and executed P commands. Direct-
action cloning/PPO is the intended first learned variant for cheap onboard
execution. A waypoint-output student can later be compared using the same data.
The observation includes the legitimate episode anchor and previous P command;
it does not include the target label. DP-owned commands are masked from the
approach-policy cloning loss. Current episode records label approach actions
`lion_mppi`; the exporter also accepts the historical `teacher` label.
Failures remain in the raw dataset and manifest.

PPO must collect new learner interactions after cloning. The critic predicts
returns under the current policy/opponents, not a certified minimax time.

