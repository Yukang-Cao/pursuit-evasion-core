# Acceleration-limited reversible game

This model is separate from the original fixed-speed six-state experiment.
The world state is `[xP, yP, thetaP, vP, xE, yE, thetaE, vE]`.
Each action is `[acceleration, curvature]`, in m/s² and 1/m. Steering/curvature
is directly commanded; signed speed changes continuously through acceleration.
Yaw rate is signed speed times curvature, so a stopped car does not rotate.

Current free-space experiment (`configs/reversible_free_a3.json`):

- P signed speed is in [-1, 2] m/s; E in [-0.5, 1] m/s.
- Minimum turn radii are 1 m for P and 0.5 m for E.
- Acceleration magnitude limits are 3 m/s² for both players.
- Both players update at 0.05 s. Capture radius is 0.3 m.
- Point vehicles in open space: use `WorldMap(None)`. The current experiment has
  no walls. In optional fixed-map experiments, first wall fault loses, with P
  fault before capture before E fault in numerical ties.

The equal acceleration bound is an explicit research setting, not a measured
hardware calibration. The earlier `ReversibleConfig()` default retains P/E
acceleration limits 2/1 m/s² for reproducing old records. Load the A3 configuration
explicitly; metadata compatibility checks reject mismatched tables.
Zero acceleration is admissible, including at zero speed; no separate parking
controller is implemented. Excluding a parked interval would be another rule,
not something silently imposed by the propagator.

## Propagation and events

`reversible/model.py` analytically integrates held acceleration until a speed
limit is reached. It then integrates the saturated speed for the remaining
interval. Constant curvature turns signed traveled distance into the exact arc.
The same code works when velocity crosses zero within an interval. It does not
replace distance by final speed times duration.

`reversible/game.py` stops at the first physical event. Its speed-bound interval
search catches events that occur between positive-clearance endpoints, including
forward-then-reverse motion that returns to the initial position. Walls do not
bounce, stop, clamp, or teleport vehicles. The `WorldMap` can represent free
space, a square, and rectangular obstacles.

## Free-space reduction

The relative state is `[x, y, thetaE-thetaP, vP, vE]`, expressed in P's body frame.
Translation/rotation invariance removes three redundant world coordinates; it
does not remove either speed or acceleration dynamics. A world-fixed map cannot
be omitted from a value function just by changing coordinates.

The free-space DP target is physical capture. The fixed-map solver's declared
P-success target additionally includes an E boundary fault; the two quantities
must not silently be called identical capture-time values. Field queries and
actual legal-opponent replays serve different validation purposes.

New tables must carry this model metadata. The old positive-speed fields cannot
be reused for the revised dynamics. Signed-field interpolation must precede a
first-crossing time query; unknown values remain explicit.
