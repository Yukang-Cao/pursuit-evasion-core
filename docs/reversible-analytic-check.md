# A small analytic check of a numerical capture claim

`reversible/analytic_bounds.py` checks a particular short-time exclusion geometry
in the acceleration-limited reversible model. It does not change the solver,
controller, handoff rule, or physical model.

## What it establishes

At some states, E can prevent physical capture through a calculable deadline by
copying a scaled version of P's current command. This gives an independent lower
bound on the signed reachability payoff: minimum separation minus capture radius.
A stored numerical field below that bound is incorrect for this physical game.

The function `copying_evader_exclusion(relative, config)` returns a bound or
`None` if the required model/state/geometry conditions do not hold. `None` means
this diagnostic supplies no conclusion; it does not mean capture is possible.

## Information and physical assumptions

- The game is free space. An obstacle-constrained game needs an additional proof
  that the copying E stays legal; this helper supplies no such proof.
- E can respond to P's current command, matching the declared P-then-E convention.
  It does not receive future P commands.
- E's signed speed and acceleration limits are a fixed factor gamma of P's.
- E has enough curvature to match P's heading rate at the lower speed.
- Initially vE = gamma vP > 0.

E chooses acceleration `gamma * aP` and curvature `kP / gamma`. Speeds remain
proportional and relative heading stays constant, including at speed saturation.
In a fixed frame aligned with P at the checked state, E's displacement is a
rotated gamma-scaled copy of P's displacement. Consequently capture becomes a
fixed disk that P must enter.

Before P can reverse and before its maximum path length reaches a quarter of
its minimum-radius circle, all admissible P paths remain outside each initial
minimum-turning circle. If the entire equivalent capture disk is strictly inside
one circle, capture is impossible through that interval. The implementation
returns the positive distance margin as well as the time bound.

This is a model-specific analytic argument with floating-point evaluation of its
constants, not interval arithmetic or a certificate for a complete numerical table.

## Reproducible example

Relative state order is `[x, y, relative_heading, vP, vE]`.
With the default reversible configuration, the exact-node state
`[0.3, -0.2, -3*pi/8, 2, 1]` has no capture through approximately 0.785398 s.
Its signed-payoff lower bound is approximately +0.001465493 m.
A numerical field that calls this state captured by 0.4 s is therefore optimistic.

Tests check this state, the recorded off-grid example, a tight copying trajectory,
random admissible P controls, reflected geometry, slower speeds, mismatched models,
and numerical stability near equal vehicle speeds. The random checks validate the
implementation of the identity; the geometric argument establishes the bound.

## What this does not settle

A false remaining-time estimate at a later recorded state does not prove that the
initial-state estimate was false: a better earlier control could avoid that state.
The bound also does not identify the dominant numerical approximation error or
provide a replacement controller. A repaired numerical method must be evaluated
separately.
