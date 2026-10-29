# Reusing the C-Free-Uniform MPPI tracker

The reversible tracker adapts Yukang Cao's existing MPPI implementation instead
of introducing a separate tracking objective. The source is the MIT-licensed
`c_free_uniform_sampling` directory of `traj_sampling_cuniform`.

## What is reused

- `controllers/mppi_pytorch_controller.py`, `_run_mppi_optimization`: Gaussian
  perturbation, bounds, softmax trajectory weights, raw-noise nominal update,
  and final command clipping. The original code uses raw sampled noise after
  evaluating clamped controls; this adaptation preserves that choice.
- The same file's circular warm-start shift.
- `controllers/torch_planner_base.py`, `_calculate_trajectory_costs`: accumulate
  squared position error until navigation goal arrival or collision, apply
  terminal distance only to safe trajectories that never reached the goal.
- The original distance weight 1, terminal weight 5, temperature 0.5 and one
  update iteration. The 0.15 m goal tolerance comes from the navigation example.

The source sums position costs per sampled step; it does not multiply them by
the time step. We preserve that convention. All compared controllers must use
the same control period when comparing these objective values.

The compact NumPy routines are in `reversible/_cfree_mppi.py`. Unmodified source
methods are retained in `tests/references/cfree_mppi.py`; optional Torch tests
check numerical agreement on frozen costs/noise. They do not import the original
ROS, map-perception or experiment harness. The MIT notice is retained alongside
the adapted code.

Source snapshots inspected:

- `mppi_pytorch_controller.py` SHA256:
  `ba0bd9738aaada1d889f5a3d5c63e4c014039373a349d6f9738127bb8d6d5231`
- `torch_planner_base.py` SHA256:
  `a7831a3688de27b5c34e95f85baea9ee80a69372ce5b92a69f6c26012005171b`

## Required adaptations and explicitly changed behavior

1. Controls are acceleration and curvature. The state contains speed. Shared
   `reversible.model.vehicle_step` propagates these commands exactly, including
   speed saturation and reversal. This is not the original direct-speed model.
2. Calculations are performed in world coordinates with a signed map-clearance
   callback. The original local costmap/perception interface is not copied.
3. A goal reached after an earlier collision can no longer erase that collision.
   The original independent cumulative masks had this failure. A blocked event
   wins a same-sample tie. A later blocked state after goal arrival is outside
   the already completed local navigation task.
4. Invalid predicted rollouts receive zero optimization weight when any valid
   sample exists. The final weighted rollout is rechecked; if invalid, use the
   best valid sampled sequence. With no valid sample, record a longest-valid-
   prefix fallback rather than claiming safety.
5. Basic nominal and constant acceleration/curvature sequences are included in
   each bank. These give forward, reverse and turning candidates when the
   initial acceleration nominal is zero. The remaining samples are Gaussian.
6. Noise units changed with the controls. The default standard deviations are
   half each physical control bound. This is a declared prototype sampling
   setting, not a tuned or theoretically optimal choice.

No steering-effort cost or terminal full-circle requirement is used. The target
has position only; heading matters through the car dynamics and map geometry.

## Navigation termination and physical events

Arrival at a waypoint ends the local tracking cost. It does not terminate the
pursuit-evasion simulator. The simulator separately detects physical capture and
wall faults with the shared game rules.

The first physical command interval is always checked for wall clearance, even
if the local waypoint is already reached. Ending a local cost does not shorten
the command that the pursuit-evasion simulator actually executes.

For each predicted interval, positive endpoint clearances whose sum exceeds the
interval's maximum speed times its duration certify safety. The signed clearance
must be 1-Lipschitz. If that inexpensive certificate fails, the tracker checks the
actual held-command path: analytic square/rectangle intersections for WorldMap,
or adaptive numerical zero finding for a general clearance callback.

An earlier prototype incorrectly treated failure of the conservative distance
bound as a collision. In a recorded boundary case it rejected a legal path that
accelerated away from a wall. That implementation is preserved in the local audit
records; the corrected code refines ambiguous intervals rather than rejecting
them. The tracking objective and MPPI update did not change.

Checks stop after the first local goal or blocked event, except that the full
first physically executed command is always protected. Diagnostics report the
number of exact refinements, cheap certificates, and skipped post-terminal checks.

## Interface

`ReversibleMPPI(config, settings, seed, role='pursuer', clearance=...)` accepts a
signed-clearance callback. `WorldMap.clearance` supplies the square/rectangle
geometry, and `WorldMap(None).clearance` supplies free space.

`command(pose4, target2, diagnostics=True)` returns `[acceleration, curvature]`.
The `last` dictionary reports timing, navigation cost, sample validity and
fallback status; detailed sampled and selected rollouts are optional.

The tracker is stateful: it retains a nominal sequence and RNG. Hypothetical
search simulations must clone or reset that state deliberately; they must not
silently consume the actual controller's future random samples.

## Optional high-variance log-MPPI sampler

`MPPISettings(sampler="log_mppi", noise_std=(2.0, 1.0), log_noise_sigma=1.0)`
selects the tested high-variance option for P. New `Baseline1` instances use this
setting by default. The low-level `MPPISettings()` default remains Gaussian; pass
`mppi_settings=MPPISettings()` to B1 to reproduce the older Gaussian approach.
Old recordings and their declared configurations are unchanged.

Perturbations are a signed normal variable multiplied by a log-normal variable,
as in the log-MPPI sampling family and the existing C-Free-Uniform variant.
The log multiplier here has log-mean zero and log-space standard deviation one.
The normal-factor scales are 2 m/s² and 1/m. The resulting pre-clipping standard
deviations are approximately 5.44 m/s² and 2.72/m: about 5.4 times the original
Gaussian standard deviations. Physical controls are still clipped to their
unchanged bounds. This changes sampling, not admissible vehicle dynamics.

These explicit parameters avoid carrying over the old implementation's conversion
of normal moments into parameters subsequently interpreted as log-space moments.
It is the same normal-times-lognormal construction, not a bit-identical port of
that parameter convention. The navigation objective, horizon, sample count,
softmax/raw-noise update, feasibility checks and early waypoint exit are unchanged.
No claim of a general escape guarantee or exact path-integral importance correction
is made for the bounded heavy-tailed variant.

Primary source: Mohamed et al., *Autonomous Navigation of AGVs in Unknown
Cluttered Environments: log-MPPI Control Strategy*,
https://arxiv.org/abs/2203.16599 ; author implementation:
https://github.com/IhabMohamed/log-MPPI_ros .

The diagnostics retain the sampler name. Generated comparisons stay outside Git;
controller timing must still be measured before any onboard claim.
