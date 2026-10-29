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

For each predicted interval, the tracker requires endpoint clearances to exceed
half the interval duration times its maximum speed. The signed clearance must
be 1-Lipschitz. This is a conservative continuous-path check: it can reject a
legal grazing path. It is not the old complete-circle constraint and is not an
exact physical wall-event calculation. Report this distinction in diagnostics.

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
