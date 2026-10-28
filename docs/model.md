# The first physical game

Two point cars move in a fixed square. Both speeds are positive and constant.
The state is `[xP, yP, thetaP, xE, yE, thetaE]`, in metres and radians.
The commands are P and E angular velocities in rad/s. They are not steering
angles. For a bicycle, steering angle = atan(wheelbase * turn_rate / speed).

The default square is `[-3, 3] x [-3, 3]` metres. P travels at 1 m/s with a
1 rad/s turn bound; E travels at 0.3 m/s with a 2 rad/s turn bound. Thus their
minimum turn radii are 1 m and 0.15 m. Capture radius is **0.3 m**, not diameter.
Both commands are held for 0.05 s. Exact constant-turn arcs are used between
updates, with separate continuous-event detection. No vehicle stops or reverses.

## Boundary rule: a declared modeling choice

A fixed-speed car may be unable to avoid a wall from some poses. This prototype
therefore terminates on the first event: physical capture, P boundary contact,
or E boundary contact. A boundary fault loses the game for the offending player.
P's fault has priority in a numerical tie; otherwise capture precedes E's fault.

**An E boundary fault is not physical capture.** Metrics must distinguish these
outcomes. The PDE target is physical capture OR E's boundary fault, subject to
avoiding P's boundary fault. Consequently its time is a time-to-P-success, not
unconditionally a physical capture time. This extension defines values even for
nonviable wall-facing states. It is not equivalent to silently removing all
future wall-violating E strategies. The choice must be reviewed before treating
the benchmark as a canonical constrained capture game.

For trajectories where E remains legal, P success means physical capture.
Replays and headline capture metrics must explicitly show whether this is so.
The code never bounces, clamps, stops, or teleports a car at a wall.

## Numerical DP convention

The controller minimizes, the evader maximizes. The continuous HJI Hamiltonian
is the minimum over P and maximum over E of gradient dot joint dynamics.
Positive lookback time is a *remaining time budget*, not physical elapsed time.
The square's absolute placement remains in the six state coordinates.

Offline PDE controls can change continuously; executed commands are held for
0.05 s. That difference needs sampled-control validation. Matching array values
between solvers does not establish an exact continuous-game capture certificate.

## Information and online evaluation

Both players observe the current joint state. A numerical P-then-E minimax backup
allows E to respond to P's candidate command. A chosen benchmark E policy must
say whether it uses that command or only the observed state. Use the same rule
across compared controllers. Never silently disclose P's entire future plan.

