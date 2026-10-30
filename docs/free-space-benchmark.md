# Reproduce the free-space comparison

The current development question is whether a reusable local game table improves
an approach controller, and whether bounded UCT improves the remaining approach.
The experiment is deliberately obstacle-free. It is not a general opponent benchmark.

Use `configs/reversible_free_a3.json`: both acceleration bounds are 3 m/s², P
speed is −1 to 2 m/s, E speed is −0.5 to 1 m/s, curvature bounds are 1 and 2 m⁻¹,
and capture radius is 0.3 m. `WorldMap(None)` explicitly removes walls. The legacy
`arena_half_width` config field does not define this experiment's environment.

The fixed dataset has E x/y in `{-15, -5, 5, 15}` m and headings
`{0, pi/2, pi, -pi/2}`: 64 conditions. P starts at `(0,0,0,0)`; E starts at
1 m/s. The cutoff is 45 s. Use controller seed 0, 2 m waypoint lookahead,
512 log-MPPI samples and a 1.5 s MPPI horizon. All methods share those settings.
The 2 m setting was selected by a separate small development check; the main
64-case dataset was fixed before final comparison.

The actual opponent is `ReactiveEvader(mode='separation', horizon=1)`.
It observes the selected current P command and updates every 0.05 s. The same
policy reacts separately in every run; it does not imply identical E paths.

From the repository root, after installing the package:

```bash
python examples/chase_free.py --method approach --output /tmp/pe-approach.json
python examples/chase_free.py --method b1 --table /path/to/compatible/table \
  --output /tmp/pe-b1.json
python examples/chase_free.py --method b2 --table /path/to/compatible/table \
  --native-rollouts --output /tmp/pe-b2.json
```

The default initial state is the selected qualitative example. Pass `--initial`
with eight world-state components to reproduce another condition. The example
refuses to overwrite an existing episode. `--root-selection most_visited` selects
the separately evaluated B2 root rule; all other settings remain explicit.

## Local numerical reference

The completed Oct 30 field uses 193 position samples on each axis over ±4 m,
96 periodic relative headings, and 13 samples for each speed. There are
604,327,776 states. Saved signed fields are 0.2 s apart up to 8 s; this is not
the controller's 0.05 s execution interval. The GPU solve used float32 local
arithmetic, so it is a declared numerical approximation. Model metadata prevents
using an earlier acceleration model with the new controller.

The solve took 40.66 minutes on the development laptop. Core GPU arrays required
7.25 GB, its allocator reserved 7.87 GB, and the host transfer buffer required
2.42 GB. Output occupied 103.94 GB. These figures distinguish working allocations
from retained history and are hardware-specific. Generated data are not in Git.

## Read the comparison correctly

Report capture count over all 64 starts, plus mean time with uncaptured episodes
counted at the 45 s cutoff. A mean restricted to captured episodes can improve
merely because hard cases became failures. Preserve every failure and mark the
DP/approach/search owner in replays.

B1 and approach-only each captured 36/64; their mean capped times were 28.39
and 28.57 s. The selected paired improvement was 30.22 to 27.67 s, with identical
executed states before the first DP command at 22.80 s. All 28 B1 timeouts
occurred without DP takeover. This shows a local finishing benefit, not improved
capture coverage in this dataset.

The initial B2 configuration captured 33/64 and had mean capped time 29.33 s.
It did not improve B1. Conditional longer continuations were also insufficient
on the complete grid. These negative results are retained; a working search
implementation is not evidence of a stronger controller.

Physical simulation advances at a fixed 0.05 s interval regardless of planning
latency. B2 currently takes far longer than that budget. Planning quality here
must not be presented as real-time Jetson performance.
