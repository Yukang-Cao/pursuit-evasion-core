# Code and numerical provenance

The package separates the reusable physical contract and controllers from the
local research workspace. It does not import private report builders, old
relative-game implementations, or paths to stored experiments. See
[migration](migration.md) for the retained and excluded responsibilities.

## Project implementations

The world-state model, event handling, signed-table query layer and native
square solver were implemented for this package. The older relative free-space
game is background implementation experience, not an interchangeable source of
values for this fixed-square game.

The MPPI tracker follows the weighted-sampling structure inspected in the
C-Free-Uniform navigation implementation. Here it is a small NumPy tracker with
angular-velocity commands, fixed-speed arcs and square-specific validity checks.
There is no runtime dependency on that navigation framework. Matching the name
MPPI does not imply matching dynamics, costs, or an exact optimality result.

The lion-guided target adapts an anchor/ray geometric idea to vehicle tracking.
The alternating-move polygon capture theorem is not claimed for these
simultaneously moving, turning-limited cars.

## External numerical code

- [OptimizedDP](https://github.com/SFU-MARS/optimized_dp), tested at commit
  `341b20c49813ea8ec1f84638bae60a70dd4d6e7b`. The adapter calls upstream
  `HJSolver` and its six-dimensional graph. Wrapping `hcl.build` records
  compilation and kernel timing; it does not replace the graph. Output history
  is reordered and converted to the package's portable float32 layout.
- [hj_reachability](https://github.com/StanfordASL/hj_reachability), tested at
  version 0.7.0 with JAX/jaxlib 0.12.0.1. The adapter supplies the game dynamics,
  a counted integration loop and the declared tube projection. Finite
  differences and Runge-Kutta stages come from the upstream package.

The projection, target/avoid signs, physical model and CFL adjustment are
part of **our adapter configuration**. Agreement between these adapters is not
independent proof of that configuration's mathematical correctness. Read
[numerics](numerics.md) before interpreting a backend comparison.

Solver metadata records native source hashes or upstream versions/revisions
where available. For an experiment also preserve the core Git revision,
configuration, exact command, environment package list, compiler version and
hardware. A floating optional-dependency minimum in `pyproject.toml` is an
installation constraint, not an experiment lock.

License for newly authored project code has not been selected for public
release. External dependency licenses remain with their respective authors;
the upstream solver sources and environment binaries are not vendored here.

