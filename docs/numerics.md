# Numerical contract and comparisons

The grid order is `(xP, yP, thetaP, xE, yE, thetaE)`. Positions include their
endpoints; headings are periodic without a repeated endpoint. The numerical
position domain extends outside the physical square so wall-fault states have
explicit values rather than being confused with an arbitrary grid truncation.

The target is capture OR an E boundary fault. P boundary fault is the avoid
set. The initial signed field is max(target, avoid), with negative values inside
the P-success target and positive values in the avoid set. See `model.md` before
interpreting these values as physical capture times.

Every internal numerical update applies a minimum with the previous budget's
field, then a maximum with the fixed avoid field. This is the declared
capture-by-time tube convention. It makes the stored fields nonincreasing in
budget. A target-only projection is a different numerical configuration and
must not silently share the same query metadata.

## Implementations

- `native_square.cpp`: ENO1/ENO2 finite differences, LF dissipation and RK1/RK2.
- `jax_backend.py`: upstream `hj_reachability` derivatives and integrators,
  with a counted loop for profiling and the same tube projection.
- `optimized_runner.py`: unmodified OptimizedDP HJSolver/6D graph in an isolated
  legacy environment. A wrapper measures compile/kernel time; it does not
  replace the numerical algorithm.

The JAX CFL parameter is adjusted to match the native/OptimizedDP conservative
sum-of-axis-maxima time bound while retaining the same pointwise dissipation
coefficients. Nominally identical ENO orders need not be bit-identical: stencil
tie choices and floating-point trigonometric evaluations can differ. Compare
fields, signs, action decisions, and grid refinement rather than requiring
bytewise equality or accepting any discrepancy without investigation.

## Output and query interface

All adapters write metadata and float32 fields in a portable format. With
history enabled, `fields.f32` is ordered `[time, *grid.shape]`, from zero budget
to the chosen horizon. `LocalTable` memory-maps that file and interpolates signed
fields before locating the first zero crossing.

`arrival.f32` is a diagnostic node-entry record. **Do not interpolate it as an
ordinary continuous capture-time function:** arrival times may jump across
capture boundaries. Different adapters record internal-step or output-frame
entry times; compare the signed histories for the main cross-solver check.

Unresolved/out-of-domain queries return NaN. They are neither zero cost nor a
proof of infinite evasion. Exact terminal event rules override interpolation.
The table must agree with the simulator's model metadata before controller use.

## Timing and validation

Offline records separate initialization, compilation, numerical updates,
transfers and writes where the backend exposes them. Online query and full
controller latency are measured separately. CUDA work must be synchronized
before its timer stops. A laptop measurement is not a Jetson result.

Small-grid implementation comparisons do not certify the continuous game.
Check simple geometric bounds, refinement, wall rules, event order and actual
sample-and-hold controller replays. Report physical captures separately from
E faults, and distinguish a timeout from a proven escape.

