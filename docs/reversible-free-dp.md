# Reusable free-space game for reversible cars

The physical state has eight coordinates, but the free-space game uses
`[relative x, relative y, thetaE-thetaP, vP, vE]`. Both signed speeds remain state
variables. Controls are acceleration and curvature. Translating or rotating the
whole pair changes neither the free-space model nor its capture condition.
A fixed map does change the game; this table contains no wall information.

## Numerical implementation

`reversible/free_dp.py` runs the C++ CPU implementation. The optional
`reversible/gpu_relative.py` uses CuPy RawKernel. Both implement the same ENO1 or
ENO2 spatial derivatives, local Lax–Friedrichs dissipation, and RK1 or RK2 updates.
The signed field begins as `distance - capture_radius`. Budget increases from
zero; a minimum over each internal update forms the capture-by-time tube.

The signed-speed Hamiltonian uses the **magnitude** of speed when optimizing
symmetric curvature bounds. Acceleration may point inward or be zero at a speed
limit. Positive and negative body speed do not change the state-order convention.
Position boundaries use one-sided extrapolation; heading is periodic. A finite
relative-position box is an approximation to free space, and domain enlargement
should be checked for the queries used by the controller.

Float32 fields are stored in C order `[vP,vE,psi,y,x]`, with x fastest. CPU local
arithmetic is double precision. GPU double-local arithmetic is the reference
default; optional float-local arithmetic is identified in metadata and requires
an accuracy check. Neither backend gives a formal capture certificate.

## Run a small field

From an installed checkout, with a C++ compiler and OpenMP:

```bash
python -m pursuit_evasion.reversible.free_dp /absolute/output/free5 \
  --positions 41 --headings 24 --p-speeds 7 --e-speeds 7 \
  --extent 4 --horizon 2 --output-dt 0.1 --threads 4
```

With a working CuPy installation matching the CUDA runtime:

```bash
python -m pursuit_evasion.reversible.gpu_relative /absolute/output/free5-gpu \
  --positions 81 --headings 48 --p-speeds 13 --e-speeds 13 \
  --extent 4 --horizon 4 --output-dt 0.1 --local-precision float64
```

These commands use the explicitly provisional defaults in `ReversibleConfig`.
Pass `--config` to select another model. Existing nonempty output directories are
never overwritten. RAM/GPU and disk estimates are checked before allocation.
The source kernel, physics, discretization, numerical options, timings, and
completion status are retained alongside each output.

## Query interface

```python
from pursuit_evasion.reversible.relative_table import RelativeTable
from pursuit_evasion.reversible.model import ReversibleConfig

table = RelativeTable('/absolute/output/free5', ReversibleConfig())
time = table.value([0.55, 0.0, 0.0, 2.0, 1.0])
signed_value = table.field([0.55, 0.0, 0.0, 2.0, 1.0], budget=0.5)
```

`value` interpolates the signed field spatially at successive budgets, then finds
its first zero crossing. It does **not** interpolate the diagnostic node-arrival
array. Unresolved and outside-grid queries return NaN. Model checks compare the
capture radius and vehicle limits; arena placement is deliberately excluded so
that the same free-space field can be reused in different world maps.

`gradient` and `hamiltonian_controls` expose an optional pointwise feedback
extraction. For minimum-time guidance, use the gradient at the state's
first-crossing budget, not an arbitrary later interior level set. A finite-hold
controller should evaluate first-event successors explicitly. Numerical gradient
ties and finite action/control periods can affect the resulting policy.

## Resource probes and validation

History-disabled runs measure allocation/update cost without writing every field.
The GPU API additionally accepts `probes=relative_states`, `history=False`, and
`save_final=False`; it saves only interpolated signed-field histories at those
specified states. This is useful for matched-step domain checks without retaining
another large full history. It is not a substitute for a general online table.

The unit tests compare native updates with an independent explicit control-pair
Hamiltonian, cover both numerical orders, test signed-speed boundaries and
periodic queries, and reject model mismatches and excessive allocation. Additional
GPU/CPU parity, analytic capture cases, grid/domain sensitivity, and executed
feedback replays belong in the experiment evidence. Agreement at one frontal
state does not establish policy reliability throughout the table.
