# Boundary between reusable code and local research

This repository is the reusable implementation for the six-dimensional
fixed-square game. It can be installed without its parent research workspace.
The migration preserved the older workspace and its results; it did not move
or delete those records.

## Responsibilities carried into the package

- Shared car dynamics, units, boundary/capture semantics and independent
  checks live in `model.py`, `game.py`, `geometry.py` and `tests/`.
- Offline solver adapters and a portable signed-field reader live in `dp/`.
  Public solvers remain separately installed dependencies.
- The geometric teacher, MPPI tracker, partial-DP finish and development evader
  live in `controllers/`. Physics comes from the common model.
- Explicit configurations, episode recording and demonstration export replace
  reliance on one notebook, a shell's current directory or report-builder state.

This is an extraction and implementation of responsibilities, not a file-for-
file copy of the old workspace. In particular, old relative free-space DP
tables cannot be relabeled as fixed-square six-dimensional values.

## Deliberately kept local

Research notes, HTML reports, videos, full episode cohorts, benchmark records,
large numerical fields, trained checkpoints, solver build environments and
temporary compatibility libraries remain outside this repository. They may be
referenced by explicit paths when running an experiment, but the installed
package must not import them or require their directory layout.

The tracked repository contains source, short documentation, configurations and
small tests. Generated output is ignored as an additional guard, not as a reason
to commit data by force. There are no empty placeholder packages for future
UCT, cloning or PPO implementations.

## Carrying a completed table to another machine

Keep `meta.json` and `fields.f32` together in one directory. Preserve their
original metadata. The reader checks completion, history availability, the
declared monotone projection and field length; the DP controller checks the
physical model against its configuration. `final.f32` and `arrival.f32` are
useful offline diagnostics but do not replace signed history for the current
online reader.

A copied table remains an approximation for its recorded model, grid and
horizon. Memory mapping avoids eagerly loading the whole history but does not
remove storage I/O or page-fault costs. Measure those costs on the actual
deployment device.

## Reproducing an experiment

Use a clean installation from [setup](setup.md), the same core revision and
physical configuration, and a new output location. Record the selected backend,
upstream version, grid, numerical order, precision, horizon, history retention,
controller settings, starting case and seed. Preserve the exact commands and
environment manifest beside the outputs.

Do not silently reuse old pilot fields with another projection convention.
Do not interpret an E boundary fault as physical capture, a timeout as proven
escape, or successful execution as scientific validation. The next repository
milestone is validated baseline work; the private report workspace retains the
evidence and unresolved questions behind it.
