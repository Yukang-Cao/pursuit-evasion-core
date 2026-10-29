# Inspect a 5D table and recorded controller handoffs

Run the Python-backed local viewer from an installed checkout:

```bash
python -m pursuit_evasion.reversible.viewer --manifest /absolute/path/viewer.json
```

It opens a browser at `http://127.0.0.1:8765`. Use `--port` to change the port,
`--no-browser` to suppress opening a window, and Ctrl+C to stop. The server binds
only to the local machine. It needs NumPy and the Python standard library, not a
GPU, plotting package, online DP solve, or remote service.

## What is implemented

- World view with draggable P/E positions and individual heading/signed-speed sliders.
- A linked relative-position heatmap. Relative heading and both speeds are fixed
  while relative x/y span the displayed slice.
- Selection of completed free-space tables and their retained time coverage.
- The numerical time at the current query, gray unresolved cells, and a capture circle.
- A separate finite-action DP decision with the square-wall/braking check.
  The colored heatmap is not the map-aware B1 takeover region.
- Recorded chases with explicit current P mode, E strategy, MPPI waypoint,
  synchronized value slices, scrubbing, and one-control-step advancement.
- Editing a recording detaches it; prospective decisions are clearly separated
  from recorded behavior. Changing coverage does not rewrite a recorded chase.
- Export of the current world/map pair to a PNG.

This version displays the reusable **5D free-space** table. It does not visualize
an 8D fixed-map field or expose a 3D surface. A square wall can be shown for the
controller feasibility check, but is never silently inserted into the value table.
Vehicles are points under the current model.

## Manifest, separate from source code

Keep large tables, recordings and their absolute paths outside Git. A manifest
has `tables`, optional `presets`, and optional `replays` lists. Each table has
`id`, `label`, and `path` pointing to a completed `RelativeTable` folder. Relative
paths are resolved against the manifest directory.

A preset supplies `id`, `label`, `description`, a world `state` of eight values,
optional `anchor` for the lion target, a `table` id, `coverage`, and `free_world`.
A recording supplies `id`, `label`, `path`, `table`, `coverage`, `p_strategy`,
and `e_strategy`. Recordings use `reversible_episode_v1` from `evaluation.py`.
The lion target comes from each recorded row, not a reconstructed guessed goal.

Example minimal manifest (replace the external table path):

```json
{
  "tables": [{"id": "free", "label": "Reusable free-space table", "path": "/data/free5"}],
  "presets": [{
    "id": "start", "label": "Editable start", "table": "free", "coverage": 2.0,
    "state": [0, 0, 0, 1, 1, 0.5, 1, 0.5],
    "description": "Drag the vehicles or change their headings and speeds."
  }]
}
```

## Query implementation and checks

`viewer.py::interpolate_slice` interpolates the three fixed axes of the signed
fields and keeps native x/y nodes. `arrival_slice` finds the first zero crossing
in time; it does not interpolate an arrival-time table. Slice fields are cached
for repeated position-only dragging. Files are memory-mapped, and no table is
written. Query latency is reported in the UI; it is separate from offline solve
cost and controller latency.

Tests compare every displayed node with the general `RelativeTable` query,
including fractional speeds, periodic headings, unknown states, and global
translation invariance. Browser checks cover drag, sliders, recordings, time
coverage, export and the distinction between recorded and hypothetical control.
A replay is a synchronized diagnostic step-through whose rate depends on queries;
it is not advertised as real-time video or robot control.
