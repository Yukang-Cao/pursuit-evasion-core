"""Local, Python-backed 5D value-table explorer. No DP recomputation or GPU needed."""

from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
import argparse
import json
import time
import webbrowser
from urllib.request import urlopen
from urllib.error import URLError
import numpy as np

from .relative_table import RelativeTable
from .model import ReversibleConfig, relative_state
from .game import WorldMap
from .controllers import action_catalogue, braking_safe_mask, dp_decision
from ..controllers.lion import LionTarget
from .evaluation import json_safe


def interpolate_slice(table, psi, vp, ve):
    """Interpolate only the three fixed axes; retain every native x/y node/time.

    Uses the same periodic heading and signed-field-before-zero-crossing
    convention as RelativeTable. It never interpolates a node-arrival table.
    """
    fixed = [vp, ve, psi]
    lower = []
    weights = []
    for k, (value, axis) in enumerate(zip(fixed, table.axes[:3])):
        if not np.isfinite(value):
            raise ValueError("Slice coordinates must be finite")
        step = axis[1] - axis[0]
        if k == 2:
            f = ((value + np.pi) % (2 * np.pi)) / step
            j = int(np.floor(f)) % len(axis)
            w = f - np.floor(f)
        else:
            if value < axis[0] - 1e-10 or value > axis[-1] + 1e-10:
                raise ValueError("Speed outside table limits")
            f = (np.clip(value, axis[0], axis[-1]) - axis[0]) / step
            j = min(int(np.floor(f)), len(axis) - 2)
            w = f - j
        lower.append(j)
        weights.append(w)
    result = np.zeros((len(table.times), *table.shape[-2:]), dtype=np.float64)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = np.prod(
                    [
                        weights[k] if v else 1 - weights[k]
                        for k, v in enumerate((a, b, c))
                    ]
                )
                if w == 0:
                    continue
                result += (
                    w
                    * table.fields[
                        :, lower[0] + a, lower[1] + b, (lower[2] + c) % table.shape[2]
                    ]
                )
    return result


def arrival_slice(fields, times, x, y, capture_radius):
    """First zero crossing at each x/y node; NaN means unresolved by last budget."""
    inside = fields <= 0
    resolved = inside[-1]
    k = np.argmax(inside, axis=0)
    prev = np.maximum(k - 1, 0)
    yy, xx = np.indices(k.shape)
    a = fields[prev, yy, xx]
    b = fields[k, yy, xx]
    fraction = np.divide(a, a - b, out=np.zeros_like(a), where=np.abs(a - b) > 1e-14)
    result = times[prev] + np.clip(fraction, 0, 1) * (times[k] - times[prev])
    result = np.where(resolved, result, np.nan)
    X, Y = np.meshgrid(x, y)
    return np.where(X * X + Y * Y <= capture_radius**2, 0, result)


class Explorer:
    def __init__(self, manifest):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.tables = {}
        self.replays = {}
        self.views = OrderedDict()
        self.slices = OrderedDict()
        self.lock = Lock()
        for entry in self.manifest["tables"]:
            table = RelativeTable(self._path(entry["path"]))
            self.tables[entry["id"]] = (entry, table)
        first = next(iter(self.tables.values()))[1]
        self.config = ReversibleConfig.from_dict(first.physics)
        for _, table in self.tables.values():
            table.assert_compatible(self.config)
        for entry in self.manifest.get("replays", []):
            self.replays[entry["id"]] = (
                entry,
                json.loads(self._path(entry["path"]).read_text()),
            )

    def _path(self, value):
        p = Path(value)
        return p if p.is_absolute() else self.manifest_path.parent / p

    def metadata(self):
        return {
            "tables": [
                {
                    "id": k,
                    "label": e["label"],
                    "max_budget": t.horizon,
                    "budgets": t.times[1:].tolist(),
                    "position_spacing": float(t.spacings[-1]),
                    "heading_spacing_degrees": float(np.degrees(t.spacings[2])),
                    "p_speed_range": [float(t.axes[0][0]), float(t.axes[0][-1])],
                    "e_speed_range": [float(t.axes[1][0]), float(t.axes[1][-1])],
                }
                for k, (e, t) in self.tables.items()
            ],
            "replays": [
                {
                    "id": k,
                    "label": e["label"],
                    "p_strategy": e["p_strategy"],
                    "e_strategy": e["e_strategy"],
                    "outcome": d["outcome"],
                    "end_time": d["end_time"],
                    "table": e["table"],
                    "coverage": e.get("coverage"),
                }
                for k, (e, d) in self.replays.items()
            ],
            "presets": self.manifest.get("presets")
            or [
                {
                    "id": "start",
                    "label": "Editable start",
                    "table": next(iter(self.tables)),
                    "coverage": next(iter(self.tables.values()))[1].horizon,
                    "state": [
                        0,
                        0,
                        0,
                        self.config.pursuer.speed_max / 2,
                        1,
                        0.5,
                        0,
                        self.config.evader.speed_max / 2,
                    ],
                    "description": "Drag either vehicle or change heading and speed.",
                }
            ],
            "config": self.config.to_dict(),
        }

    def replay(self, key):
        entry, d = self.replays[key]
        return {
            "id": key,
            "initial": d["initial"],
            "world": d["world"],
            "config": d["config"],
            "outcome": d["outcome"],
            "end_time": d["end_time"],
            "p_strategy": entry["p_strategy"],
            "e_strategy": entry["e_strategy"],
            "trace": [
                {
                    "t": r["t"],
                    "state": r["state"],
                    "p_control": r.get("p_control"),
                    "e_control": r.get("e_control"),
                    "owner": r.get("pursuer", {}).get("control_owner"),
                    "waypoint": r.get("pursuer", {}).get("lion", {}).get("target"),
                    "sampler": r.get("pursuer", {})
                    .get("mppi", {})
                    .get("sampler", "gaussian"),
                }
                for r in d["trace"]
            ],
        }

    def query(self, request):
        start = time.perf_counter()
        key = request["table"]
        entry, base = self.tables[key]
        budget = float(request.get("budget", base.horizon))
        matches = np.flatnonzero(np.isclose(base.times, budget, rtol=0, atol=1e-9))
        if not len(matches) or matches[0] == 0:
            raise ValueError("Select a saved positive coverage budget")
        stop = int(matches[0]) + 1
        state = np.asarray(request["state"], float)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite world state[8]")
        relative = relative_state(state)
        fixed = tuple(float(v) for v in relative[[2, 3, 4]])
        cache_key = (key, *fixed)
        if cache_key not in self.slices:
            self.slices[cache_key] = interpolate_slice(base, *fixed)
            while len(self.slices) > 4:
                self.slices.popitem(last=False)
        else:
            self.slices.move_to_end(cache_key)
        fields = self.slices[cache_key][:stop]
        values = arrival_slice(
            fields,
            base.times[:stop],
            base.axes[4],
            base.axes[3],
            self.config.capture_radius,
        )
        view_key = (key, budget)
        if view_key not in self.views:
            self.views[view_key] = RelativeTable(base.folder, max_budget=budget)
            while len(self.views) > 8:
                self.views.popitem(last=False)
        table = self.views[view_key]
        value = float(table.value(relative))
        world = WorldMap(
            None if request.get("free_world", False) else self.config.arena_half_width
        )
        physically_valid = bool(
            world.clearance(state[:2]) > 0 and world.clearance(state[4:6]) > 0
        )
        point_in_domain = bool(np.isfinite(table.field(relative, budget)))
        decision = None
        rows = []
        u = None
        if physically_valid and point_in_domain:
            u, decision = dp_decision(state, table, self.config, world)
            commands = action_catalogue(self.config.pursuer)
            safe = braking_safe_mask(state[:4], commands, self.config, world)
            rows = [
                {
                    "acceleration": float(a[0]),
                    "curvature": float(a[1]),
                    "wall_feasible": bool(ok),
                    "worst_time": v,
                    "selected": i == decision["selected_index"],
                }
                for i, (a, ok, v) in enumerate(
                    zip(commands, safe, decision["candidate_costs"])
                )
            ]
        capture = bool(np.linalg.norm(relative[:2]) <= self.config.capture_radius)
        can_dp = bool(
            not capture
            and np.isfinite(value)
            and decision
            and decision["resolved"]
            and decision["guard_accepts"]
        )
        anchor = np.asarray(request.get("anchor", state[:2]), float)
        if anchor.shape != (2,) or not np.isfinite(anchor).all():
            raise ValueError("Invalid lion anchor")
        lion = LionTarget(step_length=1, update_period=self.config.control_dt)
        lion.reset(state[[0, 1, 2, 4, 5, 6]])
        lion.anchor = anchor
        goal = lion.target_at(state[[0, 1, 2, 4, 5, 6]], 0)
        if not physically_valid:
            mode = "invalid: vehicle outside wall"
        elif capture:
            mode = "captured"
        elif not point_in_domain:
            mode = "lion + MPPI (outside table domain)"
        else:
            mode = "DP" if can_dp else "lion + MPPI"
        # Small maps are returned directly as numeric pixels; browser maps the colors.
        result = {
            "relative": relative.tolist(),
            "state_value": value,
            "coverage": budget,
            "x": base.axes[4].tolist(),
            "y": base.axes[3].tolist(),
            "values": values.tolist(),
            "signed_field": fields[-1].tolist(),
            "slice_fixed": {
                "relative_heading_degrees": float(np.degrees(relative[2])),
                "p_speed": float(relative[3]),
                "e_speed": float(relative[4]),
            },
            "predicted_B1_mode": mode,
            "waypoint": None if can_dp or capture else goal.tolist(),
            "anchor": anchor.tolist(),
            "dp_control": None if u is None else u.tolist(),
            "dp": decision,
            "actions": rows,
            "point_in_domain": point_in_domain,
            "physical_state_valid": physically_valid,
            "query_seconds": time.perf_counter() - start,
        }
        return json_safe(result)


def serve(manifest, port=8765, open_browser=True):
    # Re-running the launch command reopens this same local viewer, rather than
    # failing because its port is already occupied.
    if port:
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
                health = json.load(response)
            if health.get("app") == "pursuit-evasion-value-viewer" and health.get(
                "manifest"
            ) == str(Path(manifest).resolve()):
                url = f"http://127.0.0.1:{port}"
                print(f"Viewer already running: {url}", flush=True)
                if open_browser:
                    webbrowser.open(url)
                return
        except (URLError, TimeoutError, ValueError, OSError):
            pass
    explorer = Explorer(manifest)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, status=200, kind="application/json"):
            raw = (
                data
                if isinstance(data, bytes)
                else json.dumps(
                    json_safe(data), allow_nan=False, separators=(",", ":")
                ).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            try:
                if self.path in ("/", "/index.html"):
                    self.send(
                        Path(__file__).with_name("viewer.html").read_bytes(),
                        kind="text/html; charset=utf-8",
                    )
                elif self.path == "/api/meta":
                    self.send(explorer.metadata())
                elif self.path.startswith("/api/replay/"):
                    self.send(explorer.replay(self.path.rsplit("/", 1)[-1]))
                elif self.path == "/health":
                    self.send(
                        {
                            "ok": True,
                            "app": "pursuit-evasion-value-viewer",
                            "manifest": str(explorer.manifest_path),
                        }
                    )
                else:
                    self.send({"error": "not found"}, 404)
            except (ValueError, KeyError) as e:
                self.send({"error": str(e)}, 400)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if self.path != "/api/query" or not 0 < size < 10000:
                    return self.send({"error": "invalid query"}, 400)
                request = json.loads(self.rfile.read(size))
                with explorer.lock:
                    result = explorer.query(request)
                self.send(result)
            except (ValueError, KeyError, TypeError) as e:
                self.send({"error": str(e)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(
        f"Value viewer: {url}\nRead-only table queries; Ctrl+C stops the local viewer.",
        flush=True,
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Local table/replay paths; not committed with datasets",
    )
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve(args.manifest, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
