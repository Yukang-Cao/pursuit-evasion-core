"""Generate sampled 8D footprint-game labels from an explicit map suite.

Requires CuPy/CUDA. --smoke provides a small implementation check, not useful
control resolution. Full data remain outside the source repository.
"""

from pathlib import Path
import argparse, json
import numpy as np
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.world_dp import WorldGrid
from pursuit_evasion.reversible.footprint_map import FootprintMap
from pursuit_evasion.reversible.gpu_map_game import solve, node_states


def sample_queries(grid, config, geometry, count, seed=12):
    rng = np.random.default_rng(seed)
    kept = []
    n = 0
    for _ in range(100):
        idx = rng.integers(grid.states, size=max(10000, count * 2))
        s = node_states(idx, grid, config)
        valid = (
            (geometry.clearance(s[:, :4]) > 0.03)
            & (geometry.clearance(s[:, 4:], "evader") > 0.03)
            & (
                np.linalg.norm(s[:, :2] - s[:, 4:6], axis=1)
                > config.capture_radius + 0.02
            )
        )
        kept.append(idx[valid])
        n += int(valid.sum())
        if n >= count:
            return np.concatenate(kept)[:count]
    raise ValueError("Not enough legal query nodes for this model/grid")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/map_value_pilot_suite.json",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--retain-final", nargs="*", default=[])
    p.add_argument("--retain-history", nargs="*", default=[])
    args = p.parse_args()
    data = json.loads(args.suite.read_text())
    config = ReversibleConfig.from_dict(data["config"])
    grid = WorldGrid(5, 4, 3, 3, 3) if args.smoke else WorldGrid(**data["grid"])
    entries = data["maps"][:1] if args.smoke else data["maps"]
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a new dataset directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "maps.json").write_text(json.dumps(entries, indent=2))
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    for entry in entries:
        geometry = FootprintMap.from_dict(entry["geometry"])
        folder = out / "results" / entry["id"]
        if (folder / "meta.json").exists():
            raise FileExistsError("Do not overwrite a completed/incomplete solve")
        idx = sample_queries(
            grid, config, geometry, 500 if args.smoke else data["query_count_per_map"]
        )
        solve(
            config,
            grid,
            geometry,
            folder,
            horizon=0.05 if args.smoke else data["horizon"],
            output_dt=0.05 if args.smoke else data["output_dt"],
            query_indices=idx,
            save_final=entry["id"] in args.retain_final,
            save_history=entry["id"] in args.retain_history,
        )
        q = np.linspace(-grid.extent, grid.extent, 64)
        x, y = np.meshgrid(q, q)
        np.save(
            folder / "map_sdf.npy",
            geometry.point_sdf(np.stack([x, y], -1)).astype(np.float32),
        )
        print("Completed", entry["id"], flush=True)
    (out / "checks").mkdir(exist_ok=True)
    (out / "checks/dataset_ready.json").write_text(
        json.dumps({"status": "complete", "maps": len(entries)})
    )


if __name__ == "__main__":
    main()
