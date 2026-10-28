"""Thin entrypoints. Numerical backends are imported only by the solve command."""

import argparse
import json

from .model import GameConfig


def main():
    p = argparse.ArgumentParser(prog="python -m pursuit_evasion")
    commands = p.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve", help="compute a partial 6D square game field")
    solve.add_argument("--config", required=True)
    solve.add_argument(
        "--backend", choices=["native", "jax", "optimized_dp"], default="native"
    )
    solve.add_argument("--positions", type=int, default=17)
    solve.add_argument("--headings", type=int, default=16)
    solve.add_argument("--extent", type=float, default=3.5)
    solve.add_argument("--horizon", type=float, default=2.0)
    solve.add_argument("--output-dt", type=float, default=0.1)
    solve.add_argument("--order", type=int, choices=[1, 2], default=2)
    solve.add_argument("--cfl", type=float, default=0.8)
    solve.add_argument("--threads", type=int, default=8)
    solve.add_argument("--no-history", action="store_true")
    solve.add_argument("--output", required=True)
    a = p.parse_args()
    if a.command == "solve":
        from .dp.grid import GridSpec

        if a.backend == "native":
            from .dp.native import solve as run
        elif a.backend == "jax":
            from .dp.jax_backend import solve as run
        else:
            from .dp.optimized import solve as run
        result = run(
            GameConfig.load(a.config),
            GridSpec(a.positions, a.headings, a.extent),
            a.output,
            horizon=a.horizon,
            output_dt=a.output_dt,
            order=a.order,
            cfl=a.cfl,
            threads=a.threads,
            history=not a.no_history,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
