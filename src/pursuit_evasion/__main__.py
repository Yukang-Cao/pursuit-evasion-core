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
    chase = commands.add_parser(
        "chase", help="run and record the lion/MPPI/DP baseline"
    )
    chase.add_argument("--config", required=True)
    chase.add_argument("--cases", required=True)
    chase.add_argument("--case", required=True)
    chase.add_argument("--table")
    chase.add_argument("--without-dp", action="store_true")
    chase.add_argument("--seed", type=int, default=0)
    chase.add_argument("--samples", type=int, default=256)
    chase.add_argument("--horizon", type=float, default=1.5)
    chase.add_argument("--lion-step", type=float, default=1.0)
    chase.add_argument("--max-time", type=float, default=20.0)
    chase.add_argument("--diagnostic-stride", type=int, default=10)
    chase.add_argument("--output", required=True)
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
    elif a.command == "chase":
        from pathlib import Path
        from .dp.table import LocalTable
        from .controllers.hybrid import LionMPPIDP
        from .controllers.mppi import MPPISettings
        from .controllers.opponents import ReactiveEvader
        from .evaluation import run_episode, save_episode

        config = GameConfig.load(a.config)
        cases = json.loads(Path(a.cases).read_text())
        matches = [x for x in cases if x["name"] == a.case]
        if len(matches) != 1:
            raise ValueError(
                "Case name must identify exactly one starting configuration."
            )
        if not a.without_dp and not a.table:
            raise ValueError("Provide --table, or explicitly select --without-dp.")
        table = None if a.without_dp else LocalTable(a.table)
        controller = LionMPPIDP(
            config,
            table,
            seed=a.seed,
            use_dp=not a.without_dp,
            lion_step=a.lion_step,
            mppi_settings=MPPISettings(samples=a.samples, horizon=a.horizon),
        )
        episode = run_episode(
            config,
            controller,
            ReactiveEvader(config),
            matches[0]["state"],
            max_time=a.max_time,
            seed=a.seed,
            case_name=a.case,
            diagnostic_stride=a.diagnostic_stride,
        )
        episode["method"] = "lion_mppi" if a.without_dp else "lion_mppi_dp"
        episode["table"] = None if a.without_dp else str(Path(a.table).resolve())
        save_episode(a.output, episode)
        print(json.dumps({k: v for k, v in episode.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
