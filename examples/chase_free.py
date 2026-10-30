"""Run one explicit free-space approach/B1/B2 comparison case.

Install the package first. Generated tables and episodes stay outside the repo.
See docs/free-space-benchmark.md for the protocol and comparisons.
"""

from pathlib import Path
import argparse
import json

from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.game import WorldMap
from pursuit_evasion.reversible.controllers import Baseline1
from pursuit_evasion.reversible.b2 import B2Controller, B2Settings
from pursuit_evasion.reversible.relative_table import RelativeTable
from pursuit_evasion.reversible.opponents import ReactiveEvader
from pursuit_evasion.reversible.evaluation import run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/reversible_free_a3.json",
    )
    parser.add_argument("--method", choices=["approach", "b1", "b2"], required=True)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial",
        nargs=8,
        type=float,
        default=[0, 0, 0, 0, -15, 5, 0, 1],
        metavar="STATE",
    )
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead", type=float, default=2)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument(
        "--root-selection",
        choices=["worst_response_mean", "most_visited"],
        default="worst_response_mean",
    )
    parser.add_argument(
        "--native-rollouts",
        action="store_true",
        help="Use the equivalent optional C++ free-space MPPI cost evaluator (requires g++).",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path.")
    if args.method != "approach" and args.table is None:
        parser.error(
            "B1/B2 require --table pointing to a compatible completed field history."
        )
    if args.seconds <= 0:
        parser.error("--seconds must be positive.")
    config = ReversibleConfig.from_dict(json.loads(args.config.read_text()))
    world = WorldMap(None)
    table = None if args.method == "approach" else RelativeTable(args.table, config)
    if args.method == "b2":
        controller = B2Controller(
            config,
            world,
            table,
            B2Settings(
                depth=6,
                simulations=args.simulations,
                rollout_steps=args.rollout_steps,
                root_selection=args.root_selection,
                rollout_cost_backend="native_free" if args.native_rollouts else "numpy",
            ),
            seed=args.seed,
            lookahead=args.lookahead,
        )
    else:
        controller = Baseline1(
            config,
            world,
            table,
            mode="approach" if args.method == "approach" else "hybrid",
            seed=args.seed,
            lookahead=args.lookahead,
            takeover="successor",
        )
    opponent = ReactiveEvader(config, world, mode="separation", horizon=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_episode(
        controller,
        opponent,
        args.initial,
        config,
        world,
        max_time=args.seconds,
        seed=args.seed,
        case="free-space",
        output=args.output,
    )
    print(
        f"{args.method}: {result['outcome']} at {result['end_time']:.3f} s; {result['dp_steps']} DP steps; {result['handoffs']} DP entries"
    )
    print(f"Episode saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
