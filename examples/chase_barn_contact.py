"""Replay a contact-game hierarchy against a matched numerical/learned evader."""

import argparse
import json
from pathlib import Path

from pursuit_evasion.local_game.reference import ContactReference
from pursuit_evasion.local_game.hierarchy import chase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--initial",
        nargs=8,
        type=float,
        required=True,
        metavar=("xP", "yP", "headingP", "vP", "xE", "yE", "headingE", "vE"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Learned P value; numerical P otherwise"
    )
    parser.add_argument(
        "--learned-evader", action="store_true", help="Use the same learned value for E"
    )
    parser.add_argument(
        "--learned-only",
        action="store_true",
        help="Load only game metadata; both players use the checkpoint, with no numerical arrays",
    )
    parser.add_argument("--approach-only", action="store_true")
    parser.add_argument(
        "--use-contact-cost",
        action="store_true",
        help="Also use supported contact-cost guidance when capture time is unresolved; experimental",
    )
    parser.add_argument(
        "--observation",
        choices=["memory", "single", "ideal"],
        default="memory",
        help="Value-network input; ideal supplies complete local geometry as a diagnostic",
    )
    parser.add_argument("--max-time", type=float, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.learned_only:
        if not args.checkpoint:
            parser.error("--learned-only requires --checkpoint")
        from types import SimpleNamespace
        from pursuit_evasion.reversible.model import ReversibleConfig
        from pursuit_evasion.local_game.geometry import CircleArena

        meta = json.loads((args.reference / "meta.json").read_text())
        ref = SimpleNamespace(
            config=ReversibleConfig.from_dict(meta["spec"]["config"]),
            arena=CircleArena.from_dict(meta["spec"]["arena"]),
            domain=SimpleNamespace(exit_radius=meta["spec"]["exit_radius"]),
        )
        args.learned_evader = True
    else:
        ref = ContactReference(args.reference)
    p = ref
    if args.checkpoint:
        import torch

        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        from pursuit_evasion.local_game.learning import ContactNeuralField

        p = ContactNeuralField(args.checkpoint, ref.config)
    if args.learned_evader and not args.checkpoint:
        parser.error("--learned-evader requires --checkpoint")
    result = chase(
        args.initial,
        ref,
        p,
        p if args.learned_evader else ref,
        max_time=args.max_time,
        seed=args.seed,
        approach_only=args.approach_only,
        use_contact_cost=args.use_contact_cost,
        value_observation=args.observation,
    )
    result["numerical_table_loaded"] = not args.learned_only
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"{result['event']} at {result['end_time']:.3f} s; saved {args.output}")


if __name__ == "__main__":
    main()
