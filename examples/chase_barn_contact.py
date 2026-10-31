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
    parser.add_argument("--approach-only", action="store_true")
    parser.add_argument("--max-time", type=float, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ref = ContactReference(args.reference)
    p = ref
    if args.checkpoint:
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"{result['event']} at {result['end_time']:.3f} s; saved {args.output}")


if __name__ == "__main__":
    main()
