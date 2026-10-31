"""Replace reference query samples with legal continuous control successors."""

import argparse
from pathlib import Path
from pursuit_evasion.local_game.dataset import sample_control_successors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "original", type=Path, help="Existing labels and observation frames"
    )
    parser.add_argument(
        "reference", type=Path, help="Matching retained numerical history"
    )
    parser.add_argument("output", type=Path, help="Separate empty output directory")
    parser.add_argument("--seed", type=int, default=1031)
    args = parser.parse_args()
    result = sample_control_successors(
        args.original, args.reference, args.output, seed=args.seed
    )
    print(f"Saved {result['queries']:,} legal successor queries; images unchanged")


if __name__ == "__main__":
    main()
