"""Fit numerical local-game values from prepared map observations."""

import argparse
from pathlib import Path
from pursuit_evasion.local_game.training import fit_contact_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = fit_contact_values(
        args.train,
        args.validation,
        args.output,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
    )
    print("Completed supervised value fitting:", result["best_validation_mae"])


if __name__ == "__main__":
    main()
