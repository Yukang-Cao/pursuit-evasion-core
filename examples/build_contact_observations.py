"""Build local sensor inputs for a completed sampled numerical reference."""

import argparse
from pathlib import Path
from pursuit_evasion.local_game.dataset import build_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    result = build_observations(args.reference)
    print(f"Saved observation inputs in {result['seconds']:.1f} s")


if __name__ == "__main__":
    main()
