#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from profile_io import load_dektak_profile
from ui import launch_thickness_ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze thickness from a Dektak profile by selecting a high region "
            "on the top plot and a background region on the bottom plot."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        # default=Path("data/dektak/20260415/Ea2.txt"),
        help="Path to a Dektak text export or converted one-column text file.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=10,
        help="Maximum number of high-background thickness pairs to display.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input is None:
        print("Error: No input file provided.")
        return 1
    x, y = load_dektak_profile(args.input.resolve())
    launch_thickness_ui(x, y, args.input, args.max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
