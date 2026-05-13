#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

try:
    from .pulse_io import format_hdf5_summary, inspect_hdf5_file
except ImportError:
    from pulse_io import format_hdf5_summary, inspect_hdf5_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an interactive pulse analysis workflow for an HDF5 file."
    )
    parser.add_argument("input", type=Path, help="Path to a pulse HDF5 file.")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Validate the input and print the HDF5 summary without opening a GUI.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=40,
        help="Maximum number of HDF5 groups/datasets to include in the summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        summary = inspect_hdf5_file(args.input, max_items=args.max_items)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if args.non_interactive:
        print(format_hdf5_summary(summary))
        return 0

    try:
        from .ui import launch_pulse_ui
    except ImportError:
        from ui import launch_pulse_ui

    launch_pulse_ui(summary, args.input.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
