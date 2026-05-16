#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

try:
    from .config import load_config
    from .pulse_io import format_hdf5_summary, open_hdf5_pulse_data
except ImportError:
    from config import load_config
    from pulse_io import format_hdf5_summary, open_hdf5_pulse_data

DEFAULT_OUTPUT_DIR = Path("outputs/pulse")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pulse analysis for an HDF5 file.")
    parser.add_argument("input", type=Path, help="Path to a pulse HDF5 file.")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Open the interactive pulse analysis GUI.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_false",
        dest="interactive",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plot images. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a pulse analysis YAML config.",
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        help="Optional path where the effective pulse analysis YAML config is saved.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=40,
        help="Maximum number of HDF5 groups/datasets to include in the summary.",
    )
    parser.add_argument(
        "-a",
        "--all-traces",
        action="store_true",
        help=(
            "Plot every sample in each waveform trace instead of downsampling to "
            "the default display limit."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from .workflow import launch_pulse_workflow, save_pulse_plots
    except ImportError:
        from workflow import launch_pulse_workflow, save_pulse_plots

    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        pulse_data = open_hdf5_pulse_data(args.input, max_items=args.max_items)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    max_points_per_trace = None if args.all_traces else config.max_points_per_trace
    max_traces = None if args.all_traces else config.max_display_traces
    if args.interactive:
        launch_pulse_workflow(
            pulse_data,
            max_points_per_trace=max_points_per_trace,
            max_traces=max_traces,
            config=config,
            output_dir=args.output_dir,
            save_config_path=args.save_config,
        )
        return 0

    try:
        output_paths = save_pulse_plots(
            pulse_data,
            args.output_dir,
            max_points_per_trace=max_points_per_trace,
            max_traces=max_traces,
            config=config,
            save_config_path=args.save_config,
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        pulse_data.close()

    print(format_hdf5_summary(pulse_data.summary))
    print("\nSaved plot images:")
    for output_path in output_paths:
        print(f"- {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
