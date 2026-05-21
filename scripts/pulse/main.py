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
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plot images. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-c",
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
        "--array-format",
        choices=("npy", "csv"),
        default="npy",
        help="Format for generated numeric array outputs. Default: npy",
    )
    drift_group = parser.add_mutually_exclusive_group()
    drift_group.add_argument(
        "-d",
        "--drift-correction",
        dest="baseline_drift_correction",
        action="store_true",
        default=None,
        help="Enable baseline/PHA drift correction and show drift controls.",
    )
    drift_group.add_argument(
        "-n-d",
        "--no-drift-correction",
        dest="baseline_drift_correction",
        action="store_false",
        help="Disable baseline/PHA drift correction and use the original UI steps.",
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
        from .workflow import (
            launch_pulse_workflow,
            print_savefig_progress,
            save_pulse_plots,
        )
    except ImportError:
        from workflow import (
            launch_pulse_workflow,
            print_savefig_progress,
            save_pulse_plots,
        )

    try:
        config = load_config(args.config)
        if args.baseline_drift_correction is not None:
            config = config.with_updates(
                baseline_drift_correction=args.baseline_drift_correction
            )
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
            array_format=args.array_format,
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
            array_format=args.array_format,
            savefig_progress_callback=print_savefig_progress,
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        pulse_data.close()

    print(format_hdf5_summary(pulse_data.summary))
    print("\nSaved outputs:")
    for output_path in output_paths:
        print(f"- {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
