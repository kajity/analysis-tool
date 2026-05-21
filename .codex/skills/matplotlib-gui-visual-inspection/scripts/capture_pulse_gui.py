#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)


DEFAULT_OUTPUT_DIR = Path("outputs/matplotlib-gui-visual-inspection")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture rendered screenshots of the ates.pulse Matplotlib GUI."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Optional pulse HDF5 input. If omitted, a synthetic fixture is generated.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used for importing scripts.pulse. Default: current directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for screenshots and metadata. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Optional pulse analysis YAML config.",
    )
    parser.add_argument(
        "-a",
        "--all-traces",
        action="store_true",
        help="Render all traces instead of applying display downsampling.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Screenshot DPI. Default: 150.",
    )
    return parser.parse_args(argv)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "step"


def write_synthetic_hdf5(path: Path) -> None:
    """Create a deterministic pulse-shaped HDF5 fixture outside data/.

    The first half of each row is background and the second half is signal,
    matching the PulseDataSource convention used by ates.pulse.
    """
    rng = np.random.default_rng(20260520)
    trace_count = 72
    sample_count = 256
    signal_start = sample_count // 2
    x = np.arange(sample_count - signal_start, dtype=float)
    pulse = -90.0 * np.exp(-0.5 * ((x - 46.0) / 7.0) ** 2)

    wave = rng.normal(0.0, 4.0, size=(trace_count, sample_count)).astype(np.float32)
    amplitudes = rng.normal(1.0, 0.12, size=trace_count).astype(np.float32)
    wave[:, signal_start:] += amplitudes[:, np.newaxis] * pulse

    # Add a few obvious out-of-range jumps so the Reduction view is meaningful.
    for row in range(0, trace_count, 17):
        wave[row, signal_start + 8] += 180.0

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        waveform = h5_file.create_group("waveform")
        waveform.create_dataset("wave", data=wave)
        waveform.create_dataset("vres", data=np.float32(0.5))
        waveform.create_dataset("hres", data=np.float32(0.1))


def capture_gui(args: argparse.Namespace) -> tuple[Path, ...]:
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))

    from scripts.pulse.config import load_config
    from scripts.pulse.pulse_io import open_hdf5_pulse_data
    from scripts.pulse.workflow import PulseWorkflowController

    output_dir = args.output_dir.resolve()
    input_path = args.input
    synthetic = input_path is None
    if synthetic:
        input_path = output_dir / "synthetic-pulse.hdf5"
        write_synthetic_hdf5(input_path)

    config = load_config(args.config)
    max_points_per_trace = None if args.all_traces else config.max_points_per_trace
    max_traces = None if args.all_traces else config.max_display_traces

    run_dir = output_dir / input_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    screenshots: list[Path] = []

    pulse_data = open_hdf5_pulse_data(input_path)
    controller = PulseWorkflowController(
        pulse_data,
        max_points_per_trace=max_points_per_trace,
        max_traces=max_traces,
        config=config,
    )
    try:
        for index, step in enumerate(controller.workflow.steps):
            controller.workflow.go_to_step(index)
            controller.render()
            output_path = run_dir / f"{index + 1:02d}-{slugify(step)}.png"
            controller.ui.fig.savefig(output_path, dpi=args.dpi, facecolor="white")
            screenshots.append(output_path)

        metadata = {
            "input": str(input_path),
            "synthetic_input": synthetic,
            "steps": list(controller.workflow.steps),
            "screenshots": [str(path) for path in screenshots],
            "figure_size_inches": list(controller.ui.fig.get_size_inches()),
            "dpi": args.dpi,
        }
        metadata_path = run_dir / "capture-metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        screenshots.append(metadata_path)
    finally:
        controller.ui.close()
        pulse_data.close()

    return tuple(screenshots)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = capture_gui(args)
    except (OSError, ValueError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("Captured pulse GUI artifacts:")
    for path in paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
