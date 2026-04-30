#!/usr/bin/env python3
"""Create a new analysis scaffold inside a repository."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


MAIN_TEMPLATE = """#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the {analysis_name} analysis."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a raw data file or directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs/{analysis_name}"),
        help="Directory for generated figures and tables.",
    )
    return parser.parse_args()


def load_data(path: Path):
    # TODO: Inspect a real file and implement a parser that handles
    # headers, units, delimiter choices, and malformed rows explicitly.
    raise NotImplementedError("Implement a parser for the target instrument export.")


def analyze(data):
    # TODO: Implement the core analysis using pure functions where possible.
    return data


def export_results(results, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Save figures, CSV summaries, or fit parameters here.


def main() -> int:
    args = parse_args()
    data = load_data(args.input)
    results = analyze(data)
    export_results(results, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def normalize_name(raw_name: str) -> str:
    normalized = raw_name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    if not normalized:
        raise ValueError("analysis name must contain at least one letter or digit")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create scripts/<analysis-name>/main.py and outputs/<analysis-name>/."
    )
    parser.add_argument("analysis_name", help="Name for the new analysis workflow.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root where scripts/ and outputs/ live.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_name = normalize_name(args.analysis_name)
    root = args.root.resolve()

    scripts_dir = root / "scripts" / analysis_name
    outputs_dir = root / "outputs" / analysis_name
    main_py = scripts_dir / "main.py"
    gitkeep = outputs_dir / ".gitkeep"

    if main_py.exists():
        print(f"[ERROR] Refusing to overwrite existing file: {main_py}")
        return 1

    scripts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    main_py.write_text(MAIN_TEMPLATE.format(analysis_name=analysis_name))
    gitkeep.write_text("")
    main_py.chmod(0o755)

    print(f"[OK] Created {main_py}")
    print(f"[OK] Created {gitkeep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
