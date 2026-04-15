from __future__ import annotations

import re
from pathlib import Path

import numpy as np


NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")


def _extract_first_number(text: str) -> float:
    match = NUMBER_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Could not parse a numeric value from: {text!r}")
    return float(match.group())


def load_dektak_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    num_points: int | None = None
    step_size: float | None = None
    data_started = False
    values: list[float] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("NumPts:"):
            num_points = int(_extract_first_number(line))
            continue

        if line.startswith("Hsf:"):
            step_size = _extract_first_number(line)
            continue

        if line.startswith("SCALED DATA:"):
            data_started = True
            continue

        if not data_started:
            continue

        tokens = [float(token) for token in NUMBER_PATTERN.findall(line)]
        if not tokens:
            continue

        if num_points is None:
            values.extend(tokens)
            continue

        remaining = num_points - len(values)
        if remaining <= 0:
            break

        values.extend(tokens[:remaining])
        if len(values) >= num_points:
            break

    if not values:
        converted_values: list[float] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if NUMBER_PATTERN.fullmatch(line) is None:
                continue
            converted_values.append(float(line))
        values = converted_values

    if not values:
        raise ValueError(
            f"Could not parse profile data from {path}. Expected a Dektak export "
            "with 'SCALED DATA:' or a plain one-column numeric text file."
        )

    y = np.asarray(values, dtype=float)
    if num_points is not None and num_points != len(y):
        raise ValueError(
            f"NumPts={num_points} but parsed {len(y)} points from {path}"
        )

    if step_size is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.arange(len(y), dtype=float) * step_size

    return x, y
