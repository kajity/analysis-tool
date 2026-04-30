from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

TEMPERATURE_PATTERN = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*mK", re.IGNORECASE)


@dataclass(frozen=True)
class IvData:
    current_uA: np.ndarray
    voltage_V: np.ndarray
    temperature_mK: float | None


def load_iv_data(path: Path) -> IvData:
    """Load IV data exported as whitespace-delimited Itesb(uA), SqVout(V) rows."""
    rows: list[tuple[float, float]] = []
    temperature_mK = temperature_from_name(path)

    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            parsed_temperature = temperature_from_header_line(line)
            if parsed_temperature is not None and (
                "Set Bath Temp" in line or temperature_mK is None
            ):
                temperature_mK = parsed_temperature
            continue

        columns = line.split()
        if len(columns) < 2:
            continue

        try:
            current_uA = float(columns[0])
            voltage_V = float(columns[1])
        except ValueError:
            continue

        rows.append((current_uA, voltage_V))

    if not rows:
        raise ValueError(
            f"Could not parse IV rows from {path}. Expected two numeric columns: "
            "Itesb(uA) and SqVout(V)."
        )

    data = np.asarray(rows, dtype=float)
    return IvData(
        current_uA=data[:, 0],
        voltage_V=data[:, 1],
        temperature_mK=temperature_mK,
    )


def temperature_from_header_line(line: str) -> float | None:
    if "Bath Temp" not in line:
        return None

    match = TEMPERATURE_PATTERN.search(line)
    if match is None:
        return None

    value = float(match.group(1))
    # Some exports write Current Bath Temp as 0.0997 mK while the setpoint is
    # 100 mK. Prefer the explicit setpoint scale when available.
    if "Current Bath Temp" in line and abs(value) < 1:
        value *= 1000
    return value


def temperature_from_name(path: Path) -> float | None:
    match = TEMPERATURE_PATTERN.search(path.stem)
    if match is None:
        return None
    return float(match.group(1))
