from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from matplotlib import font_manager

try:
    from .iv_io import IvData
except ImportError:
    from iv_io import IvData


def load_pyplot(show: bool):
    if not show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def resolve_input_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    missing_inputs: list[Path] = []
    empty_patterns: list[str] = []
    empty_directories: list[Path] = []

    for input_path in inputs:
        raw = str(input_path)
        if any(character in raw for character in "*?[]"):
            matches = [Path(match) for match in glob.glob(raw)]
            if not matches:
                empty_patterns.append(raw)
            paths.extend(matches)
            continue

        if input_path.is_dir():
            matches = sorted(input_path.glob("IV_*mK_*.txt"))
            if not matches:
                empty_directories.append(input_path)
            paths.extend(matches)
            continue

        if not input_path.exists() or not input_path.is_file():
            missing_inputs.append(input_path)
            continue

        paths.append(input_path)

    if missing_inputs:
        missing = ", ".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(f"Input file not found: {missing}")

    if empty_patterns:
        patterns = ", ".join(empty_patterns)
        raise FileNotFoundError(f"No IV input files matched pattern: {patterns}")

    if empty_directories:
        directories = ", ".join(str(path) for path in empty_directories)
        raise FileNotFoundError(
            f"No IV input files found in directory: {directories}. "
            "Expected files like IV_*mK_*.txt."
        )

    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise FileNotFoundError(f"No IV input files found from: {inputs}")
    return unique_paths


def temperature_sort_key(item: tuple[Path, IvData]) -> tuple[float, str]:
    path, iv_data = item
    if iv_data.temperature_mK is None:
        return (float("inf"), path.name)
    return (iv_data.temperature_mK, path.name)


def label_for(path: Path, iv_data: IvData) -> str:
    if iv_data.temperature_mK is None:
        return path.stem
    return f"{iv_data.temperature_mK:g} mK"


def normalize_excluded_temperatures(exclude: list[float] | None) -> list[float]:
    if not exclude:
        return []
    return [float(value) for value in exclude]


def is_excluded_temperature(
    temperature_mK: float | None, exclude: list[float] | None
) -> bool:
    if temperature_mK is None:
        return False
    return any(
        np.isclose(temperature_mK, value, rtol=0, atol=1e-6)
        for value in normalize_excluded_temperatures(exclude)
    )


def filter_excluded_temperatures(
    datasets: list[tuple[Path, IvData]],
    exclude: list[float] | None,
) -> list[tuple[Path, IvData]]:
    excluded = normalize_excluded_temperatures(exclude)
    if not excluded:
        return datasets

    filtered = [
        (path, iv_data)
        for path, iv_data in datasets
        if not is_excluded_temperature(iv_data.temperature_mK, excluded)
    ]
    if not filtered:
        raise ValueError("All datasets were excluded by --exclude.")
    return filtered


def configure_plot_style(plt) -> None:
    preferred_fonts = [
        "Noto Sans CJK JP",
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
        "DejaVu Sans",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = [font for font in preferred_fonts if font in available_fonts]
    if not font_family:
        font_family = ["sans-serif"]

    plt.rcParams.update(
        {
            "font.family": font_family,
            "axes.unicode_minus": False,
        }
    )


def voltage_for_plot(iv_data: IvData, raw_voltage: bool) -> np.ndarray:
    if raw_voltage:
        return iv_data.voltage_V

    zero_current_index = int(np.argmin(iv_data.current_uA))
    return iv_data.voltage_V - iv_data.voltage_V[zero_current_index]


def color_for_dataset(plt, index: int, total: int):
    colormap = plt.get_cmap("turbo")
    if total == 1:
        position = 0.5
    else:
        ratio = index / (total - 1)
        position = 0.06 + ratio * (0.88 - 0.06)
    return colormap(position)


def export_plot(fig, output_dir: Path, output_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=200)
    return output_path
