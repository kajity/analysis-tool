from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py


@dataclass(frozen=True)
class Hdf5ItemSummary:
    path: str
    kind: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    children: int | None = None


@dataclass(frozen=True)
class Hdf5Summary:
    file_path: Path
    groups: int
    datasets: int
    items: tuple[Hdf5ItemSummary, ...] = field(default_factory=tuple)
    truncated: bool = False


def inspect_hdf5_file(input_path: Path, max_items: int = 40) -> Hdf5Summary:
    """Open an HDF5 file and return a compact structural summary.

    This intentionally records structure only. Pulse waveform interpretation is a
    later step once the instrument export layout is known.
    """
    if max_items < 1:
        raise ValueError("max_items must be at least 1.")

    file_path = input_path.resolve()
    if not file_path.exists():
        raise ValueError(f"Input file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Input path is not a file: {file_path}")

    items: list[Hdf5ItemSummary] = []
    groups = 0
    datasets = 0
    truncated = False

    try:
        with h5py.File(file_path, "r") as h5_file:
            for name, node in h5_file.items():
                stack: list[tuple[str, h5py.Group | h5py.Dataset]] = [(name, node)]
                while stack:
                    path, current = stack.pop()
                    if isinstance(current, h5py.Group):
                        groups += 1
                        if len(items) < max_items:
                            items.append(
                                Hdf5ItemSummary(
                                    path=f"/{path}",
                                    kind="group",
                                    children=len(current),
                                )
                            )
                        else:
                            truncated = True

                        for child_name in reversed(list(current.keys())):
                            stack.append((f"{path}/{child_name}", current[child_name]))
                    elif isinstance(current, h5py.Dataset):
                        datasets += 1
                        if len(items) < max_items:
                            items.append(
                                Hdf5ItemSummary(
                                    path=f"/{path}",
                                    kind="dataset",
                                    shape=tuple(int(size) for size in current.shape),
                                    dtype=str(current.dtype),
                                )
                            )
                        else:
                            truncated = True
    except OSError as error:
        raise ValueError(f"Could not open HDF5 file: {file_path}") from error

    return Hdf5Summary(
        file_path=file_path,
        groups=groups,
        datasets=datasets,
        items=tuple(items),
        truncated=truncated,
    )


def format_hdf5_summary(summary: Hdf5Summary) -> str:
    lines = [
        f"Input file: {summary.file_path}",
        f"Groups: {summary.groups}",
        f"Datasets: {summary.datasets}",
        "",
        "HDF5 structure:",
    ]

    if not summary.items:
        lines.append("(root contains no groups or datasets)")
        return "\n".join(lines)

    for item in summary.items:
        if item.kind == "group":
            lines.append(f"- {item.path} [group, children={item.children}]")
        else:
            lines.append(
                f"- {item.path} [dataset, shape={item.shape}, dtype={item.dtype}]"
            )

    if summary.truncated:
        lines.append("- ... summary truncated")

    return "\n".join(lines)
