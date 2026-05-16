from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

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


@dataclass
class Hdf5PulseData:
    file_path: Path
    h5_file: h5py.File
    summary: Hdf5Summary

    def close(self) -> None:
        self.h5_file.close()

    def __enter__(self) -> Hdf5PulseData:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def datasets(self) -> list[h5py.Dataset]:
        found: list[h5py.Dataset] = []

        def collect(_: str, node: h5py.Group | h5py.Dataset) -> None:
            if isinstance(node, h5py.Dataset):
                found.append(node)

        self.h5_file.visititems(collect)
        return found


def _resolve_hdf5_input(input_path: Path) -> Path:
    file_path = input_path.resolve()
    if not file_path.exists():
        raise ValueError(f"Input file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Input path is not a file: {file_path}")
    return file_path


def _summarize_hdf5_file(
    h5_file: h5py.File,
    file_path: Path,
    max_items: int,
) -> Hdf5Summary:
    items: list[Hdf5ItemSummary] = []
    groups = 0
    datasets = 0
    truncated = False

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

    return Hdf5Summary(
        file_path=file_path,
        groups=groups,
        datasets=datasets,
        items=tuple(items),
        truncated=truncated,
    )


def open_hdf5_pulse_data(input_path: Path, max_items: int = 40) -> Hdf5PulseData:
    """Open an HDF5 file for interactive pulse analysis.

    The returned object owns the file handle. Keep it open while analyzer code is
    reading h5py.Dataset contents, and close it when the UI exits.
    """
    if max_items < 1:
        raise ValueError("max_items must be at least 1.")

    file_path = _resolve_hdf5_input(input_path)
    try:
        h5_file = h5py.File(file_path, "r")
    except OSError as error:
        raise ValueError(f"Could not open HDF5 file: {file_path}") from error

    try:
        summary = _summarize_hdf5_file(h5_file, file_path, max_items)
    except Exception:
        h5_file.close()
        raise

    return Hdf5PulseData(file_path=file_path, h5_file=h5_file, summary=summary)


def inspect_hdf5_file(input_path: Path, max_items: int = 40) -> Hdf5Summary:
    """Open an HDF5 file and return a compact structural summary."""
    with open_hdf5_pulse_data(input_path, max_items=max_items) as pulse_data:
        return pulse_data.summary


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
