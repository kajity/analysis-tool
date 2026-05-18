from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import h5py
import numpy as np

try:
    from .pulse_io import Hdf5PulseData
except ImportError:
    from pulse_io import Hdf5PulseData

WAVE_DATASET = "/waveform/wave"
VERTICAL_RESOLUTION_DATASET = "/waveform/vres"
HORIZONTAL_RESOLUTION_DATASET = "/waveform/hres"


@dataclass
class PulseDataSource:
    pulse_data: Hdf5PulseData

    def __post_init__(self) -> None:
        self.input_path = self.pulse_data.file_path
        self.wave_dataset = self._dataset(WAVE_DATASET)
        self._validate_dataset(self.wave_dataset)
        self.vertical_resolution = self._scalar_float(VERTICAL_RESOLUTION_DATASET)
        self.horizontal_resolution = self._scalar_float(HORIZONTAL_RESOLUTION_DATASET)
        self.trace_count = int(self.wave_dataset.shape[0])
        self.sample_count = int(self.wave_dataset.shape[-1])
        self.signal_start = self.sample_count // 2
        self.signal_count = self.sample_count - self.signal_start

    def _dataset(self, path: str) -> h5py.Dataset:
        node = self.pulse_data.h5_file[path]
        if not isinstance(node, h5py.Dataset):
            raise ValueError(f"HDF5 path is not a dataset: {path}")
        return node

    def _validate_dataset(self, dataset: h5py.Dataset) -> None:
        if not np.issubdtype(dataset.dtype, np.number):
            raise ValueError(f"HDF5 dataset is not numeric: {dataset.name}")
        if dataset.ndim < 2:
            raise ValueError(
                f"HDF5 dataset must be a 2D array of shape (traces, samples): {dataset.name}"
            )

    def _scalar_float(self, path: str) -> float:
        dataset = self._dataset(path)
        return float(np.asarray(dataset[()]).reshape(-1)[0])

    def display_trace_indices(self, max_display_traces: int | None) -> np.ndarray:
        if max_display_traces is None or self.trace_count <= max_display_traces:
            return np.arange(self.trace_count, dtype=int)
        return np.unique(
            np.linspace(0, self.trace_count - 1, max_display_traces, dtype=int)
        )

    def sample_indices(self, size: int, max_points: int | None) -> np.ndarray:
        if max_points is None or size <= max_points:
            return np.arange(size, dtype=int)
        return np.linspace(0, size - 1, max_points, dtype=int)

    def read_wave_rows(self, row_indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.wave_dataset[row_indices])

    def iter_wave_chunks(self, chunk_size: int) -> Iterator[np.ndarray]:
        for start in range(0, self.trace_count, chunk_size):
            stop = min(start + chunk_size, self.trace_count)
            yield np.asarray(self.wave_dataset[start:stop])

    def aligned_signal_from_wave(self, wave: np.ndarray) -> np.ndarray:
        background_data = wave[:, : self.signal_start]
        signal_data = wave[:, self.signal_start :]
        offset = np.average(background_data, axis=1)
        return signal_data - offset[:, np.newaxis]

    def background_from_wave(self, wave: np.ndarray) -> np.ndarray:
        return wave[:, : self.signal_start]
