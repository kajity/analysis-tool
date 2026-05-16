from __future__ import annotations

import h5py
import numpy as np
from matplotlib.axes import Axes

try:
    from .pulse_io import Hdf5PulseData
except ImportError:
    from pulse_io import Hdf5PulseData

WAVE_DATASET = "/waveform/wave"
VERTICAL_RESOLUTION_DATASET = "/waveform/vres"
HORIZONTAL_RESOLUTION_DATASET = "/waveform/hres"
MAX_TRACES_PER_DATASET = 20
MAX_TOTAL_TRACES = 40
MAX_POINTS_PER_TRACE = 1000


class PulseAnalyzer:
    def __init__(
        self,
        pulse_data: Hdf5PulseData,
        max_points_per_trace: int | None = MAX_POINTS_PER_TRACE,
        max_traces: int | None = MAX_TRACES_PER_DATASET,
    ) -> None:
        self.pulse_data = pulse_data
        self.input_path = pulse_data.file_path
        self.max_points_per_trace = max_points_per_trace
        self.max_traces = max_traces

        self.vertical_resolution = self._scalar_float(VERTICAL_RESOLUTION_DATASET)
        self.horizontal_resolution = self._scalar_float(HORIZONTAL_RESOLUTION_DATASET)
        data = self._dataset(WAVE_DATASET)
        self._validate_dataset(data)
        self.trace_count = int(data.shape[0])
        if self.max_traces is None:
            self.trace_indices = np.arange(self.trace_count)
        else:
            trace_sample_count = min(self.trace_count, self.max_traces)
            self.trace_indices = np.sort(
                np.random.default_rng().choice(
                    self.trace_count,
                    size=trace_sample_count,
                    replace=False,
                )
            )
        points = int(data.shape[-1])
        self.background_data: np.ndarray = data[self.trace_indices, : points // 2]
        self.signal_data: np.ndarray = data[self.trace_indices, points // 2 :]
        self.offset: np.ndarray = np.average(self.background_data, axis=1)

    def _validate_dataset(self, dataset: h5py.Dataset) -> None:
        if not np.issubdtype(dataset.dtype, np.number):
            raise ValueError(f"HDF5 dataset is not numeric: {dataset.name}")
        if dataset.ndim < 2:
            raise ValueError(
                f"HDF5 dataset must be a 2D array of shape (traces, samples): {dataset.name}"
            )

    def _dataset(self, path: str) -> h5py.Dataset:
        node = self.pulse_data.h5_file[path]
        if not isinstance(node, h5py.Dataset):
            raise ValueError(f"HDF5 path is not a dataset: {path}")
        return node

    def _scalar_float(self, path: str) -> float:
        dataset = self._dataset(path)
        return float(np.asarray(dataset[()]).reshape(-1)[0])

    def _sample_indices(self, size: int) -> np.ndarray:
        if self.max_points_per_trace is None or size <= self.max_points_per_trace:
            return np.arange(size, dtype=int)
        return np.linspace(0, size - 1, self.max_points_per_trace, dtype=int)

    def _sample_1d_dataset(self, dataset: h5py.Dataset) -> np.ndarray:
        if (
            self.max_points_per_trace is None
            or int(dataset.shape[0]) <= self.max_points_per_trace
        ):
            return np.asarray(dataset[()], dtype=float)
        indices = self._sample_indices(int(dataset.shape[0]))
        return np.asarray(dataset[indices], dtype=float)

    def _draw_all(self, ax: Axes) -> None:
        signal_data_length = self.signal_data.shape[1]
        signal_data = self.signal_data[:, self._sample_indices(signal_data_length)]
        for trace in signal_data:
            ax.plot(
                np.arange(len(trace)) * self.horizontal_resolution,
                trace * self.vertical_resolution,
                alpha=0.25,
            )

    def _draw_offset_aligned(self, ax: Axes) -> None:
        aligned_signal = self.signal_data - self.offset[:, np.newaxis]
        signal_data_length = aligned_signal.shape[1]
        aligned_signal = aligned_signal[:, self._sample_indices(signal_data_length)]
        for trace in aligned_signal:
            ax.plot(
                np.arange(len(trace)) * self.horizontal_resolution,
                trace * self.vertical_resolution,
                alpha=0.25,
            )

    def _draw_differential(self, ax: Axes) -> None: 
        aligned_signal = self.signal_data - self.offset[:, np.newaxis]
        differential_signal = np.diff(aligned_signal, axis=1)
        signal_data_length = differential_signal.shape[1]
        differential_signal = differential_signal[:, self._sample_indices(signal_data_length)]
        for trace in differential_signal:
            ax.plot(
                np.arange(len(trace)) * self.horizontal_resolution,
                trace * self.vertical_resolution,
                alpha=0.25,
            )   

    def draw_plot(self, ax: Axes, step: str) -> None:
        ax.set_title(
            f"{step}: {self.input_path.name}",
            loc="left",
            fontsize=13,
            pad=14,
        )
        ax.set_xlabel(f"Sample index * {HORIZONTAL_RESOLUTION_DATASET}")
        ax.set_ylabel(f"ADC count * {VERTICAL_RESOLUTION_DATASET}")
        ax.grid(True, alpha=0.25)

        if step == "Show all":
            self._draw_all(ax)
            return
        if step == "Preprocess":
            self._draw_offset_aligned(ax)
            return
        if step == "Differential":
            self._draw_differential(ax)
            return
        if step == "Spectrum":
            self._draw_all(ax)
            return

        ax.text(
            0.5,
            0.5,
            "Pulse plot area",
            transform=ax.transAxes,
            va="center",
            ha="center",
            fontsize=12,
            color="0.35",
        )
