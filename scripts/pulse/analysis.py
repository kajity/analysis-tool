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
SPECTRUM_BINS = "auto"
SPECTRUM_CHUNK_SIZE = 512


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
        self.wave_dataset = self._dataset(WAVE_DATASET)
        self._validate_dataset(self.wave_dataset)
        self.trace_count = int(self.wave_dataset.shape[0])
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
        points = int(self.wave_dataset.shape[-1])
        self.signal_start = points // 2
        self.background_data: np.ndarray = self.wave_dataset[
            self.trace_indices, : self.signal_start
        ]
        self.signal_data: np.ndarray = self.wave_dataset[
            self.trace_indices, self.signal_start :
        ]
        self.offset: np.ndarray = np.average(self.background_data, axis=1)

        self.valid_pulse_range = (320, 330)
        self.valid_pulse_diff_threshold = 1000

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

    def _aligned_signal(self) -> np.ndarray:
        return self.signal_data - self.offset[:, np.newaxis]

    def _aligned_signal_chunk(self, wave_chunk: np.ndarray) -> np.ndarray:
        background_data = wave_chunk[:, : self.signal_start]
        signal_data = wave_chunk[:, self.signal_start :]
        offset = np.average(background_data, axis=1)
        return signal_data - offset[:, np.newaxis]

    def _differential_signal(self, aligned_signal: np.ndarray) -> np.ndarray:
        return np.diff(aligned_signal, axis=1)

    def _sample_traces(self, traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        signal_data_length = traces.shape[1]
        sample_indices = self._sample_indices(signal_data_length)
        sample_times = sample_indices * self.horizontal_resolution
        return sample_times, traces[:, sample_indices]

    def _valid_pulse_mask(self, differential_signal: np.ndarray) -> np.ndarray:
        start, stop = self.valid_pulse_range
        if start < 0 or stop < start:
            raise ValueError(
                "valid_pulse_range must be a non-negative (start, stop) pair."
            )

        diff_count = differential_signal.shape[1]
        outside_valid_range = np.ones(diff_count, dtype=bool)
        outside_valid_range[min(start, diff_count) : min(stop, diff_count)] = False
        if not np.any(outside_valid_range):
            return np.ones(differential_signal.shape[0], dtype=bool)

        # A pulse is valid only when every diff outside valid_pulse_range stays
        # within the allowed threshold.
        outside_diff = np.abs(differential_signal[:, outside_valid_range])
        return np.all(outside_diff <= self.valid_pulse_diff_threshold, axis=1)

    def _shaped_signal(self) -> np.ndarray:
        aligned_signal = self._aligned_signal()
        differential_signal = np.diff(aligned_signal, axis=1)
        valid_pulse_mask = self._valid_pulse_mask(differential_signal)
        return aligned_signal[valid_pulse_mask]

    def _pulse_heights(self, shaped_signal: np.ndarray | None = None) -> np.ndarray:
        if shaped_signal is None:
            shaped_signal = self._shaped_signal()
        if shaped_signal.size == 0:
            return np.array([], dtype=float)
        return np.min(shaped_signal, axis=1) * self.vertical_resolution * -1

    def _spectrum_pulse_heights(self) -> np.ndarray:
        pulse_height_chunks: list[np.ndarray] = []
        for start in range(0, self.trace_count, SPECTRUM_CHUNK_SIZE):
            stop = min(start + SPECTRUM_CHUNK_SIZE, self.trace_count)
            wave_chunk = np.asarray(self.wave_dataset[start:stop])
            aligned_signal = self._aligned_signal_chunk(wave_chunk)
            differential_signal = self._differential_signal(aligned_signal)
            shaped_signal = aligned_signal[self._valid_pulse_mask(differential_signal)]
            pulse_heights = self._pulse_heights(shaped_signal)
            if pulse_heights.size:
                pulse_height_chunks.append(pulse_heights)

        if not pulse_height_chunks:
            return np.array([], dtype=float)
        return np.concatenate(pulse_height_chunks)

    def _pulse_height_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        pulse_heights = self._spectrum_pulse_heights()
        if pulse_heights.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        counts, bin_edges = np.histogram(pulse_heights, bins=SPECTRUM_BINS)
        return counts, bin_edges

    def _draw_spectrum_plot(self, ax: Axes) -> None:
        counts, bin_edges = self._pulse_height_spectrum()
        ax.set_xlabel(
            f"Pulse height minimum (ADC count * {VERTICAL_RESOLUTION_DATASET})"
        )
        ax.set_ylabel("Counts")

        if counts.size == 0:
            ax.set_ylim(bottom=0)
            ax.text(
                0.5,
                0.5,
                "No shaped pulses available",
                transform=ax.transAxes,
                va="center",
                ha="center",
                fontsize=12,
                color="0.35",
            )
            return

        ax.stairs(
            counts,
            bin_edges,
            fill=True,
            alpha=0.4,
            linewidth=1.5,
        )
        ax.set_ylim(bottom=0)

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
            sample_times, traces = self._sample_traces(self._aligned_signal())
        elif step == "Differential":
            sample_times, traces = self._sample_traces(
                self._differential_signal(self._aligned_signal())
            )
        elif step == "Shaped":
            sample_times, traces = self._sample_traces(self._shaped_signal())
        elif step == "Spectrum":
            self._draw_spectrum_plot(ax)
            return
        else:
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
            return

        for trace in traces:
            ax.plot(
                sample_times,
                trace * self.vertical_resolution,
                alpha=0.25,
            )


try:
    from .config import PulseAnalysisConfig, default_config
    from .datasource import PulseDataSource
    from .pipeline import PulsePipeline
    from .rendering import PulsePlotRenderer
except ImportError:
    from config import PulseAnalysisConfig, default_config
    from datasource import PulseDataSource
    from pipeline import PulsePipeline
    from rendering import PulsePlotRenderer


class PulseAnalyzer:  # type: ignore[no-redef]
    """Compatibility wrapper around the pipeline-based pulse analysis stack."""

    def __init__(
        self,
        pulse_data: Hdf5PulseData,
        max_points_per_trace: int | None = MAX_POINTS_PER_TRACE,
        max_traces: int | None = MAX_TRACES_PER_DATASET,
        config: PulseAnalysisConfig | None = None,
    ) -> None:
        self.pulse_data = pulse_data
        self.input_path = pulse_data.file_path
        base_config = config or default_config()
        self.config = base_config.with_updates(
            max_points_per_trace=max_points_per_trace,
            max_display_traces=max_traces,
        )
        self.source = PulseDataSource(pulse_data)
        self.pipeline = PulsePipeline(self.source, self.config)
        self.renderer = PulsePlotRenderer(self.pipeline, self.input_path)
        self.vertical_resolution = self.source.vertical_resolution
        self.horizontal_resolution = self.source.horizontal_resolution
        self.wave_dataset = self.source.wave_dataset
        self.trace_count = self.source.trace_count
        self.signal_start = self.source.signal_start
        self.valid_pulse_range = (
            self.config.valid_pulse_range_start,
            self.config.valid_pulse_range_stop,
        )
        self.valid_pulse_diff_threshold = self.config.valid_pulse_diff_threshold

    def _shaped_signal(self) -> np.ndarray:
        return self.pipeline.rejection_view().shaped_traces

    def _pulse_heights(self, shaped_signal: np.ndarray | None = None) -> np.ndarray:
        if shaped_signal is None:
            return self.pipeline.spectrum().pulse_heights
        return self.pipeline._pulse_heights(shaped_signal)

    def _spectrum_pulse_heights(self) -> np.ndarray:
        return self.pipeline.spectrum().pulse_heights

    def _pulse_height_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        spectrum = self.pipeline.spectrum()
        return spectrum.counts, spectrum.bin_edges

    def draw_plot(self, ax: Axes, step: str) -> None:
        self.renderer.draw_plot(ax, step)
