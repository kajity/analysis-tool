from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

try:
    from .config import PulseAnalysisConfig
    from .datasource import PulseDataSource
except ImportError:
    from config import PulseAnalysisConfig
    from datasource import PulseDataSource


class PulseStage(StrEnum):
    RAW_VIEW = "Raw View"
    REJECT_SHAPING = "Reject/Shaping"
    PREPROCESS = "Preprocess"
    SPECTRUM = "Spectrum"
    OPTIMAL_FILTER_PREP = "Optimal Filter Prep"


DEFAULT_STAGES = tuple(stage.value for stage in PulseStage)


@dataclass(frozen=True)
class TracePlotResult:
    sample_times: np.ndarray
    traces: np.ndarray
    vertical_resolution: float
    ylabel: str


@dataclass(frozen=True)
class RejectionResult:
    sample_times: np.ndarray
    shaped_traces: np.ndarray
    accepted_count: int
    rejected_count: int
    vertical_resolution: float


@dataclass(frozen=True)
class SpectrumResult:
    pulse_heights: np.ndarray
    counts: np.ndarray
    bin_edges: np.ndarray
    accepted_count: int
    rejected_count: int


@dataclass(frozen=True)
class OptimalFilterPrepResult:
    accepted_count: int
    status: str


class PulsePipeline:
    def __init__(self, source: PulseDataSource, config: PulseAnalysisConfig) -> None:
        self.source = source
        self.config = config.validated()
        self._cache: dict[str, Any] = {}

    def update_config(self, config: PulseAnalysisConfig) -> None:
        old_config = self.config
        self.config = config.validated()
        if self.config == old_config:
            return

        display_keys = {"max_points_per_trace", "max_display_traces"}
        rejection_keys = {
            "valid_pulse_range_start",
            "valid_pulse_range_stop",
            "valid_pulse_diff_threshold",
            "negative_pulses",
        }
        spectrum_keys = {"spectrum_bins", "spectrum_chunk_size"}
        changed = {
            key
            for key, value in self.config.to_dict().items()
            if value != old_config.to_dict()[key]
        }

        if changed & display_keys:
            self._cache.pop("display_aligned", None)
            self._cache.pop(PulseStage.RAW_VIEW.value, None)
            self._cache.pop(PulseStage.PREPROCESS.value, None)
            self._cache.pop(PulseStage.REJECT_SHAPING.value, None)
        if changed & rejection_keys:
            self._cache.pop(PulseStage.REJECT_SHAPING.value, None)
            self._cache.pop(PulseStage.SPECTRUM.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_PREP.value, None)
        if changed & spectrum_keys:
            self._cache.pop(PulseStage.SPECTRUM.value, None)

    def result_for_stage(self, stage: str) -> Any:
        if stage == PulseStage.RAW_VIEW.value:
            return self.raw_view()
        if stage == PulseStage.PREPROCESS.value:
            return self.preprocess_view()
        if stage == PulseStage.REJECT_SHAPING.value:
            return self.rejection_view()
        if stage == PulseStage.SPECTRUM.value:
            return self.spectrum()
        if stage == PulseStage.OPTIMAL_FILTER_PREP.value:
            return self.optimal_filter_prep()
        raise ValueError(f"Unknown pulse stage: {stage}")

    def raw_view(self) -> TracePlotResult:
        key = PulseStage.RAW_VIEW.value
        if key not in self._cache:
            sample_times, traces = self._sample_traces(self._display_aligned_signal())
            self._cache[key] = TracePlotResult(
                sample_times=sample_times,
                traces=traces,
                vertical_resolution=self.source.vertical_resolution,
                ylabel="ADC count * /waveform/vres",
            )
        return self._cache[key]

    def preprocess_view(self) -> RejectionResult:
        key = PulseStage.PREPROCESS.value
        if key not in self._cache:
            self._cache[key] = self.rejection_view()
        return self._cache[key]

    def rejection_view(self) -> RejectionResult:
        key = PulseStage.REJECT_SHAPING.value
        if key not in self._cache:
            aligned_signal = self._display_aligned_signal()
            mask = self._valid_pulse_mask(self._differential_signal(aligned_signal))
            sample_times, shaped_traces = self._sample_traces(aligned_signal[mask])
            self._cache[key] = RejectionResult(
                sample_times=sample_times,
                shaped_traces=shaped_traces,
                accepted_count=int(np.count_nonzero(mask)),
                rejected_count=int(mask.size - np.count_nonzero(mask)),
                vertical_resolution=self.source.vertical_resolution,
            )
        return self._cache[key]

    def spectrum(self) -> SpectrumResult:
        key = PulseStage.SPECTRUM.value
        if key not in self._cache:
            pulse_height_chunks: list[np.ndarray] = []
            accepted_count = 0
            rejected_count = 0
            for wave_chunk in self.source.iter_wave_chunks(
                self.config.spectrum_chunk_size
            ):
                aligned_signal = self.source.aligned_signal_from_wave(wave_chunk)
                mask = self._valid_pulse_mask(self._differential_signal(aligned_signal))
                accepted_count += int(np.count_nonzero(mask))
                rejected_count += int(mask.size - np.count_nonzero(mask))
                pulse_heights = self._pulse_heights(aligned_signal[mask])
                if pulse_heights.size:
                    pulse_height_chunks.append(pulse_heights)

            if pulse_height_chunks:
                pulse_heights = np.concatenate(pulse_height_chunks)
                counts, bin_edges = np.histogram(
                    pulse_heights,
                    bins=self.config.spectrum_bins,
                )
            else:
                pulse_heights = np.array([], dtype=float)
                counts = np.array([], dtype=int)
                bin_edges = np.array([], dtype=float)

            self._cache[key] = SpectrumResult(
                pulse_heights=pulse_heights,
                counts=counts,
                bin_edges=bin_edges,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
            )
        return self._cache[key]

    def optimal_filter_prep(self) -> OptimalFilterPrepResult:
        key = PulseStage.OPTIMAL_FILTER_PREP.value
        if key not in self._cache:
            spectrum = self.spectrum()
            self._cache[key] = OptimalFilterPrepResult(
                accepted_count=spectrum.accepted_count,
                status=(
                    "Optimal filter is not implemented yet. Accepted pulses are "
                    "available for future template and noise PSD estimation."
                ),
            )
        return self._cache[key]

    def status_text(self, stage: str) -> str:
        config = self.config
        lines = [
            f"stage: {stage}",
            f"traces: {self.source.trace_count}",
            f"samples/trace: {self.source.sample_count}",
            f"valid range: {config.valid_pulse_range_start}:{config.valid_pulse_range_stop}",
            f"diff threshold: {config.valid_pulse_diff_threshold:g}",
            f"spectrum bins: {config.spectrum_bins}",
            f"chunk size: {config.spectrum_chunk_size}",
        ]
        cached = self._cache.get(stage)
        if isinstance(cached, RejectionResult | SpectrumResult):
            lines.extend(
                [
                    f"accepted: {cached.accepted_count}",
                    f"rejected: {cached.rejected_count}",
                ]
            )
        return "\n".join(lines)

    def _display_aligned_signal(self) -> np.ndarray:
        if "display_aligned" not in self._cache:
            row_indices = self.source.display_trace_indices(
                self.config.max_display_traces
            )
            wave = self.source.read_wave_rows(row_indices)
            self._cache["display_aligned"] = self.source.aligned_signal_from_wave(wave)
        return self._cache["display_aligned"]

    def _differential_signal(self, aligned_signal: np.ndarray) -> np.ndarray:
        return np.diff(aligned_signal, axis=1)

    def _sample_traces(self, traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sample_indices = self.source.sample_indices(
            traces.shape[1],
            self.config.max_points_per_trace,
        )
        sample_times = sample_indices * self.source.horizontal_resolution
        return sample_times, traces[:, sample_indices]

    def _valid_pulse_mask(self, differential_signal: np.ndarray) -> np.ndarray:
        start = self.config.valid_pulse_range_start
        stop = self.config.valid_pulse_range_stop
        diff_count = differential_signal.shape[1]
        outside_valid_range = np.ones(diff_count, dtype=bool)
        outside_valid_range[min(start, diff_count) : min(stop, diff_count)] = False
        if not np.any(outside_valid_range):
            return np.ones(differential_signal.shape[0], dtype=bool)

        outside_diff = np.abs(differential_signal[:, outside_valid_range])
        return np.all(
            outside_diff <= self.config.valid_pulse_diff_threshold,
            axis=1,
        )

    def _pulse_heights(self, shaped_signal: np.ndarray) -> np.ndarray:
        if shaped_signal.size == 0:
            return np.array([], dtype=float)
        heights = np.min(shaped_signal, axis=1) * self.source.vertical_resolution
        if self.config.negative_pulses:
            heights *= -1
        return heights
