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
    OPTIMAL_FILTER_SIGNAL_FFT = "Optimal Filter Signal FFT"
    OPTIMAL_FILTER_NOISE_FFT = "Optimal Filter Noise FFT"
    OPTIMAL_FILTER_TEMPLATE = "Optimal Filter Template"
    OPTIMAL_FILTER_PULSE_HEIGHT = "Optimal Filter Pulse Height"


DEFAULT_STAGES = tuple(stage.value for stage in PulseStage)
OPTIMAL_FILTER_PREP_CACHE_KEY = "Optimal Filter Prep"


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
    rejected_count: int
    template_times: np.ndarray
    template: np.ndarray
    template_frequencies: np.ndarray
    template_fft: np.ndarray
    noise_fft: np.ndarray
    noise_frequencies: np.ndarray
    noise_psd: np.ndarray
    filter_template_times: np.ndarray
    filter_template: np.ndarray
    filter_template_fft: np.ndarray
    status: str


@dataclass(frozen=True)
class OptimalFilterHeightResult:
    pulse_heights: np.ndarray
    counts: np.ndarray
    bin_edges: np.ndarray
    accepted_count: int
    rejected_count: int
    normalization: float


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
        spectrum_keys = {
            "spectrum_bins",
            "histogram_min",
            "histogram_max",
            "spectrum_chunk_size",
        }
        optimal_filter_keys = {
            "spectrum_chunk_size",
            "optimal_filter_template_normalize",
        }
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
            self._cache.pop(OPTIMAL_FILTER_PREP_CACHE_KEY, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_SIGNAL_FFT.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_NOISE_FFT.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_TEMPLATE.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_PULSE_HEIGHT.value, None)
        if changed & spectrum_keys:
            self._cache.pop(PulseStage.SPECTRUM.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_PULSE_HEIGHT.value, None)
        if changed & optimal_filter_keys:
            self._cache.pop(OPTIMAL_FILTER_PREP_CACHE_KEY, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_SIGNAL_FFT.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_NOISE_FFT.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_TEMPLATE.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_PULSE_HEIGHT.value, None)

    def result_for_stage(self, stage: str) -> Any:
        if stage == PulseStage.RAW_VIEW.value:
            return self.raw_view()
        if stage == PulseStage.PREPROCESS.value:
            return self.preprocess_view()
        if stage == PulseStage.REJECT_SHAPING.value:
            return self.rejection_view()
        if stage == PulseStage.SPECTRUM.value:
            return self.spectrum()
        if stage == PulseStage.OPTIMAL_FILTER_SIGNAL_FFT.value:
            return self.optimal_filter_prep()
        if stage == PulseStage.OPTIMAL_FILTER_NOISE_FFT.value:
            return self.optimal_filter_prep()
        if stage == PulseStage.OPTIMAL_FILTER_TEMPLATE.value:
            return self.optimal_filter_prep()
        if stage == PulseStage.OPTIMAL_FILTER_PULSE_HEIGHT.value:
            return self.optimal_filter_pulse_height()
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

    def optimal_filter_pulse_height(self) -> OptimalFilterHeightResult:
        key = PulseStage.OPTIMAL_FILTER_PULSE_HEIGHT.value
        if key not in self._cache:
            prep = self.optimal_filter_prep()
            pulse_height_chunks: list[np.ndarray] = []
            accepted_count = 0
            rejected_count = 0
            normalization = self._filter_height_normalization(prep)

            if prep.filter_template.size and normalization != 0:
                for wave_chunk in self.source.iter_wave_chunks(
                    self.config.spectrum_chunk_size
                ):
                    aligned_signal = self.source.aligned_signal_from_wave(wave_chunk)
                    mask = self._valid_pulse_mask(
                        self._differential_signal(aligned_signal)
                    )
                    accepted = aligned_signal[mask]
                    accepted_count += int(np.count_nonzero(mask))
                    rejected_count += int(mask.size - np.count_nonzero(mask))
                    if accepted.size == 0:
                        continue
                    template_source = (
                        -accepted if self.config.negative_pulses else accepted
                    )
                    pulse_heights = (
                        template_source
                        @ prep.filter_template
                        / normalization
                        * self.source.vertical_resolution
                    )
                    pulse_height_chunks.append(pulse_heights)
            else:
                accepted_count = prep.accepted_count
                rejected_count = prep.rejected_count

            if pulse_height_chunks:
                pulse_heights = np.concatenate(pulse_height_chunks)
                counts, bin_edges = np.histogram(
                    pulse_heights,
                    bins=self.config.spectrum_bins,
                    range=self._histogram_range(pulse_heights),
                )
            else:
                pulse_heights = np.array([], dtype=float)
                counts = np.array([], dtype=int)
                bin_edges = np.array([], dtype=float)

            self._cache[key] = OptimalFilterHeightResult(
                pulse_heights=pulse_heights,
                counts=counts,
                bin_edges=bin_edges,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                normalization=float(normalization),
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
                    range=self._histogram_range(pulse_heights),
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
        key = OPTIMAL_FILTER_PREP_CACHE_KEY
        if key not in self._cache:
            template_sum: np.ndarray | None = None
            noise_power_sum: np.ndarray | None = None
            accepted_count = 0
            rejected_count = 0

            for wave_chunk in self.source.iter_wave_chunks(
                self.config.spectrum_chunk_size
            ):
                aligned_signal = self.source.aligned_signal_from_wave(wave_chunk)
                mask = self._valid_pulse_mask(self._differential_signal(aligned_signal))
                accepted = aligned_signal[mask]
                background = self.source.background_from_wave(wave_chunk)[mask]
                accepted_count += int(np.count_nonzero(mask))
                rejected_count += int(mask.size - np.count_nonzero(mask))
                if accepted.size == 0:
                    continue

                template_source = -accepted if self.config.negative_pulses else accepted
                chunk_template_sum = np.sum(template_source, axis=0)
                if template_sum is None:
                    template_sum = chunk_template_sum
                else:
                    template_sum += chunk_template_sum

                noise_records = self._noise_records(background)
                if noise_records.size == 0:
                    continue
                noise_fft = np.fft.rfft(noise_records, axis=1)
                chunk_noise_power = np.sum(
                    np.abs(noise_fft) ** 2 / noise_records.shape[1],
                    axis=0,
                )
                if noise_power_sum is None:
                    noise_power_sum = chunk_noise_power
                else:
                    noise_power_sum += chunk_noise_power

            if accepted_count == 0 or template_sum is None:
                template = np.array([], dtype=float)
                template_frequencies = np.array([], dtype=float)
                template_fft = np.array([], dtype=complex)
                noise_fft = np.array([], dtype=float)
                noise_frequencies = np.array([], dtype=float)
                noise_psd = np.array([], dtype=float)
                filter_template_times = np.array([], dtype=float)
                filter_template = np.array([], dtype=float)
                filter_template_fft = np.array([], dtype=complex)
                status = "No accepted pulses are available for optimal filter prep."
            else:
                template = template_sum / accepted_count
                if self.config.optimal_filter_template_normalize:
                    peak = float(np.max(np.abs(template)))
                    if peak > 0:
                        template = template / peak
                template_frequencies = np.fft.rfftfreq(
                    template.size,
                    d=self.source.horizontal_resolution,
                )
                template_fft = np.fft.rfft(template)

                if noise_power_sum is None:
                    noise_frequencies = np.array([], dtype=float)
                    noise_psd = np.array([], dtype=float)
                    noise_fft = np.array([], dtype=float)
                    filter_template_fft = np.array([], dtype=complex)
                    filter_template = np.array([], dtype=float)
                else:
                    noise_psd = noise_power_sum / accepted_count
                    noise_fft = np.sqrt(noise_psd)
                    noise_frequencies = np.fft.rfftfreq(
                        self.source.signal_start,
                        d=self.source.horizontal_resolution,
                    )
                    filter_template_fft = self._filter_template_fft(
                        template_fft,
                        noise_psd,
                    )
                    filter_template = np.fft.irfft(
                        filter_template_fft,
                        n=template.size,
                    )
                    peak = float(np.max(np.abs(filter_template)))
                    if peak > 0:
                        filter_template = filter_template / peak
                status = (
                    "Template and noise PSD are estimated. Optimal filter "
                    "application is not implemented yet."
                )

            template_times = (
                np.arange(template.size) * self.source.horizontal_resolution
                if template.size
                else np.array([], dtype=float)
            )
            filter_template_times = (
                np.arange(filter_template.size) * self.source.horizontal_resolution
                if filter_template.size
                else np.array([], dtype=float)
            )
            self._cache[key] = OptimalFilterPrepResult(
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                template_times=template_times,
                template=template,
                template_frequencies=template_frequencies,
                template_fft=template_fft,
                noise_fft=noise_fft,
                noise_frequencies=noise_frequencies,
                noise_psd=noise_psd,
                filter_template_times=filter_template_times,
                filter_template=filter_template,
                filter_template_fft=filter_template_fft,
                status=status,
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
            f"histogram range: {self._histogram_range_text()}",
            f"chunk size: {config.spectrum_chunk_size}",
        ]
        if stage in {
            PulseStage.OPTIMAL_FILTER_SIGNAL_FFT.value,
            PulseStage.OPTIMAL_FILTER_NOISE_FFT.value,
            PulseStage.OPTIMAL_FILTER_TEMPLATE.value,
        }:
            cached = self._cache.get(OPTIMAL_FILTER_PREP_CACHE_KEY)
        else:
            cached = self._cache.get(stage)
        if isinstance(
            cached,
            RejectionResult
            | SpectrumResult
            | OptimalFilterPrepResult
            | OptimalFilterHeightResult,
        ):
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

    def _noise_records(self, background_signal: np.ndarray) -> np.ndarray:
        if background_signal.size == 0:
            return np.array([], dtype=float)
        return background_signal - np.average(background_signal, axis=1, keepdims=True)

    def _filter_template_fft(
        self,
        template_fft: np.ndarray,
        noise_psd: np.ndarray,
    ) -> np.ndarray:
        usable_bins = min(template_fft.size, noise_psd.size)
        if usable_bins == 0:
            return np.array([], dtype=complex)

        result = np.zeros_like(template_fft)
        noise_power = noise_psd[:usable_bins]
        positive = noise_power > 0
        # Noise records are mean-subtracted, so the DC bin has no useful variance
        # estimate and would otherwise dominate signal/noise^2.
        positive[0] = False
        usable_result = result[:usable_bins]
        usable_result[positive] = (
            template_fft[:usable_bins][positive] / noise_power[positive]
        )
        return result

    def _filter_height_normalization(self, prep: OptimalFilterPrepResult) -> float:
        if prep.template.size == 0 or prep.filter_template.size == 0:
            return 0.0
        usable_samples = min(prep.template.size, prep.filter_template.size)
        return float(
            prep.template[:usable_samples] @ prep.filter_template[:usable_samples]
        )

    def _histogram_range(self, values: np.ndarray) -> tuple[float, float] | None:
        if self.config.histogram_min is None and self.config.histogram_max is None:
            return None
        if values.size == 0:
            return None
        lower = (
            float(np.min(values))
            if self.config.histogram_min is None
            else self.config.histogram_min
        )
        upper = (
            float(np.max(values))
            if self.config.histogram_max is None
            else self.config.histogram_max
        )
        if upper <= lower:
            raise ValueError("histogram range must have max greater than min.")
        return lower, upper

    def _configured_histogram_range(self) -> tuple[float | None, float | None] | None:
        if self.config.histogram_min is None or self.config.histogram_max is None:
            if self.config.histogram_min is None and self.config.histogram_max is None:
                return None
            return self.config.histogram_min, self.config.histogram_max
        return self.config.histogram_min, self.config.histogram_max

    def _histogram_range_text(self) -> str:
        histogram_range = self._configured_histogram_range()
        if histogram_range is None:
            return "auto"
        lower = "auto" if histogram_range[0] is None else f"{histogram_range[0]:g}"
        upper = "auto" if histogram_range[1] is None else f"{histogram_range[1]:g}"
        return f"{lower}:{upper}"
