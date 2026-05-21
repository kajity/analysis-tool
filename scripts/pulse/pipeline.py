from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

try:
    from . import analysis as pulse_analysis
    from .config import PulseAnalysisConfig
    from .datasource import PulseDataSource
except ImportError:
    import analysis as pulse_analysis
    from config import PulseAnalysisConfig
    from datasource import PulseDataSource


class PulseStage(StrEnum):
    RAW_VIEW = "Raw View"
    REDUCTION = "Reduction"
    PH = "PH"
    OPTIMAL_FILTER_SIGNAL_FFT = "Optimal Filter Signal FFT"
    OPTIMAL_FILTER_NOISE_FFT = "Optimal Filter Noise FFT"
    OPTIMAL_FILTER_TEMPLATE = "Optimal Filter Template"
    OPTIMAL_FILTER_PULSE_HEIGHT = "Optimal Filter Pulse Height"
    BASELINE_OPTIMAL_FILTER_PULSE_HEIGHT = "Baseline vs Optimal Filter Pulse Height"
    DRIFT_CORRECTED_OPTIMAL_FILTER_PULSE_HEIGHT = (
        "Drift-Corrected Optimal Filter Pulse Height"
    )


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
    pha: np.ndarray
    counts: np.ndarray
    bin_edges: np.ndarray
    accepted_count: int
    rejected_count: int
    normalization: float

    @property
    def pulse_heights(self) -> np.ndarray:
        return self.pha


@dataclass(frozen=True)
class BaselineOptimalFilterHeightResult:
    baseline: np.ndarray
    pha: np.ndarray
    pha_corrected: np.ndarray
    drift_slope: float
    drift_intercept: float
    drift_reference_baseline: float
    drift_correction_enabled: bool
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
        drift_correction_keys = {"baseline_drift_correction"}
        changed = {
            key
            for key, value in self.config.to_dict().items()
            if value != old_config.to_dict()[key]
        }

        if changed & display_keys:
            self._cache.pop("display_aligned", None)
            self._cache.pop(PulseStage.RAW_VIEW.value, None)
            self._cache.pop(PulseStage.REDUCTION.value, None)
        if changed & rejection_keys:
            self._cache.pop(PulseStage.REDUCTION.value, None)
            self._cache.pop(PulseStage.PH.value, None)
            self._cache.pop(OPTIMAL_FILTER_PREP_CACHE_KEY, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_SIGNAL_FFT.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_NOISE_FFT.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_TEMPLATE.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_PULSE_HEIGHT.value, None)
            self._cache.pop(PulseStage.BASELINE_OPTIMAL_FILTER_PULSE_HEIGHT.value, None)
            self._cache.pop(
                PulseStage.DRIFT_CORRECTED_OPTIMAL_FILTER_PULSE_HEIGHT.value,
                None,
            )
        if changed & spectrum_keys:
            self._cache.pop(PulseStage.PH.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_PULSE_HEIGHT.value, None)
            self._cache.pop(
                PulseStage.DRIFT_CORRECTED_OPTIMAL_FILTER_PULSE_HEIGHT.value,
                None,
            )
        if changed & optimal_filter_keys:
            self._cache.pop(OPTIMAL_FILTER_PREP_CACHE_KEY, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_SIGNAL_FFT.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_NOISE_FFT.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_TEMPLATE.value, None)
            self._cache.pop(PulseStage.OPTIMAL_FILTER_PULSE_HEIGHT.value, None)
            self._cache.pop(PulseStage.BASELINE_OPTIMAL_FILTER_PULSE_HEIGHT.value, None)
            self._cache.pop(
                PulseStage.DRIFT_CORRECTED_OPTIMAL_FILTER_PULSE_HEIGHT.value,
                None,
            )
        if changed & drift_correction_keys:
            self._cache.pop(PulseStage.BASELINE_OPTIMAL_FILTER_PULSE_HEIGHT.value, None)
            self._cache.pop(
                PulseStage.DRIFT_CORRECTED_OPTIMAL_FILTER_PULSE_HEIGHT.value,
                None,
            )

    def result_for_stage(self, stage: str) -> Any:
        if stage == PulseStage.RAW_VIEW.value:
            return self.raw_view()
        if stage == PulseStage.REDUCTION.value:
            return self.reduction_view()
        if stage == PulseStage.PH.value:
            return self.spectrum()
        if stage == PulseStage.OPTIMAL_FILTER_SIGNAL_FFT.value:
            return self.optimal_filter_prep()
        if stage == PulseStage.OPTIMAL_FILTER_NOISE_FFT.value:
            return self.optimal_filter_prep()
        if stage == PulseStage.OPTIMAL_FILTER_TEMPLATE.value:
            return self.optimal_filter_prep()
        if stage == PulseStage.OPTIMAL_FILTER_PULSE_HEIGHT.value:
            return self.optimal_filter_pulse_height()
        if stage == PulseStage.BASELINE_OPTIMAL_FILTER_PULSE_HEIGHT.value:
            return self.baseline_optimal_filter_pulse_height()
        if stage == PulseStage.DRIFT_CORRECTED_OPTIMAL_FILTER_PULSE_HEIGHT.value:
            return self.drift_corrected_optimal_filter_pulse_height()
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
            baseline_pha = self.baseline_optimal_filter_pulse_height()
            if baseline_pha.pha.size:
                counts, bin_edges = np.histogram(
                    baseline_pha.pha,
                    bins=self.config.spectrum_bins,
                    range=pulse_analysis.histogram_range(
                        baseline_pha.pha,
                        self.config.histogram_min,
                        self.config.histogram_max,
                    ),
                )
            else:
                counts = np.array([], dtype=int)
                bin_edges = np.array([], dtype=float)

            self._cache[key] = OptimalFilterHeightResult(
                pha=baseline_pha.pha,
                counts=counts,
                bin_edges=bin_edges,
                accepted_count=baseline_pha.accepted_count,
                rejected_count=baseline_pha.rejected_count,
                normalization=baseline_pha.normalization,
            )
        return self._cache[key]

    def drift_corrected_optimal_filter_pulse_height(self) -> OptimalFilterHeightResult:
        key = PulseStage.DRIFT_CORRECTED_OPTIMAL_FILTER_PULSE_HEIGHT.value
        if key not in self._cache:
            baseline_pha = self.baseline_optimal_filter_pulse_height()
            pha = baseline_pha.pha_corrected
            if pha.size:
                counts, bin_edges = np.histogram(
                    pha,
                    bins=self.config.spectrum_bins,
                    range=pulse_analysis.histogram_range(
                        pha,
                        self.config.histogram_min,
                        self.config.histogram_max,
                    ),
                )
            else:
                counts = np.array([], dtype=int)
                bin_edges = np.array([], dtype=float)

            self._cache[key] = OptimalFilterHeightResult(
                pha=pha,
                counts=counts,
                bin_edges=bin_edges,
                accepted_count=baseline_pha.accepted_count,
                rejected_count=baseline_pha.rejected_count,
                normalization=baseline_pha.normalization,
            )
        return self._cache[key]

    def baseline_optimal_filter_pulse_height(
        self,
    ) -> BaselineOptimalFilterHeightResult:
        key = PulseStage.BASELINE_OPTIMAL_FILTER_PULSE_HEIGHT.value
        if key not in self._cache:
            prep = self.optimal_filter_prep()
            baseline_chunks: list[np.ndarray] = []
            pha_chunks: list[np.ndarray] = []
            accepted_count = 0
            rejected_count = 0
            normalization = pulse_analysis.filter_height_normalization(
                prep.template,
                prep.filter_template,
            )

            if prep.filter_template.size and normalization != 0:
                for wave_chunk in self.source.iter_wave_chunks(
                    self.config.spectrum_chunk_size
                ):
                    aligned_signal = self.source.aligned_signal_from_wave(wave_chunk)
                    mask = pulse_analysis.valid_pulse_mask(
                        pulse_analysis.differential_signal(aligned_signal),
                        self.config.valid_pulse_range_start,
                        self.config.valid_pulse_range_stop,
                        self.config.valid_pulse_diff_threshold,
                    )
                    accepted = aligned_signal[mask]
                    accepted_count += int(np.count_nonzero(mask))
                    rejected_count += int(mask.size - np.count_nonzero(mask))
                    if accepted.size == 0:
                        continue

                    background = self.source.background_from_wave(wave_chunk)
                    baseline = (
                        np.average(background, axis=1)[mask]
                        * self.source.vertical_resolution
                    )
                    template_source = (
                        -accepted if self.config.negative_pulses else accepted
                    )
                    pha = (
                        template_source
                        @ prep.filter_template
                        / normalization
                        * self.source.vertical_resolution
                    )
                    baseline_chunks.append(baseline)
                    pha_chunks.append(pha)
            else:
                accepted_count = prep.accepted_count
                rejected_count = prep.rejected_count

            if pha_chunks:
                baseline = np.concatenate(baseline_chunks)
                pha = np.concatenate(pha_chunks)
            else:
                baseline = np.array([], dtype=float)
                pha = np.array([], dtype=float)

            drift = pulse_analysis.baseline_drift_corrected_pulse_heights(
                baseline,
                pha,
                self.config.baseline_drift_correction,
            )
            self._cache[key] = BaselineOptimalFilterHeightResult(
                baseline=baseline,
                pha=pha,
                pha_corrected=drift.pha_corrected,
                drift_slope=drift.slope,
                drift_intercept=drift.intercept,
                drift_reference_baseline=drift.reference_baseline,
                drift_correction_enabled=self.config.baseline_drift_correction,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                normalization=float(normalization),
            )
        return self._cache[key]

    def reduction_view(self) -> RejectionResult:
        key = PulseStage.REDUCTION.value
        if key not in self._cache:
            aligned_signal = self._display_aligned_signal()
            mask = pulse_analysis.valid_pulse_mask(
                pulse_analysis.differential_signal(aligned_signal),
                self.config.valid_pulse_range_start,
                self.config.valid_pulse_range_stop,
                self.config.valid_pulse_diff_threshold,
            )
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
        key = PulseStage.PH.value
        if key not in self._cache:
            pulse_height_chunks: list[np.ndarray] = []
            accepted_count = 0
            rejected_count = 0
            for wave_chunk in self.source.iter_wave_chunks(
                self.config.spectrum_chunk_size
            ):
                aligned_signal = self.source.aligned_signal_from_wave(wave_chunk)
                mask = pulse_analysis.valid_pulse_mask(
                    pulse_analysis.differential_signal(aligned_signal),
                    self.config.valid_pulse_range_start,
                    self.config.valid_pulse_range_stop,
                    self.config.valid_pulse_diff_threshold,
                )
                accepted_count += int(np.count_nonzero(mask))
                rejected_count += int(mask.size - np.count_nonzero(mask))
                pulse_heights = pulse_analysis.pulse_heights(
                    aligned_signal[mask],
                    self.source.vertical_resolution,
                    self.config.negative_pulses,
                )
                if pulse_heights.size:
                    pulse_height_chunks.append(pulse_heights)

            if pulse_height_chunks:
                pulse_heights = np.concatenate(pulse_height_chunks)
                counts, bin_edges = np.histogram(
                    pulse_heights,
                    bins=self.config.spectrum_bins,
                    range=pulse_analysis.histogram_range(
                        pulse_heights,
                        self.config.histogram_min,
                        self.config.histogram_max,
                    ),
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
                mask = pulse_analysis.valid_pulse_mask(
                    pulse_analysis.differential_signal(aligned_signal),
                    self.config.valid_pulse_range_start,
                    self.config.valid_pulse_range_stop,
                    self.config.valid_pulse_diff_threshold,
                )
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

                noise_records = pulse_analysis.noise_records(background)
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
                    filter_template_fft = pulse_analysis.filter_template_fft(
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
            f"PH bins: {config.spectrum_bins}",
            f"histogram range: {self._histogram_range_text()}",
            f"chunk size: {config.spectrum_chunk_size}",
            f"baseline drift correction: {config.baseline_drift_correction}",
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
            | OptimalFilterHeightResult
            | BaselineOptimalFilterHeightResult,
        ):
            lines.extend(
                [
                    f"accepted: {cached.accepted_count}",
                    f"rejected: {cached.rejected_count}",
                ]
            )
            if isinstance(cached, BaselineOptimalFilterHeightResult):
                lines.extend(
                    [
                        f"drift correction: {cached.drift_correction_enabled}",
                        f"drift slope: {cached.drift_slope:g}",
                        f"drift reference baseline: {cached.drift_reference_baseline:g}",
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

    def _sample_traces(self, traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sample_indices = self.source.sample_indices(
            traces.shape[1],
            self.config.max_points_per_trace,
        )
        sample_times = sample_indices * self.source.horizontal_resolution
        return sample_times, traces[:, sample_indices]

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
