from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes

try:
    from .datasource import HORIZONTAL_RESOLUTION_DATASET, VERTICAL_RESOLUTION_DATASET
    from .pipeline import (
        OptimalFilterHeightResult,
        OptimalFilterPrepResult,
        PulsePipeline,
        PulseStage,
        RejectionResult,
        SpectrumResult,
        TracePlotResult,
    )
except ImportError:
    from datasource import HORIZONTAL_RESOLUTION_DATASET, VERTICAL_RESOLUTION_DATASET
    from pipeline import (
        OptimalFilterHeightResult,
        OptimalFilterPrepResult,
        PulsePipeline,
        PulseStage,
        RejectionResult,
        SpectrumResult,
        TracePlotResult,
    )


class PulsePlotRenderer:
    def __init__(self, pipeline: PulsePipeline, input_path: Path) -> None:
        self.pipeline = pipeline
        self.input_path = input_path

    def draw_plot(self, ax: Axes, stage: str) -> None:
        ax.set_title(
            f"{stage}: {self.input_path.name}",
            loc="left",
            fontsize=13,
            pad=14,
        )
        ax.grid(True, alpha=0.25)

        result = self.pipeline.result_for_stage(stage)
        if isinstance(result, TracePlotResult):
            self._draw_traces(ax, result)
        elif isinstance(result, RejectionResult):
            self._draw_rejection(ax, result)
        elif isinstance(result, SpectrumResult):
            self._draw_spectrum(ax, result)
        elif isinstance(result, OptimalFilterHeightResult):
            self._draw_optimal_filter_height(ax, result)
        elif isinstance(result, OptimalFilterPrepResult):
            self._draw_optimal_filter(ax, stage, result)
        else:
            raise TypeError(f"Unsupported pulse plot result: {type(result).__name__}")

    def _draw_traces(self, ax: Axes, result: TracePlotResult) -> None:
        ax.set_xlabel(f"Sample index * {HORIZONTAL_RESOLUTION_DATASET}")
        ax.set_ylabel(result.ylabel)
        self._draw_trace_collection(
            ax,
            result.sample_times,
            result.traces * result.vertical_resolution,
        )

    def _draw_rejection(self, ax: Axes, result: RejectionResult) -> None:
        ax.set_xlabel(f"Sample index * {HORIZONTAL_RESOLUTION_DATASET}")
        ax.set_ylabel(f"ADC count * {VERTICAL_RESOLUTION_DATASET}")
        self._draw_trace_collection(
            ax,
            result.sample_times,
            result.shaped_traces * result.vertical_resolution,
        )
        ax.text(
            0.02,
            0.98,
            f"display accepted={result.accepted_count}, rejected={result.rejected_count}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="0.25",
        )

    def _draw_trace_collection(
        self,
        ax: Axes,
        sample_times,
        traces,
    ) -> None:
        if traces.size == 0:
            return
        ax.plot(sample_times, traces.T, alpha=0.25)

    def _draw_spectrum(self, ax: Axes, result: SpectrumResult) -> None:
        ax.set_xlabel(f"Pulse height (ADC count * {VERTICAL_RESOLUTION_DATASET})")
        ax.set_ylabel("Counts")
        if result.counts.size == 0:
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
            result.counts,
            result.bin_edges,
            fill=False,
            alpha=0.4,
            linewidth=1.5,
        )
        ax.set_ylim(bottom=0)
        ax.text(
            0.02,
            0.98,
            f"accepted={result.accepted_count}, rejected={result.rejected_count}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="0.25",
        )

    def _draw_optimal_filter(
        self,
        ax: Axes,
        stage: str,
        result: OptimalFilterPrepResult,
    ) -> None:
        if stage == PulseStage.OPTIMAL_FILTER_SIGNAL_FFT.value:
            self._draw_optimal_filter_signal_fft(ax, result)
            return
        if stage == PulseStage.OPTIMAL_FILTER_NOISE_FFT.value:
            self._draw_optimal_filter_noise_fft(ax, result)
            return
        if stage == PulseStage.OPTIMAL_FILTER_TEMPLATE.value:
            self._draw_optimal_filter_template(ax, result)
            return
        raise ValueError(f"Unknown optimal filter stage: {stage}")

    def _draw_optimal_filter_signal_fft(
        self,
        ax: Axes,
        result: OptimalFilterPrepResult,
    ) -> None:
        ax.set_xlabel("Frequency")
        ax.set_ylabel("|signal FFT|")
        if result.template_fft.size == 0:
            self._draw_empty_optimal_filter(ax)
            return
        ax.plot(result.template_frequencies, abs(result.template_fft))
        ax.set_yscale("log")

    def _draw_optimal_filter_noise_fft(
        self,
        ax: Axes,
        result: OptimalFilterPrepResult,
    ) -> None:
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Noise FFT magnitude")
        if result.noise_fft.size == 0:
            self._draw_empty_optimal_filter(ax)
            return
        ax.plot(result.noise_frequencies, result.noise_fft)
        ax.set_yscale("log")

    def _draw_optimal_filter_template(
        self,
        ax: Axes,
        result: OptimalFilterPrepResult,
    ) -> None:
        ax.set_xlabel(f"Sample index * {HORIZONTAL_RESOLUTION_DATASET}")
        ax.set_ylabel("Normalized irfft(signal FFT / noise FFT^2)")
        if result.filter_template.size == 0:
            self._draw_empty_optimal_filter(ax)
            return
        ax.plot(result.filter_template_times, result.filter_template)

    def _draw_optimal_filter_height(
        self,
        ax: Axes,
        result: OptimalFilterHeightResult,
    ) -> None:
        ax.set_xlabel(
            f"Optimized pulse height (ADC count * {VERTICAL_RESOLUTION_DATASET})"
        )
        ax.set_ylabel("Counts")
        if result.counts.size == 0:
            ax.set_ylim(bottom=0)
            self._draw_empty_optimal_filter(ax)
            return
        ax.stairs(
            result.counts,
            result.bin_edges,
            fill=False,
            alpha=0.4,
            linewidth=1.5,
        )
        ax.set_ylim(bottom=0)
        ax.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"accepted={result.accepted_count}, rejected={result.rejected_count}",
                    f"normalization={result.normalization:g}",
                ]
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="0.25",
        )

    def _draw_empty_optimal_filter(self, ax: Axes) -> None:
        ax.text(
            0.5,
            0.5,
            "No optimal filter prep data available",
            transform=ax.transAxes,
            va="center",
            ha="center",
            fontsize=12,
            color="0.35",
        )

    def _optimal_filter_text(self, result: OptimalFilterPrepResult) -> str:
        noise_bins = result.noise_psd.size
        template_bins = result.template_fft.size
        filter_template_bins = result.filter_template.size
        return "\n".join(
            [
                result.status,
                f"accepted: {result.accepted_count}",
                f"rejected: {result.rejected_count}",
                "noise source: background",
                f"template FFT bins: {template_bins}",
                f"noise PSD bins: {noise_bins}",
                f"filter template samples: {filter_template_bins}",
            ]
        )
