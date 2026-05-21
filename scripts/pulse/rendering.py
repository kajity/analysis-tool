from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.widgets import RectangleSelector

try:
    from .datasource import HORIZONTAL_RESOLUTION_DATASET, VERTICAL_RESOLUTION_DATASET
    from .pipeline import (
        BaselineOptimalFilterHeightResult,
        OptimalFilterHeightResult,
        OptimalFilterPrepResult,
        PhaTimelineResult,
        PulsePipeline,
        PulseStage,
        RejectionResult,
        SpectrumResult,
        TracePlotResult,
    )
except ImportError:
    from datasource import HORIZONTAL_RESOLUTION_DATASET, VERTICAL_RESOLUTION_DATASET
    from pipeline import (
        BaselineOptimalFilterHeightResult,
        OptimalFilterHeightResult,
        OptimalFilterPrepResult,
        PhaTimelineResult,
        PulsePipeline,
        PulseStage,
        RejectionResult,
        SpectrumResult,
        TracePlotResult,
    )


BaselineDriftRangeCallback = Callable[[float, float, float, float], None]


class PulsePlotRenderer:
    def __init__(
        self,
        pipeline: PulsePipeline,
        input_path: Path,
        baseline_drift_range_callback: BaselineDriftRangeCallback | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.input_path = input_path
        self.baseline_drift_range_callback = baseline_drift_range_callback
        self._baseline_selector: RectangleSelector | None = None

    def draw_plot(self, ax: Axes, stage: str) -> None:
        ax.set_title(
            f"{stage}: {self.input_path.name}",
            loc="left",
            fontsize=13,
            pad=14,
        )
        ax.grid(True, alpha=0.25)
        self._baseline_selector = None

        result = self.pipeline.result_for_stage(stage)
        if isinstance(result, TracePlotResult):
            self._draw_traces(ax, result)
        elif isinstance(result, RejectionResult):
            self._draw_rejection(ax, result)
        elif isinstance(result, SpectrumResult):
            self._draw_spectrum(ax, result)
        elif isinstance(result, OptimalFilterHeightResult):
            self._draw_optimal_filter_height(ax, result)
        elif isinstance(result, PhaTimelineResult):
            self._draw_pha_timeline(ax, result)
        elif isinstance(result, BaselineOptimalFilterHeightResult):
            self._draw_baseline_optimal_filter_height(ax, result)
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
            linewidth=1.5,
        )
        ax.set_ylim(bottom=0)

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
            linewidth=1.5,
        )
        ax.set_ylim(bottom=0)

    def _draw_pha_timeline(
        self,
        ax: Axes,
        result: PhaTimelineResult,
    ) -> None:
        ax.set_xlabel("Accepted pulse index")
        ax.set_ylabel(
            f"Optimized pulse height (ADC count * {VERTICAL_RESOLUTION_DATASET})"
        )
        if result.pha.size == 0:
            self._draw_empty_optimal_filter(ax)
            return
        ax.scatter(
            result.pulse_indices,
            result.pha,
            s=10,
            alpha=0.85,
            linewidths=0,
        )

    def _draw_baseline_optimal_filter_height(
        self,
        ax: Axes,
        result: BaselineOptimalFilterHeightResult,
    ) -> None:
        ax.set_xlabel(f"Baseline average (ADC count * {VERTICAL_RESOLUTION_DATASET})")
        ax.set_ylabel(
            f"Optimized pulse height (ADC count * {VERTICAL_RESOLUTION_DATASET})"
        )
        if result.pha.size == 0:
            self._draw_empty_optimal_filter(ax)
            return
        ax.scatter(
            result.baseline,
            result.pha,
            s=6,
            alpha=0.35,
            linewidths=0,
            label="raw",
        )
        finite = np.isfinite(result.baseline) & np.isfinite(result.pha)
        drift = result.drift
        if drift is not None:
            fit_mask = drift.fit_mask & finite
            if np.any(fit_mask):
                ax.scatter(
                    result.baseline[fit_mask],
                    result.pha[fit_mask],
                    s=12,
                    alpha=0.75,
                    linewidths=0,
                    color="tab:orange",
                    label="drift fit",
                )
            if (
                drift.fit_count >= 2
                and np.count_nonzero(fit_mask) >= 2
                and np.ptp(result.baseline[fit_mask]) > 0
            ):
                x_values = np.array(
                    [
                        np.min(result.baseline[fit_mask]),
                        np.max(result.baseline[fit_mask]),
                    ]
                )
                y_values = drift.slope * x_values + drift.intercept
                ax.plot(
                    x_values,
                    y_values,
                    color="tab:red",
                    linewidth=1.2,
                    label=f"linear fit slope={drift.slope:.3g}",
                )
            ax.legend(loc="best", framealpha=0.9, fontsize="small")
            if self.baseline_drift_range_callback is not None:
                self._baseline_selector = RectangleSelector(
                    ax,
                    self._on_baseline_range_selected,
                    useblit=True,
                    button=[1],
                    minspanx=0,
                    minspany=0,
                    interactive=True,
                    props={"facecolor": "tab:orange", "alpha": 0.15},
                )

    def _on_baseline_range_selected(self, eclick, erelease) -> None:
        if self.baseline_drift_range_callback is None:
            return
        if None in {eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata}:
            return
        baseline_min, baseline_max = sorted(
            (float(eclick.xdata), float(erelease.xdata))
        )
        pha_min, pha_max = sorted((float(eclick.ydata), float(erelease.ydata)))
        if baseline_min == baseline_max or pha_min == pha_max:
            return
        self.baseline_drift_range_callback(
            baseline_min,
            baseline_max,
            pha_min,
            pha_max,
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
