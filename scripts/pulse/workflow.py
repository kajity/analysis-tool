from __future__ import annotations

import csv
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, cast

import numpy as np

try:
    from .config import (
        PulseAnalysisConfig,
        default_config,
        parse_config_updates,
        save_config,
    )
    from .datasource import PulseDataSource
    from .pipeline import DEFAULT_STAGES, PulsePipeline, PulseStage
    from .pulse_io import Hdf5PulseData, format_hdf5_summary
    from .rendering import PulsePlotRenderer
    from .ui import PulseViewState, PulseWizardUI
except ImportError:
    from config import (
        PulseAnalysisConfig,
        default_config,
        parse_config_updates,
        save_config,
    )
    from datasource import PulseDataSource
    from pipeline import DEFAULT_STAGES, PulsePipeline, PulseStage
    from pulse_io import Hdf5PulseData, format_hdf5_summary
    from rendering import PulsePlotRenderer
    from ui import PulseViewState, PulseWizardUI

MAX_POINTS_PER_TRACE = 1000
MAX_TRACES_PER_DATASET = 20
DEFAULT_STEPS = DEFAULT_STAGES
ArrayOutputFormat = Literal["npy", "csv"]
SaveProgressCallback = Callable[[int, int, str, Path], None]
STEP_INFO_TEXT = {
    "Raw View": (
        "Raw View\n\n"
        "Baseline-subtracted waveforms are plotted from /waveform/wave using "
        "/waveform/vres and /waveform/hres."
    ),
    "Reduction": (
        "Reduction\n\n"
        "A trace is accepted when every diff outside the valid pulse range stays "
        "within the configured threshold. Accepted traces are baseline-subtracted "
        "for downstream pulse-height analysis."
    ),
    "PH": (
        "PH\n\n"
        "Pulse heights are calculated from reduced pulses by taking the minimum "
        "sample value in each trace, then counting those heights in histogram bins. "
        "The bins, min, and max controls set the histogram range."
    ),
    "Optimal Filter Signal FFT": (
        "Optimal Filter Signal FFT\n\n"
        "Magnitude of the FFT of the accepted-pulse average signal template."
    ),
    "Optimal Filter Noise FFT": (
        "Optimal Filter Noise FFT\n\n"
        "Noise FFT magnitude estimated from the background records of accepted traces."
    ),
    "Optimal Filter Template": (
        "Optimal Filter Template\n\n"
        "Time-domain inverse FFT of signal FFT divided by noise FFT squared."
    ),
    "PHA": (
        "PHA\n\n"
        "Accepted pulses are projected onto the optimal-filter template to estimate "
        "pulse heights, then counted in histogram bins. The bins, min, and max "
        "controls set the histogram range."
    ),
    "PHA Timeline": (
        "PHA Timeline\n\n"
        "Accepted-pulse PHA values are plotted in the same order as the pulse "
        "array, using accepted pulse index on the horizontal axis."
    ),
    "PHA Cluster": (
        "PHA Cluster\n\n"
        "The drift-corrected PHA timeline is filtered by the configured PHA "
        "range, then each selected point is assigned to a lower or upper cluster "
        "from the boundary and its adjacent selected points."
    ),
    "Lower Cluster PHA": (
        "Lower Cluster PHA\n\n"
        "Only PHA values assigned to the lower cluster are counted in histogram "
        "bins. The bins, min, and max controls set the histogram range."
    ),
    "Baseline/PHA": (
        "Baseline/PHA\n\n"
        "The background-window average for each accepted trace is plotted against "
        "that trace's optimal-filter pulse height."
    ),
    "Drift-Corrected PHA": (
        "Drift-Corrected PHA\n\n"
        "Optimal-filter pulse heights are corrected by subtracting the fitted "
        "linear dependence on baseline around the mean baseline."
    ),
}


def pulse_steps(config: PulseAnalysisConfig) -> tuple[str, ...]:
    steps = list(DEFAULT_STEPS)
    if config.baseline_drift_correction:
        steps.append(PulseStage.DRIFT_CORRECTED_PHA.value)
    if config.pha_clustering:
        steps.extend(
            [
                PulseStage.PHA_CLUSTER.value,
                PulseStage.LOWER_CLUSTER_PHA.value,
            ]
        )
    return tuple(steps)


@dataclass
class PulseWorkflow:
    steps: tuple[str, ...] = DEFAULT_STEPS
    step_index: int = 0
    finished: bool = False
    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("PulseWorkflow requires at least one step.")
        if not 0 <= self.step_index < len(self.steps):
            raise ValueError("step_index is outside the configured steps.")

    @property
    def current_step(self) -> str:
        return self.steps[self.step_index]

    @property
    def can_go_back(self) -> bool:
        return self.step_index > 0 and not self.finished

    @property
    def can_go_next(self) -> bool:
        return self.step_index < len(self.steps) - 1 and not self.finished

    @property
    def can_finish(self) -> bool:
        return not self.finished

    def go_to_step(self, step_index: int) -> bool:
        if self.finished or not 0 <= step_index < len(self.steps):
            return False
        self.step_index = step_index
        self.events.append(f"step:{self.current_step}")
        return True

    def back(self) -> bool:
        if not self.can_go_back:
            return False
        self.step_index -= 1
        self.events.append(f"back:{self.current_step}")
        return True

    def next(self) -> bool:
        if not self.can_go_next:
            return False
        self.step_index += 1
        self.events.append(f"next:{self.current_step}")
        return True

    def finish(self) -> bool:
        if not self.can_finish:
            return False
        self.finished = True
        self.events.append("finish")
        return True


class PulseWorkflowController:
    def __init__(
        self,
        pulse_data: Hdf5PulseData,
        max_points_per_trace: int | None = MAX_POINTS_PER_TRACE,
        max_traces: int | None = MAX_TRACES_PER_DATASET,
        config: PulseAnalysisConfig | None = None,
        output_dir: Path | None = None,
        save_config_path: Path | None = None,
        array_format: ArrayOutputFormat = "npy",
    ) -> None:
        self.pulse_data = pulse_data
        base_config = config or default_config()
        self.config = base_config.with_updates(
            max_points_per_trace=max_points_per_trace,
            max_display_traces=max_traces,
        )
        self.workflow = PulseWorkflow(steps=pulse_steps(self.config))
        self.output_dir = output_dir
        self.save_config_path = save_config_path
        self.array_format = _validated_array_format(array_format)
        self.status_message = ""
        self.source = PulseDataSource(pulse_data)
        self.pipeline = PulsePipeline(self.source, self.config)
        self.renderer = PulsePlotRenderer(
            self.pipeline,
            pulse_data.file_path,
            baseline_drift_range_callback=self.apply_baseline_drift_range,
        )
        self.ui = PulseWizardUI(callbacks=self)

    def start(self) -> None:
        try:
            self.render()
            self.ui.show()
        finally:
            self.pulse_data.close()

    def render(self) -> None:
        self.ui.render(self.view_state(), self.renderer.draw_plot)

    def view_state(self) -> PulseViewState:
        step = self.workflow.current_step
        return PulseViewState(
            steps=self.workflow.steps,
            step_index=self.workflow.step_index,
            current_step=step,
            can_go_back=self.workflow.can_go_back,
            can_go_next=self.workflow.can_go_next,
            can_finish=self.workflow.can_finish,
            finished=self.workflow.finished,
            info_text=self.info_text(step),
            config_values=self.config_values(),
            status_text=self.pipeline.status_text(step),
            error_text=self.status_message,
        )

    def info_text(self, step: str) -> str:
        if step == "Show all":
            return "\n\n".join(
                [format_hdf5_summary(self.pulse_data.summary), STEP_INFO_TEXT[step]]
            )
        return STEP_INFO_TEXT.get(step, step)

    def back(self) -> None:
        if self.workflow.back():
            self.render()

    def next(self) -> None:
        if self.workflow.next():
            self.render()

    def go_to_step(self, step_index: int) -> None:
        if self.workflow.go_to_step(step_index):
            self.render()

    def finish(self) -> None:
        if self.workflow.finish():
            if self.output_dir is not None:
                output_paths = _save_pipeline_outputs(
                    self.pulse_data,
                    self.pipeline,
                    self.renderer,
                    self.output_dir,
                    self.config,
                    self.workflow.steps,
                    array_format=self.array_format,
                    savefig_progress_callback=print_savefig_progress,
                )
                if self.save_config_path is not None:
                    output_paths = output_paths + (
                        save_config(self.config, self.save_config_path),
                    )
                print("\nSaved pulse outputs:")
                for output_path in output_paths:
                    print(f"- {output_path}")
            print(f"\n{self.analysis_summary_text()}")
            self.ui.close()

    def analysis_summary_text(self) -> str:
        baseline_pha = self.pipeline.baseline_pha()
        pha = (
            baseline_pha.drift.pha_corrected
            if baseline_pha.drift is not None
            else baseline_pha.pha
        )
        finite_pha = pha[np.isfinite(pha)]
        lines = [
            "[pulse] Analysis summary",
            f"traces: {self.source.trace_count}",
            f"accepted: {baseline_pha.accepted_count}",
            f"rejected: {baseline_pha.rejected_count}",
            f"PHA points: {pha.size}",
            f"optimal filter normalization: {baseline_pha.normalization:g}",
        ]
        if finite_pha.size:
            lines.extend(
                [
                    f"PHA mean: {float(np.mean(finite_pha)):g}",
                    f"PHA std: {float(np.std(finite_pha)):g}",
                ]
            )
        if baseline_pha.drift is not None:
            lines.extend(
                [
                    "drift correction: enabled",
                    f"drift slope: {baseline_pha.drift.slope:g}",
                    f"drift reference baseline: {baseline_pha.drift.reference_baseline:g}",
                    f"drift fit points: {baseline_pha.drift.fit_count}",
                ]
            )
            if baseline_pha.drift.cluster_labels is not None:
                lines.append("baseline/PHA clustering: enabled")
                lines.append(
                    f"baseline/PHA cluster count: {self.config.baseline_drift_cluster_count}"
                )
                for cluster_index in range(self.config.baseline_drift_cluster_count):
                    lines.append(
                        f"baseline/PHA cluster {cluster_index + 1} points: "
                        f"{int(np.count_nonzero(baseline_pha.drift.cluster_labels == cluster_index))}"
                    )
        else:
            lines.append("drift correction: disabled")
        if self.config.pha_clustering:
            cluster = self.pipeline.pha_cluster()
            lines.extend(
                [
                    "PHA clustering: enabled",
                    f"cluster selected points: {int(np.count_nonzero(cluster.selected_mask))}",
                    f"lower cluster points: {int(np.count_nonzero(cluster.lower_cluster_mask))}",
                    f"upper cluster points: {int(np.count_nonzero(cluster.upper_cluster_mask))}",
                ]
            )
            if cluster.boundary is not None:
                lines.append(f"cluster boundary: {cluster.boundary:g}")
        return "\n".join(lines)

    def apply_baseline_drift_range(
        self,
        baseline_min: float,
        baseline_max: float,
        pha_min: float,
        pha_max: float,
    ) -> None:
        try:
            self.config = self.config.with_updates(
                baseline_drift_baseline_min=baseline_min,
                baseline_drift_baseline_max=baseline_max,
                baseline_drift_pha_min=pha_min,
                baseline_drift_pha_max=pha_max,
            )
        except ValueError as error:
            self.status_message = f"Error: {error}"
            self.render()
            return
        self.status_message = "Drift fit range selected."
        self.pipeline.update_config(self.config)
        self.render()

    def apply_settings(self, updates: dict[str, str]) -> None:
        try:
            self.config = parse_config_updates(self.config, updates)
        except (TypeError, ValueError) as error:
            self.status_message = f"Error: {error}"
            self.render()
            return
        self.status_message = "Settings applied."
        self.workflow = PulseWorkflow(
            steps=pulse_steps(self.config),
            step_index=min(self.workflow.step_index, len(pulse_steps(self.config)) - 1),
        )
        self.pipeline.update_config(self.config)
        self.render()

    def config_values(self) -> dict[str, str]:
        values = self.config.to_dict()
        config_values = {
            "spectrum_bins": str(values["spectrum_bins"]),
            "histogram_min": self._optional_config_text(values["histogram_min"]),
            "histogram_max": self._optional_config_text(values["histogram_max"]),
        }
        if self.config.pha_clustering:
            config_values.update(
                {
                    "pha_cluster_pha_min": self._optional_config_text(
                        values["pha_cluster_pha_min"]
                    ),
                    "pha_cluster_pha_max": self._optional_config_text(
                        values["pha_cluster_pha_max"]
                    ),
                    "pha_cluster_boundary": self._optional_config_text(
                        values["pha_cluster_boundary"]
                    ),
                }
            )
        if self.config.baseline_drift_correction:
            drift_values = {
                "baseline_drift_baseline_min": self._optional_config_text(
                    values["baseline_drift_baseline_min"]
                ),
                "baseline_drift_baseline_max": self._optional_config_text(
                    values["baseline_drift_baseline_max"]
                ),
                "baseline_drift_pha_min": self._optional_config_text(
                    values["baseline_drift_pha_min"]
                ),
                "baseline_drift_pha_max": self._optional_config_text(
                    values["baseline_drift_pha_max"]
                ),
            }
            if self.config.baseline_drift_clustering:
                drift_values["baseline_drift_cluster_count"] = str(
                    values["baseline_drift_cluster_count"]
                )
            config_values.update(drift_values)
        return config_values

    def _optional_config_text(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)


def launch_pulse_workflow(
    pulse_data: Hdf5PulseData,
    max_points_per_trace: int | None = MAX_POINTS_PER_TRACE,
    max_traces: int | None = MAX_TRACES_PER_DATASET,
    config: PulseAnalysisConfig | None = None,
    output_dir: Path | None = None,
    save_config_path: Path | None = None,
    array_format: ArrayOutputFormat = "npy",
) -> None:
    PulseWorkflowController(
        pulse_data,
        max_points_per_trace,
        max_traces,
        config=config,
        output_dir=output_dir,
        save_config_path=save_config_path,
        array_format=array_format,
    ).start()


def _slugify_step(step: str) -> str:
    return "-".join(step.lower().replace("/", "-").split())


def _run_output_dir(output_dir: Path, pulse_data: Hdf5PulseData) -> Path:
    return output_dir / pulse_data.file_path.stem


def _fit_terminal_line(text: str, width: int) -> str:
    width = max(1, width)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    keep = width - 3
    head = keep // 2
    tail = keep - head
    return f"{text[:head]}...{text[-tail:]}"


def print_savefig_progress(
    current: int,
    total: int,
    step: str,
    output_path: Path,
) -> None:
    columns = shutil.get_terminal_size(fallback=(80, 24)).columns
    max_width = max(1, columns - 1)
    message = _fit_terminal_line(
        f"Saving figure {current}/{total}: {step} -> {output_path}",
        max_width,
    )
    end = "\n" if current >= total else ""
    sys.stdout.write(f"\r{message}\033[K{end}")
    sys.stdout.flush()


def _validated_array_format(array_format: str) -> ArrayOutputFormat:
    if array_format not in {"npy", "csv"}:
        raise ValueError('array_format must be "npy" or "csv".')
    return cast(ArrayOutputFormat, array_format)


def _save_table_csv(output_path: Path, columns: dict[str, np.ndarray]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns.keys())
        if not columns:
            return output_path
        arrays = [np.asarray(column) for column in columns.values()]
        for row in zip(*arrays):
            writer.writerow(row)
    return output_path


def _spectrum_columns(pipeline: PulsePipeline) -> dict[str, np.ndarray]:
    spectrum = pipeline.ph_spectrum()
    return {
        "bin_left": spectrum.bin_edges[:-1],
        "bin_right": spectrum.bin_edges[1:],
        "count": spectrum.counts,
    }


def _optimal_filter_output_columns(
    pipeline: PulsePipeline,
) -> dict[str, dict[str, np.ndarray]]:
    prep = pipeline.optimal_filter_prep()
    heights = pipeline.pha_spectrum()
    baseline_pha = pipeline.baseline_pha()
    pha_timeline = pipeline.pha_timeline()
    outputs = {
        "optimal_filter_template": {
            "time": prep.template_times,
            "template": prep.template,
        },
        "optimal_filter_template_fft": {
            "frequency": prep.template_frequencies,
            "template_fft": prep.template_fft,
        },
        "optimal_filter_noise_psd": {
            "frequency": prep.noise_frequencies,
            "noise_fft": prep.noise_fft,
            "noise_psd": prep.noise_psd,
        },
        "optimal_filter_filter_template": {
            "time": prep.filter_template_times,
            "filter_template": prep.filter_template,
        },
        "optimal_filter_pulse_height": {
            "bin_left": heights.bin_edges[:-1],
            "bin_right": heights.bin_edges[1:],
            "count": heights.counts,
        },
        "optimal_filter_pha_timeline": {
            "pulse_index": pha_timeline.pulse_indices,
            "pha": pha_timeline.pha,
        },
        "optimal_filter_baseline_pulse_height": {
            "baseline": baseline_pha.baseline,
            "pha": baseline_pha.pha,
        },
    }
    if pipeline.config.pha_clustering:
        cluster = pipeline.pha_cluster()
        outputs["optimal_filter_pha_cluster"] = {
            "pulse_index": cluster.pulse_indices,
            "pha": cluster.pha,
            "selected": cluster.selected_mask.astype(int),
            "lower_cluster": cluster.lower_cluster_mask.astype(int),
            "upper_cluster": cluster.upper_cluster_mask.astype(int),
        }
        lower_cluster_pha = pipeline.lower_cluster_pha_spectrum()
        outputs["optimal_filter_lower_cluster_pulse_height"] = {
            "bin_left": lower_cluster_pha.bin_edges[:-1],
            "bin_right": lower_cluster_pha.bin_edges[1:],
            "count": lower_cluster_pha.counts,
        }
    if baseline_pha.drift is not None:
        drift_corrected = pipeline.drift_corrected_pha_spectrum()
        outputs["optimal_filter_baseline_pulse_height"][
            "pha_corrected"
        ] = baseline_pha.drift.pha_corrected
        if baseline_pha.drift.cluster_labels is not None:
            outputs["optimal_filter_baseline_pulse_height"][
                "drift_cluster"
            ] = baseline_pha.drift.cluster_labels
        outputs["optimal_filter_drift_correction"] = {
            "enabled": np.array([True]),
            "slope": np.array([baseline_pha.drift.slope]),
            "intercept": np.array([baseline_pha.drift.intercept]),
            "reference_baseline": np.array([baseline_pha.drift.reference_baseline]),
            "fit_count": np.array([baseline_pha.drift.fit_count]),
        }
        outputs["optimal_filter_drift_corrected_pulse_height"] = {
            "bin_left": drift_corrected.bin_edges[:-1],
            "bin_right": drift_corrected.bin_edges[1:],
            "count": drift_corrected.counts,
        }
    return outputs


def _save_npy_outputs(
    pipeline: PulsePipeline,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "pulse-results.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spectrum": _spectrum_columns(pipeline),
        **_optimal_filter_output_columns(pipeline),
    }
    np.save(output_path, payload, allow_pickle=True)
    return output_path


def _save_csv_outputs(
    pipeline: PulsePipeline,
    output_dir: Path,
) -> tuple[Path, ...]:
    spectrum_path = _save_table_csv(
        output_dir / "spectrum.csv",
        _spectrum_columns(pipeline),
    )
    optimal_filter_outputs = _optimal_filter_output_columns(pipeline)
    template_path = _save_table_csv(
        output_dir / "optimal-filter-template.csv",
        optimal_filter_outputs["optimal_filter_template"],
    )
    template_fft_columns = optimal_filter_outputs["optimal_filter_template_fft"]
    template_fft = template_fft_columns["template_fft"]
    template_fft_path = _save_table_csv(
        output_dir / "optimal-filter-template-fft.csv",
        {
            "frequency": template_fft_columns["frequency"],
            "template_fft_real": template_fft.real,
            "template_fft_imag": template_fft.imag,
        },
    )
    noise_path = _save_table_csv(
        output_dir / "optimal-filter-noise-psd.csv",
        optimal_filter_outputs["optimal_filter_noise_psd"],
    )
    filter_template_path = _save_table_csv(
        output_dir / "optimal-filter-filter-template.csv",
        optimal_filter_outputs["optimal_filter_filter_template"],
    )
    pulse_height_path = _save_table_csv(
        output_dir / "optimal-filter-pulse-height.csv",
        optimal_filter_outputs["optimal_filter_pulse_height"],
    )
    pha_timeline_path = _save_table_csv(
        output_dir / "optimal-filter-pha-timeline.csv",
        optimal_filter_outputs["optimal_filter_pha_timeline"],
    )
    baseline_pha_path = _save_table_csv(
        output_dir / "optimal-filter-baseline-pulse-height.csv",
        optimal_filter_outputs["optimal_filter_baseline_pulse_height"],
    )

    paths = [
        spectrum_path,
        template_path,
        template_fft_path,
        noise_path,
        filter_template_path,
        pulse_height_path,
        pha_timeline_path,
        baseline_pha_path,
    ]
    if "optimal_filter_pha_cluster" in optimal_filter_outputs:
        paths.append(
            _save_table_csv(
                output_dir / "optimal-filter-pha-cluster.csv",
                optimal_filter_outputs["optimal_filter_pha_cluster"],
            )
        )
    if "optimal_filter_lower_cluster_pulse_height" in optimal_filter_outputs:
        paths.append(
            _save_table_csv(
                output_dir / "optimal-filter-lower-cluster-pulse-height.csv",
                optimal_filter_outputs["optimal_filter_lower_cluster_pulse_height"],
            )
        )
    if "optimal_filter_drift_corrected_pulse_height" in optimal_filter_outputs:
        paths.append(
            _save_table_csv(
                output_dir / "optimal-filter-drift-corrected-pulse-height.csv",
                optimal_filter_outputs["optimal_filter_drift_corrected_pulse_height"],
            )
        )
    return tuple(paths)


def _save_pipeline_outputs(
    pulse_data: Hdf5PulseData,
    pipeline: PulsePipeline,
    renderer: PulsePlotRenderer,
    output_dir: Path,
    config: PulseAnalysisConfig,
    steps: tuple[str, ...] = DEFAULT_STEPS,
    dpi: int = 150,
    array_format: ArrayOutputFormat = "npy",
    savefig_progress_callback: SaveProgressCallback | None = None,
) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt

    array_format = _validated_array_format(array_format)
    run_output_dir = _run_output_dir(output_dir, pulse_data)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    total_steps = len(steps)
    for index, step in enumerate(steps, start=1):
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        try:
            renderer.draw_plot(ax, step)
            output_path = run_output_dir / f"{_slugify_step(step)}.png"
            if savefig_progress_callback is not None:
                savefig_progress_callback(index, total_steps, step, output_path)
            fig.savefig(output_path, dpi=dpi)
            output_paths.append(output_path)
        finally:
            plt.close(fig)

    if array_format == "csv":
        output_paths.extend(_save_csv_outputs(pipeline, run_output_dir))
    else:
        output_paths.append(_save_npy_outputs(pipeline, run_output_dir))
    output_paths.append(save_config(config, run_output_dir / "config.yaml"))
    return tuple(output_paths)


def save_pulse_plots(
    pulse_data: Hdf5PulseData,
    output_dir: Path,
    max_points_per_trace: int | None = MAX_POINTS_PER_TRACE,
    max_traces: int | None = MAX_TRACES_PER_DATASET,
    steps: tuple[str, ...] | None = None,
    dpi: int = 150,
    config: PulseAnalysisConfig | None = None,
    save_config_path: Path | None = None,
    array_format: ArrayOutputFormat = "npy",
    savefig_progress_callback: SaveProgressCallback | None = None,
) -> tuple[Path, ...]:
    """Render pulse pipeline outputs without opening a GUI."""
    base_config = config or default_config()
    effective_config = base_config.with_updates(
        max_points_per_trace=max_points_per_trace,
        max_display_traces=max_traces,
    )
    source = PulseDataSource(pulse_data)
    pipeline = PulsePipeline(source, effective_config)
    renderer = PulsePlotRenderer(pipeline, pulse_data.file_path)
    output_steps = pulse_steps(effective_config) if steps is None else steps
    output_paths = _save_pipeline_outputs(
        pulse_data,
        pipeline,
        renderer,
        output_dir,
        effective_config,
        output_steps,
        dpi,
        array_format=array_format,
        savefig_progress_callback=savefig_progress_callback,
    )
    if save_config_path is not None:
        output_paths = output_paths + (save_config(effective_config, save_config_path),)
    return output_paths
