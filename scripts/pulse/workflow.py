from __future__ import annotations

import csv
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
    from .pipeline import DEFAULT_STAGES, PulsePipeline
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
    from pipeline import DEFAULT_STAGES, PulsePipeline
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
    "Reject/Shaping": (
        "Reject/Shaping\n\n"
        "A trace is accepted when every diff outside the valid pulse range stays "
        "within the configured threshold."
    ),
    "Preprocess": (
        "Preprocess\n\n"
        "This stage is reserved for corrections after pulse rejection. Current "
        "preprocessing is baseline subtraction only."
    ),
    "Spectrum": (
        "Spectrum\n\n"
        "Pulse heights are calculated from shaped pulses by taking the minimum "
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
    "Optimal Filter Pulse Height": (
        "Optimal Filter Pulse Height\n\n"
        "Accepted pulses are projected onto the optimal-filter template to estimate "
        "pulse heights, then counted in histogram bins. The bins, min, and max "
        "controls set the histogram range."
    ),
    "Baseline vs Optimal Filter Pulse Height": (
        "Baseline vs Optimal Filter Pulse Height\n\n"
        "The background-window average for each accepted trace is plotted against "
        "that trace's optimal-filter pulse height."
    ),
}


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
        return self.step_index == len(self.steps) - 1 and not self.finished

    @property
    def can_reset(self) -> bool:
        return self.step_index != 0 or self.finished

    def reset(self) -> None:
        self.step_index = 0
        self.finished = False
        self.events.append("reset")

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
        self.workflow = PulseWorkflow()
        base_config = config or default_config()
        self.config = base_config.with_updates(
            max_points_per_trace=max_points_per_trace,
            max_display_traces=max_traces,
        )
        self.output_dir = output_dir
        self.save_config_path = save_config_path
        self.array_format = _validated_array_format(array_format)
        self.status_message = ""
        self.source = PulseDataSource(pulse_data)
        self.pipeline = PulsePipeline(self.source, self.config)
        self.renderer = PulsePlotRenderer(self.pipeline, pulse_data.file_path)
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
            can_reset=self.workflow.can_reset,
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

    def reset(self) -> None:
        self.workflow.reset()
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
                    DEFAULT_STEPS,
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
            self.ui.close()

    def apply_settings(self, updates: dict[str, str]) -> None:
        try:
            self.config = parse_config_updates(self.config, updates)
        except (TypeError, ValueError) as error:
            self.status_message = f"Error: {error}"
            self.render()
            return
        self.status_message = "Settings applied."
        self.pipeline.update_config(self.config)
        self.render()

    def config_values(self) -> dict[str, str]:
        values = self.config.to_dict()
        return {
            "spectrum_bins": str(values["spectrum_bins"]),
            "histogram_min": self._optional_config_text(values["histogram_min"]),
            "histogram_max": self._optional_config_text(values["histogram_max"]),
        }

    def _optional_config_text(self, value: object) -> str:
        return "" if value is None else str(value)


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


def print_savefig_progress(
    current: int,
    total: int,
    step: str,
    output_path: Path,
) -> None:
    end = "\n" if current >= total else ""
    sys.stdout.write(
        f"\rSaving figure {current}/{total}: {step} -> {output_path}\033[K{end}"
    )
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
    spectrum = pipeline.spectrum()
    return {
        "bin_left": spectrum.bin_edges[:-1],
        "bin_right": spectrum.bin_edges[1:],
        "count": spectrum.counts,
    }


def _optimal_filter_output_columns(
    pipeline: PulsePipeline,
) -> dict[str, dict[str, np.ndarray]]:
    prep = pipeline.optimal_filter_prep()
    heights = pipeline.optimal_filter_pulse_height()
    baseline_pha = pipeline.baseline_optimal_filter_pulse_height()
    return {
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
        "optimal_filter_baseline_pulse_height": {
            "baseline": baseline_pha.baseline,
            "pha": baseline_pha.pha,
        },
    }


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
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
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
    baseline_pha_path = _save_table_csv(
        output_dir / "optimal-filter-baseline-pulse-height.csv",
        optimal_filter_outputs["optimal_filter_baseline_pulse_height"],
    )

    return (
        spectrum_path,
        template_path,
        template_fft_path,
        noise_path,
        filter_template_path,
        pulse_height_path,
        baseline_pha_path,
    )


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
    steps: tuple[str, ...] = DEFAULT_STEPS,
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
    output_paths = _save_pipeline_outputs(
        pulse_data,
        pipeline,
        renderer,
        output_dir,
        effective_config,
        steps,
        dpi,
        array_format=array_format,
        savefig_progress_callback=savefig_progress_callback,
    )
    if save_config_path is not None:
        output_paths = output_paths + (save_config(effective_config, save_config_path),)
    return output_paths
