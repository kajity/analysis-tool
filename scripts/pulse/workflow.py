from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    from .analysis import MAX_POINTS_PER_TRACE, MAX_TRACES_PER_DATASET, PulseAnalyzer
    from .pulse_io import Hdf5PulseData, format_hdf5_summary
    from .ui import PulseViewState, PulseWizardUI
except ImportError:
    from analysis import MAX_POINTS_PER_TRACE, MAX_TRACES_PER_DATASET, PulseAnalyzer
    from pulse_io import Hdf5PulseData, format_hdf5_summary
    from ui import PulseViewState, PulseWizardUI

DEFAULT_STEPS = ("Show all", "Preprocess", "Differential", "Spectrum")
STEP_INFO_TEXT = {
    "Show all": (
        "Show all\n\n"
        "Waveforms are plotted from /waveform/wave using /waveform/vres "
        "for the vertical scale and /waveform/hres for the horizontal scale."
    ),
    "Preprocess": (
        "Preprocess\n\n"
        "Future controls will expose baseline correction, filtering, and clipping parameters."
    ),
    "Differential": (
        "Differential\n\n"
        "Future controls will calculate pulse derivatives from selected datasets."
    ),
    "Spectrum": (
        "Spectrum\n\n"
        "Future controls will calculate spectra from selected pulse traces."
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
    ) -> None:
        self.pulse_data = pulse_data
        self.workflow = PulseWorkflow()
        self.analyzer = PulseAnalyzer(pulse_data, max_points_per_trace, max_traces)
        self.ui = PulseWizardUI(callbacks=self)

    def start(self) -> None:
        try:
            self.render()
            self.ui.show()
        finally:
            self.pulse_data.close()

    def render(self) -> None:
        self.ui.render(self.view_state(), self.analyzer.draw_plot)

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
            self.pulse_data.close()
            self.ui.close()


def launch_pulse_workflow(
    pulse_data: Hdf5PulseData,
    max_points_per_trace: int | None = MAX_POINTS_PER_TRACE,
    max_traces: int | None = MAX_TRACES_PER_DATASET,
) -> None:
    PulseWorkflowController(pulse_data, max_points_per_trace, max_traces).start()


def _slugify_step(step: str) -> str:
    return "-".join(step.lower().split())


def save_pulse_plots(
    pulse_data: Hdf5PulseData,
    output_dir: Path,
    max_points_per_trace: int | None = MAX_POINTS_PER_TRACE,
    max_traces: int | None = MAX_TRACES_PER_DATASET,
    steps: tuple[str, ...] = DEFAULT_STEPS,
    dpi: int = 150,
) -> tuple[Path, ...]:
    """Render pulse workflow plots to image files without opening a GUI."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = PulseAnalyzer(pulse_data, max_points_per_trace, max_traces)
    output_paths: list[Path] = []

    for step in steps:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        try:
            analyzer.draw_plot(ax, step)
            output_path = output_dir / (
                f"{pulse_data.file_path.stem}-{_slugify_step(step)}.png"
            )
            fig.savefig(output_path, dpi=dpi)
            output_paths.append(output_path)
        finally:
            plt.close(fig)

    return tuple(output_paths)
