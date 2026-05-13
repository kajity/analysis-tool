from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Button

try:
    from .pulse_io import Hdf5Summary, format_hdf5_summary
    from .workflow import PulseWorkflow
except ImportError:
    from pulse_io import Hdf5Summary, format_hdf5_summary
    from workflow import PulseWorkflow


PLACEHOLDER_TEXT = {
    "Preview": (
        "Pulse preview\n\n"
        "Waveform display will be added after the HDF5 pulse dataset layout is fixed."
    ),
    "Configure": (
        "Analysis configuration\n\n"
        "Future controls will expose pulse detection, filtering, baseline, and fitting parameters."
    ),
    "Review": (
        "Review\n\n"
        "No pulse analysis settings are active in this framework-only implementation."
    ),
}


class PulseWizardUI:
    def __init__(self, summary: Hdf5Summary, input_path: Path) -> None:
        self.summary = summary
        self.input_path = input_path
        self.workflow = PulseWorkflow()

        self.fig: Figure = plt.figure(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title("Pulse interactive workflow")
        self.ax_content = self.fig.add_axes([0.07, 0.22, 0.60, 0.68])
        self.ax_info = self.fig.add_axes([0.71, 0.22, 0.23, 0.68])
        self.ax_steps = self.fig.add_axes([0.07, 0.08, 0.38, 0.06])
        self.ax_info.axis("off")
        self.ax_steps.axis("off")

        self.back_button = self._make_button([0.50, 0.06, 0.10, 0.06], "Back")
        self.next_button = self._make_button([0.61, 0.06, 0.10, 0.06], "Next")
        self.reset_button = self._make_button([0.72, 0.06, 0.10, 0.06], "Reset")
        self.finish_button = self._make_button([0.83, 0.06, 0.11, 0.06], "Finish")

        self.back_button.on_clicked(self.on_back)
        self.next_button.on_clicked(self.on_next)
        self.reset_button.on_clicked(self.on_reset)
        self.finish_button.on_clicked(self.on_finish)

        # Keep widget instances alive for the full lifetime of the figure.
        self.fig._pulse_widgets = {
            "back": self.back_button,
            "next": self.next_button,
            "reset": self.reset_button,
            "finish": self.finish_button,
        }

        self.refresh()

    def _make_button(self, bounds: list[float], label: str) -> Button:
        ax = self.fig.add_axes(bounds)
        return Button(ax, label)

    def _set_button_enabled(self, button: Button, enabled: bool) -> None:
        button.set_active(enabled)
        button.label.set_color("black" if enabled else "0.55")
        button.ax.set_facecolor("0.92" if enabled else "0.82")

    def _step_line(self) -> str:
        labels = []
        for index, step in enumerate(self.workflow.steps):
            marker = ">" if index == self.workflow.step_index else " "
            labels.append(f"{marker} {step}")
        return "    ".join(labels)

    def _info_text(self) -> str:
        step = self.workflow.current_step
        if step == "Input":
            return format_hdf5_summary(self.summary)
        return PLACEHOLDER_TEXT.get(step, step)

    def _draw_current_plot(self) -> None:
        step = self.workflow.current_step
        self.ax_content.set_title(
            f"{step}: {self.input_path.name}",
            loc="left",
            fontsize=13,
            pad=14,
        )
        self.ax_content.set_xlabel("Sample index")
        self.ax_content.set_ylabel("Signal")
        self.ax_content.grid(True, alpha=0.25)

        if step == "Preview":
            x = [0, 1, 2, 3, 4, 5, 6]
            y = [0.0, 0.0, -0.15, 1.0, 0.38, 0.08, 0.0]
            self.ax_content.plot(x, y, color="tab:blue", lw=1.8)
            self.ax_content.text(
                0.02,
                0.95,
                "placeholder trace",
                transform=self.ax_content.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="0.35",
            )
            return

        self.ax_content.text(
            0.5,
            0.5,
            "Pulse plot area",
            transform=self.ax_content.transAxes,
            va="center",
            ha="center",
            fontsize=12,
            color="0.35",
        )

    def refresh(self) -> None:
        self.ax_content.clear()
        self._draw_current_plot()

        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.text(
            0.0,
            1.0,
            self._info_text(),
            transform=self.ax_info.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )

        self.ax_steps.clear()
        self.ax_steps.axis("off")
        self.ax_steps.text(
            0.0,
            0.5,
            self._step_line(),
            va="center",
            ha="left",
            fontsize=10,
            family="monospace",
        )

        self._set_button_enabled(self.back_button, self.workflow.can_go_back)
        self._set_button_enabled(self.next_button, self.workflow.can_go_next)
        self._set_button_enabled(self.finish_button, self.workflow.can_finish)
        self._set_button_enabled(
            self.reset_button, self.workflow.step_index != 0 or self.workflow.finished
        )
        self.fig.canvas.draw_idle()

    def on_back(self, _: object) -> None:
        if self.workflow.back():
            self.refresh()

    def on_next(self, _: object) -> None:
        if self.workflow.next():
            self.refresh()

    def on_reset(self, _: object) -> None:
        self.workflow.reset()
        self.refresh()

    def on_finish(self, _: object) -> None:
        if self.workflow.finish():
            plt.close(self.fig)


def launch_pulse_ui(summary: Hdf5Summary, input_path: Path) -> None:
    PulseWizardUI(summary, input_path)
    plt.show()
