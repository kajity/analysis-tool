from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button, TextBox

PlotRenderer = Callable[[Axes, str], None]


@dataclass(frozen=True)
class PulseViewState:
    steps: tuple[str, ...]
    step_index: int
    current_step: str
    can_go_back: bool
    can_go_next: bool
    can_finish: bool
    can_reset: bool
    info_text: str
    config_values: dict[str, str]
    status_text: str
    error_text: str = ""


class PulseUiCallbacks(Protocol):
    def back(self) -> None: ...

    def next(self) -> None: ...

    def reset(self) -> None: ...

    def finish(self) -> None: ...

    def apply_settings(self, updates: dict[str, str]) -> None: ...


class PulseWizardUI:
    def __init__(self, callbacks: PulseUiCallbacks) -> None:
        self.callbacks = callbacks
        self.fig: Figure = plt.figure(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title("Pulse interactive workflow")
        self.ax_content = self.fig.add_axes((0.07, 0.22, 0.58, 0.68))
        self.ax_info = self.fig.add_axes((0.69, 0.50, 0.27, 0.40))
        self.ax_steps = self.fig.add_axes((0.07, 0.08, 0.38, 0.06))
        self.ax_info.axis("off")
        self.ax_steps.axis("off")

        self.control_boxes: dict[str, TextBox] = {
            "spectrum_bins": self._make_text_box((0.80, 0.30, 0.12, 0.035), "bins"),
            "histogram_min": self._make_text_box((0.80, 0.25, 0.12, 0.035), "min"),
            "histogram_max": self._make_text_box((0.80, 0.20, 0.12, 0.035), "max"),
        }
        self.apply_button = self._make_button((0.80, 0.13, 0.12, 0.045), "Apply")

        self.back_button = self._make_button((0.50, 0.06, 0.10, 0.06), "Back")
        self.next_button = self._make_button((0.61, 0.06, 0.10, 0.06), "Next")
        self.reset_button = self._make_button((0.72, 0.06, 0.10, 0.06), "Reset")
        self.finish_button = self._make_button((0.83, 0.06, 0.11, 0.06), "Finish")

        self.back_button.on_clicked(self.on_back)
        self.next_button.on_clicked(self.on_next)
        self.reset_button.on_clicked(self.on_reset)
        self.finish_button.on_clicked(self.on_finish)
        self.apply_button.on_clicked(self.on_apply_settings)

        # Keep widget instances alive for the full lifetime of the figure.
        # self.fig._pulse_widgets = {
        #     "back": self.back_button,
        #     "next": self.next_button,
        #     "reset": self.reset_button,
        #     "finish": self.finish_button,
        # }

    def _make_button(
        self, bounds: tuple[float, float, float, float], label: str
    ) -> Button:
        ax = self.fig.add_axes(bounds)
        return Button(ax, label)

    def _make_text_box(
        self, bounds: tuple[float, float, float, float], label: str
    ) -> TextBox:
        ax = self.fig.add_axes(bounds)
        return TextBox(ax, label)

    def _set_button_enabled(self, button: Button, enabled: bool) -> None:
        button.set_active(enabled)
        button.label.set_color("black" if enabled else "0.55")
        button.ax.set_facecolor("0.92" if enabled else "0.82")

    def _step_line(self, state: PulseViewState) -> str:
        labels = []
        for index, step in enumerate(state.steps):
            marker = ">" if index == state.step_index else " "
            labels.append(f"{marker} {step}")
        return "    ".join(labels)

    def render(self, state: PulseViewState, draw_plot: PlotRenderer) -> None:
        self.ax_content.clear()
        draw_plot(self.ax_content, state.current_step)

        self.ax_info.clear()
        self.ax_info.axis("off")
        info_text = "\n\n".join(
            text
            for text in [state.info_text, state.status_text, state.error_text]
            if text
        )
        self.ax_info.text(
            0.0,
            1.0,
            info_text,
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
            self._step_line(state),
            va="center",
            ha="left",
            fontsize=10,
            family="monospace",
        )

        self._set_button_enabled(self.back_button, state.can_go_back)
        self._set_button_enabled(self.next_button, state.can_go_next)
        self._set_button_enabled(self.finish_button, state.can_finish)
        self._set_button_enabled(self.reset_button, state.can_reset)
        for key, text_box in self.control_boxes.items():
            if key in state.config_values:
                text_box.set_val(state.config_values[key])
        self.fig.canvas.draw_idle()

    def on_back(self, _: object) -> None:
        self.callbacks.back()

    def on_next(self, _: object) -> None:
        self.callbacks.next()

    def on_reset(self, _: object) -> None:
        self.callbacks.reset()

    def on_finish(self, _: object) -> None:
        self.callbacks.finish()

    def on_apply_settings(self, _: object) -> None:
        self.callbacks.apply_settings(
            {key: text_box.text for key, text_box in self.control_boxes.items()}
        )

    def close(self) -> None:
        plt.close(self.fig)

    def show(self) -> None:
        plt.show()
