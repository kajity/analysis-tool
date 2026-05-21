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
    finished: bool
    info_text: str
    config_values: dict[str, str]
    status_text: str
    error_text: str = ""


class PulseUiCallbacks(Protocol):
    def back(self) -> None: ...

    def next(self) -> None: ...

    def go_to_step(self, step_index: int) -> None: ...

    def finish(self) -> None: ...

    def apply_settings(self, updates: dict[str, str]) -> None: ...


class PulseWizardUI:
    STEP_BUTTON_LABELS = {
        "Raw View": "Raw",
        "Reduction": "Reduction",
        "PH": "PH",
        "Optimal Filter Signal FFT": "Signal FFT",
        "Optimal Filter Noise FFT": "Noise FFT",
        "Optimal Filter Template": "Template",
        "Optimal Filter Pulse Height": "PHA",
        "Baseline vs Optimal Filter Pulse Height": "Baseline/PHA",
        "Drift-Corrected Optimal Filter Pulse Height": "Drift PHA",
    }

    def __init__(self, callbacks: PulseUiCallbacks) -> None:
        self.callbacks = callbacks
        self.fig: Figure = plt.figure(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title("Pulse interactive workflow")
        self.ax_content = self.fig.add_axes((0.07, 0.24, 0.58, 0.66))
        self.ax_info = self.fig.add_axes((0.69, 0.50, 0.27, 0.40))
        self.ax_info.axis("off")

        self.control_boxes: dict[str, TextBox] = {
            "spectrum_bins": self._make_text_box((0.81, 0.325, 0.10, 0.030), "bins"),
            "histogram_min": self._make_text_box((0.81, 0.282, 0.10, 0.030), "min"),
            "histogram_max": self._make_text_box((0.81, 0.239, 0.10, 0.030), "max"),
            "baseline_drift_correction": self._make_text_box(
                (0.81, 0.196, 0.10, 0.030),
                "drift",
            ),
        }
        self.apply_button = self._make_button((0.81, 0.145, 0.10, 0.038), "Apply")

        self.next_button = self._make_button((0.07, 0.105, 0.075, 0.040), "Next")
        self.back_button = self._make_button((0.07, 0.055, 0.075, 0.040), "Prev")
        self.finish_button = self._make_button((0.84, 0.055, 0.080, 0.040), "Finish")
        self.step_buttons: list[Button] = []
        self.step_button_axes: list[Axes] = []

        self.back_button.on_clicked(self.on_back)
        self.next_button.on_clicked(self.on_next)
        self.finish_button.on_clicked(self.on_finish)
        self.apply_button.on_clicked(self.on_apply_settings)

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

    def _set_step_button_state(
        self,
        button: Button,
        enabled: bool,
        selected: bool,
    ) -> None:
        button.set_active(enabled)
        button.label.set_color("black" if enabled or selected else "0.55")
        if selected:
            button.ax.set_facecolor("0.72")
        else:
            button.ax.set_facecolor("0.92" if enabled else "0.82")

    def _step_button_label(self, step: str) -> str:
        return self.STEP_BUTTON_LABELS.get(step, step)

    def _ensure_step_buttons(self, steps: tuple[str, ...]) -> None:
        if len(self.step_buttons) == len(steps):
            return
        for ax in self.step_button_axes:
            ax.remove()
        self.step_buttons = []
        self.step_button_axes = []

        columns = (len(steps) + 1) // 2
        start_x = 0.16
        total_width = 0.64
        gap = 0.006
        width = (total_width - gap * (columns - 1)) / columns
        row_y = [0.105, 0.055]
        height = 0.040
        for index, step in enumerate(steps):
            row = index // columns
            column = index % columns
            bounds = (
                start_x + column * (width + gap),
                row_y[row],
                width,
                height,
            )
            button = self._make_button(bounds, self._step_button_label(step))
            button.label.set_fontsize(7)
            button.on_clicked(self._on_step_button(index))
            self.step_buttons.append(button)
            self.step_button_axes.append(button.ax)

    def _on_step_button(self, step_index: int) -> Callable[[object], None]:
        def go_to_step(_: object) -> None:
            self.callbacks.go_to_step(step_index)

        return go_to_step

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

        self._ensure_step_buttons(state.steps)
        self._set_button_enabled(self.back_button, state.can_go_back)
        self._set_button_enabled(self.next_button, state.can_go_next)
        self._set_button_enabled(self.finish_button, state.can_finish)
        for index, button in enumerate(self.step_buttons):
            selected = index == state.step_index
            self._set_step_button_state(
                button,
                enabled=not selected and not state.finished,
                selected=selected,
            )
        for key, text_box in self.control_boxes.items():
            if key in state.config_values:
                text_box.set_val(state.config_values[key])
        self.fig.canvas.draw_idle()

    def on_back(self, _: object) -> None:
        self.callbacks.back()

    def on_next(self, _: object) -> None:
        self.callbacks.next()

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
