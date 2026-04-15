from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button, SpanSelector

from analysis import SelectionStats, selection_from_range, summarize_thickness


@dataclass
class SelectionAction:
    target: str
    selection: SelectionStats


def _reindex(selections: list[SelectionStats]) -> None:
    for idx, selection in enumerate(selections, start=1):
        selection.index = idx


def _draw_spans(
    ax: Axes,
    selections: list[SelectionStats],
    color: str,
    alpha: float,
    spans: list[object],
) -> list[object]:
    for patch in spans:
        patch.remove()

    new_spans: list[object] = []
    for selection in selections:
        patch = ax.axvspan(selection.xmin, selection.xmax, color=color, alpha=alpha)
        new_spans.append(patch)
    return new_spans


def launch_thickness_ui(
    x: np.ndarray,
    y: np.ndarray,
    input_path: Path,
    max_history: int,
) -> None:
    fig: Figure = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(3.4, 1.6),
        height_ratios=(1.0, 1.0),
        left=0.06,
        right=0.97,
        bottom=0.12,
        top=0.92,
        wspace=0.18,
        hspace=0.24,
    )

    ax_high = fig.add_subplot(gs[0, 0])
    ax_low = fig.add_subplot(gs[1, 0], sharex=ax_high, sharey=ax_high)
    ax_info = fig.add_subplot(gs[:, 1])
    ax_info.axis("off")

    for ax in (ax_high, ax_low):
        ax.plot(x, y, color="tab:blue", lw=1.0)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Height")

    ax_high.set_title(f"High region selection: {input_path.name}")
    ax_low.set_title("Background region selection")
    ax_low.set_xlabel("Position")

    high_regions: list[SelectionStats] = []
    low_regions: list[SelectionStats] = []
    history: list[SelectionAction] = []
    high_spans: list[object] = []
    low_spans: list[object] = []

    info_text = ax_info.text(
        0.0,
        1.0,
        summarize_thickness(high_regions, low_regions, y, max_history),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    def refresh() -> None:
        nonlocal high_spans, low_spans
        high_spans = _draw_spans(ax_high, high_regions, "tab:red", 0.22, high_spans)
        low_spans = _draw_spans(ax_low, low_regions, "tab:green", 0.22, low_spans)
        info_text.set_text(summarize_thickness(high_regions, low_regions, y, max_history))
        fig.canvas.draw_idle()

    def add_selection(target: str, xmin: float, xmax: float) -> None:
        if target == "high":
            selection = selection_from_range(x, y, xmin, xmax, len(high_regions) + 1)
            if selection is None:
                return
            high_regions.append(selection)
        else:
            selection = selection_from_range(x, y, xmin, xmax, len(low_regions) + 1)
            if selection is None:
                return
            low_regions.append(selection)
        history.append(SelectionAction(target=target, selection=selection))
        refresh()

    def onselect_high(xmin: float, xmax: float) -> None:
        add_selection("high", xmin, xmax)

    def onselect_low(xmin: float, xmax: float) -> None:
        add_selection("low", xmin, xmax)

    def reset(_: object) -> None:
        high_regions.clear()
        low_regions.clear()
        history.clear()
        refresh()

    def undo(_: object) -> None:
        if not history:
            return

        last_action = history.pop()
        if last_action.target == "high" and high_regions:
            high_regions.pop()
            _reindex(high_regions)
        elif last_action.target == "low" and low_regions:
            low_regions.pop()
            _reindex(low_regions)
        refresh()

    high_selector = SpanSelector(
        ax_high,
        onselect_high,
        "horizontal",
        useblit=True,
        interactive=True,
        drag_from_anywhere=True,
        props=dict(alpha=0.18, facecolor="tab:red"),
    )
    low_selector = SpanSelector(
        ax_low,
        onselect_low,
        "horizontal",
        useblit=True,
        interactive=True,
        drag_from_anywhere=True,
        props=dict(alpha=0.18, facecolor="tab:green"),
    )

    reset_ax = fig.add_axes([0.68, 0.03, 0.12, 0.055])
    undo_ax = fig.add_axes([0.82, 0.03, 0.12, 0.055])
    reset_button = Button(reset_ax, "Reset")
    undo_button = Button(undo_ax, "Undo")
    reset_button.on_clicked(reset)
    undo_button.on_clicked(undo)

    # Keep widget instances alive for the full lifetime of the figure.
    fig._thickness_widgets = {
        "high_selector": high_selector,
        "low_selector": low_selector,
        "reset_button": reset_button,
        "undo_button": undo_button,
    }

    plt.show()
