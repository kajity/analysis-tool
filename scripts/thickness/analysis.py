from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionStats:
    index: int
    xmin: float
    xmax: float
    start_idx: int
    end_idx: int
    count: int
    mean: float
    variance: float


def selection_from_range(
    x: np.ndarray,
    y: np.ndarray,
    xmin: float,
    xmax: float,
    index: int,
) -> SelectionStats | None:
    left, right = sorted((xmin, xmax))
    indices = np.flatnonzero((x >= left) & (x <= right))
    region = y[indices]

    if region.size < 2:
        return None

    return SelectionStats(
        index=index,
        xmin=float(x[indices[0]]),
        xmax=float(x[indices[-1]]),
        start_idx=int(indices[0]),
        end_idx=int(indices[-1]),
        count=int(region.size),
        mean=float(np.mean(region)),
        variance=float(np.var(region, ddof=0)),
    )


def _concat_values(selections: list[SelectionStats], y: np.ndarray) -> np.ndarray:
    if not selections:
        return np.asarray([], dtype=float)

    merged_indices = np.concatenate(
        [
            np.arange(selection.start_idx, selection.end_idx + 1, dtype=int)
            for selection in selections
        ]
    )
    unique_indices = np.unique(merged_indices)
    return y[unique_indices]


def summarize_thickness(
    high_regions: list[SelectionStats],
    low_regions: list[SelectionStats],
    y: np.ndarray,
    max_history: int,
) -> str:
    if not high_regions and not low_regions:
        return (
            "Top plot: select a high region.\n"
            "Bottom plot: select a background region.\n\n"
            "Displayed values:\n"
            "- mean: average height across all selected samples\n"
            "- variance: variance across all selected samples\n"
            "- thickness: mean(high) - mean(background)\n\n"
            "Buttons:\n"
            "Reset: remove all regions\n"
            "Undo: remove the latest region"
        )

    lines: list[str] = []

    lines.append("High regions (top plot):")
    if high_regions:
        for selection in high_regions[-max_history:]:
            lines.append(
                f"[H{selection.index}] x={selection.xmin:.3f}..{selection.xmax:.3f} "
                f"(n={selection.count})"
            )
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("Background regions (bottom plot):")
    if low_regions:
        for selection in low_regions[-max_history:]:
            lines.append(
                f"[B{selection.index}] x={selection.xmin:.3f}..{selection.xmax:.3f} "
                f"(n={selection.count})"
            )
    else:
        lines.append("(none)")

    if high_regions and low_regions:
        high_values = _concat_values(high_regions, y)
        low_values = _concat_values(low_regions, y)
        high_mean = float(np.mean(high_values))
        low_mean = float(np.mean(low_values))
        high_variance = float(np.var(high_values, ddof=0))
        low_variance = float(np.var(low_values, ddof=0))
        thickness_mean = high_mean - low_mean
        thickness_variance = high_variance**2 + low_variance**2
        thickness_sigma = float(np.sqrt(thickness_variance))

        print(f"High mean: {high_mean:.3f}, variance: {high_variance:.3f}, n: {high_values.size}")
        print(f"Background mean: {low_mean:.3f}, variance: {low_variance:.3f}, n: {low_values.size}")
        print(f"Thickness: {thickness_mean:.2f} ± {thickness_sigma:.2f}")

        lines.append("")
        lines.append("Combined statistics:")
        lines.append(f"High n: {high_values.size}")
        lines.append(f"High mean: {high_mean:.3f}")
        lines.append(f"High variance: {high_variance:.3f}")
        lines.append(f"Background n: {low_values.size}")
        lines.append(f"Background mean: {low_mean:.3f}")
        lines.append(f"Background variance: {low_variance:.3f}")
        lines.append("")
        lines.append(f"Thickness: {thickness_mean:.2f} ± {thickness_sigma:.2f}")
        lines.append(
            "Thickness variance is propagated as "
            "Var(mean_high - mean_background) = Var(high)/n_high + Var(background)/n_background."
        )

    return "\n".join(lines)
