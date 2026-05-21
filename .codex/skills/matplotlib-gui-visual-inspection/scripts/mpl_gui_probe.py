#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg", force=True)


@dataclass(frozen=True)
class ArtistCounts:
    lines: int
    images: int
    collections: int
    patches: int
    texts: int
    legends: int


@dataclass(frozen=True)
class AxisSummary:
    index: int
    bounds: tuple[float, float, float, float]
    title: str
    xlabel: str
    ylabel: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    visible: bool
    artist_counts: ArtistCounts
    text_samples: tuple[str, ...]
    line_labels: tuple[str, ...]


@dataclass(frozen=True)
class FigureSummary:
    number: int | None
    size_inches: tuple[float, float]
    dpi: float
    axes: tuple[AxisSummary, ...]


def _figure_number(fig: Any) -> int | None:
    number = getattr(fig, "number", None)
    return int(number) if number is not None else None


def _axis_summary(index: int, ax: Any) -> AxisSummary:
    legend = ax.get_legend()
    return AxisSummary(
        index=index,
        bounds=tuple(float(value) for value in ax.get_position().bounds),
        title=ax.get_title(),
        xlabel=ax.get_xlabel(),
        ylabel=ax.get_ylabel(),
        xlim=tuple(float(value) for value in ax.get_xlim()),
        ylim=tuple(float(value) for value in ax.get_ylim()),
        visible=bool(ax.get_visible()),
        artist_counts=ArtistCounts(
            lines=len(ax.lines),
            images=len(ax.images),
            collections=len(ax.collections),
            patches=len(ax.patches),
            texts=len(ax.texts),
            legends=0 if legend is None else 1,
        ),
        text_samples=tuple(text.get_text() for text in ax.texts[:8]),
        line_labels=tuple(line.get_label() for line in ax.lines[:8]),
    )


def summarize_figure(fig: Any) -> FigureSummary:
    """Return a compact, JSON-serializable summary of a Matplotlib Figure."""
    fig.canvas.draw()
    size = fig.get_size_inches()
    return FigureSummary(
        number=_figure_number(fig),
        size_inches=(float(size[0]), float(size[1])),
        dpi=float(fig.dpi),
        axes=tuple(_axis_summary(index, ax) for index, ax in enumerate(fig.axes)),
    )


def summarize_pyplot_figures() -> tuple[FigureSummary, ...]:
    """Summarize all live pyplot-managed figures."""
    import matplotlib.pyplot as plt

    return tuple(summarize_figure(plt.figure(number)) for number in plt.get_fignums())


def summary_to_dict(summary: FigureSummary | Sequence[FigureSummary]) -> Any:
    if isinstance(summary, FigureSummary):
        return asdict(summary)
    return [asdict(item) for item in summary]


def write_summary(
    summary: FigureSummary | Sequence[FigureSummary],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_to_dict(summary), indent=2) + "\n")
    return output_path


def save_figure(fig: Any, output_path: Path, dpi: int = 150) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize live pyplot figures. For most GUI apps, import this module "
            "from a REPL after constructing the GUI instead of running it directly."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/matplotlib-gui-visual-inspection/pyplot-summary.json"),
        help="JSON summary output path.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = summarize_pyplot_figures()
    write_summary(summaries, args.output)
    print(f"Wrote {len(summaries)} figure summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
