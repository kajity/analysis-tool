from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    color_for_dataset,
    configure_plot_style,
    export_plot,
    label_for,
    load_pyplot,
    resolve_input_paths,
    temperature_sort_key,
    voltage_for_plot,
)
from iv_io import IvData, load_iv_data


def plot_iv(plt, datasets: list[tuple[Path, IvData]], raw_voltage: bool):
    configure_plot_style(plt)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    for index, (path, iv_data) in enumerate(datasets):
        ax.plot(
            iv_data.current_uA,
            voltage_for_plot(iv_data, raw_voltage),
            linestyle="None",
            marker="o",
            markersize=4,
            color=color_for_dataset(plt, index, len(datasets), iv_data),
            label=label_for(path, iv_data),
        )

    ax.set_xlabel(r"$I_{bias}$ ($\mu$A)", fontsize=13)
    ax.set_ylabel(r"$V_{out}$ (V)", fontsize=13)
    ax.set_title(r"$I_{bias}$-$V_{out}$", fontsize=18)
    ax.grid(True, linestyle="--", color="0.65", alpha=0.85)
    ax.legend(loc="lower right", framealpha=0.9, fontsize="small")
    return fig


def run_iv_step(args: argparse.Namespace) -> int:
    try:
        input_paths = resolve_input_paths(args.inputs)
        datasets = [(path, load_iv_data(path)) for path in input_paths]
        datasets.sort(key=temperature_sort_key)

        plt = load_pyplot(args.show)
        output_name = (
            "iv_by_temperature.png"
            if len(datasets) > 1
            else f"{input_paths[0].stem}_iv.png"
        )
        fig = plot_iv(plt, datasets, args.raw_voltage)
        output_path = export_plot(fig, args.output_dir, output_name)
        print(f"Saved IV plot: {output_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)

        return 0
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
