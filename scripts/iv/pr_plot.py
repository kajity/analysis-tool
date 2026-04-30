from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import MISSING, OmegaConf

from common import (
    color_for_dataset,
    configure_plot_style,
    export_plot,
    filter_excluded_temperatures,
    label_for,
    load_pyplot,
    resolve_input_paths,
    temperature_sort_key,
    voltage_for_plot,
)
from iv_io import IvData, load_iv_data


@dataclass
class PrConfig:
    R_sh: float = MISSING
    R_FB: float = MISSING
    M_in: float = MISSING
    M_FB: float = MISSING


@dataclass
class IvAnalysisConfig:
    pr: PrConfig


def load_pr_config(args: argparse.Namespace) -> PrConfig:
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    schema = OmegaConf.structured(IvAnalysisConfig(pr=PrConfig()))
    config = OmegaConf.merge(schema, OmegaConf.load(args.config))
    overrides = {
        "pr": {
            "R_sh": args.r_sh,
            "R_FB": args.r_fb,
            "M_in": args.m_in,
            "M_FB": args.m_fb,
        }
    }
    clean_overrides = {
        "pr": {
            key: value
            for key, value in overrides["pr"].items()
            if value is not None
        }
    }
    config = OmegaConf.merge(config, clean_overrides)

    missing = [
        key
        for key in ("R_sh", "R_FB", "M_in", "M_FB")
        if OmegaConf.select(config, f"pr.{key}") is None
    ]
    if missing:
        raise ValueError(f"Missing PR config parameter(s): {', '.join(missing)}")

    for key in ("R_sh", "R_FB", "M_in", "M_FB"):
        value = float(OmegaConf.select(config, f"pr.{key}"))
        if value == 0:
            raise ValueError(f"PR config parameter {key} must be non-zero.")
        config.pr[key] = value

    typed_config = OmegaConf.to_object(config)
    if not isinstance(typed_config, IvAnalysisConfig):
        raise ValueError("Could not convert PR config to typed config.")
    return typed_config.pr


def calculate_pr(iv_data: IvData, config: PrConfig, raw_voltage: bool) -> tuple[np.ndarray, np.ndarray]:
    xi = config.M_in / config.M_FB * config.R_FB
    voltage_v = voltage_for_plot(iv_data, raw_voltage)
    i_tes_a = voltage_v / xi
    i_bias_a = iv_data.current_uA * 1e-6

    with np.errstate(divide="ignore", invalid="ignore"):
        r_tes_ohm = config.R_sh * (i_bias_a / i_tes_a - 1)
        p_b_w = r_tes_ohm * i_tes_a**2

    valid = np.isfinite(r_tes_ohm) & np.isfinite(p_b_w)
    valid &= r_tes_ohm >= 0
    valid &= p_b_w >= 0
    return r_tes_ohm[valid], p_b_w[valid]


def normalize_resistance(r_tes_ohm: np.ndarray) -> np.ndarray:
    if r_tes_ohm.size == 0:
        raise ValueError("Cannot normalize R_TES because no valid R_TES points were calculated.")

    r_normal = float(np.max(r_tes_ohm))
    if r_normal <= 0:
        raise ValueError("Cannot normalize R_TES because its maximum is non-positive.")
    return r_tes_ohm / r_normal


def plot_pr(plt, datasets: list[tuple[Path, IvData]], config: PrConfig, raw_voltage: bool):
    configure_plot_style(plt)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    for index, (path, iv_data) in enumerate(datasets):
        r_tes_ohm, p_b_w = calculate_pr(iv_data, config, raw_voltage)
        ax.plot(
            normalize_resistance(r_tes_ohm),
            p_b_w * 1e12,
            linestyle="None",
            marker="o",
            markersize=4,
            color=color_for_dataset(plt, index, len(datasets), iv_data),
            label=label_for(path, iv_data),
        )

    ax.set_xlabel(r"$R_{TES}/R_N$", fontsize=13)
    ax.set_ylabel(r"$P_b$ (pW)", fontsize=13)
    ax.set_title(r"$P_b$-$R_{TES}/R_N$", fontsize=18)
    ax.grid(True, linestyle="--", color="0.65", alpha=0.85)
    ax.legend(loc="best", framealpha=0.9, fontsize="small")
    return fig


def run_pr_step(args: argparse.Namespace) -> int:
    try:
        input_paths = resolve_input_paths(args.inputs)
        datasets = [(path, load_iv_data(path)) for path in input_paths]
        datasets.sort(key=temperature_sort_key)
        datasets = filter_excluded_temperatures(datasets, args.exclude)
        config = load_pr_config(args)

        plt = load_pyplot(args.show)
        output_name = (
            "pr_by_temperature.png"
            if len(datasets) > 1
            else f"{input_paths[0].stem}_pr.png"
        )
        fig = plot_pr(plt, datasets, config, args.raw_voltage)
        output_path = export_plot(fig, args.output_dir, output_name)
        print(f"Saved PR plot: {output_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)

        return 0
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
