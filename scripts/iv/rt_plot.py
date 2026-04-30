from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from .common import (
        color_for_dataset,
        configure_plot_style,
        export_plot,
        filter_excluded_temperatures,
        label_for,
        load_pyplot,
        resolve_input_paths,
        temperature_sort_key,
    )
    from .context import AnalysisContext
    from .iv_io import IvData, load_iv_data
    from .pr_plot import PrConfig, calculate_pr, load_pr_config
    from .pt_plot import PtFitResult, build_pc_points, fit_pc_temperature
except ImportError:
    from common import (
        color_for_dataset,
        configure_plot_style,
        export_plot,
        filter_excluded_temperatures,
        label_for,
        load_pyplot,
        resolve_input_paths,
        temperature_sort_key,
    )
    from context import AnalysisContext
    from iv_io import IvData, load_iv_data
    from pr_plot import PrConfig, calculate_pr, load_pr_config
    from pt_plot import PtFitResult, build_pc_points, fit_pc_temperature


def calculate_t_tes(
    p_b_w: np.ndarray,
    temperature_mK: float,
    g0_W_per_K: float,
    n: float,
) -> np.ndarray:
    t_bath_K = temperature_mK * 1e-3
    t_tes_power = n * p_b_w / g0_W_per_K + t_bath_K**n
    valid = np.isfinite(t_tes_power) & (t_tes_power >= 0)
    t_tes_K = np.full_like(p_b_w, np.nan, dtype=float)
    t_tes_K[valid] = t_tes_power[valid] ** (1.0 / n)
    return t_tes_K


def build_rt_curve(
    iv_data: IvData,
    config: PrConfig,
    raw_voltage: bool,
    g0_W_per_K: float,
    n: float,
) -> tuple[np.ndarray, np.ndarray]:
    if iv_data.temperature_mK is None:
        raise ValueError("Cannot calculate T_TES without a bath temperature.")

    r_tes_ohm, p_b_w = calculate_pr(iv_data, config, raw_voltage)
    t_tes_K = calculate_t_tes(p_b_w, iv_data.temperature_mK, g0_W_per_K, n)
    valid = np.isfinite(r_tes_ohm) & np.isfinite(t_tes_K)
    valid &= r_tes_ohm >= 0
    return t_tes_K[valid], r_tes_ohm[valid]


def plot_rt(
    plt,
    datasets: list[tuple[Path, IvData]],
    config: PrConfig,
    raw_voltage: bool,
    g0_W_per_K: float,
    n: float,
):
    configure_plot_style(plt)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    for index, (path, iv_data) in enumerate(datasets):
        t_tes_K, r_tes_ohm = build_rt_curve(iv_data, config, raw_voltage, g0_W_per_K, n)
        ax.plot(
            t_tes_K * 1e3,
            r_tes_ohm * 1e3,
            linestyle="None",
            marker="o",
            markersize=3.5,
            color=color_for_dataset(plt, index, len(datasets), iv_data),
            label=label_for(path, iv_data),
        )

    ax.set_xlabel(r"$T_{TES}$ (mK)", fontsize=13)
    ax.set_ylabel(r"$R_{TES}$ (m$\Omega$)", fontsize=13)
    ax.set_title(r"$R_{TES}$-$T_{TES}$", fontsize=18)
    ax.grid(True, linestyle="--", color="0.65", alpha=0.85)
    ax.legend(loc="best", framealpha=0.9, fontsize="small")
    return fig


def get_pt_fit_for_rt(
    args: argparse.Namespace,
    datasets: list[tuple[Path, IvData]],
    config: PrConfig,
    context: AnalysisContext,
) -> tuple[PtFitResult, bool]:
    if context.pt_fit_result is not None:
        return context.pt_fit_result, True

    pc_points = build_pc_points(datasets, config, args.raw_voltage, args.ratio)
    return fit_pc_temperature(pc_points), False


def run_rt_step(
    args: argparse.Namespace, context: AnalysisContext | None = None
) -> int:
    if context is None:
        context = AnalysisContext()

    try:
        input_paths = resolve_input_paths(args.inputs)
        datasets = [(path, load_iv_data(path)) for path in input_paths]
        datasets.sort(key=temperature_sort_key)
        datasets = filter_excluded_temperatures(datasets, args.exclude)

        config = load_pr_config(args)
        fit, reused_pt_fit = get_pt_fit_for_rt(args, datasets, config, context)

        plt = load_pyplot(args.show)
        fig = plot_rt(plt, datasets, config, args.raw_voltage, fit.g0_W_per_K, fit.n)
        output_path = export_plot(fig, args.output_dir, "rt_rtes_ttes.png")
        print(f"Saved RT plot: {output_path}")
        source = "reused PT step fit" if reused_pt_fit else "new PT fit"
        print(
            f"Using {source}: Tc = {fit.tc_K * 1e3:.6g} mK, "
            f"G0 = {fit.g0_W_per_K:.6g} W/K, n = {fit.n:.6g}"
        )

        if args.show:
            plt.show()
        else:
            plt.close(fig)

        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
