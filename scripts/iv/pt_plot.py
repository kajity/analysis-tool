from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

try:
    from .common import (
        color_for_dataset,
        configure_plot_style,
        export_plot,
        filter_excluded_temperatures,
        load_pyplot,
        resolve_input_paths,
        temperature_sort_key,
    )
    from .context import AnalysisContext
    from .iv_io import IvData, load_iv_data
    from .pr_plot import PrConfig, calculate_pr, load_pr_config, normalize_resistance
except ImportError:
    from common import (
        color_for_dataset,
        configure_plot_style,
        export_plot,
        filter_excluded_temperatures,
        load_pyplot,
        resolve_input_paths,
        temperature_sort_key,
    )
    from context import AnalysisContext
    from iv_io import IvData, load_iv_data
    from pr_plot import PrConfig, calculate_pr, load_pr_config, normalize_resistance


@dataclass(frozen=True)
class CriticalPowerPoint:
    temperature_mK: float
    pc_w: float
    iv_data: IvData


@dataclass(frozen=True)
class PtFitResult:
    tc_K: float
    g0_W_per_K: float
    n: float
    tc_K_err: float | None = None
    g0_W_per_K_err: float | None = None
    n_err: float | None = None


def thermal_model(
    t_bath_K: np.ndarray, tc_K: float, g0_W_per_K: float, n: float
) -> np.ndarray:
    return g0_W_per_K / n * (tc_K**n - t_bath_K**n)


def thermal_model_pW(
    t_bath_K: np.ndarray, tc_K: float, g0_W_per_K: float, n: float
) -> np.ndarray:
    return thermal_model(t_bath_K, tc_K, g0_W_per_K, n) * 1e12


def covariance_standard_error(
    covariance: np.ndarray, parameter_index: int
) -> float | None:
    try:
        variance = float(covariance[parameter_index, parameter_index])
    except (IndexError, TypeError, ValueError):
        return None

    if not np.isfinite(variance) or variance < 0:
        return None
    return float(np.sqrt(variance))


def format_value_with_error(
    value: float,
    error: float | None,
    *,
    scale: float = 1.0,
    unit: str = "",
    precision: int = 6,
) -> str:
    scaled_value = value * scale
    suffix = f" {unit}" if unit else ""
    if error is None:
        return f"{scaled_value:.{precision}g}{suffix}"

    scaled_error = error * scale
    if not np.isfinite(scaled_error):
        return f"{scaled_value:.{precision}g}{suffix}"
    return f"{scaled_value:.{precision}g} +/- {scaled_error:.{precision}g}{suffix}"


def format_fit_result_lines(fit: PtFitResult, precision: int = 6) -> list[str]:
    return [
        "Tc = "
        + format_value_with_error(
            fit.tc_K, fit.tc_K_err, scale=1e3, unit="mK", precision=precision
        ),
        "G0 = "
        + format_value_with_error(
            fit.g0_W_per_K,
            fit.g0_W_per_K_err,
            unit="W/K",
            precision=precision,
        ),
        "n = " + format_value_with_error(fit.n, fit.n_err, precision=precision),
    ]


def latex_number(value: float, precision: int = 4) -> str:
    if not np.isfinite(value):
        return str(value)

    if value == 0:
        return "0"

    abs_value = abs(value)
    if 1e-3 <= abs_value < 1e4:
        return f"{value:.{precision}g}"

    mantissa, exponent = f"{value:.{precision}e}".split("e")
    return rf"{float(mantissa):.{precision}g}\times10^{{{int(exponent)}}}"


def latex_value_with_error(
    value: float,
    error: float | None,
    *,
    scale: float = 1.0,
    unit: str = "",
    precision: int = 4,
) -> str:
    scaled_value = value * scale
    unit_text = rf"\ \mathrm{{{unit}}}" if unit else ""
    if error is None:
        return rf"{latex_number(scaled_value, precision)}{unit_text}"

    scaled_error = error * scale
    if not np.isfinite(scaled_error):
        return rf"{latex_number(scaled_value, precision)}{unit_text}"
    return (
        rf"{latex_number(scaled_value, precision)}"
        rf"\pm{latex_number(scaled_error, precision)}{unit_text}"
    )


def interpolate_pc_at_resistance_ratio(
    ratio: np.ndarray,
    p_b_w: np.ndarray,
    target_ratio: float,
) -> float:
    order = np.argsort(ratio)
    sorted_r = ratio[order]
    sorted_p = p_b_w[order]

    unique_r, unique_indices = np.unique(sorted_r, return_index=True)
    unique_p = sorted_p[unique_indices]
    if unique_r.size < 2:
        raise ValueError(
            "Need at least two distinct R_TES/R_N points to interpolate Pc."
        )

    if target_ratio < unique_r[0] or target_ratio > unique_r[-1]:
        raise ValueError(
            f"Target R_TES/R_N={target_ratio:g} is outside calculated range "
            f"{unique_r[0]:g}-{unique_r[-1]:g}."
        )

    return float(np.interp(target_ratio, unique_r, unique_p))


def build_pc_points(
    datasets,
    config: PrConfig,
    raw_voltage: bool,
    target_ratio: float,
    min_points: int = 3,
) -> list[CriticalPowerPoint]:
    points: list[CriticalPowerPoint] = []
    for path, iv_data in datasets:
        if iv_data.temperature_mK is None:
            raise ValueError(f"Could not determine bath temperature for {path}.")

        r_tes_ohm, p_b_w = calculate_pr(iv_data, config, raw_voltage)
        ratio = normalize_resistance(r_tes_ohm)
        pc_w = interpolate_pc_at_resistance_ratio(ratio, p_b_w, target_ratio)
        points.append(
            CriticalPowerPoint(
                temperature_mK=iv_data.temperature_mK, pc_w=pc_w, iv_data=iv_data
            )
        )

    if len(points) < min_points:
        raise ValueError(
            f"Need at least {min_points} temperature points to fit this model."
        )
    return points


def fit_pc_temperature(points: list[CriticalPowerPoint]) -> PtFitResult:
    t_bath_K = np.asarray(
        [point.temperature_mK * 1e-3 for point in points], dtype=float
    )
    pc_pW = np.asarray([point.pc_w * 1e12 for point in points], dtype=float)

    initial_tc = float(np.max(t_bath_K) * 1.05)
    initial_n = 3.0
    denominator = max(
        initial_tc**initial_n - float(np.min(t_bath_K)) ** initial_n,
        np.finfo(float).eps,
    )
    initial_g0 = float(initial_n * np.max(pc_pW) * 1e-12 / denominator)

    lower_bounds = [float(np.max(t_bath_K) * 1.0001), 0.0, 0.5]
    upper_bounds = [1.0, np.inf, 10.0]
    popt, pcov = curve_fit(
        thermal_model_pW,
        t_bath_K,
        pc_pW,
        p0=[initial_tc, initial_g0, initial_n],
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )
    return PtFitResult(
        tc_K=float(popt[0]),
        g0_W_per_K=float(popt[1]),
        n=float(popt[2]),
        tc_K_err=covariance_standard_error(pcov, 0),
        g0_W_per_K_err=covariance_standard_error(pcov, 1),
        n_err=covariance_standard_error(pcov, 2),
    )


def plot_pt(
    plt, points: list[CriticalPowerPoint], fit: PtFitResult, target_ratio: float
):
    configure_plot_style(plt)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    sorted_points = sorted(points, key=lambda point: point.temperature_mK)
    for index, point in enumerate(sorted_points):
        ax.plot(
            point.temperature_mK,
            point.pc_w * 1e12,
            linestyle="None",
            marker="o",
            markersize=7,
            color=color_for_dataset(plt, index, len(sorted_points)),
            label=f"{point.temperature_mK:g} mK",
        )

    t_bath_mK = np.asarray(
        [point.temperature_mK for point in sorted_points], dtype=float
    )
    fit_t_mK = np.linspace(float(np.min(t_bath_mK)), float(np.max(t_bath_mK)), 300)
    fit_pc_pW = thermal_model(fit_t_mK * 1e-3, fit.tc_K, fit.g0_W_per_K, fit.n) * 1e12
    ax.plot(fit_t_mK, fit_pc_pW, linewidth=2, label="fit")

    ax.set_xlabel(r"$T_{bath}$ (mK)", fontsize=13)
    ax.set_ylabel(r"$P_c$ (pW)", fontsize=13)
    ax.set_title(r"$P_c$-$T_{bath}$", fontsize=18)
    ax.grid(True, linestyle="--", color="0.65", alpha=0.85)
    ax.legend(loc="best", framealpha=0.9, fontsize="small")
    ax.text(
        0.02,
        0.02,
        (
            rf"$R_{{TES}}/R_N={target_ratio:g}$"
            "\n"
            rf"$T_c={latex_value_with_error(fit.tc_K, fit.tc_K_err, scale=1e3, unit='mK')}$"
            "\n"
            rf"$G_0={latex_value_with_error(fit.g0_W_per_K, fit.g0_W_per_K_err, unit='W/K')}$"
            "\n"
            rf"$n={latex_value_with_error(fit.n, fit.n_err)}$"
        ),
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
    )
    return fig


def export_fit_summary(
    output_dir: Path,
    fit: PtFitResult,
    points: list[CriticalPowerPoint],
    target_ratio: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pt_fit_parameters.txt"
    lines = [
        f"R_TES/R_N target = {target_ratio:g}",
        "Tc = "
        + format_value_with_error(
            fit.tc_K, fit.tc_K_err, scale=1e3, unit="[mK]", precision=12
        ),
        "G0 = "
        + format_value_with_error(
            fit.g0_W_per_K,
            fit.g0_W_per_K_err,
            unit="[W/K]",
            precision=12,
        ),
        "n = " + format_value_with_error(fit.n, fit.n_err, precision=12),
        "",
        "Sampled Pc points:",
    ]
    for point in points:
        lines.append(
            f"T_bath = {point.temperature_mK:.12g} [mK], Pc = {point.pc_w:.12g} [W]"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_pt_step(
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
        points = build_pc_points(datasets, config, args.raw_voltage, args.ratio)
        fit = fit_pc_temperature(points)
        context.pt_fit_result = fit

        plt = load_pyplot(args.show)
        fig = plot_pt(plt, points, fit, args.ratio)
        output_path = export_plot(fig, args.output_dir, "pt_pc_fit.png")
        summary_path = export_fit_summary(args.output_dir, fit, points, args.ratio)
        print(f"Saved PT plot: {output_path}")
        print(f"Saved PT fit summary: {summary_path}")
        for line in format_fit_result_lines(fit):
            print(line)

        if args.show:
            plt.show()
        else:
            plt.close(fig)

        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
