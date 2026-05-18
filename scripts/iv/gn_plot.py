from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

try:
    from .common import (
        configure_plot_style,
        export_plot,
        filter_excluded_temperatures,
        load_pyplot,
        resolve_input_paths,
        temperature_sort_key,
    )
    from .iv_io import load_iv_data
    from .pt_plot import (
        CriticalPowerPoint,
        PtFitResult,
        build_pc_points,
        covariance_standard_error,
        format_value_with_error,
        thermal_model_pW,
    )
    from .pr_plot import load_pr_config
except ImportError:
    from common import (
        configure_plot_style,
        export_plot,
        filter_excluded_temperatures,
        load_pyplot,
        resolve_input_paths,
        temperature_sort_key,
    )
    from iv_io import load_iv_data
    from pt_plot import (
        CriticalPowerPoint,
        PtFitResult,
        build_pc_points,
        covariance_standard_error,
        format_value_with_error,
        thermal_model_pW,
    )
    from pr_plot import load_pr_config


@dataclass(frozen=True)
class GnFitPoint:
    n: float
    tc_K: float
    g0_W_per_K: float
    g_at_tc_W_per_K: float
    tc_K_err: float | None = None
    g0_W_per_K_err: float | None = None
    g_at_tc_W_per_K_err: float | None = None


def calculate_g_at_tc(g0_W_per_K: float, tc_K: float, n: float) -> float:
    return float(g0_W_per_K * tc_K ** (n - 1.0))


def calculate_g_at_tc_error(
    g0_W_per_K: float, tc_K: float, n: float, covariance: np.ndarray
) -> float | None:
    if covariance.shape != (2, 2):
        return None

    d_g_d_tc = 0.0
    if n != 1.0:
        d_g_d_tc = g0_W_per_K * (n - 1.0) * tc_K ** (n - 2.0)
    d_g_d_g0 = tc_K ** (n - 1.0)

    jacobian = np.asarray([d_g_d_tc, d_g_d_g0], dtype=float)
    variance = float(jacobian @ covariance @ jacobian)
    if not np.isfinite(variance) or variance < 0:
        return None
    return float(np.sqrt(variance))


def format_sigfig_value_with_error(
    value: float, error: float | None, *, scale: float = 1.0, unit: str = ""
) -> str:
    scaled_value = value * scale
    suffix = rf"\ \mathrm{{{unit}}}" if unit else ""
    if error is None or not np.isfinite(error):
        return rf"{scaled_value:.4g}{suffix}"

    scaled_error = error * scale
    if scaled_error == 0 or not np.isfinite(scaled_error):
        return rf"{scaled_value:.4g}{suffix}"

    exponent = int(np.floor(np.log10(abs(scaled_error))))
    first_digit = int(abs(scaled_error) / 10**exponent)
    error_sigfigs = 2 if first_digit in (1, 2) else 1
    decimals = max(0, -(exponent - error_sigfigs + 1))
    return (
        rf"{scaled_value:.{decimals}f}" rf"\pm{scaled_error:.{decimals}f}" rf"{suffix}"
    )


def fit_pc_temperature_fixed_n(
    points: list[CriticalPowerPoint], n: float
) -> GnFitPoint:
    if n <= 0:
        raise ValueError("Fixed n values must be positive.")

    t_bath_K = np.asarray(
        [point.temperature_mK * 1e-3 for point in points], dtype=float
    )
    pc_pW = np.asarray([point.pc_w * 1e12 for point in points], dtype=float)

    initial_tc = float(np.max(t_bath_K) * 1.05)
    denominator = max(
        initial_tc**n - float(np.min(t_bath_K)) ** n,
        np.finfo(float).eps,
    )
    initial_g0 = float(n * np.max(pc_pW) * 1e-12 / denominator)

    lower_bounds = [float(np.max(t_bath_K) * 1.0001), 0.0]
    upper_bounds = [1.0, np.inf]
    popt, pcov = curve_fit(
        lambda t_bath_K, tc_K, g0_W_per_K: thermal_model_pW(
            t_bath_K, tc_K, g0_W_per_K, n
        ),
        t_bath_K,
        pc_pW,
        p0=[initial_tc, initial_g0],
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )

    tc_K = float(popt[0])
    g0_W_per_K = float(popt[1])
    return GnFitPoint(
        n=float(n),
        tc_K=tc_K,
        g0_W_per_K=g0_W_per_K,
        g_at_tc_W_per_K=calculate_g_at_tc(g0_W_per_K, tc_K, n),
        tc_K_err=covariance_standard_error(pcov, 0),
        g0_W_per_K_err=covariance_standard_error(pcov, 1),
        g_at_tc_W_per_K_err=calculate_g_at_tc_error(g0_W_per_K, tc_K, n, pcov),
    )


def resolve_n_values(args: argparse.Namespace) -> np.ndarray:
    if args.n_values:
        n_values = np.asarray(args.n_values, dtype=float)
    else:
        if args.n_count < 2:
            raise ValueError("--n-count must be at least 2.")
        if args.n_min <= 0 or args.n_max <= 0:
            raise ValueError("--n-min and --n-max must be positive.")
        if args.n_min >= args.n_max:
            raise ValueError("--n-min must be smaller than --n-max.")
        n_values = np.linspace(args.n_min, args.n_max, args.n_count)

    if np.any(~np.isfinite(n_values)) or np.any(n_values <= 0):
        raise ValueError("n values must be finite positive numbers.")
    return np.unique(n_values)


def fit_gn_points(
    points: list[CriticalPowerPoint], n_values: np.ndarray
) -> list[GnFitPoint]:
    if len(points) < 2:
        raise ValueError("Need at least two temperature points to fit Tc and G0.")
    return [fit_pc_temperature_fixed_n(points, float(n)) for n in n_values]


def plot_gn(
    plt,
    fit_points: list[GnFitPoint],
    target_ratio: float,
):
    configure_plot_style(plt)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    n_values = np.asarray([point.n for point in fit_points], dtype=float)
    g_values_pW_per_K = np.asarray(
        [point.g_at_tc_W_per_K * 1e12 for point in fit_points], dtype=float
    )
    tc_values_mK = np.asarray([point.tc_K * 1e3 for point in fit_points], dtype=float)
    g_errors_pW_per_K = np.asarray(
        [
            (
                0.0
                if point.g_at_tc_W_per_K_err is None
                else point.g_at_tc_W_per_K_err * 1e12
            )
            for point in fit_points
        ],
        dtype=float,
    )
    norm = plt.Normalize(
        vmin=float(np.min(tc_values_mK)),
        vmax=float(np.max(tc_values_mK)),
    )
    colormap = plt.get_cmap("turbo")
    colors = colormap(norm(tc_values_mK))

    for n, g_value, g_error, color in zip(
        n_values, g_values_pW_per_K, g_errors_pW_per_K, colors
    ):
        ax.errorbar(
            n,
            g_value,
            yerr=g_error,
            linestyle="None",
            marker="o",
            markersize=5,
            color=color,
            ecolor=color,
            capsize=3,
            elinewidth=1,
        )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    scalar_mappable.set_array(tc_values_mK)
    colorbar = fig.colorbar(scalar_mappable, ax=ax)
    colorbar.set_label(r"Fitted $T_c$ (mK)", fontsize=12)

    ax.set_xlabel(r"$n$", fontsize=13)
    ax.set_ylabel(r"$G_0T_c^{n-1}$ (pW/K)", fontsize=13)
    ax.set_title(r"$G$-$n$", fontsize=18)
    ax.grid(True, linestyle="--", color="0.65", alpha=0.85)

    first = fit_points[0]
    last = fit_points[-1]
    ax.text(
        0.02,
        0.98,
        (
            rf"$R_{{TES}}/R_N={target_ratio:g}$"
            "\n"
            rf"$n={first.n:g}: T_c={format_sigfig_value_with_error(first.tc_K, first.tc_K_err, scale=1e3, unit='mK')}$"
            "\n"
            rf"$n={last.n:g}: T_c={format_sigfig_value_with_error(last.tc_K, last.tc_K_err, scale=1e3, unit='mK')}$"
        ),
        transform=ax.transAxes,
        fontsize=10,
        va="top",
    )
    return fig


def export_gn_summary(
    output_dir: Path,
    fit_points: list[GnFitPoint],
    target_ratio: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gn_fit_parameters.txt"
    lines = [
        f"R_TES/R_N target = {target_ratio:g}",
        "n, Tc [mK], G0 [W/K], G0*Tc^(n-1) [W/K]",
    ]
    for point in fit_points:
        lines.append(
            ", ".join(
                [
                    f"{point.n:.12g}",
                    format_value_with_error(
                        point.tc_K,
                        point.tc_K_err,
                        scale=1e3,
                        precision=12,
                    ),
                    format_value_with_error(
                        point.g0_W_per_K,
                        point.g0_W_per_K_err,
                        precision=12,
                    ),
                    format_value_with_error(
                        point.g_at_tc_W_per_K,
                        point.g_at_tc_W_per_K_err,
                        precision=12,
                    ),
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_gn_step(args: argparse.Namespace) -> int:
    try:
        input_paths = resolve_input_paths(args.inputs)
        datasets = [(path, load_iv_data(path)) for path in input_paths]
        datasets.sort(key=temperature_sort_key)
        datasets = filter_excluded_temperatures(datasets, args.exclude)

        config = load_pr_config(args)
        points = build_pc_points(
            datasets, config, args.raw_voltage, args.ratio, min_points=2
        )
        n_values = resolve_n_values(args)
        fit_points = fit_gn_points(points, n_values)

        plt = load_pyplot(args.show)
        fig = plot_gn(plt, fit_points, args.ratio)
        output_path = export_plot(fig, args.output_dir, "gn_g_at_tc_by_n.png")
        summary_path = export_gn_summary(args.output_dir, fit_points, args.ratio)
        print(f"Saved GN plot: {output_path}")
        print(f"Saved GN fit summary: {summary_path}")

        best = min(fit_points, key=lambda point: point.g_at_tc_W_per_K)
        fit = PtFitResult(
            tc_K=best.tc_K,
            g0_W_per_K=best.g0_W_per_K,
            n=best.n,
            tc_K_err=best.tc_K_err,
            g0_W_per_K_err=best.g0_W_per_K_err,
            n_err=None,
        )
        print("Lowest G0*Tc^(n-1) point:")
        print(
            "  G0*Tc^(n-1) = "
            f"{format_value_with_error(best.g_at_tc_W_per_K, best.g_at_tc_W_per_K_err, unit='W/K')}"
        )
        print(
            "  Tc = "
            f"{format_value_with_error(fit.tc_K, fit.tc_K_err, scale=1e3, unit='mK')}"
        )
        print(
            "  G0 = "
            f"{format_value_with_error(fit.g0_W_per_K, fit.g0_W_per_K_err, unit='W/K')}"
        )
        print(f"  n = {fit.n:.6g}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)

        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
