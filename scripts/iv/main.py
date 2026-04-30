#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from context import AnalysisContext
from common import resolve_input_paths
from iv_plot import run_iv_step
from pr_plot import run_pr_step
from pt_plot import run_pt_step
from rt_plot import run_rt_step

DEFAULT_OUTPUT_DIR = Path("outputs/iv")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.yaml"
STEP_NAMES = ("all", "iv", "pr", "pt", "rt")


def add_common_step_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help=(
            "Paths to IV text exports, directories, or glob patterns. "
            "Example: data/iv or data/iv/IV_*mK_*.txt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )


def add_iv_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Display plots interactively after saving them.",
    )
    parser.add_argument(
        "--raw-voltage",
        action="store_true",
        help="Use raw SqVout values instead of offsetting each curve to Vout=0 at the minimum current.",
    )


def add_pr_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"OmegaConf YAML file for PR parameters. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--r-sh", type=float, default=None, help="Override R_sh in ohm."
    )
    parser.add_argument(
        "--r-fb", type=float, default=None, help="Override R_FB in ohm."
    )
    parser.add_argument(
        "--m-in", type=float, default=None, help="Override M_in in henry."
    )
    parser.add_argument(
        "--m-fb", type=float, default=None, help="Override M_FB in henry."
    )


def add_post_iv_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-e",
        "--exclude",
        type=float,
        action="append",
        default=[],
        metavar="TEMP_MK",
        help="Exclude a temperature in mK from pr and later steps. Can be specified multiple times.",
    )


def add_pt_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=0.5,
        help="R_TES/R_N value used to sample one P_b value per temperature. Default: 0.5.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IV analysis steps.")
    subparsers = parser.add_subparsers(dest="step", metavar="step")

    iv_parser = subparsers.add_parser(
        "iv",
        help="Plot current-voltage curves by temperature.",
        description="Plot IV data with current on the x-axis and voltage on the y-axis.",
    )
    add_common_step_arguments(iv_parser)
    add_iv_options(iv_parser)
    iv_parser.set_defaults(handler=run_iv_step)

    pr_parser = subparsers.add_parser(
        "pr",
        help="Plot bias-power curves by normalized TES resistance.",
        description="Calculate P_b and normalized R_TES from IV data and plot P_b versus R_TES/R_N.",
    )
    add_common_step_arguments(pr_parser)
    add_pr_options(pr_parser)
    add_post_iv_options(pr_parser)
    add_iv_options(pr_parser)
    pr_parser.set_defaults(handler=run_pr_step)

    pt_parser = subparsers.add_parser(
        "pt",
        help="Fit critical power versus bath temperature.",
        description="Sample Pc from P_b at a target R_TES/R_N and fit Pc ~ G0 / n * (Tc^n - T_bath^n).",
    )
    add_common_step_arguments(pt_parser)
    add_pr_options(pt_parser)
    add_post_iv_options(pt_parser)
    add_iv_options(pt_parser)
    add_pt_options(pt_parser)
    pt_parser.set_defaults(handler=run_pt_step)

    rt_parser = subparsers.add_parser(
        "rt",
        help="Plot TES resistance versus TES temperature.",
        description="Calculate T_TES by inverting P_b = G0 / n * (T_TES^n - T_bath^n), then plot R_TES versus T_TES.",
    )
    add_common_step_arguments(rt_parser)
    add_pr_options(rt_parser)
    add_post_iv_options(rt_parser)
    add_iv_options(rt_parser)
    add_pt_options(rt_parser)
    rt_parser.set_defaults(handler=run_rt_step)

    all_parser = subparsers.add_parser(
        "all",
        help="Run iv, pr, pt, and rt in order.",
        description="Run all IV analysis steps in order: iv, pr, pt, rt.",
    )
    add_common_step_arguments(all_parser)
    add_pr_options(all_parser)
    add_post_iv_options(all_parser)
    add_iv_options(all_parser)
    add_pt_options(all_parser)
    all_parser.set_defaults(handler=run_all_steps)

    args_list = list(sys.argv[1:] if argv is None else argv)
    if (
        args_list
        and args_list[0] not in STEP_NAMES
        and args_list[0] not in ("-h", "--help")
    ):
        args_list.insert(0, "all")

    args = parser.parse_args(args_list)
    if args.step is None:
        parser.error("inputs are required when no step is specified.")
    return args


def parse_exclude_prompt(text: str) -> list[float]:
    values: list[float] = []
    for token in text.replace(",", " ").split():
        values.append(float(token))
    return values


def confirm_exclusions(args: argparse.Namespace) -> None:
    if args.exclude:
        temperatures = ", ".join(f"{temperature:g} mK" for temperature in args.exclude)
        print(f"Excluding temperatures from pr/pt/rt: {temperatures}")
        return

    try:
        answer = input(
            "Exclude temperatures from pr/pt/rt? "
            "Enter mK values separated by spaces or commas, or press Enter for none: "
        ).strip()
    except EOFError:
        print("No exclusion input received; continuing without exclusions.")
        return

    if not answer:
        print("No temperatures excluded from pr/pt/rt.")
        return

    try:
        args.exclude = parse_exclude_prompt(answer)
    except ValueError:
        print(
            "Invalid exclusion input; continuing without exclusions.", file=sys.stderr
        )
        args.exclude = []
        return

    temperatures = ", ".join(f"{temperature:g} mK" for temperature in args.exclude)
    print(f"Excluding temperatures from pr/pt/rt: {temperatures}")


def run_unimplemented_step(args: argparse.Namespace) -> int:
    try:
        resolve_input_paths(args.inputs)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Error: The '{args.step}' step is available as a subcommand, "
        "but its analysis logic is not implemented yet.",
        file=sys.stderr,
    )
    return 2


def run_all_steps(args: argparse.Namespace) -> int:
    context = AnalysisContext()
    first_error = 0
    print("Running step: iv")
    first_error = run_iv_step(args)
    confirm_exclusions(args)

    args.step = "pr"
    print("Running step: pr")
    result = run_pr_step(args)
    if result != 0 and first_error == 0:
        first_error = result

    args.step = "pt"
    print("Running step: pt")
    result = run_pt_step(args, context)
    if result != 0:
        if first_error == 0:
            first_error = result
        print(
            "Skipping step: rt because pt did not produce fit parameters.",
            file=sys.stderr,
        )
        args.step = "all"
        return first_error

    args.step = "rt"
    print("Running step: rt")
    result = run_rt_step(args, context)
    if result != 0 and first_error == 0:
        first_error = result

    args.step = "all"
    return first_error


def main() -> int:
    args = parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
