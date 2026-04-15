# Project Conventions

## Repository Shape

- Keep raw instrument exports in `data/`.
- Put runnable analysis entry points in `scripts/<analysis-name>/main.py`.
- Put generated figures, tables, and intermediate exports in `outputs/<analysis-name>/`.

## Parser Checklist

Read a small sample before coding.

- Confirm text encoding.
- Confirm delimiter and decimal format.
- Count header and metadata lines.
- Identify whether units are embedded in headers or separate rows.
- Check for empty lines, comments, sentinel values, or trailing delimiters.
- Verify whether columns are fixed-width or delimited.

## Analysis Structure

Prefer this split inside each analysis script:

1. `parse_args()`
2. `load_*()` functions for raw input
3. pure transformation/statistics functions
4. `plot_*()` and `export_*()` helpers
5. `main()`

Keep file-system side effects inside `main()` or dedicated export helpers.

## Scientific Hygiene

- Record unit conversions explicitly.
- Make smoothing, baseline correction, clipping, and fit windows parameterized.
- Avoid hidden constants; promote them to named variables or CLI flags.
- Log enough metadata to reproduce figures and summary tables later.

## Plotting

- Label axes with units.
- Put the sample or run identifier in figure titles or output filenames.
- Save plots to disk when they are part of the workflow, even if interactive display is also used.

## Current Repository Context

As of April 15, 2026, the repository contains Dektak data under `data/dektak/20260415/` and an early interactive prototype at `scripts/thickness/main.py`.
Treat that script as a starting point, not as a fixed architecture.
