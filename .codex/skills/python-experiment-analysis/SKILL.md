---
name: python-experiment-analysis
description: Build reproducible Python analysis workflows for experimental data in this repository. Use when Codex needs to inspect raw lab exports such as `.txt`, `.csv`, or instrument-specific text files under `data/`, determine their structure, create or extend scripts under `scripts/`, generate plots or summary tables, or scaffold a new analysis task without modifying raw data.
---

# Python Experiment Analysis

## Overview

Turn raw experimental exports into reproducible Python scripts instead of one-off manual steps.
Use this skill to inspect file structure, implement parsers, keep scientific assumptions explicit, and save generated outputs separately from raw data.

## Workflow

1. Inspect representative files in `data/` before writing code.
2. Identify delimiter, header rows, units, missing-value conventions, and whether the file is row- or column-oriented.
3. Create or extend `scripts/<analysis-name>/main.py` and keep parsing, analysis, plotting, and export logic separated.
4. Save generated artifacts under `outputs/<analysis-name>/` or another explicit generated-data directory.
5. Run the script against at least one real file and check for obvious scale, unit, and indexing mistakes.

## Operating Rules

- Never edit or overwrite files under `data/`.
- Prefer command-line entry points with explicit input and output paths.
- Make unit conversions, smoothing windows, baseline corrections, and fitting ranges explicit parameters or named constants.
- Add short comments or docstrings where assumptions affect scientific interpretation.
- If interactive plotting is useful, preserve a non-interactive path for reproducibility when possible.

## Scaffolding

When a new analysis area is needed, run:

```bash
python3 .codex/skills/python-experiment-analysis/scripts/init_analysis.py <analysis-name> --root .
```

This creates `scripts/<analysis-name>/main.py` and `outputs/<analysis-name>/.gitkeep`.

## References

- Read `references/project-conventions.md` when choosing output layout, parser structure, or plotting conventions.
- Use `scripts/init_analysis.py` when the repository needs a fresh analysis entry point instead of an ad hoc file.
