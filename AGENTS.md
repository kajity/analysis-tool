# AGENTS.md

## Purpose

This repository is for Python-based analysis of experimental data.
Treat `data/` as raw input and implement reproducible analysis code under `scripts/`.

## Project Layout

- `data/`: raw measurement files exported from instruments
- `scripts/`: analysis entry points grouped by task or instrument
- `.codex/skills/python-experiment-analysis/`: project-local skill for analysis workflow and scaffolding

## Working Rules

- Inspect a few representative files before writing parsers; do not assume delimiter, header length, encoding, or units.
- Never modify or overwrite files under `data/`.
- Keep parsing, transformation, plotting, and export logic separated so analyses stay testable.
- Prefer command-line entry points that accept explicit input paths and output directories.
- Save derived artifacts under `outputs/<analysis-name>/` or another clearly named generated-data directory; create it if needed.
- Document assumptions about columns, units, filters, baseline correction, and fitting windows in code comments or function docstrings when they affect scientific interpretation.
- When adding a new analysis workflow, prefer creating `scripts/<analysis-name>/main.py` and related helpers instead of growing unrelated scripts.
- If the task matches the project workflow, use the local skill `$python-experiment-analysis` at `.codex/skills/python-experiment-analysis`.

## Validation

- Run the analysis script against at least one real file from `data/`.
- Verify generated numbers and plots for obvious scale, unit, and indexing mistakes.
- If an analysis uses interactive selection, provide a non-interactive fallback or a clearly documented default path.
