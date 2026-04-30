# analysis-tool

Python-based analysis workflows for experimental data.

## Environment

This project is managed with `uv`.

Install or update the project environment:

```powershell
uv sync
```

Run the IV analysis commands:

```powershell
uv run python scripts\iv\main.py data\iv
uv run python scripts\iv\main.py iv data\iv
uv run python scripts\iv\main.py pr data\iv
uv run python scripts\iv\main.py pt data\iv
uv run python scripts\iv\main.py rt data\iv
```

Install the analysis commands as uv tools:

```powershell
uv tool install .
```

After installation, use:

```powershell
ates.iv data\iv
ates.iv rt data\iv -e 100 -e 110 --ratio 0.5
ates.thickness path\to\profile.txt
```

When the step is omitted, the script runs `iv`, `pr`, `pt`, and `rt` in order.
After the `iv` plot, it asks on the command line which temperatures should be
excluded from `pr` and later steps. You can also specify exclusions directly:

```powershell
uv run python scripts\iv\main.py data\iv -e 100 -e 110
uv run python scripts\iv\main.py pr data\iv -e 100 -e 110
```

The `iv` step plots current-voltage curves by temperature. The `pr` step
calculates and plots `P_b` versus normalized `R_TES`.
The `pt` step samples one `P_b` value per temperature at a target `R_TES/R_N`
and fits:

```text
Pc ~ G0 / n * (Tc^n - T_bath^n)
```

The target resistance ratio defaults to `0.5` and can be changed:

```powershell
uv run python scripts\iv\main.py pt data\iv --ratio 0.5 -e 100 -e 110
```

The `rt` step reuses the PR calculation for `R_TES` and `P_b`, fits the PT
model to obtain `G0` and `n`, then inverts:

```text
P_b = G0 / n * (T_TES^n - T_bath^n)
```

to plot `R_TES` versus `T_TES`.

PR parameters are loaded with OmegaConf from `scripts/iv/config/default.yaml`.
They can also be overridden from the command line:

```powershell
uv run python scripts\iv\main.py pr data\iv --r-sh 0.003 --r-fb 100000 --m-in 104.12e-12 --m-fb 85.21e-12
```

Scripts read raw input files from `data/` and write generated artifacts under
`outputs/`.

## Dependency Management

Add runtime dependencies with:

```powershell
uv add <package>
```

Update the lockfile after dependency changes with:

```powershell
uv lock
```
