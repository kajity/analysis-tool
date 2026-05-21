---
name: matplotlib-gui-visual-inspection
description: "Analyze Matplotlib GUI applications by constructing them from a Python REPL, inspecting live Figure/Axes/widget objects, driving state changes, and saving representative screenshots. Use for Matplotlib Button/TextBox/Slider apps, plot layout debugging, visual regression checks, or target-specific adapters such as ates.pulse interactive mode."
---

# Matplotlib GUI Visual Inspection

## Overview

Use this skill when a Matplotlib GUI needs to be understood through its live objects and rendered output. The preferred loop is: construct the GUI without entering a blocking `plt.show()`, inspect the raw `Figure`, `Axes`, artists, and widgets from a REPL, drive a few states, then save screenshots for visual review.

The generic workflow applies to any Matplotlib GUI. This skill also includes an `ates.pulse` adapter because that app has a known workflow controller and representative states.

## Generic REPL Workflow

1. Read the GUI entry point and identify where the `Figure` is created, where widgets are stored, and where `plt.show()` is called.
2. Start Python/IPython from the repository root and import the GUI module directly instead of launching the CLI.
3. Use a non-blocking backend for headless inspection when needed:

```python
import matplotlib
matplotlib.use("Agg", force=True)
```

4. Construct the app object or call the setup function, but avoid calling `show()` until after inspection.
5. Get the live figure from an app/controller attribute, from a returned value, or from pyplot:

```python
import matplotlib.pyplot as plt

fig = app.fig              # common for GUI classes
fig = controller.ui.fig    # common for controller-owned UIs
fig = plt.figure(plt.get_fignums()[-1])
```

6. Inspect structure and state before guessing from source coordinates:

```python
fig.get_size_inches()
[(ax.get_position().bounds, ax.get_title()) for ax in fig.axes]
[(line.get_label(), len(line.get_xdata())) for ax in fig.axes for line in ax.lines]
[(text.get_text(), text.get_position()) for ax in fig.axes for text in ax.texts]
```

7. Drive state through the app's public methods or callback targets, then redraw:

```python
app.next(None)
fig.canvas.draw()
fig.savefig("outputs/matplotlib-gui-visual-inspection/state.png", dpi=150)
```

8. Repeat for the states that matter: initial view, each tab/step/mode, changed settings, empty data, error/status messages, and final/disabled states.

## Helper Script

Use `scripts/mpl_gui_probe.py` as a REPL helper for Figure summaries:

```python
import matplotlib.pyplot as plt
from pathlib import Path
from importlib.machinery import SourceFileLoader

probe = SourceFileLoader(
    "mpl_gui_probe",
    ".codex/skills/matplotlib-gui-visual-inspection/scripts/mpl_gui_probe.py",
).load_module()

summary = probe.summarize_figure(plt.gcf())
probe.write_summary(summary, Path("outputs/matplotlib-gui-visual-inspection/summary.json"))
probe.save_figure(plt.gcf(), Path("outputs/matplotlib-gui-visual-inspection/current.png"))
```

Prefer this helper when a GUI does not have a target-specific adapter yet. Add a small adapter script only after the REPL sequence is stable and worth repeating.

## Visual Review Checklist

- Figure and axes: expected number of axes, stable figure size, no unintended empty axes.
- Layout: controls, labels, titles, legends, and status text do not overlap or clip.
- Artists: expected lines, images, collections, patches, and text objects are present.
- Widgets: buttons, text boxes, sliders, check boxes, radio buttons, and selectors are reachable from app attributes or axes.
- State: selected/disabled controls look distinct, and callbacks update the rendered plot.
- Data: empty-data messages appear only when the underlying result is truly empty.
- Reproducibility: screenshots and summaries are written under `outputs/`, with enough metadata to rerun the same inspection.

## ates.pulse Adapter

For `ates.pulse`, the bundled adapter captures one screenshot per workflow step:

```bash
python3 .codex/skills/matplotlib-gui-visual-inspection/scripts/capture_pulse_gui.py
```

This creates a synthetic pulse HDF5 fixture and PNGs under:

```text
outputs/matplotlib-gui-visual-inspection/
```

Use a real pulse HDF5 file when available:

```bash
python3 .codex/skills/matplotlib-gui-visual-inspection/scripts/capture_pulse_gui.py path/to/pulse.hdf5
```

For `ates.pulse`, inspect these files first:

- `scripts/pulse/ui.py`: Figure, axes bounds, Button/TextBox widgets, and render orchestration.
- `scripts/pulse/workflow.py`: workflow state, callbacks, status text, config application, and interactive launch.
- `scripts/pulse/rendering.py`: stage-specific plot drawing.
- `scripts/pulse/pipeline.py`: stage names and cached analysis results.

Keep generated fixtures and screenshots under `outputs/`; never write generated files under `data/`.
