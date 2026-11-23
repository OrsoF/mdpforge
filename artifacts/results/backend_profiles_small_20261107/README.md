# Solver performance profiles, November 2026

The campaign ran on 2026-11-07 and covered 102 catalogue recipes and 17 solvers from MDPToolbox, Marmote and MDPSolver. It used `discount=0.99`, `precision=1e-3`, seed 0, a 20-second trial timeout and `validate=True`. It selected small presets where available and default dimensions otherwise. The [published figure](../../figures/gallery_performance_profiles.png) uses the first repeat only. The results reflect performance on the recorded machine, and a single repeat gives no estimate of timing variability.

At 2×, a curve shows the share of recipes that the solver completed within twice the fastest eligible runtime among all 17 solvers. Each panel uses this same reference. An eligible run completed and passed validation in the selected repeat. Failed and missing pairs stay in the denominator. The recipes include related model families and source variants.

The [campaign manifest](campaign.json) records model and solver selection, settings and exclusions. This directory contains each model's trial CSV and .experiment.json manifest with dimensions, solver options, versions and machine information. [coverage.csv](coverage.csv) contains the summaries used by the plot.

## Replot the saved curves

From the repository root, install the notebook dependencies with `python -m pip install -e ".[notebooks]"` and run the code below. It reads the saved summaries and includes every recipe in the denominator. Colors and layout may differ from the published figure.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

source = Path("artifacts/results/backend_profiles_small_20261107")
campaign = json.loads((source / "campaign.json").read_text())
coverage = pd.read_csv(source / "coverage.csv")
times = coverage.pivot(
    index="catalogue_model", columns="solver", values="median_runtime"
).reindex(index=campaign["models"], columns=campaign["solvers"])
times = times.fillna(np.inf)
ratios = times.div(times.min(axis=1), axis=0).fillna(np.inf)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, backend in zip(axes, ["mdptoolbox", "marmote", "mdpsolver"]):
    for solver in ratios.columns:
        if solver.startswith(backend + "_"):
            values = ratios[solver].to_numpy()
            factors = np.unique(
                np.r_[1, values[np.isfinite(values) & (values <= 100)], 100]
            )
            shares = (values[:, None] <= factors).mean(axis=0)
            ax.step(factors, shares, where="post", label=solver)
    ax.set(xscale="log", xlim=(1, 100), ylim=(0, 1.04), title=backend)
    ax.set_xlabel("Runtime / fastest certified runtime")
    ax.legend(fontsize=7)
axes[0].set_ylabel("Share of all catalogue recipes")
plt.tight_layout()
plt.show()
```
