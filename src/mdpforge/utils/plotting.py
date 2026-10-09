"""Plot benchmark results with the optional Matplotlib dependency."""

import numpy as np


def plot_heat(
    results,
    model_names,
    solver_names,
    *,
    reference=None,
    metric="speedup",
    ax=None,
    show=True,
):
    """Plot median runtimes or speedups, marking any failed repeat as FAIL."""
    if not results:
        raise ValueError("No results available; call run() before plot_heat()")
    if metric not in {"speedup", "runtime"}:
        raise ValueError("metric must be 'speedup' or 'runtime'")
    if reference is None:
        reference = next(iter(solver_names))
    if metric == "speedup" and reference not in solver_names:
        raise ValueError(f"Unknown reference solver: {reference!r}")

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    times = np.full((len(solver_names), len(model_names)), np.nan)
    failures = np.zeros_like(times, dtype=bool)
    for i, solver in enumerate(solver_names):
        for j, model in enumerate(model_names):
            rows = [
                row
                for row in results
                if row["solver"] == solver and row["model"] == model
            ]
            if not rows or any(row["status"] != "success" for row in rows):
                failures[i, j] = True
                continue
            times[i, j] = float(np.median([row["runtime"] for row in rows]))

    if metric == "speedup":
        ref_times = times[solver_names.index(reference)]
        with np.errstate(divide="ignore", invalid="ignore"):
            values = ref_times[np.newaxis, :] / times
            colors = np.log2(values)
        colors[~np.isfinite(colors)] = np.nan
        finite = np.abs(colors[np.isfinite(colors)])
        limit = max(1.0, float(finite.max())) if finite.size else 1.0
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        cmap = plt.get_cmap("RdYlGn").copy()
        color_label = f"log₂ speedup vs {reference}"
    else:
        values = times
        colors = times
        norm = None
        cmap = plt.get_cmap("viridis_r").copy()
        color_label = "Runtime (s)"

    cmap.set_bad("#e5e7eb")
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(
                max(6, 1.4 * len(model_names) + 2),
                max(3, 0.55 * len(solver_names) + 2),
            )
        )
    else:
        fig = ax.figure
    im = ax.imshow(np.ma.masked_invalid(colors), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(model_names)), model_names, rotation=35, ha="right")
    ax.set_yticks(range(len(solver_names)), solver_names)
    ax.set_xlabel("MDP")
    ax.set_ylabel("Solver")
    settings = results[0]
    ax.set_title(
        f"Runtime at fixed precision — γ={settings['discount']:g}, "
        f"ε={settings['epsilon']:g}"
    )
    for i in range(len(solver_names)):
        for j in range(len(model_names)):
            if failures[i, j]:
                label = "FAIL"
            elif not np.isfinite(values[i, j]):
                label = "N/A"
            elif metric == "speedup":
                label = f"{values[i, j]:.2f}×"
            else:
                label = f"{values[i, j]:.3g} s"
            ax.text(j, i, label, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label=color_label)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
