"""Plot benchmark results with the optional Matplotlib dependency."""

import numpy as np

from mdpforge.core.result import (
    _aggregate_rows,
    _performance_ratios,
    _scaling_statistics,
)


def _ordered_solver_labels(solver_names, experiment):
    """Order registered solver IDs and resolve their captured display names."""
    metadata = {
        name: record.get("metadata") or {}
        for name, record in (experiment or {}).get("solvers", {}).items()
    }
    solver_labels = {
        name: fields.get("display_name") or name for name, fields in metadata.items()
    }

    def solver_order(name):
        fields = metadata.get(name, {})
        family = {"vi": 0, "pi": 1, "mpi": 2, "lp": 3}.get(fields.get("family"), 4)
        if family == 4:
            return (family, "", "", ())
        parts = solver_labels[name].split(" · ")
        return (
            family,
            parts[0].casefold(),
            (fields.get("backend") or "").casefold(),
            tuple(part.casefold() for part in parts[2:]),
        )

    solver_names = sorted(solver_names, key=solver_order)
    labels = [solver_labels.get(name, name) for name in solver_names]
    labels = [
        f"{label} [{name}]" if labels.count(label) > 1 else label
        for name, label in zip(solver_names, labels)
    ]
    return solver_names, labels


def plot_heat(
    results,
    experiment=None,
    *,
    reference=None,
    metric="speedup",
    ax=None,
    show=True,
):
    """Plot median runtimes or speedups, marking incomplete pairs as FAIL.

    Model/solver names are derived from the measurements.
    Display labels and family/method/backend/variant sorting use the manifest.
    Without a manifest, technical IDs and their original order are preserved.
    Duplicate display labels are disambiguated with their registered IDs.
    Abbreviations in catalogue labels are explained beneath the plot.
    """
    if not results:
        raise ValueError("No results available; call run() before plot_heat()")
    if metric not in {"speedup", "runtime"}:
        raise ValueError("metric must be 'speedup' or 'runtime'")
    model_names, solver_names, aggregated = _aggregate_rows(results, experiment)
    reference = solver_names[0] if reference is None else reference
    if metric == "speedup" and reference not in solver_names:
        raise ValueError(f"Unknown reference solver: {reference!r}")
    solver_names, labels = _ordered_solver_labels(solver_names, experiment)

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    times = np.array(
        [
            [aggregated[model, solver]["median_runtime"] for model in model_names]
            for solver in solver_names
        ]
    )
    failures = np.array(
        [
            [not aggregated[model, solver]["complete"] for model in model_names]
            for solver in solver_names
        ]
    )

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
        color_label = f"log₂ speedup vs {labels[solver_names.index(reference)]}"
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
    ax.set_yticks(range(len(solver_names)), labels)
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
    notes = []
    for marker, explanation in (
        (" · GS", "GS = Gauss-Seidel"),
        (" · SOR", "SOR = successive over-relaxation"),
        (" · V₀=0", "V₀=0 = zero initialization"),
        ("QVI", "QVI = Q-value iteration"),
    ):
        if any(marker in label for label in labels):
            notes.append(explanation)
    if notes:
        lines = ["; ".join(notes[i : i + 2]) for i in range(0, len(notes), 2)]
        fig.text(0.5, 0.01, "\n".join(lines), ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.05 * len(lines), 1, 1))
    else:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_performance_profile(
    results, experiment=None, *, max_factor=100, ax=None, show=True
) -> tuple:
    """Plot model-weighted Dolan-Moré runtime performance profiles.

    max_factor: finite float greater than one; maximum displayed runtime ratio.
    ax: Matplotlib axes, or None to create a figure.
    show: True displays the figure; False only returns (figure, axes).
    """
    if (
        isinstance(max_factor, (bool, np.bool_))
        or not isinstance(max_factor, (int, float, np.integer, np.floating))
        or not np.isfinite(max_factor)
        or max_factor <= 1
    ):
        raise ValueError("max_factor must be finite and greater than one")
    models, solvers, _, ratios = _performance_ratios(results, experiment)
    ordered, labels = _ordered_solver_labels(solvers, experiment)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    for solver, label in zip(ordered, labels):
        values = ratios[:, solvers.index(solver)]
        finite = values[np.isfinite(values) & (values <= max_factor)]
        factors = np.unique(np.r_[1.0, finite, max_factor])
        fractions = (values[:, None] <= factors[None, :]).mean(axis=0)
        ax.step(factors, fractions, where="post", linewidth=2, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xlim(1, max_factor)
    ticks = [factor for factor in (1, 2, 4, 8, 16, 32, 64, 100) if factor <= max_factor]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda factor, _: f"{factor:g}×"))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Runtime / fastest eligible median")
    ax.set_ylabel("Fraction of all MDPs")
    ax.set_title(f"Performance profiles — {len(models)} MDPs")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_scaling(
    results,
    experiment=None,
    *,
    x="state_dim",
    metric="runtime",
    reference=None,
    ax=None,
    show=True,
):
    """Plot median instance runtimes or paired speedups for one size sweep.

    x: "state_dim" or "nnz" (median nonzero transition entries).
    metric: "runtime" (seconds) or "speedup" (geometric reference/solver).
    reference: solver ID; None uses the first solver in the measurements.
    ax: Matplotlib axes, or None to create a figure.
    show: True displays the figure; False returns (figure, axes) without showing.
    """
    if x not in {"state_dim", "nnz"}:
        raise ValueError("x must be 'state_dim' or 'nnz'")
    if metric not in {"runtime", "speedup"}:
        raise ValueError("metric must be 'runtime' or 'speedup'")
    stats = _scaling_statistics(results, experiment, reference=reference)
    solver_names = list(dict.fromkeys(row["solver"] for row in stats))
    names, labels = _ordered_solver_labels(solver_names, experiment)

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    field = "median_runtime" if metric == "runtime" else "geomean_speedup"
    for solver, label in zip(names, labels):
        records = sorted(
            (row for row in stats if row["solver"] == solver),
            key=lambda row: row[x],
        )
        xs = np.array([row[x] for row in records], dtype=float)
        ys = np.array([row[field] for row in records], dtype=float)
        (line,) = ax.plot(xs, ys, marker="o", label=label)
        for row in records:
            available = row["n_success"] if metric == "runtime" else row["n_compared"]
            if available < row["n_instances"]:
                caption = f"{available}/{row['n_instances']}"
                if np.isfinite(row[field]):
                    ax.annotate(
                        caption,
                        (row[x], row[field]),
                        xytext=(4, 6),
                        textcoords="offset points",
                        fontsize=8,
                        color=line.get_color(),
                    )
                else:
                    # Keep fully failed sizes visible even with logarithmic y.
                    ax.plot(
                        row[x],
                        0.04,
                        marker="x",
                        linestyle="None",
                        color=line.get_color(),
                        transform=ax.get_xaxis_transform(),
                    )
                    ax.annotate(
                        caption,
                        (row[x], 0.04),
                        xycoords=ax.get_xaxis_transform(),
                        xytext=(4, 4),
                        textcoords="offset points",
                        color=line.get_color(),
                        fontsize=8,
                    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("States" if x == "state_dim" else "Transition nonzeros (nnz)")
    if metric == "runtime":
        ax.set_ylabel("Median runtime per completed instance (s)")
    else:
        ax.set_ylabel("Paired geometric speedup vs reference")
        ax.axhline(1, linestyle="--", linewidth=1, color="0.5")
    ax.set_title("Solver scalability")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
