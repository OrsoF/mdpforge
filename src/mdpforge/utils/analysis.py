"""Aggregate and plot benchmark measurements with optional Matplotlib."""

from collections import Counter

import numpy as np


def _aggregate_rows(rows, experiment=None):
    """Aggregate eligible, fully successful trials by model and solver."""
    if not rows:
        raise ValueError("No benchmark measurements available")
    models = list(dict.fromkeys(row["model"] for row in rows))
    solvers = list(dict.fromkeys(row["solver"] for row in rows))
    settings = {
        (row.get("criterion", "discounted"), row.get("discount"), row.get("epsilon"))
        for row in rows
    }
    if len(settings) > 1:
        raise ValueError(
            "Cannot compare different criterion, discount or precision settings"
        )
    if len({row.get("experiment_id") for row in rows}) > 1:
        raise ValueError("Cannot compare multiple benchmark runs")

    grouped = {}
    by_model = {model: [] for model in models}
    for row in rows:
        grouped.setdefault((row["model"], row["solver"]), []).append(row)
        by_model[row["model"]].append(row)

    recorded_repeats = (experiment or {}).get("settings", {}).get("repeats")
    if recorded_repeats is not None and (
        type(recorded_repeats) is not int or recorded_repeats < 1
    ):
        raise ValueError("Manifest repeats must be a positive integer")

    aggregated = {}
    for model in models:
        model_rows = by_model[model]
        repeat_fields = {"repeat" in row for row in model_rows}
        if len(repeat_fields) > 1:
            raise ValueError(f"Inconsistent repeat fields for model {model!r}")
        has_repeats = True in repeat_fields
        if recorded_repeats is not None:
            expected = set(range(1, recorded_repeats + 1))
        elif has_repeats:
            # Legacy CSVs cannot prove that all intended repeats were recorded.
            expected = {row["repeat"] for row in model_rows}
        else:
            expected = max(len(grouped.get((model, solver), ())) for solver in solvers)

        for solver in solvers:
            group = grouped.get((model, solver), [])
            if has_repeats:
                actual = [row["repeat"] for row in group]
                present = len(actual) == len(set(actual)) and set(actual) == expected
            else:
                present = len(group) == (
                    len(expected) if isinstance(expected, set) else expected
                )
            valid = all(
                row["status"] == "success"
                and np.isfinite(row["runtime"])
                and row["runtime"] > 0
                for row in group
            )
            complete = bool(group) and present and valid
            aggregated[model, solver] = {
                "model": model,
                "solver": solver,
                "n_success": sum(row["status"] == "success" for row in group),
                "n_trials": len(group),
                "median_runtime": (
                    float(np.median([row["runtime"] for row in group]))
                    if complete
                    else np.nan
                ),
                "complete": complete,
            }
    return models, solvers, aggregated


def _performance_ratios(rows, experiment=None):
    """Return per-model ratios to the fastest eligible median runtime."""
    models, solvers, aggregated = _aggregate_rows(rows, experiment)
    ratios = np.full((len(models), len(solvers)), np.inf)
    for i, model in enumerate(models):
        runtimes = np.array(
            [aggregated[model, solver]["median_runtime"] for solver in solvers]
        )
        eligible = np.isfinite(runtimes)
        if eligible.any():
            ratios[i, eligible] = runtimes[eligible] / runtimes[eligible].min()
    return models, solvers, aggregated, ratios


def _scaling_statistics(rows, experiment, *, reference=None):
    """Aggregate paired instance medians by actual state dimension."""
    study = (experiment or {}).get("study")
    if not isinstance(study, dict) or study.get("type") != "scaling":
        raise ValueError("Scaling analysis requires a scaling experiment manifest")
    instances = study.get("models", {})
    sizes = study.get("sizes", [])
    if not instances or not sizes:
        raise ValueError("Scaling manifest has no registered models or sizes")

    observed, solvers, cells = _aggregate_rows(rows, experiment)
    if set(observed) - set(instances):
        raise ValueError(
            "Benchmark contains models not registered in the scaling study"
        )
    reference = solvers[0] if reference is None else reference
    if reference not in solvers:
        raise ValueError(f"Unknown reference solver: {reference!r}")

    by_size = {size: [] for size in sizes}
    requested_sizes = study.get("requested_state_dims", sizes)
    by_request = {size: [] for size in requested_sizes}
    model_metadata = experiment.get("models", {})
    for model, entry in instances.items():
        size = entry["state_dim"]
        if size not in by_size:
            raise ValueError(f"Unexpected state dimension {size} for {model!r}")
        metadata = model_metadata.get(model, {})
        if metadata.get("state_dim") != size or "nnz" not in metadata:
            raise ValueError(f"Missing dimension or nnz metadata for {model!r}")
        requested = entry.get("requested_state_dim", size)
        if requested not in by_request:
            raise ValueError(f"Unexpected requested state dimension {requested}")
        by_request[requested].append(model)
        by_size[size].append(model)
    if any(len(names) != study.get("instances") for names in by_request.values()):
        raise ValueError("Scaling manifest has an inconsistent instance count")
    if any(not names for names in by_size.values()):
        raise ValueError("Scaling manifest has an actual size without instances")

    def runtime(model, solver):
        return cells.get((model, solver), {}).get("median_runtime", np.nan)

    records = []
    for size in sorted(by_size):
        names = by_size[size]
        nnz = [model_metadata[name]["nnz"] for name in names]
        for solver in solvers:
            times = np.array([runtime(name, solver) for name in names], dtype=float)
            ref_times = np.array(
                [runtime(name, reference) for name in names], dtype=float
            )
            succeeded = np.isfinite(times)
            paired = succeeded & np.isfinite(ref_times)
            valid = times[succeeded]
            records.append(
                {
                    "state_dim": size,
                    "nnz": float(np.median(nnz)),
                    "solver": solver,
                    "n_instances": len(names),
                    "n_success": int(succeeded.sum()),
                    "n_failed": int((~succeeded).sum()),
                    "success_rate": float(succeeded.mean()),
                    "median_runtime": float(np.median(valid)) if valid.size else np.nan,
                    "q25_runtime": float(np.quantile(valid, 0.25))
                    if valid.size
                    else np.nan,
                    "q75_runtime": float(np.quantile(valid, 0.75))
                    if valid.size
                    else np.nan,
                    "n_compared": int(paired.sum()),
                    "geomean_speedup": float(
                        np.exp(
                            np.mean(np.log(ref_times[paired]) - np.log(times[paired]))
                        )
                    )
                    if paired.any()
                    else np.nan,
                }
            )
    return records


def _ordered_solver_labels(solver_names, experiment):
    """Order registered solver IDs and resolve their captured display names."""
    records = (experiment or {}).get("solvers", {})
    metadata = {name: record.get("metadata") or {} for name, record in records.items()}
    solver_labels = {
        name: fields.get("display_name") or name for name, fields in metadata.items()
    }

    def solver_order(name):
        fields = metadata.get(name, {})
        family = {"vi": 0, "pi": 1, "mpi": 2, "lp": 3, "experimental": 4}.get(
            fields.get("family"), 5
        )
        if family == 5:
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
    counts = Counter(labels)
    seen = Counter()
    for index, label in enumerate(labels):
        if counts[label] > 1:
            seen[label] += 1
            labels[index] = f"{label} [{seen[label]}]"
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
    Duplicate display labels are numbered [1], [2], etc.
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
    model_metadata = (experiment or {}).get("models", {})
    model_labels = [
        model_metadata.get(name, {}).get("display_name", name) for name in model_names
    ]
    ax.set_xticks(range(len(model_names)), model_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(solver_names)), labels)
    ax.set_xlabel("MDP")
    ax.set_ylabel("Solver")
    settings = results[0]
    if settings.get("criterion", "discounted") == "total":
        ax.set_title(f"Total reward runtime — ε={settings['epsilon']:g}")
    else:
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

    ink, muted, grid = "#142c40", "#526879", "#dae4ec"
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": ink,
        "axes.labelcolor": ink,
        "xtick.color": muted,
        "ytick.color": muted,
        "axes.edgecolor": grid,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#f8fafc",
        "svg.fonttype": "none",
    }
    palettes = {"mdptoolbox": "Blues", "marmote": "Greens", "mdpsolver": "Oranges"}
    records = (experiment or {}).get("solvers", {})
    backends = {}
    groups = {}
    for solver in ordered:
        metadata = records.get(solver, {}).get("metadata") or {}
        backend = metadata.get("backend") or solver.split("_", 1)[0]
        backends[solver] = backend
        groups.setdefault(backend, []).append(solver)
    fallback_palettes = ("Purples", "Reds", "viridis", "cividis")
    for backend in groups:
        if backend not in palettes:
            palettes[backend] = fallback_palettes[0]
            fallback_palettes = fallback_palettes[1:] + fallback_palettes[:1]
    styles = ["-", "--", "-.", ":"]

    with plt.rc_context(style):
        own_axes = ax is None
        if own_axes:
            fig, ax = plt.subplots(figsize=(9, 5.2))
        else:
            fig = ax.figure
        for solver, label in zip(ordered, labels):
            backend = backends[solver]
            group = groups[backend]
            index = group.index(solver)
            palette = plt.get_cmap(palettes[backend])
            values = ratios[:, solvers.index(solver)]
            finite = values[np.isfinite(values) & (values <= max_factor)]
            factors = np.unique(np.r_[1.0, finite, max_factor])
            fractions = (values[:, None] <= factors[None, :]).mean(axis=0)
            ax.step(
                factors,
                fractions,
                where="post",
                linewidth=2,
                label=label,
                color=palette(0.45 + 0.5 * index / max(1, len(group) - 1)),
                linestyle=styles[index % len(styles)],
            )

        ax.set_facecolor("#ffffff")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xlim(1, max_factor)
        ticks = [
            factor for factor in (1, 2, 4, 8, 16, 32, 64, 100) if factor <= max_factor
        ]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda factor, _: f"{factor:g}×"))
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("Runtime / fastest eligible median", color=ink)
        ax.set_ylabel("Share of all benchmark models", color=ink)
        ax.set_title(
            "How often is a solver close to the fastest?",
            loc="left",
            fontsize=16,
            fontweight="bold",
            color=ink,
            pad=16,
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(color=grid, alpha=0.65)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=False,
            fontsize=9,
        )
        if own_axes:
            fig.text(
                0.02,
                0.02,
                f"{len(models)} models · {len(solvers)} solvers · equal model weights\n"
                "Failed / missing pairs stay in the denominator",
                fontsize=9,
                color=muted,
            )
            fig.tight_layout(rect=(0, 0.09, 1, 1))
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
