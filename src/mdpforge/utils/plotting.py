"""Plot benchmark results with the optional Matplotlib dependency."""

import numpy as np


def plot_heat(
    results,
    experiment=None,
    *,
    reference=None,
    metric="speedup",
    ax=None,
    show=True,
):
    """Plot median runtimes or speedups, marking failed repeats as FAIL.

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
    model_names = list(dict.fromkeys(row["model"] for row in results))
    solver_names = list(dict.fromkeys(row["solver"] for row in results))
    if reference is None:
        reference = solver_names[0]
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
    if metric == "speedup" and reference not in solver_names:
        raise ValueError(f"Unknown reference solver: {reference!r}")
    labels = [solver_labels.get(name, name) for name in solver_names]
    labels = [
        f"{label} [{name}]" if labels.count(label) > 1 else label
        for name, label in zip(solver_names, labels)
    ]

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
