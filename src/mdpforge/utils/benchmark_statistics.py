"""Aggregate benchmark measurements for summaries and plots."""

import numpy as np


def _aggregate_rows(rows, experiment=None):
    """Aggregate eligible, fully successful trials by model and solver."""
    if not rows:
        raise ValueError("No benchmark measurements available")
    models = list(dict.fromkeys(row["model"] for row in rows))
    solvers = list(dict.fromkeys(row["solver"] for row in rows))
    if len({(row.get("discount"), row.get("epsilon")) for row in rows}) > 1:
        raise ValueError("Cannot compare different discount or precision settings")
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


