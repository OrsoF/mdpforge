"""Measurements and provenance of one finite discounted MDP benchmark run."""

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mdpforge.utils.paths import BENCHMARKS_PATH

_FIELD_TYPES = {
    "state_dim": int,
    "action_dim": int,
    "repeat": int,
    "seed": int,
    "discount": float,
    "epsilon": float,
    "runtime": float,
    "residual": float,
    "error_bound": float,
}
_REQUIRED_FIELDS = {"model", "solver", "status", "runtime", "discount", "epsilon"}


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
    """Aggregate paired instance medians by requested state dimension."""
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
    model_metadata = experiment.get("models", {})
    for model, entry in instances.items():
        size = entry["state_dim"]
        if size not in by_size:
            raise ValueError(f"Unexpected state dimension {size} for {model!r}")
        metadata = model_metadata.get(model, {})
        if metadata.get("state_dim") != size or "nnz" not in metadata:
            raise ValueError(f"Missing dimension or nnz metadata for {model!r}")
        by_size[size].append(model)
    if any(len(names) != study.get("instances") for names in by_size.values()):
        raise ValueError("Scaling manifest has an inconsistent instance count")

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
                            np.mean(
                                np.log(ref_times[paired]) - np.log(times[paired])
                            )
                        )
                    )
                    if paired.any()
                    else np.nan,
                }
            )
    return records


@dataclass
class BenchmarkResult:
    """A run's rows and captured experiment metadata, independent of Benchmark.

    rows: trial dictionaries, with numerical values and technical solver IDs.
    experiment: captured manifest, or None for a CSV without a companion manifest.
    """

    rows: list[dict]
    experiment: dict | None = None

    def to_dataframe(self):
        """Return raw trials as a pandas DataFrame (pandas optional)."""
        import pandas as pd

        return pd.DataFrame(self.rows)

    def summary(self):
        """Return per-model/solver trial counts and eligible median runtimes."""
        import pandas as pd

        models, solvers, aggregated = _aggregate_rows(self.rows, self.experiment)
        return pd.DataFrame(
            [aggregated[model, solver] for model in models for solver in solvers]
        )

    def solver_summary(self, *, reference=None):
        """Compare model coverage, runtime, and speedup against a fixed solver.

        reference: registered solver ID; None uses the first solver in the rows.
        Geometric speedups use only MDPs solved by both solvers; n_compared
        records the number of such MDPs. All models count toward failure rates.
        """
        import pandas as pd

        models, solvers, aggregated, ratios = _performance_ratios(
            self.rows, self.experiment
        )
        reference = solvers[0] if reference is None else reference
        if reference not in solvers:
            raise ValueError(f"Unknown reference solver: {reference!r}")
        reference_times = np.array(
            [aggregated[model, reference]["median_runtime"] for model in models]
        )
        records = []
        for j, solver in enumerate(solvers):
            values = ratios[:, j]
            runtimes = np.array(
                [aggregated[model, solver]["median_runtime"] for model in models]
            )
            successful = np.isfinite(runtimes)
            paired = np.isfinite(reference_times) & successful
            records.append(
                {
                    "solver": solver,
                    "models_solved": int(successful.sum()),
                    "n_models": len(models),
                    "n_failed": int((~successful).sum()),
                    "success_rate": float(successful.mean()),
                    "fastest_rate": float((values <= 1).mean()),
                    "within_2x": float((values <= 2).mean()),
                    "median_ratio": float(np.median(values[successful]))
                    if successful.any()
                    else np.nan,
                    "median_runtime": float(np.median(runtimes[successful]))
                    if successful.any()
                    else np.nan,
                    "n_compared": int(paired.sum()),
                    "geomean_speedup": float(
                        np.exp(
                            np.mean(
                                np.log(reference_times[paired])
                                - np.log(runtimes[paired])
                            )
                        )
                    )
                    if paired.any()
                    else np.nan,
                }
            )
        return pd.DataFrame(records)

    def scaling_summary(self, *, reference=None):
        """Summarize instance runtimes and paired speedups by state dimension.

        reference: solver ID; None uses the first solver in the results.
        Each MDP contributes its median successful trial runtime, only if all
        expected repeats succeeded. The quantiles aggregate eligible instances.
        """
        import pandas as pd

        return pd.DataFrame(
            _scaling_statistics(self.rows, self.experiment, reference=reference)
        )

    def plot_scaling(
        self, *, x="state_dim", metric="runtime", reference=None, ax=None, show=True
    ):
        """Plot scaling of median runtime or paired geometric speedup.

        x: "state_dim" or "nnz" (median transition nonzeros across instances).
        metric: "runtime" (seconds) or "speedup" (reference / solver).
        reference: solver ID; None uses the first solver in the results.
        ax: Matplotlib axes, or None to create a figure.
        show: True displays the figure; False only returns (figure, axes).
        """
        from mdpforge.utils.plotting import plot_scaling

        return plot_scaling(
            self.rows,
            self.experiment,
            x=x,
            metric=metric,
            reference=reference,
            ax=ax,
            show=show,
        )

    def plot_performance_profile(self, *, max_factor=100, ax=None, show=True):
        """Plot each solver's share of models completed within a runtime factor.

        max_factor: finite float greater than one; maximum displayed factor.
        ax: Matplotlib axes, or None to create a figure.
        show: True displays the figure; False only returns (figure, axes).
        """
        from mdpforge.utils.plotting import plot_performance_profile

        return plot_performance_profile(
            self.rows,
            self.experiment,
            max_factor=max_factor,
            ax=ax,
            show=show,
        )

    def plot_heat(self, *, reference=None, metric="speedup", ax=None, show=True):
        """Plot eligible medians or speedups; failed or missing trials mark FAIL.

        reference: registered solver ID; None uses the first solver in the rows.
        metric: "speedup" or "runtime" (seconds).
        ax: Matplotlib axes, or None to create a figure.
        show: True displays the figure; False only returns (figure, axes).

        Rows are grouped by family (VI, PI, MPI, LP), method, backend and variant.
        Labels come from the recorded manifest; missing labels fall back to IDs.
        Matplotlib is optional. Plotting never imports models or solver backends.
        """
        from mdpforge.utils.plotting import plot_heat

        return plot_heat(
            self.rows,
            self.experiment,
            reference=reference,
            metric=metric,
            ax=ax,
            show=show,
        )

    def export_csv(self, path=None):
        """Save trial rows and their companion manifest; return the CSV Path.

        path: explicit CSV destination. None saves to artifacts/benchmark/
            bench_<UTC date and time>.csv, without overwriting a previous export.
        A result loaded without a manifest exports only its CSV.
        """
        if not self.rows:
            raise ValueError("No benchmark measurements to export")
        automatic = path is None
        if automatic:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
            path = BENCHMARKS_PATH / f"bench_{stamp}.csv"
            suffix = 1
            while path.exists() or path.with_suffix(".experiment.json").exists():
                path = BENCHMARKS_PATH / f"bench_{stamp}_{suffix}.csv"
                suffix += 1
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.experiment is None and path.with_suffix(".experiment.json").exists():
            raise ValueError("Cannot replace a CSV with an unrelated existing manifest")
        mode = "x" if automatic else "w"
        with path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        if self.experiment is not None:
            payload = self.experiment | {"results_file": path.name}
            text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
            path.with_suffix(".experiment.json").write_text(
                text + "\n", encoding="utf-8"
            )
        return path

    @classmethod
    def from_csv(cls, path=None):
        """Load one run and its companion manifest, if present.

        path: CSV to load. None selects the most recently modified CSV in
            artifacts/benchmark/; ties are resolved by filename.
        Integer/float fields are restored, including NaN for failed trials.
        Missing exports raise FileNotFoundError.
        Invalid or mixed-run CSVs raise ValueError.
        """
        if path is None:
            candidates = [
                item for item in BENCHMARKS_PATH.glob("*.csv") if item.is_file()
            ]
            if not candidates:
                raise FileNotFoundError(f"No benchmark CSV found in {BENCHMARKS_PATH}")
            path = max(
                candidates, key=lambda item: (item.stat().st_mtime_ns, item.name)
            )
        path = Path(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = _REQUIRED_FIELDS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing CSV columns: {sorted(missing)}")
            rows = []
            for number, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path}: invalid CSV row {number}")
                try:
                    for field, convert in _FIELD_TYPES.items():
                        if field in row:
                            row[field] = convert(row[field])
                except ValueError as exc:
                    raise ValueError(
                        f"{path}: invalid number in CSV row {number}"
                    ) from exc
                rows.append(row)
        if not rows:
            raise ValueError(f"{path}: no benchmark measurements")
        experiment_ids = {row.get("experiment_id") for row in rows}
        settings = {(row["discount"], row["epsilon"]) for row in rows}
        if len(experiment_ids) > 1 or len(settings) > 1:
            raise ValueError(f"{path}: CSV contains multiple benchmark runs")
        manifest_path = path.with_suffix(".experiment.json")
        experiment = None
        if manifest_path.exists():
            experiment = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(experiment, dict):
                raise ValueError(f"{manifest_path}: experiment manifest must be a dict")
            if experiment_ids != {None} and experiment_ids != {
                experiment.get("experiment_id")
            }:
                raise ValueError(
                    f"{path}: CSV and experiment manifest IDs do not match"
                )
        return cls(rows=rows, experiment=experiment)
