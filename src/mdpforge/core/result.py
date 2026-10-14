"""Measurements and provenance of one finite discounted MDP benchmark run."""

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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


@dataclass
class BenchmarkResult:
    """A run's rows and captured experiment metadata, independent of Benchmark.

    rows: trial dictionaries, with numerical values and technical solver IDs.
    experiment: captured manifest, or None for a CSV without a companion manifest.
    """

    rows: list[dict]
    experiment: dict | None = None

    def plot_heat(self, *, reference=None, metric="speedup", ax=None, show=True):
        """Plot median runtimes or speedups; any failed repeat marks a cell FAIL.

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
