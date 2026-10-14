"""Deterministic solver comparisons without running any solvers."""

import numpy as np
import pytest

from mdpforge import BenchmarkResult


def comparison_result():
    """Three MDPs, two repeats, one failure and one missing trial."""
    rows = []
    cases = {
        "m1": {"VI": [2.0, 2.0], "MPI": [1.0, 1.0], "PI": [4.0, 4.0]},
        "m2": {"VI": [1.0, 1.0], "MPI": [3.0, 3.0], "PI": [8.0, None]},
        "m3": {"VI": [4.0, 4.0], "MPI": [2.0], "PI": [8.0, 8.0]},
    }
    for model, solvers in cases.items():
        for solver, values in solvers.items():
            for repeat, value in enumerate(values, 1):
                rows.append(
                    {
                        "experiment_id": "example",
                        "model": model,
                        "solver": solver,
                        "repeat": repeat,
                        "seed": repeat - 1,
                        "discount": 0.9,
                        "epsilon": 1e-3,
                        "runtime": value if value is not None else np.nan,
                        "status": "success" if value is not None else "error",
                    }
                )
    manifest = {"experiment_id": "example", "settings": {"repeats": 2}}
    return BenchmarkResult(rows=rows, experiment=manifest)


def test_summary_marks_missing_and_failed_repeats_ineligible():
    pd = pytest.importorskip("pandas")
    result = comparison_result()
    data = result.summary().set_index(["model", "solver"])

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 9
    assert data.loc[("m1", "MPI"), "median_runtime"] == 1.0
    assert data.loc[("m1", "MPI"), "complete"]
    assert data.loc[("m2", "PI"), "n_success"] == 1
    assert data.loc[("m2", "PI"), "n_trials"] == 2
    assert not data.loc[("m2", "PI"), "complete"]
    assert np.isnan(data.loc[("m2", "PI"), "median_runtime"])
    assert data.loc[("m3", "MPI"), "n_success"] == 1
    assert data.loc[("m3", "MPI"), "n_trials"] == 1
    assert not data.loc[("m3", "MPI"), "complete"]
    assert np.isnan(data.loc[("m3", "MPI"), "median_runtime"])
    assert len(result.to_dataframe()) == len(result.rows)


def test_solver_summary_uses_all_models_as_denominator():
    pytest.importorskip("pandas")
    result = comparison_result()
    summary = result.solver_summary().set_index("solver")
    assert list(summary.index) == ["VI", "MPI", "PI"]
    assert summary.loc["VI", "models_solved"] == 3
    assert summary.loc["VI", "success_rate"] == 1.0
    assert summary.loc["VI", "fastest_rate"] == pytest.approx(2 / 3)
    assert summary.loc["VI", "within_2x"] == 1.0
    assert summary.loc["VI", "median_ratio"] == 1.0
    assert summary.loc["MPI", "models_solved"] == 2
    assert summary.loc["MPI", "success_rate"] == pytest.approx(2 / 3)
    assert summary.loc["MPI", "fastest_rate"] == pytest.approx(1 / 3)
    assert summary.loc["MPI", "within_2x"] == pytest.approx(1 / 3)
    assert summary.loc["MPI", "median_ratio"] == 2.0
    assert summary.loc["PI", "models_solved"] == 2
    assert summary.loc["PI", "fastest_rate"] == 0.0
    assert summary.loc["PI", "within_2x"] == pytest.approx(1 / 3)
    assert summary.loc["PI", "median_ratio"] == 3.0


def test_solver_summary_reports_paired_geomean_and_failure_counts():
    pytest.importorskip("pandas")
    result = comparison_result()
    summary = result.solver_summary(reference="VI").set_index("solver")

    assert summary.loc["VI", "n_failed"] == 0
    assert summary.loc["VI", "median_runtime"] == 2.0
    assert summary.loc["VI", "n_compared"] == 3
    assert summary.loc["VI", "geomean_speedup"] == pytest.approx(1.0)

    assert summary.loc["MPI", "n_failed"] == 1  # Missing repeat on m3.
    assert summary.loc["MPI", "median_runtime"] == 2.0
    assert summary.loc["MPI", "n_compared"] == 2
    assert summary.loc["MPI", "geomean_speedup"] == pytest.approx(np.sqrt(2 / 3))

    assert summary.loc["PI", "n_failed"] == 1  # Failed repeat on m2.
    assert summary.loc["PI", "median_runtime"] == 6.0
    assert summary.loc["PI", "n_compared"] == 2
    assert summary.loc["PI", "geomean_speedup"] == pytest.approx(0.5)

    against_mpi = result.solver_summary(reference="MPI").set_index("solver")
    assert against_mpi.loc["VI", "n_compared"] == 2
    assert against_mpi.loc["VI", "geomean_speedup"] == pytest.approx(np.sqrt(1.5))
    assert against_mpi.loc["MPI", "geomean_speedup"] == pytest.approx(1.0)
    assert against_mpi.loc["PI", "n_compared"] == 1
    assert against_mpi.loc["PI", "geomean_speedup"] == pytest.approx(0.25)

    assert result.solver_summary().equals(result.solver_summary(reference="VI"))
    with pytest.raises(ValueError, match="Unknown reference solver"):
        result.solver_summary(reference="not_registered")


def test_solver_summary_without_shared_successes():
    pytest.importorskip("pandas")
    result = comparison_result()
    for row in result.rows:
        if row["solver"] == "MPI" and row["model"] == "m1":
            row["status"] = "error"
        if row["solver"] == "MPI" and row["model"] == "m2":
            row["status"] = "error"
    summary = result.solver_summary(reference="MPI").set_index("solver")

    assert summary.loc["MPI", "n_failed"] == 3
    assert summary.loc["MPI", "n_compared"] == 0
    assert np.isnan(summary.loc["MPI", "median_runtime"])
    assert np.isnan(summary.loc["MPI", "geomean_speedup"])
    assert summary.loc["VI", "n_compared"] == 0
    assert np.isnan(summary.loc["VI", "geomean_speedup"])
    assert summary.loc["VI", "median_runtime"] == 2.0


def test_profile_fractions_use_missing_and_failed_trials_in_denominator():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = comparison_result()
    fig, ax = result.plot_performance_profile(max_factor=4, show=False)
    try:
        assert ax.get_xscale() == "log"
        curves = {line.get_label(): line for line in ax.lines}
        for solver, factor, expected in (
            ("VI", 1, 2 / 3),
            ("VI", 2, 1.0),
            ("MPI", 1, 1 / 3),
            ("MPI", 3, 2 / 3),
            ("PI", 2, 1 / 3),
            ("PI", 4, 2 / 3),
        ):
            x, y = curves[solver].get_data()
            assert y[np.where(x == factor)[0][0]] == pytest.approx(expected)
    finally:
        plt.close(fig)


def test_heatmap_marks_missing_repeats_as_fail():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = comparison_result().plot_heat(reference="VI", show=False)
    try:
        # Three solvers by three models, with M3/MPI missing repeat 2.
        assert [text.get_text() for text in ax.texts].count("FAIL") == 2
    finally:
        plt.close(fig)


def test_manifest_absent_uses_observed_repeat_ids():
    result = comparison_result()
    result.experiment = None
    from mdpforge.core.result import _aggregate_rows

    _, _, cells = _aggregate_rows(result.rows)
    assert cells["m3", "MPI"]["complete"] is False
    assert cells["m1", "MPI"]["complete"] is True


def test_cross_experiment_comparisons_are_rejected():
    result = comparison_result()
    result.rows[-1]["discount"] = 0.8
    with pytest.raises(ValueError, match="discount or precision"):
        result.plot_performance_profile(show=False)


def test_profile_rejects_invalid_max_factor():
    result = comparison_result()
    for factor in (0, 1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_factor"):
            result.plot_performance_profile(max_factor=factor, show=False)


def test_import_export_preserves_comparisons(tmp_path):
    pytest.importorskip("pandas")
    original = comparison_result()
    path = original.export_csv(tmp_path / "benchmark.csv")
    loaded = BenchmarkResult.from_csv(path)
    assert len(loaded.rows) == len(original.rows)
    assert loaded.experiment["settings"]["repeats"] == 2
    assert loaded.solver_summary(reference="MPI").equals(
        original.solver_summary(reference="MPI")
    )
