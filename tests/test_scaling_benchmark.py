"""Deterministic checks for size-sweep registration and aggregation."""

import numpy as np
import pytest
from scipy.sparse import eye

from mdpforge import BenchmarkResult, ScalingBenchmark
from mdpforge.core.mdp import MDP


def _family(size, seed):
    rng = np.random.default_rng(seed)
    return MDP.from_matrices(
        f"toy_{size}_{seed}",
        [eye(size, format="csr")],
        rng.uniform(size=(size, 1)),
    )


def _zero_family(size, seed):
    return MDP.from_matrices(
        f"zero_{size}_{seed}",
        [eye(size, format="csr")],
        np.zeros((size, 1)),
    )


def _zero_solver(transitions, rewards, discount, precision):
    return np.zeros(rewards.shape[0])


def _rounded_zero_family(size, seed):
    actual = size // 2 * 2
    return MDP.from_matrices(
        f"rounded_{size}_{seed}",
        [eye(actual, format="csr")],
        np.zeros((actual, 1)),
    )


def _scaling_result():
    rows = []
    sizes = {"a": 2, "b": 2, "c": 4, "d": 4}
    durations = {
        "a": {"VI": [4, 4], "MPI": [2, 2]},
        "b": {"VI": [8, 8], "MPI": [4, 4]},
        "c": {"VI": [8, 8], "MPI": [4, 4]},
        "d": {"VI": [16, 16], "MPI": [8, None]},
    }
    for model, solver_times in durations.items():
        for solver, values in solver_times.items():
            for repeat, value in enumerate(values, 1):
                rows.append(
                    {
                        "experiment_id": "scale",
                        "model": model,
                        "state_dim": sizes[model],
                        "action_dim": 1,
                        "solver": solver,
                        "repeat": repeat,
                        "seed": repeat,
                        "discount": 0.9,
                        "epsilon": 1e-3,
                        "runtime": value if value is not None else np.nan,
                        "status": "success" if value is not None else "imprecise",
                    }
                )
    experiment = {
        "experiment_id": "scale",
        "settings": {"repeats": 2},
        "models": {
            model: {"state_dim": size, "nnz": size} for model, size in sizes.items()
        },
        "study": {
            "type": "scaling",
            "variable": "state_dim",
            "family": "toy",
            "sizes": [2, 4],
            "instances": 2,
            "seed": 0,
            "models": {
                model: {
                    "state_dim": size,
                    "instance": index,
                    "model_seed": index,
                }
                for index, (model, size) in enumerate(sizes.items())
            },
        },
    }
    return BenchmarkResult(rows, experiment)


def test_scaling_family_registers_distinct_seeded_models():
    bench = ScalingBenchmark(default_solvers=False)
    assert (
        bench.add_family(_family, [2, 4], instances=2, seed=10, family="toy") is bench
    )
    assert len(bench._mdps) == 4
    assert [m.state_dim for m in bench._mdps] == [2, 2, 4, 4]
    assert [m.name for m in bench._mdps] == [
        "toy_2_10",
        "toy_2_11",
        "toy_4_12",
        "toy_4_13",
    ]
    assert all(
        model.transition_matrix[0].nnz == model.state_dim for model in bench._mdps
    )
    with pytest.raises(ValueError, match="one family"):
        bench.add_family(_family, [5])
    assert bench.reset(models=False, solvers=False)._scaling_study is not None
    assert bench.reset()._scaling_study is None


@pytest.mark.parametrize(
    "sizes, instances, seed",
    [
        ([], 2, 0),
        ([2, 2], 2, 0),
        ([0, 2], 2, 0),
        ([2], 0, 0),
        ([2], 1, -1),
    ],
)
def test_invalid_study_parameters(sizes, instances, seed):
    with pytest.raises(ValueError):
        ScalingBenchmark().add_family(_family, sizes, instances=instances, seed=seed)


def test_scaling_benchmark_returns_standard_result():
    bench = ScalingBenchmark(default_solvers=False)
    bench.add_family(_zero_family, [2, 3], instances=2, seed=17, family="identity")
    bench.add_solver("zeros", solve_function=_zero_solver)
    result = bench.run(0.9, repeats=1)
    assert type(result) is BenchmarkResult
    assert all(row["status"] == "success" for row in result.rows)
    assert result.experiment["study"]["sizes"] == [2, 3]
    assert result.experiment["study"]["models"]["zero_2_17"]["model_seed"] == 17
    assert result.experiment["models"]["zero_3_19"]["nnz"] == 3


def test_scaling_merges_rounded_requests_at_actual_dimensions(tmp_path):
    pytest.importorskip("pandas")
    bench = ScalingBenchmark(default_solvers=False)
    bench.add_family(_rounded_zero_family, [5, 6, 7], instances=2)
    bench.add_solver("zeros", solve_function=_zero_solver)
    result = bench.run(0.9, repeats=1)
    study = result.experiment["study"]
    assert study["sizes"] == [4, 6]
    assert study["requested_state_dims"] == [5, 6, 7]
    assert [row["state_dim"] for row in result.rows] == [4, 4, 6, 6, 6, 6]
    assert [row["requested_state_dim"] for row in result.rows] == [5, 5, 6, 6, 7, 7]
    table = result.scaling_summary().set_index("state_dim")
    assert table.loc[4, "n_instances"] == 2
    assert table.loc[6, "n_instances"] == 4
    assert table.loc[6, "success_rate"] == 1
    restored = BenchmarkResult.from_csv(result.export_csv(tmp_path / "rounded.csv"))
    assert restored.experiment["study"] == study
    assert restored.scaling_summary().equals(result.scaling_summary())


def test_scaling_plot_uses_actual_dimensions():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench = ScalingBenchmark(default_solvers=False)
    bench.add_family(_rounded_zero_family, [5, 6, 7], instances=1)
    bench.add_solver("zeros", solve_function=_zero_solver)
    result = bench.run(0.9, repeats=1)
    fig, ax = result.plot_scaling(show=False)
    try:
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [4, 6])
    finally:
        plt.close(fig)


def test_scaling_summary_aggregates_instances_after_repeats():
    pytest.importorskip("pandas")
    table = (
        _scaling_result()
        .scaling_summary(reference="VI")
        .set_index(["state_dim", "solver"])
    )
    assert table.loc[(2, "VI"), "median_runtime"] == 6
    assert table.loc[(2, "VI"), "q25_runtime"] == 5
    assert table.loc[(2, "VI"), "q75_runtime"] == 7
    assert table.loc[(2, "MPI"), "geomean_speedup"] == pytest.approx(2)
    assert table.loc[(4, "MPI"), "n_failed"] == 1
    assert table.loc[(4, "MPI"), "n_compared"] == 1
    assert table.loc[(4, "MPI"), "success_rate"] == 0.5
    assert table.loc[(4, "MPI"), "median_runtime"] == 4
    assert table.loc[(4, "MPI"), "nnz"] == 4


def test_scaling_validation_and_export(tmp_path):
    result = _scaling_result()
    loaded = BenchmarkResult.from_csv(result.export_csv(tmp_path / "scale.csv"))
    assert loaded.experiment["study"] == result.experiment["study"]
    pytest.importorskip("pandas")
    assert loaded.scaling_summary().equals(result.scaling_summary())
    with pytest.raises(ValueError, match="Unknown reference solver"):
        result.scaling_summary(reference="missing")
    result.experiment["study"]["models"]["a"]["state_dim"] = 7
    with pytest.raises(ValueError, match="Unexpected state dimension"):
        result.scaling_summary()


def test_scaling_plot_uses_optional_matplotlib():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = _scaling_result()
    for x, metric in (("state_dim", "runtime"), ("nnz", "speedup")):
        fig, ax = result.plot_scaling(x=x, metric=metric, reference="VI", show=False)
        try:
            assert ax.get_xscale() == "log"
            assert ax.get_yscale() == "log"
            assert any(line.get_label() == "MPI" for line in ax.lines)
            assert any(text.get_text() == "1/2" for text in ax.texts)
        finally:
            plt.close(fig)
    with pytest.raises(ValueError, match="x must"):
        result.plot_scaling(x="invalid", show=False)
    with pytest.raises(ValueError, match="metric must"):
        result.plot_scaling(metric="invalid", show=False)
