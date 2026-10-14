import pytest

from mdpforge import Benchmark
from mdpforge.solvers.mdpforge_vi import Solver as VI
from mdpforge.utils.plotting import plot_heat


def test_heatmap_display_labels_preserve_ids_and_disambiguate_aliases():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        {
            "model": "chain",
            "solver": name,
            "status": "success",
            "runtime": runtime,
            "discount": 0.9,
            "epsilon": 1e-3,
        }
        for name, runtime in (("VI", 2.0), ("warm VI", 1.0), ("custom", 0.5))
    ]
    fig, ax = plot_heat(
        rows,
        ["chain"],
        ["VI", "warm VI", "custom"],
        reference="VI",
        solver_labels={"VI": "VI · mdpforge", "warm VI": "VI · mdpforge"},
        show=False,
    )
    try:
        assert [label.get_text() for label in ax.get_yticklabels()] == [
            "VI · mdpforge [VI]",
            "VI · mdpforge [warm VI]",
            "custom",
        ]
        assert [label.get_text() for label in ax.texts] == ["1.00×", "2.00×", "4.00×"]
        assert [row["solver"] for row in rows] == ["VI", "warm VI", "custom"]
    finally:
        plt.close(fig)


def test_benchmark_plot_uses_recorded_metadata_labels(chain):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("mdpforge_vi", solve_function=VI)
    bench.run(0.9, repeats=1)
    metadata = bench.experiment["solvers"]["mdpforge_vi"]["metadata"]
    metadata["display_name"] = "Saved label"
    fig, ax = bench.plot_heat(reference="mdpforge_vi", show=False)
    try:
        assert [label.get_text() for label in ax.get_yticklabels()] == ["Saved label"]
        assert bench.results[0]["solver"] == "mdpforge_vi"
    finally:
        plt.close(fig)


def test_benchmark_plot_sorts_rows_without_changing_reference_or_results(chain):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = [
        ("lp", "lp", "gurobi", "LP · Gurobi · dual", 8.0),
        ("mpi_gs", "mpi", "mdpsolver", "MPI · MDPSolver · GS", 4.0),
        ("vi", "vi", "marmote", "VI · Marmote", 2.0),
        ("mpi_forge", "mpi", "mdpforge", "MPI · mdpforge", 8.0),
        ("mpi", "mpi", "mdpsolver", "MPI · MDPSolver", 2.0),
        ("pi", "pi", "mdptoolbox", "PI · MDPToolbox", 4.0),
        ("envelope", "vi", "mdpforge", "Envelope VI · mdpforge", 16.0),
        ("custom_z", None, None, "custom_z", 1.0),
        ("custom_a", None, None, "custom_a", 0.5),
    ]
    bench = Benchmark(default_solvers=False).add_mdp(chain)
    for name, *_ in entries:
        bench.add_solver(name, solve_function=VI)
    bench.results = [
        {
            "model": chain.name,
            "solver": name,
            "status": "success",
            "runtime": runtime,
            "discount": 0.9,
            "epsilon": 1e-3,
        }
        for name, _, _, _, runtime in entries
    ]
    bench.experiment = {
        "solvers": {
            name: {
                "metadata": {
                    "family": family,
                    "backend": backend,
                    "display_name": label,
                }
            }
            for name, family, backend, label, _ in entries
        }
    }
    results = bench.results.copy()
    registrations = list(bench._solvers)
    fig, ax = bench.plot_heat(show=False)
    try:
        assert [label.get_text() for label in ax.get_yticklabels()] == [
            "Envelope VI · mdpforge",
            "VI · Marmote",
            "PI · MDPToolbox",
            "MPI · mdpforge",
            "MPI · MDPSolver",
            "MPI · MDPSolver · GS",
            "LP · Gurobi · dual",
            "custom_z",
            "custom_a",
        ]
        assert [label.get_text() for label in ax.texts] == [
            "0.50×",
            "4.00×",
            "2.00×",
            "1.00×",
            "4.00×",
            "2.00×",
            "1.00×",
            "8.00×",
            "16.00×",
        ]
        assert fig.axes[-1].get_ylabel() == "log₂ speedup vs LP · Gurobi · dual"
        assert "GS = Gauss-Seidel" in fig.texts[0].get_text()
        assert bench.results == results
        assert list(bench._solvers) == registrations
    finally:
        plt.close(fig)
