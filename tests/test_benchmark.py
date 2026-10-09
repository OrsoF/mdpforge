import csv
import random
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from mdpforge.core.benchmark import Benchmark, benchmark, export_csv
from mdpforge.core.mdp import MDP
from mdpforge.core.model import MDPProtocol
from mdpforge.solvers.personal_mpi import Solver as MPI
from mdpforge.solvers.personal_qvi import Solver as QVI
from mdpforge.solvers.personal_vi import Solver as VI


@pytest.mark.parametrize("dense_input", [False, True])
def test_benchmark_accepts_generated_matrix_and_external_mdps(chain, dense_input):
    transitions = chain.transition_matrix
    if dense_input:
        transitions = np.array([matrix.toarray() for matrix in transitions])
    matrix_mdp: MDPProtocol = MDP.from_matrices(
        "matrix_chain", transitions, chain.reward_matrix
    )
    external_mdp: MDPProtocol = SimpleNamespace(
        name="external_chain",
        state_dim=chain.state_dim,
        action_dim=chain.action_dim,
        transition_matrix=chain.transition_matrix,
        reward_matrix=chain.reward_matrix,
    )
    bench = Benchmark()
    for model in (chain, matrix_mdp, external_mdp):
        bench.add_mdp(model)
    for name, constructor in (("VI", VI), ("QVI", QVI), ("MPI", MPI)):
        bench.add_solver(constructor, name=name)

    results = bench.run(0.9, precision=1e-4, repeats=2, seed=0)
    assert len(results) == 18
    for row in results:
        assert row["status"] == "success", row["error"]
        assert row["residual"] <= 1e-4 * (1 - 0.9)
    for generated, matrix, external in zip(results[:6], results[6:12], results[12:]):
        for field in ("solver", "repeat", "seed", "residual", "error_bound", "status"):
            assert generated[field] == matrix[field] == external[field]
    assert all(
        isinstance(matrix, csr_matrix) for matrix in matrix_mdp.transition_matrix
    )


def test_benchmark_measures_vi_and_qvi(chain):
    chain = SimpleNamespace(
        name=chain.name,
        state_dim=chain.state_dim,
        action_dim=chain.action_dim,
        transition_matrix=chain.transition_matrix,
        reward_matrix=chain.reward_matrix,
    )
    results = benchmark(chain, {"VI": VI, "QVI": QVI}, discount=0.9, repeats=2)
    assert len(results) == 4
    assert [(row["solver"], row["repeat"]) for row in results] == [
        ("VI", 1),
        ("VI", 2),
        ("QVI", 1),
        ("QVI", 2),
    ]
    for row in results:
        assert row["model"] == chain.name
        assert row["status"] == "success"
        assert row["error"] == ""
        assert np.isfinite(row["runtime"]) and row["runtime"] >= 0
        assert row["residual"] <= row["epsilon"] * (1 - row["discount"])
        assert row["error_bound"] <= row["epsilon"]
    assert all(isinstance(matrix, csr_matrix) for matrix in chain.transition_matrix)


def test_benchmark_checks_original_model_and_isolates_trials(chain):
    transitions = [matrix.copy() for matrix in chain.transition_matrix]
    rewards = chain.reward_matrix.copy()

    class MutatingSolver:
        def __init__(self, model, discount, final_precision):
            model.reward_matrix[:] = 0
            for matrix in model.transition_matrix:
                matrix.data[:] = 0
            self.value = np.zeros(model.state_dim)

        def run(self):
            pass

    results = benchmark(
        chain, {"mutating": MutatingSolver, "VI": VI}, discount=0.9, repeats=2
    )
    assert [row["status"] for row in results] == [
        "imprecise",
        "imprecise",
        "success",
        "success",
    ]
    assert results[0]["residual"] == 2
    assert results[0]["error_bound"] == pytest.approx(20)
    for matrix, original in zip(chain.transition_matrix, transitions):
        assert_allclose(matrix.toarray(), original.toarray(), rtol=0, atol=0)
    assert_allclose(chain.reward_matrix, rewards, rtol=0, atol=0)


@pytest.mark.parametrize("fail_in_constructor", [False, True])
def test_benchmark_records_exceptions_and_continues(chain, fail_in_constructor):
    class BrokenSolver:
        def __init__(self, model, discount, final_precision):
            if fail_in_constructor:
                raise RuntimeError("broken constructor")

        def run(self):
            raise RuntimeError("broken run")

    results = benchmark(
        chain, {"broken": BrokenSolver, "VI": VI}, discount=0.9, repeats=1
    )
    failed, successful = results
    assert failed["status"] == "error"
    assert "RuntimeError: broken" in failed["error"]
    assert np.isfinite(failed["runtime"])
    assert np.isnan(failed["residual"])
    assert successful["status"] == "success"


@pytest.mark.parametrize("value", [[0, 0], [0, np.nan, 0], [0, np.inf, 0]])
def test_benchmark_rejects_invalid_values(chain, value):
    class InvalidSolver:
        def __init__(self, model, discount, final_precision):
            self.value = value

        def run(self):
            pass

    row = benchmark(chain, {"invalid": InvalidSolver}, discount=0.9, repeats=1)[0]
    assert row["status"] == "error"
    assert "finite vector" in row["error"]


def test_benchmark_pairs_seeds_and_restores_rng_states(chain):
    samples = []

    class RandomSolver(VI):
        def run(self):
            samples.append((np.random.random(), random.random()))
            super().run()

    numpy_state, random_state = np.random.get_state(), random.getstate()
    results = benchmark(
        chain,
        {"first": RandomSolver, "second": RandomSolver},
        discount=0.9,
        repeats=2,
        seed=7,
    )
    assert samples[:2] == samples[2:]
    assert samples[0] != samples[1]
    assert [row["seed"] for row in results] == [7, 8, 7, 8]
    current = np.random.get_state()
    assert current[0] == numpy_state[0]
    assert np.array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]
    assert random.getstate() == random_state


def test_export_csv_round_trip(chain, tmp_path):
    results = benchmark(chain, {"VI": VI}, discount=0.9, repeats=1)
    path = export_csv(results, tmp_path / "nested" / "benchmark.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["solver"] == "VI"
    assert rows[0]["status"] == "success"
    assert float(rows[0]["runtime"]) == results[0]["runtime"]
