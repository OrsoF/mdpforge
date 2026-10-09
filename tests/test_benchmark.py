import csv
import random
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from mdpforge import Benchmark
from mdpforge.core.mdp import MDP
from mdpforge.core.model import MDPProtocol
from mdpforge.core.validation import validate_model
from mdpforge.solvers.personal_mpi import Solver as MPI
from mdpforge.solvers.personal_vi import Solver as VI


def test_catalogue_solver_loads_class_and_forwards_options(chain):
    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver(solver_name="personal_vi", initial_value=np.array([1.8, 2.0, 0.0]))
    results = bench.run(discount=0.9, precision=1e-4, repeats=1)
    assert len(results) == 1
    assert results[0]["solver"] == "personal_vi"
    assert results[0]["status"] == "success", results[0]["error"]
    assert results[0]["error_bound"] <= 1e-4


def test_add_all_models_builds_defaults_and_preserves_existing(chain, monkeypatch):
    existing = SimpleNamespace(**vars(chain))
    existing.name = "built_alpha"
    bench = Benchmark(default_solvers=False).add_mdp(existing)

    def import_catalogue(path):
        name = path.rsplit(".", 1)[1]
        if name == "missing":
            raise ModuleNotFoundError("Missing dependency", name="missing_backend")
        model = type(chain)(3, 2)
        model.name = f"built_{name}"
        return SimpleNamespace(Model=lambda: model)

    monkeypatch.setattr(
        "mdpforge.core.benchmark._catalogue",
        lambda package: {"zeta", "missing", "alpha"},
    )
    monkeypatch.setattr("mdpforge.core.benchmark.import_module", import_catalogue)
    for _ in range(2):
        with pytest.warns(
            UserWarning, match="Skipping model 'missing'.*missing_backend"
        ):
            assert bench.add_all_models() is bench
    assert [model.name for model in bench._mdps] == ["built_alpha", "built_zeta"]
    assert bench._mdps[0] is existing
    for model in bench._mdps:
        validate_model(model)


def test_add_all_solvers_preserves_default_vi_and_skips_missing(monkeypatch):
    bench = Benchmark()

    def import_catalogue(path):
        name = path.rsplit(".", 1)[1]
        if name == "missing":
            raise ModuleNotFoundError("Missing dependency", name="missing_backend")
        return SimpleNamespace(Solver={"baseline": VI, "other": MPI}[name])

    monkeypatch.setattr(
        "mdpforge.core.benchmark._catalogue",
        lambda package: {"other", "baseline", "missing"},
    )
    monkeypatch.setattr("mdpforge.core.benchmark.import_module", import_catalogue)
    for _ in range(2):
        with pytest.warns(
            UserWarning, match="Skipping solver 'missing'.*missing_backend"
        ):
            assert bench.add_all_solvers() is bench
    assert list(bench._solvers) == ["VI", "other"]
    assert bench._solvers["VI"] == (VI, {})
    assert bench._solvers["other"] == (MPI, {})


def test_unknown_solver_lists_catalogue_without_importing_it(monkeypatch):
    def fail_import(*args, **kwargs):
        pytest.fail("Listing solvers must not import backend dependencies")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    with pytest.raises(
        ValueError,
        match="Unknown solver 'missing_solver'.*Available solvers:.*personal_vi",
    ):
        Benchmark(default_solvers=False).add_solver("missing_solver")


def test_custom_solver_takes_priority_and_rejects_duplicates(chain, monkeypatch):
    def fail_import(*args, **kwargs):
        pytest.fail("A custom callable must bypass catalogue imports")

    def custom_solver(transitions, rewards, discount, precision):
        return np.array([2 * discount, 2, 0])

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("personal_vi", solve_function=custom_solver)
    results = bench.run(discount=0.9, repeats=1)
    assert results[0]["solver"] == "personal_vi"
    assert results[0]["status"] == "success", results[0]["error"]
    with pytest.raises(ValueError, match="already registered"):
        bench.add_solver("personal_vi")
    with pytest.raises(TypeError, match="solve_function must be a callable"):
        bench.add_solver("invalid_function", solve_function=42)
    with pytest.raises(ValueError, match="parameters controlled by run"):
        bench.add_solver(
            "invalid_precision", solve_function=custom_solver, precision=0.1
        )


@pytest.mark.parametrize("solver_name", [None, 0, "", " "])
def test_add_solver_requires_nonempty_name(solver_name):
    with pytest.raises(ValueError, match="nonempty string name"):
        Benchmark(default_solvers=False).add_solver(solver_name)


def test_function_solver_receives_matrices_and_options_alongside_classes(chain):
    calls = []

    def mon_solver(transitions, rewards, discount, precision, *, scale):
        assert all(isinstance(matrix, csr_matrix) for matrix in transitions)
        assert isinstance(rewards, np.ndarray)
        calls.append((discount, precision, scale))
        # Action zero is optimal on this deterministic chain.
        return scale * np.linalg.solve(
            np.eye(rewards.shape[0]) - discount * transitions[0].toarray(),
            rewards[:, 0],
        )

    bench = Benchmark().add_mdp(chain)
    bench.add_solver(solver_name="mon_solver", solve_function=mon_solver, scale=1.0)
    results = bench.run(discount=0.9, precision=1e-4, repeats=2)
    assert len(results) == 4
    assert {row["solver"] for row in results} == {"VI", "mon_solver"}
    assert calls == [(0.9, 1e-4, 1.0)] * 2
    for row in results:
        assert row["status"] == "success", row["error"]
        assert row["error_bound"] <= 1e-4


@pytest.mark.parametrize("outcome", ["imprecise", "invalid", "exception"])
def test_function_solver_failures_are_recorded_and_trials_isolated(chain, outcome):
    rewards = chain.reward_matrix.copy()
    transitions = [matrix.copy() for matrix in chain.transition_matrix]

    def broken_solver(transitions, rewards, discount, precision):
        assert_allclose(rewards, chain.reward_matrix, rtol=0, atol=0)
        rewards[:] = 0
        for matrix in transitions:
            matrix.data[:] = 0
        if outcome == "exception":
            raise RuntimeError("broken function")
        return np.zeros(rewards.shape[0] - (outcome == "invalid"))

    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("broken", solve_function=broken_solver)
    bench.add_solver("VI", solve_function=VI)
    results = bench.run(discount=0.9, repeats=2)
    expected = "imprecise" if outcome == "imprecise" else "error"
    statuses = [row["status"] for row in results]
    assert statuses == [expected, expected, "success", "success"]
    if outcome == "exception":
        assert all(
            "RuntimeError: broken function" in row["error"] for row in results[:2]
        )
    elif outcome == "invalid":
        assert all("finite vector" in row["error"] for row in results[:2])
    else:
        assert results[0]["error_bound"] == pytest.approx(20)
    assert_allclose(chain.reward_matrix, rewards, rtol=0, atol=0)
    for matrix, original in zip(chain.transition_matrix, transitions):
        assert_allclose(matrix.toarray(), original.toarray(), rtol=0, atol=0)


def test_catalogue_model_uses_own_defaults(isolated_model_cache):
    from mdpforge.models.rooms import Model

    expected = Model()
    bench = Benchmark().add_mdp("rooms")
    model = bench._mdps[0]
    assert model.name == expected.name
    assert (model.state_dim, model.action_dim) == (
        expected.state_dim,
        expected.action_dim,
    )
    validate_model(model)
    results = bench.run(discount=0.9, repeats=1, seed=0)
    assert len(results) == 1
    for row in results:
        assert row["status"] == "success", row["error"]
        assert row["error_bound"] <= 1e-3


def test_impatience_catalogue_model_uses_current_mdp_contract(isolated_model_cache):
    from mdpforge.models.impatience import Model

    expected = Model()
    bench = Benchmark(default_solvers=False).add_mdp("impatience")
    model = bench._mdps[0]
    assert isinstance(model, MDP)
    assert (model.state_dim, model.action_dim) == (
        expected.state_dim,
        expected.action_dim,
    )
    validate_model(model)


def test_unknown_model_lists_catalogue_without_importing_it(monkeypatch):
    def fail_import(*args, **kwargs):
        pytest.fail("Listing model names must not import model dependencies")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    with pytest.raises(
        ValueError, match="Unknown model 'roooms'.*Available models:.*rooms"
    ):
        Benchmark().add_mdp("roooms")


def test_catalogue_model_rejects_matrices_and_duplicates(chain, isolated_model_cache):
    bench = Benchmark()
    with pytest.raises(ValueError, match="choose a custom name"):
        bench.add_mdp(
            "rooms", transitions=chain.transition_matrix, reward=chain.reward_matrix
        )
    bench.add_mdp("rooms")
    with pytest.raises(ValueError, match="already registered"):
        bench.add_mdp("rooms")


@pytest.mark.parametrize("option", ["state_dim", "action_dim"])
def test_add_mdp_leaves_dimensions_to_model(option):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Benchmark().add_mdp("rooms", **{option: 10})


@pytest.mark.parametrize("dense_input", [False, True])
def test_matrix_benchmark_uses_default_solvers(chain, dense_input):
    transitions = chain.transition_matrix
    if dense_input:
        transitions = np.array([matrix.toarray() for matrix in transitions])
    bench = Benchmark().add_mdp(
        "my_mdp", transitions=transitions, reward=chain.reward_matrix.tolist()
    )

    results = bench.run(discount=0.9)
    assert len(results) == 3
    assert {row["solver"] for row in results} == {"VI"}
    for row in results:
        assert row["model"] == "my_mdp"
        assert (row["state_dim"], row["action_dim"]) == (3, 2)
        assert row["status"] == "success", row["error"]
        assert row["error_bound"] <= 1e-3


def test_matrix_benchmark_rejects_missing_matrices_and_duplicate_names(chain):
    bench = Benchmark()
    with pytest.raises(ValueError, match="both transitions and reward"):
        bench.add_mdp("incomplete", transitions=chain.transition_matrix)
    with pytest.raises(ValueError, match="Pass a name"):
        bench.add_mdp(chain, reward=chain.reward_matrix)
    bench.add_mdp(
        "my_mdp", transitions=chain.transition_matrix, reward=chain.reward_matrix
    )
    with pytest.raises(ValueError, match="already registered"):
        bench.add_mdp(
            "my_mdp", transitions=chain.transition_matrix, reward=chain.reward_matrix
        )


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
    bench = Benchmark(default_solvers=False)
    for model in (chain, matrix_mdp, external_mdp):
        bench.add_mdp(model)
    for name, constructor in (("VI", VI), ("MPI", MPI)):
        bench.add_solver(name, solve_function=constructor)

    results = bench.run(0.9, precision=1e-4, repeats=2, seed=0)
    assert len(results) == 12
    for row in results:
        assert row["status"] == "success", row["error"]
        assert row["residual"] <= 1e-4 * (1 - 0.9)
    for generated, matrix, external in zip(results[:4], results[4:8], results[8:]):
        for field in ("solver", "repeat", "seed", "residual", "error_bound", "status"):
            assert generated[field] == matrix[field] == external[field]
    assert all(
        isinstance(matrix, csr_matrix) for matrix in matrix_mdp.transition_matrix
    )


def test_benchmark_measures_vi_and_mpi(chain):
    chain = SimpleNamespace(
        name=chain.name,
        state_dim=chain.state_dim,
        action_dim=chain.action_dim,
        transition_matrix=chain.transition_matrix,
        reward_matrix=chain.reward_matrix,
    )
    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("VI", solve_function=VI)
    bench.add_solver("MPI", solve_function=MPI)
    results = bench.run(discount=0.9, repeats=2)
    assert len(results) == 4
    assert [(row["solver"], row["repeat"]) for row in results] == [
        ("VI", 1),
        ("VI", 2),
        ("MPI", 1),
        ("MPI", 2),
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

    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("mutating", solve_function=MutatingSolver)
    bench.add_solver("VI", solve_function=VI)
    results = bench.run(discount=0.9, repeats=2)
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

    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("broken", solve_function=BrokenSolver)
    bench.add_solver("VI", solve_function=VI)
    results = bench.run(discount=0.9, repeats=1)
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

    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("invalid", solve_function=InvalidSolver)
    row = bench.run(discount=0.9, repeats=1)[0]
    assert row["status"] == "error"
    assert "finite vector" in row["error"]


def test_benchmark_pairs_seeds_and_restores_rng_states(chain):
    samples = []

    class RandomSolver(VI):
        def run(self):
            samples.append((np.random.random(), random.random()))
            super().run()

    numpy_state, random_state = np.random.get_state(), random.getstate()
    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("first", solve_function=RandomSolver)
    bench.add_solver("second", solve_function=RandomSolver)
    results = bench.run(discount=0.9, repeats=2, seed=7)
    assert samples[:2] == samples[2:]
    assert samples[0] != samples[1]
    assert [row["seed"] for row in results] == [7, 8, 7, 8]
    current = np.random.get_state()
    assert current[0] == numpy_state[0]
    assert np.array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]
    assert random.getstate() == random_state


def test_export_csv_round_trip(chain, tmp_path):
    bench = Benchmark().add_mdp(chain)
    results = bench.run(discount=0.9, repeats=1)
    path = bench.export_csv(tmp_path / "nested" / "benchmark.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["solver"] == "VI"
    assert rows[0]["status"] == "success"
    assert float(rows[0]["runtime"]) == results[0]["runtime"]
