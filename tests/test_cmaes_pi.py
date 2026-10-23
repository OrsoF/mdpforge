"""Public options and scientific output checks for CMA-ES policy iteration."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

from mdpforge import Benchmark, list_solvers
from mdpforge.core.operators import optimal_bellman_operator


@pytest.fixture
def solver_class():
    pytest.importorskip("cma")
    from mdpforge.solvers.mdpforge_cmaes_pi import Solver

    return Solver


@pytest.fixture
def switching_model():
    return SimpleNamespace(
        name="cmaes_two_states",
        state_dim=2,
        action_dim=2,
        transition_matrix=[
            csr_matrix(np.eye(2)),
            csr_matrix([[0.0, 1.0], [0.0, 1.0]]),
        ],
        reward_matrix=np.array([[1.0, 0.0], [2.0, 0.0]]),
    )


@pytest.mark.parametrize("discount", [0.9, 0.99])
def test_optimal_solution_and_input_preservation(
    solver_class, switching_model, discount
):
    reference = deepcopy(switching_model)
    dense = np.stack([matrix.toarray() for matrix in reference.transition_matrix])

    solver = solver_class(switching_model, discount, final_precision=1e-4, seed=17)
    solver.run()
    expected = np.array([2 * discount, 2.0]) / (1 - discount)
    assert_allclose(solver.value, expected, atol=1e-4, rtol=0)
    assert_array_equal(solver.policy, [1, 0])
    residual = (
        optimal_bellman_operator(reference, solver.value, discount) - solver.value
    )
    assert np.max(np.abs(residual)) / (1 - discount) <= solver.final_precision
    # Evaluate the returned policy using an independent dense two-state oracle.
    states = np.arange(2)
    transition = dense[solver.policy, states]
    reward = reference.reward_matrix[states, solver.policy]
    policy_value = np.linalg.solve(np.eye(2) - discount * transition, reward)
    assert_allclose(policy_value, expected, atol=1e-4, rtol=0)
    assert_array_equal(switching_model.reward_matrix, reference.reward_matrix)
    for actual, original in zip(
        switching_model.transition_matrix, reference.transition_matrix
    ):
        for field in ("data", "indices", "indptr"):
            assert_array_equal(getattr(actual, field), getattr(original, field))


@pytest.mark.parametrize("seed", [0, 17])
def test_reproducible_benchmark_seeds(solver_class, switching_model, seed):
    numpy_state = np.random.get_state()
    try:
        results = []
        for _ in range(2):
            np.random.seed(seed)
            solver = solver_class(switching_model, 0.9, final_precision=1e-4)
            solver.run()
            results.append((solver.value.copy(), solver.policy.copy()))
        assert_array_equal(results[0][0], results[1][0])
        assert_array_equal(results[0][1], results[1][1])
    finally:
        np.random.set_state(numpy_state)


def test_explicit_seed_preserves_global_rng(solver_class, switching_model):
    before = np.random.get_state()
    solver_class(switching_model, 0.9, seed=17).run()
    after = np.random.get_state()
    assert before[0] == after[0]
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_inner_budget_is_an_error(solver_class, switching_model):
    solver = solver_class(
        switching_model,
        0.9,
        sigma0=1e-12,
        max_generations=1,
        seed=17,
    )
    with pytest.raises(RuntimeError, match="policy evaluation precision not reached"):
        solver.run()


def test_outer_budget_is_an_error(solver_class, switching_model):
    solver = solver_class(
        switching_model,
        0.9,
        initial_policy=np.ones(2, dtype=int),
        max_iterations=1,
        seed=17,
    )
    # Zero is the exact value of this policy, but not the optimal value.
    with pytest.raises(RuntimeError, match="policy iteration precision not reached"):
        solver.run()


@pytest.mark.parametrize(
    "options",
    [
        {"max_generations": 0},
        {"max_generations": 1.5},
        {"max_iterations": True},
        {"sigma0": np.nan},
        {"sigma0": 0},
        {"initial_policy": [0.5, 0]},
        {"initial_policy": [0, 2]},
        {"initial_value": [0, np.inf]},
        {"initial_value": [0]},
        {"seed": -1},
        {"seed": 2**32},
    ],
)
def test_invalid_options(solver_class, switching_model, options):
    with pytest.raises(ValueError):
        solver_class(switching_model, 0.9, **options)


def test_single_state_is_explicitly_unsupported(solver_class):
    model = SimpleNamespace(state_dim=1)
    with pytest.raises(ValueError, match="at least two states"):
        solver_class(model, 0.9)


def test_catalogue_and_benchmark(solver_class, switching_model):
    entries = list_solvers(family="mpi", backend="cma")
    assert [entry["name"] for entry in entries] == ["mdpforge_cmaes_pi"]
    bench = Benchmark(default_solvers=False).add_mdp(switching_model)
    bench.add_solver("mdpforge_cmaes_pi")
    result = bench.run(discount=0.9, precision=1e-4, repeats=1, seed=17)
    assert result.rows[0]["status"] == "success", result.rows[0]["error"]
    assert result.rows[0]["error_bound"] <= 1e-4
