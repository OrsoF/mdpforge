from importlib import import_module

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import (
    compute_transition_reward_policy,
    optimal_bellman_operator,
)
from mdpforge.utils.bellman import iterative_policy_evaluation
from tests._contracts import assert_optimal_solution


class RecurrentModel(MDP):
    def _build_model(self):
        self.transition_matrix = [csr_matrix(np.eye(2)), csr_matrix(np.eye(2))]
        self.reward_matrix = np.array([[1.0, 0.0], [2.0, 0.0]])


@pytest.fixture
def recurrent_model(isolated_model_cache):
    model = RecurrentModel(2, 2)
    model.create_model(save=False)
    return model


def test_bertsekas_pi_final_value_precision(recurrent_model):
    from mdpforge.solvers.bertsekas_pi import Solver

    discount, precision = 0.99, 1e-3
    # One aggregate region leaves state-dependent evaluation error to resolve.
    solver = Solver(recurrent_model, discount, precision, n_regions=1)
    solver.run()

    residual = (
        optimal_bellman_operator(recurrent_model, solver.value, discount) - solver.value
    )
    assert np.max(np.abs(residual)) / (1 - discount) <= precision
    assert_allclose(solver.value, [100, 200], atol=precision, rtol=0)
    np.testing.assert_array_equal(solver.policy, [0, 0])


@pytest.mark.parametrize(
    "module_name, options",
    [
        ("mdpforge_mpi", {}),
        ("mdpforge_mpi_reward_init", {}),
        ("aggregated_mpi", {}),
        ("aggregated_mpi", {"refinement": "tiles", "n_tiles": 2}),
    ],
)
def test_mpi_precision_with_stable_policy(
    module_name, options, recurrent_model, monkeypatch
):
    monkeypatch.setattr(np.random, "randint", np.random.RandomState(0).randint)
    solver = import_module(f"mdpforge.solvers.{module_name}").Solver(
        recurrent_model, 0.9, final_precision=1e-4, **options
    )
    # Truncation of the inner evaluation must not allow a stable policy to
    # bypass the outer precision criterion.
    if module_name == "mdpforge_mpi":
        solver.max_evaluation_iterations = 1
    elif module_name == "mdpforge_mpi_reward_init":
        solver.max_evaluation_iterations = 1
    solver.run()

    assert_optimal_solution(solver, recurrent_model, 0.9, 1e-4)
    assert (
        np.max(
            np.abs(
                optimal_bellman_operator(recurrent_model, solver.value, 0.9)
                - solver.value
            )
        )
        < 1e-5
    )
    transition, reward = compute_transition_reward_policy(
        recurrent_model, solver.policy
    )
    policy_value = np.linalg.solve(np.eye(2) - 0.9 * transition.toarray(), reward)
    assert_allclose(policy_value, [10, 20], atol=1e-12, rtol=0)


@pytest.mark.parametrize("max_iterations", [None, 10])
def test_inner_evaluation_threshold(recurrent_model, max_iterations):
    discount = 0.5
    transition, reward = compute_transition_reward_policy(
        recurrent_model, np.zeros(2, dtype=int)
    )
    threshold = 0.6 * (1 - discount)
    value = iterative_policy_evaluation(
        transition,
        reward,
        discount,
        threshold,
        np.zeros(2),
        max_iterations=max_iterations,
    )
    assert_allclose(value, [1.875, 3.75])
    residual = optimal_bellman_operator(recurrent_model, value, discount) - value
    assert np.max(np.abs(residual)) < threshold


def test_inner_evaluation_keeps_last_update_at_cap(recurrent_model):
    transition, reward = compute_transition_reward_policy(
        recurrent_model, np.zeros(2, dtype=int)
    )
    initial_value = np.zeros(2)
    initial_value.flags.writeable = False
    value = iterative_policy_evaluation(
        transition, reward, 0.9, 1e-5, initial_value, max_iterations=2
    )
    assert_allclose(value, [1.9, 3.8])
    assert_allclose(initial_value, [0, 0], rtol=0, atol=0)


def test_inner_evaluation_without_initial_value(recurrent_model):
    transition, reward = compute_transition_reward_policy(
        recurrent_model, np.zeros(2, dtype=int)
    )
    value = iterative_policy_evaluation(transition, reward, 0.5, 1e-7, None)
    assert_allclose(value, [2, 4], atol=1e-7, rtol=0)


@pytest.mark.parametrize("max_iterations", [0, -1, 1.5, True])
def test_inner_evaluation_rejects_invalid_budget(recurrent_model, max_iterations):
    transition, reward = compute_transition_reward_policy(
        recurrent_model, np.zeros(2, dtype=int)
    )
    with pytest.raises(ValueError, match="max_iterations"):
        iterative_policy_evaluation(
            transition, reward, 0.9, 1e-5, None, max_iterations=max_iterations
        )


def test_reward_initialized_mpi_rejects_uncertified_iteration_limit(recurrent_model):
    from mdpforge.solvers.mdpforge_mpi_reward_init import Solver

    solver = Solver(recurrent_model, 0.9, final_precision=1e-4)
    solver.max_evaluation_iterations = 1
    solver.max_iterations = 1

    with pytest.raises(RuntimeError, match="max_iterations"):
        solver.run()


@pytest.mark.parametrize(
    "module_name",
    ["mdpforge_mpi", "mdpforge_mpi_reward_init"],
)
def test_truncated_evaluation_can_improve_initially_stable_policy(
    module_name, recurrent_model, monkeypatch
):
    recurrent_model.transition_matrix[1] = csr_matrix([[0.0, 1.0], [0.0, 1.0]])
    monkeypatch.setattr(
        np.random, "randint", lambda *args, **kwargs: np.zeros(2, dtype=int)
    )
    solver = import_module(f"mdpforge.solvers.{module_name}").Solver(
        recurrent_model, 0.9, final_precision=1e-4
    )
    if module_name == "mdpforge_mpi":
        solver.max_evaluation_iterations = 1
    elif module_name == "mdpforge_mpi_reward_init":
        solver.max_evaluation_iterations = 1
    solver.run()
    assert_allclose(solver.value, [18, 20], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(solver.policy, [1, 0])


@pytest.mark.parametrize(
    "module_name",
    ["mdpforge_mpi", "mdpforge_mpi_reward_init"],
)
@pytest.mark.parametrize("discount", [0.1, 0.99])
def test_mdpforge_mpi_transient_rewards(module_name, discount, chain, monkeypatch):
    monkeypatch.setattr(np.random, "randint", np.random.RandomState(0).randint)
    solver = import_module(f"mdpforge.solvers.{module_name}").Solver(
        chain, discount, final_precision=1e-4
    )
    solver.run()
    assert_allclose(solver.value, [2 * discount, 2, 0], atol=1e-4, rtol=0)
