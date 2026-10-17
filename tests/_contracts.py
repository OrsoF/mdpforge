"""Scientific contracts shared by every compatible finite discounted MDP solver."""

from contextlib import contextmanager
from copy import deepcopy
from inspect import signature
from itertools import product
import random

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import bellman_operator, optimal_bellman_operator


@contextmanager
def reproducible_rng(seed=17):
    """Fix both public RNGs and restore the caller's state afterwards."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def matrix_model(name, transitions, rewards):
    """Create a CSR-backed model without touching the persistent model cache."""
    model = MDP.from_matrices(name, transitions, rewards)
    model.create_model(save=False)
    return model


def sample_models():
    """Exercise ties, negative rewards, stochastic transitions and general CSR rows."""
    yield matrix_model(
        "ties_and_negative_rewards",
        [np.eye(3), np.eye(3)],
        [[1, 0], [-2, -3], [3, 3]],
    )
    yield matrix_model(
        "stochastic_switches",
        [
            [[0.5, 0.5, 0], [0, 0.5, 0.5], [0.25, 0, 0.75]],
            [[0, 0, 1], [0.5, 0.5, 0], [0, 1, 0]],
        ],
        [[-1, 0], [2, -0.5], [0.5, 1]],
    )
    # A deterministic pseudo-random instance tests structure without a
    # fixed, hand-coded expected solution or global RNG side effects.
    rng = np.random.default_rng(481)
    matrices = []
    for _ in range(2):
        rows = rng.uniform(0.1, 1.0, size=(3, 3))
        matrices.append(rows / rows.sum(axis=1, keepdims=True))
    yield matrix_model("generated", matrices, rng.uniform(-2, 2, (3, 2)))


def assert_model_unchanged(model, reference):
    """Compare the exact CSR representation and rewards, not just their values."""
    assert (model.name, model.state_dim, model.action_dim) == (
        reference.name,
        reference.state_dim,
        reference.action_dim,
    )
    assert isinstance(model.reward_matrix, np.ndarray)
    assert model.reward_matrix.dtype == reference.reward_matrix.dtype
    assert_array_equal(model.reward_matrix, reference.reward_matrix)
    assert isinstance(model.transition_matrix, list)
    assert len(model.transition_matrix) == len(reference.transition_matrix)
    for matrix, original in zip(model.transition_matrix, reference.transition_matrix):
        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == original.shape
        assert matrix.dtype == original.dtype
        for field in ("data", "indices", "indptr"):
            assert_array_equal(getattr(matrix, field), getattr(original, field))


def evaluate_policy(model, policy, discount):
    """Evaluate a policy with a dense linear solve on tiny oracle instances."""
    states = np.arange(model.state_dim)
    matrices = np.stack([matrix.toarray() for matrix in model.transition_matrix])
    policy = np.asarray(policy, dtype=int)
    return np.linalg.solve(
        np.eye(model.state_dim) - discount * matrices[policy, states],
        model.reward_matrix[states, policy],
    )


def enumerate_optimal_value(model, discount):
    """Compute V* independently of all production solvers for tiny MDPs."""
    if model.action_dim**model.state_dim > 1024:
        raise ValueError("oracle is reserved for small MDPs")
    return np.max(
        [
            evaluate_policy(model, policy, discount)
            for policy in product(range(model.action_dim), repeat=model.state_dim)
        ],
        axis=0,
    )


def assert_optimal_solution(solver, model, discount, precision, *, expected=None):
    """Assert the value bound and the returned policy's actual optimality."""
    value = np.asarray(solver.value)
    assert value.shape == (model.state_dim,)
    assert np.isfinite(value).all()
    if expected is None:
        expected = enumerate_optimal_value(model, discount)
    assert_allclose(value, expected, atol=precision, rtol=0)
    residual = np.max(np.abs(optimal_bellman_operator(model, value, discount) - value))
    assert residual <= precision * (1 - discount) * (1 + 1e-8)

    policy = solver.policy
    if policy is None:
        policy = bellman_operator(model, value, discount).argmax(axis=1)
    policy = np.asarray(policy)
    assert policy.shape == (model.state_dim,)
    assert np.issubdtype(policy.dtype, np.integer)
    assert np.all((0 <= policy) & (policy < model.action_dim))
    policy_value = evaluate_policy(model, policy, discount)
    assert_allclose(policy_value, expected, atol=precision, rtol=0)
    assert np.isfinite(solver.runtime) and solver.runtime >= 0
    return value.copy(), policy_value


def run_solver_preserving_input(
    solver_class, reference, discount, precision, **options
):
    """Exercise the external interface and preserve inputs even on failure."""
    model = deepcopy(reference)
    with reproducible_rng():
        precision_options = (
            {"final_precision": precision}
            if "final_precision" in signature(solver_class).parameters
            else {}
        )
        solver = solver_class(model, discount, **precision_options, **options)
        assert_model_unchanged(model, reference)
        try:
            solver.run()
        finally:
            assert_model_unchanged(model, reference)
    return solver
