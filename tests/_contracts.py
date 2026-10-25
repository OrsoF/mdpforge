"""Shared model fixtures and solver input checks."""

import random
from contextlib import contextmanager
from copy import deepcopy
from inspect import signature

import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP


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
    model.create_model(cache="readonly")
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
            result = solver.run()
        finally:
            assert_model_unchanged(model, reference)
    return result
