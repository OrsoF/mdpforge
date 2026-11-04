"""Model and optimal-value cache persistence."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import coo_matrix, csr_array, csr_matrix, lil_matrix

from mdpforge.core.mdp import MDP
from mdpforge.utils import persistence


@pytest.mark.parametrize("matrix_type", [np.array, coo_matrix, lil_matrix, csr_array])
def test_create_model_finalizes_and_saves_csr(chain, matrix_type, monkeypatch):
    matrices = [matrix.toarray() for matrix in chain.transition_matrix]
    model = type(chain)(3, 2)

    def build():
        model.transition_matrix = [matrix_type(matrix) for matrix in matrices]
        model.reward_matrix = chain.reward_matrix.copy()

    monkeypatch.setattr(model, "_build_model", build)
    model.create_model()
    assert persistence.model_cache_path(model).is_file()

    def forbidden_build():
        pytest.fail("A cached model must be loaded without rebuilding its matrices")

    restored = type(chain)(3, 2)
    monkeypatch.setattr(restored, "_build_model", forbidden_build)
    restored.create_model(cache="readonly")
    for current in (model, restored):
        assert isinstance(current.transition_matrix, list)
        assert len(current.transition_matrix) == len(matrices)
        assert all(isinstance(matrix, csr_matrix) for matrix in current.transition_matrix)
        for matrix, expected in zip(current.transition_matrix, matrices):
            assert_allclose(matrix.toarray(), expected, rtol=0, atol=0)
        assert_allclose(current.reward_matrix, chain.reward_matrix, rtol=0, atol=0)


def test_matrix_mdp_values_are_cached_by_discount_and_name(chain):
    model = MDP.from_matrices(
        "matrix_chain", chain.transition_matrix, chain.reward_matrix
    )
    expected = model.optimal_value_function(0.9)
    assert_allclose(expected, [1.8, 2, 0], rtol=0, atol=1e-3)
    path = persistence.value_function_cache_path(model, 0.9)
    assert path.is_file()
    assert path.name == f"0.9_{model.name}.npz"

    other_discount = model.optimal_value_function(0.5)
    assert_allclose(other_discount, [1, 2, 0], rtol=0, atol=1e-3)
    other_discount_path = persistence.value_function_cache_path(model, 0.5)
    assert other_discount_path != path
    assert other_discount_path.is_file()

    equivalent = MDP.from_matrices(
        "some_other_name", chain.transition_matrix, chain.reward_matrix
    )
    equivalent_path = persistence.value_function_cache_path(equivalent, 0.9)
    assert equivalent_path != path
    assert_allclose(equivalent.optimal_value_function(0.9), expected, rtol=0, atol=0)
    assert equivalent_path.is_file()

    def forbidden_compute():
        pytest.fail("A matching cached value function must not be recomputed")

    for current, discount, values in (
        (model, 0.9, expected),
        (model, 0.5, other_discount),
        (equivalent, 0.9, expected),
    ):
        restored = persistence.get_cached_value_function(
            current, discount, 1e-3, forbidden_compute, cache="readonly"
        )
        assert_allclose(restored, values, rtol=0, atol=0)

    unbuilt = MDP(3, 2)
    unbuilt.name = model.name
    with pytest.raises(ValueError, match="Build the MDP"):
        unbuilt.optimal_value_function(0.9)
