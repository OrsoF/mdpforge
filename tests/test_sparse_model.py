from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import coo_matrix, csr_array, csr_matrix, lil_matrix

from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.solvers.personal_mpi_init import Solver as MPI
from mdpforge.solvers.personal_pi import Solver as PI
from mdpforge.solvers.personal_stochasticmpi import Solver as StochasticMPI
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
    stored = persistence.load_pickle(persistence.model_cache_path(model.name))
    for transitions in (model.transition_matrix, stored[2]):
        assert isinstance(transitions, list)
        assert all(isinstance(matrix, csr_matrix) for matrix in transitions)
        for matrix, expected in zip(transitions, matrices):
            assert_allclose(matrix.toarray(), expected, rtol=0, atol=0)
    restored = type(chain)(3, 2)
    restored.create_model(save=False)
    assert all(isinstance(matrix, csr_matrix) for matrix in restored.transition_matrix)
    assert_allclose(restored.reward_matrix, chain.reward_matrix, rtol=0, atol=0)


def test_dense_cache_is_loaded_as_csr_without_rebuilding(chain, monkeypatch):
    matrices = np.array([matrix.toarray() for matrix in chain.transition_matrix])
    persistence.save_pickle(
        (3, 2, matrices, chain.reward_matrix), persistence.model_cache_path(chain.name)
    )
    model = type(chain)(3, 2)

    def fail_build():
        pytest.fail("A cached model should not be rebuilt")

    monkeypatch.setattr(model, "_build_model", fail_build)
    model.create_model(save=False)
    assert all(isinstance(matrix, csr_matrix) for matrix in model.transition_matrix)
    for matrix, expected in zip(model.transition_matrix, matrices):
        assert_allclose(matrix.toarray(), expected, rtol=0, atol=0)


def test_create_model_finalizes_already_built_transitions(chain):
    chain.transition_matrix = np.array(
        [matrix.toarray() for matrix in chain.transition_matrix]
    )
    chain.create_model(save=False)
    assert all(isinstance(matrix, csr_matrix) for matrix in chain.transition_matrix)


def test_permutation_preserves_csr_and_bellman_values(chain):
    value = np.array([1, 3, 2], dtype=float)
    expected = optimal_bellman_operator(chain, value, 0.9)
    permutation = np.array([2, 0, 1])
    chain.randomize_states(permutation=permutation)
    assert all(isinstance(matrix, csr_matrix) for matrix in chain.transition_matrix)
    assert_allclose(
        optimal_bellman_operator(chain, value[permutation], 0.9), expected[permutation]
    )


@pytest.mark.parametrize("solver_class", [PI, MPI, StochasticMPI])
def test_policy_solvers_do_not_densify_transitions(chain, solver_class, monkeypatch):
    def fail_toarray(*args, **kwargs):
        pytest.fail("A policy solver must not densify transitions")

    monkeypatch.setattr(csr_matrix, "toarray", fail_toarray)
    solver = solver_class(deepcopy(chain), 0.9, final_precision=1e-4)
    solver.run()
    assert_allclose(solver.value, [1.8, 2, 0], atol=1e-4, rtol=0)
    assert all(
        isinstance(matrix, csr_matrix) for matrix in solver.model.transition_matrix
    )
