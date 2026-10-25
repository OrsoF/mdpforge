import pickle
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import coo_matrix, csr_array, csr_matrix, lil_matrix

from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.solvers.mdpforge_mpi import Solver as PlainMPI
from mdpforge.solvers.mdpforge_mpi_reward_init import Solver as MPI
from mdpforge.solvers.mdpforge_pi import Solver as PI
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
    stored = type(chain)(3, 2)
    stored.create_model(cache="readonly")
    for transitions in (model.transition_matrix, stored.transition_matrix):
        assert isinstance(transitions, list)
        assert all(isinstance(matrix, csr_matrix) for matrix in transitions)
        for matrix, expected in zip(transitions, matrices):
            assert_allclose(matrix.toarray(), expected, rtol=0, atol=0)
    restored = type(chain)(3, 2)
    restored.create_model(save=False)
    assert all(isinstance(matrix, csr_matrix) for matrix in restored.transition_matrix)
    assert_allclose(restored.reward_matrix, chain.reward_matrix, rtol=0, atol=0)


def test_legacy_dense_cache_is_ignored(chain, monkeypatch):
    matrices = np.array([matrix.toarray() for matrix in chain.transition_matrix])
    path = persistence.SAVED_MODELS_PATH / f"{chain.name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps((3, 2, matrices, chain.reward_matrix)))
    model = type(chain)(3, 2)
    built = []
    real_build = model._build_model

    def build():
        built.append(True)
        real_build()

    monkeypatch.setattr(model, "_build_model", build)
    model.create_model(cache="readwrite")
    assert built == [True]
    assert all(isinstance(matrix, csr_matrix) for matrix in model.transition_matrix)


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


@pytest.mark.parametrize("solver_class", [PI, MPI, PlainMPI])
def test_policy_solvers_do_not_densify_transitions(chain, solver_class, monkeypatch):
    def fail_toarray(*args, **kwargs):
        pytest.fail("A policy solver must not densify transitions")

    monkeypatch.setattr(csr_matrix, "toarray", fail_toarray)
    precision_options = {} if solver_class is PI else {"final_precision": 1e-4}
    solver = solver_class(deepcopy(chain), 0.9, **precision_options)
    solver.run()
    assert all(
        isinstance(matrix, csr_matrix) for matrix in solver.model.transition_matrix
    )
