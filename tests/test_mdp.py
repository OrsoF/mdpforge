import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import coo_matrix, csr_matrix

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.utils import persistence


def test_generated_model_uses_mdp_and_preserves_identity(chain):
    assert isinstance(chain, MDP)
    assert chain.name == "3_2_chainmodel"


def test_matrix_mdp_create_model_keeps_supplied_matrices(chain, monkeypatch):
    model = MDP.from_matrices(
        "matrix_chain", chain.transition_matrix, chain.reward_matrix
    )

    def fail(*args, **kwargs):
        pytest.fail("A matrix-defined MDP must not load or rebuild its matrices")

    monkeypatch.setattr("mdpforge.core.mdp.load_model", fail)
    monkeypatch.setattr(model, "_build_model", fail)
    model.create_model()
    assert_allclose(model.reward_matrix, chain.reward_matrix, rtol=0, atol=0)
    for actual, expected in zip(model.transition_matrix, chain.transition_matrix):
        assert_allclose(actual.toarray(), expected.toarray(), rtol=0, atol=0)


def test_matrix_mdp_shares_densities_and_state_relabeling(chain):
    model = MDP.from_matrices(
        "matrix_chain", chain.transition_matrix, chain.reward_matrix
    )
    assert model.get_transition_density() == chain.get_transition_density()
    assert model.get_reward_density() == chain.get_reward_density()
    value = np.array([1, 3, 2], dtype=float)
    expected = optimal_bellman_operator(model, value, 0.9)
    permutation = model.randomize_states(seed=0)
    assert_allclose(
        optimal_bellman_operator(model, value[permutation], 0.9),
        expected[permutation],
        rtol=0,
        atol=0,
    )
    assert all(isinstance(matrix, csr_matrix) for matrix in model.transition_matrix)


def test_matrix_mdp_reuses_standard_model_and_value_caches(chain):
    model = MDP.from_matrices(
        "matrix_chain", chain.transition_matrix, chain.reward_matrix
    )
    persistence.save_model(model)
    expected = model.optimal_value_function(0.9)
    assert isinstance(expected, np.ndarray)
    assert_allclose(expected, [1.8, 2, 0], rtol=0, atol=1e-3)
    assert persistence.value_function_cache_path("0.9_matrix_chain").is_file()

    restored = MDP(3, 2)
    restored.name = model.name
    # The value cache can be reused before loading the matrices.
    assert_allclose(restored.optimal_value_function(0.9), expected, rtol=0, atol=0)
    restored.create_model(save=False)
    assert_allclose(restored.reward_matrix, model.reward_matrix, rtol=0, atol=0)
    for actual, original in zip(restored.transition_matrix, model.transition_matrix):
        assert isinstance(actual, csr_matrix)
        assert_allclose(actual.toarray(), original.toarray(), rtol=0, atol=0)


@pytest.mark.parametrize("input_format", ["dense", "csr", "coo"])
def test_matrix_mdp_infers_dimensions_and_finalizes_csr(chain, input_format):
    transitions = [matrix.toarray() for matrix in chain.transition_matrix]
    if input_format == "dense":
        transitions = np.array(transitions)
    else:
        matrix_type = csr_matrix if input_format == "csr" else coo_matrix
        transitions = [matrix_type(matrix) for matrix in transitions]
    model = MDP.from_matrices("matrix_chain", transitions, chain.reward_matrix.tolist())

    assert model.name == "matrix_chain"
    assert (model.state_dim, model.action_dim) == (3, 2)
    assert isinstance(model.transition_matrix, list)
    assert isinstance(model.reward_matrix, np.ndarray)
    assert_allclose(model.reward_matrix, chain.reward_matrix, rtol=0, atol=0)
    for actual, expected in zip(model.transition_matrix, chain.transition_matrix):
        assert isinstance(actual, csr_matrix)
        assert_allclose(actual.toarray(), expected.toarray(), rtol=0, atol=0)


@pytest.mark.parametrize("name", [None, 0, "", " "])
def test_matrix_mdp_rejects_invalid_names(chain, name):
    with pytest.raises(ValueError, match="nonempty string"):
        MDP.from_matrices(name, chain.transition_matrix, chain.reward_matrix)


@pytest.mark.parametrize(
    "invalid_input, message",
    [
        ("reward_dimensions", "reward_matrix must have shape"),
        ("action_count", "contains 1 actions, expected 2"),
        ("transition_shape", "has shape"),
        ("negative_probability", "negative probabilities"),
        ("row_sum", "not stochastic"),
        ("transition_nan", "NaN or infinite values"),
        ("reward_inf", "NaN or infinite values"),
    ],
)
def test_matrix_mdp_rejects_invalid_data(chain, invalid_input, message):
    transitions = [matrix.toarray() for matrix in chain.transition_matrix]
    rewards = chain.reward_matrix.copy()
    if invalid_input == "reward_dimensions":
        rewards = rewards.ravel()
    elif invalid_input == "action_count":
        transitions = transitions[:1]
    elif invalid_input == "transition_shape":
        transitions[0] = np.eye(2)
    elif invalid_input == "negative_probability":
        transitions[0][0, 1] = -1
    elif invalid_input == "row_sum":
        transitions[0][0, 1] = 0.5
    elif invalid_input == "transition_nan":
        transitions[0][0, 1] = np.nan
    elif invalid_input == "reward_inf":
        rewards[0, 0] = np.inf

    with pytest.raises(ValueError, match=message):
        MDP.from_matrices("invalid_mdp", transitions, rewards)
