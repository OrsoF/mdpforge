import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import issparse

from mdpforge.core.validation import validate_model
from mdpforge.utils import persistence


def test_model_contract(model_spec, isolated_model_cache):
    model_class, state_dim, action_dim, expected_dimensions = model_spec
    model = model_class(state_dim, action_dim)
    model.create_model(save=False)
    assert (model.state_dim, model.action_dim) == expected_dimensions
    assert model.reward_matrix.shape == (model.state_dim, model.action_dim)
    assert np.all(np.isfinite(model.reward_matrix))
    assert len(model.transition_matrix) == model.action_dim
    for transition in model.transition_matrix:
        assert transition.shape == (model.state_dim, model.state_dim)
        probabilities = (
            transition.data if issparse(transition) else np.asarray(transition)
        )
        assert np.all(np.isfinite(probabilities))
        assert np.all(probabilities >= 0)
        assert_allclose(np.asarray(transition.sum(axis=1)).ravel(), 1, atol=1e-12)
    validate_model(model)
    assert not persistence.SAVED_MODELS_PATH.exists()
