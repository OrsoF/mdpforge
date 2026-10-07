import numpy as np
import pytest
from numpy.testing import assert_allclose

from core.model import GenericModel
from core.operators import optimal_bellman_operator
from utils import persistence


class ChainModel(GenericModel):
    def _build_model(self):
        self.transition_matrix = np.array(
            [[[0, 1, 0], [0, 0, 1], [0, 0, 1]], np.eye(3)], dtype=float
        )
        self.reward_matrix = np.array([[0, 0], [2, 0], [0, 0]], dtype=float)


@pytest.fixture
def chain(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "SAVED_MODELS_PATH", tmp_path / "models")
    monkeypatch.setattr(persistence, "SAVED_VALUE_FUNCTIONS_PATH", tmp_path / "values")
    model = ChainModel(3, 2)
    model.create_model(save=False)
    return model


@pytest.mark.parametrize("mode", ["numpy", "sparse"])
@pytest.mark.parametrize("discount", [0.0, 0.9, 1.0])
def test_optimal_value_on_cache_miss(chain, mode, discount):
    chain._convert_model(mode)
    value = chain.optimal_value_function(discount)
    assert value.shape == (chain.state_dim,)
    assert_allclose(value, [2 * discount, 2, 0], atol=1e-3, rtol=0)
    assert_allclose(
        optimal_bellman_operator(chain, value, discount), value, atol=1e-4, rtol=0
    )
    assert chain.get_model_type() == mode
    assert persistence.value_function_cache_path(f"{discount}_{chain.name}").is_file()


def test_optimal_value_reuses_persisted_cache(chain):
    expected = chain.optimal_value_function(0.9)
    # A fresh model can reuse the persisted value without rebuilding its matrices.
    cached_model = ChainModel(3, 2)
    assert_allclose(cached_model.optimal_value_function(0.9), expected, rtol=0, atol=0)


def test_optimal_value_cache_distinguishes_discounts(chain):
    assert_allclose(chain.optimal_value_function(0.5), [1, 2, 0], rtol=0, atol=1e-3)
    assert_allclose(chain.optimal_value_function(0.9), [1.8, 2, 0], rtol=0, atol=1e-3)
