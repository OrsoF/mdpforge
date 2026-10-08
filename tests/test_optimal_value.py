import pytest
from numpy.testing import assert_allclose

from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.utils import persistence
from mdpforge.utils.exact_value_function import distance_to_optimal, get_exact_value


@pytest.mark.parametrize("mode", ["numpy", "sparse"])
@pytest.mark.parametrize("discount", [0.5, 0.9, 0.99])
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
    cached_model = type(chain)(3, 2)
    assert_allclose(cached_model.optimal_value_function(0.9), expected, rtol=0, atol=0)


def test_optimal_value_cache_distinguishes_discounts(chain):
    assert_allclose(chain.optimal_value_function(0.5), [1, 2, 0], rtol=0, atol=1e-3)
    assert_allclose(chain.optimal_value_function(0.9), [1.8, 2, 0], rtol=0, atol=1e-3)


@pytest.mark.parametrize("discount", [-0.1, 0, 1, 1.1])
def test_reference_values_reject_invalid_discount(chain, discount):
    with pytest.raises(AssertionError, match="discount"):
        chain.optimal_value_function(discount)
    with pytest.raises(AssertionError, match="discount"):
        get_exact_value(chain, discount)


def test_distance_to_optimal_counts_uniform_error(chain, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mdpforge.utils.exact_value_function.SAVED_VALUE_FUNCTIONS_PATH", tmp_path
    )
    value = get_exact_value(chain, 0.9) + 2
    original = value.copy()
    assert distance_to_optimal(value, chain, 0.9) == pytest.approx(2)
    assert_allclose(value, original, rtol=0, atol=0)


def test_exact_value_ignores_old_span_reference(chain, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mdpforge.utils.exact_value_function.SAVED_VALUE_FUNCTIONS_PATH", tmp_path
    )
    persistence.save_pickle(
        [43.8, 44, 42], tmp_path / f"discounted_0.9_{chain.name}.pkl"
    )
    assert_allclose(get_exact_value(chain, 0.9), [1.8, 2, 0], atol=1e-6, rtol=0)
