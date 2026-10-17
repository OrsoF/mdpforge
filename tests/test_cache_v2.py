"""Identity, integrity and policy contracts for versioned numerical caches."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP
from mdpforge.utils import persistence
from mdpforge.utils.cache_keys import numerical_identity
from mdpforge.utils.cache_store import CacheCorruptError


class RecipeModel(MDP):
    CACHE_VERSION = 1
    builds = 0

    def __init__(self, reward=2.0):
        super().__init__(2, 2)
        self.reward = reward
        self.name = "shared_name"

    def _build_model(self):
        type(self).builds += 1
        self.transition_matrix = [csr_matrix(np.eye(2)) for _ in range(2)]
        self.reward_matrix = np.array([[self.reward, 0], [0, 0]], dtype=float)


class UnspecifiedModel(RecipeModel):
    CACHE_VERSION = None


@pytest.fixture
def caches(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "SAVED_MODELS_PATH", tmp_path / "models")
    monkeypatch.setattr(persistence, "SAVED_VALUE_FUNCTIONS_PATH", tmp_path / "values")
    RecipeModel.builds = 0
    return tmp_path


def test_recipes_same_name_with_different_parameters_do_not_collide(caches):
    first = RecipeModel(reward=2.0)
    second = RecipeModel(reward=4.0)
    first.create_model(cache="readwrite")
    second.create_model(cache="readwrite")
    assert RecipeModel.builds == 2
    assert persistence.model_cache_path(first) != persistence.model_cache_path(second)
    restored = RecipeModel(reward=2.0)
    restored.create_model(cache="readwrite")
    assert RecipeModel.builds == 2
    assert_allclose(restored.reward_matrix, first.reward_matrix)


@pytest.mark.parametrize(
    "mode, expected_builds, has_cache",
    [
        ("off", 1, False),
        ("readonly", 1, False),
        ("readwrite", 1, True),
        ("refresh", 1, True),
    ],
)
def test_cache_modes_on_miss(caches, mode, expected_builds, has_cache):
    model = RecipeModel()
    model.create_model(cache=mode)
    assert RecipeModel.builds == expected_builds
    assert persistence.model_cache_path(model).exists() is has_cache


def test_refresh_rebuilds_while_readonly_reuses_without_writing(caches):
    RecipeModel().create_model(cache="readwrite")
    RecipeModel().create_model(cache="readonly")
    assert RecipeModel.builds == 1
    RecipeModel().create_model(cache="refresh")
    assert RecipeModel.builds == 2


def test_uncacheable_recipe_never_reads_or_writes(caches):
    UnspecifiedModel().create_model(cache="readwrite")
    UnspecifiedModel().create_model(cache="readwrite")
    assert UnspecifiedModel.builds == 2
    assert persistence.model_cache_path(UnspecifiedModel()) is None


def test_corrupt_model_is_rebuilt_or_raises_in_readonly_mode(caches):
    model = RecipeModel()
    model.create_model(cache="readwrite")
    path = persistence.model_cache_path(model)
    path.write_bytes(b"broken ZIP")
    with pytest.raises(CacheCorruptError):
        RecipeModel().create_model(cache="readonly")
    repaired = RecipeModel()
    repaired.create_model(cache="readwrite")
    assert RecipeModel.builds == 2
    assert_allclose(repaired.reward_matrix, model.reward_matrix)


def test_cache_hit_and_miss_apply_identical_validation(caches):
    original = RecipeModel()
    original.create_model(check_transition=True, cache="readwrite")
    cached = RecipeModel()
    cached.create_model(check_transition=True, cache="readonly")
    assert_allclose(cached.transition_matrix[0].toarray(), np.eye(2))


def test_normalization_changes_model_cache_key(caches):
    raw = RecipeModel()
    raw.create_model(cache="readwrite")
    normalized = RecipeModel()
    normalized.create_model(normalize_reward=True, cache="readwrite")
    assert persistence.model_cache_path(raw) != persistence.model_cache_path(normalized)
    assert_allclose(normalized.reward_matrix, [[1, 0], [0, 0]])
    cached = RecipeModel()
    cached.create_model(normalize_reward=True, cache="readonly")
    assert_allclose(cached.reward_matrix, normalized.reward_matrix)


def test_value_cache_tracks_matrices_and_precision_not_model_names(caches):
    model = RecipeModel()
    model.create_model(cache="off")
    coarse = model.optimal_value_function(0.9)
    coarse_path = persistence.value_function_cache_path(model, 0.9)
    assert coarse_path.exists()
    precise_path = persistence.value_function_cache_path(model, 0.9, 1e-6)
    assert coarse_path != precise_path
    precise = model.optimal_value_function(0.9, cache="off")
    assert_allclose(coarse, precise)

    same = RecipeModel()
    same.name = "different_label"
    same.create_model(cache="off")
    assert numerical_identity(same) == numerical_identity(model)
    assert persistence.value_function_cache_path(same, 0.9) == coarse_path

    changed = RecipeModel(reward=4.0)
    changed.create_model(cache="off")
    assert persistence.value_function_cache_path(changed, 0.9) != coarse_path
    changed.reward_matrix[0, 0] = 3.0
    assert persistence.value_function_cache_path(changed, 0.9) != coarse_path


def test_value_cache_corruption_and_cache_off(caches):
    model = RecipeModel()
    model.create_model(cache="off")
    expected = model.optimal_value_function(0.9)
    path = persistence.value_function_cache_path(model, 0.9)
    path.write_bytes(b"broken ZIP")
    with pytest.raises(CacheCorruptError):
        model.optimal_value_function(0.9, cache="readonly")
    assert_allclose(model.optimal_value_function(0.9, cache="off"), expected)
    assert path.read_bytes() == b"broken ZIP"
    assert_allclose(model.optimal_value_function(0.9), expected)
    assert path.read_bytes() != b"broken ZIP"


def test_cache_rejects_name_only_value_lookups(caches):
    with pytest.raises(TypeError, match="model instance"):
        persistence.model_cache_path("model_name")
    with pytest.raises(ValueError, match="Build the MDP"):
        persistence.value_function_cache_path(RecipeModel(), 0.9)


def test_legacy_import_requires_explicit_trust(caches):
    model = RecipeModel()
    with pytest.raises(ValueError, match="trusted=True"):
        persistence.import_legacy_model_cache(model)
