from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mdpforge import list_models, load_model, models
from mdpforge.core.mdp import MDP
from mdpforge.core.validation import validate_model
from mdpforge.models.unichain import Model as Unichain
from mdpforge.utils import persistence


@pytest.fixture
def preset_catalogue(tmp_path, monkeypatch):
    # Include a missing backend recipe to verify metadata filtering imports only
    # selected models, plus an unavailable large preset to avoid silent fallback.
    metadata = {
        "category": "synthetic",
        "tags": ["random"],
        "sizes": {
            "small": {
                "state_dim": 4,
                "parameters": {"state_dim": 4},
                "source": "measured",
            },
            "large": {
                "state_dim": None,
                "parameters": None,
                "source": "unavailable",
                "reason": "No calibration",
            },
        },
    }
    (tmp_path / "unichain.py").write_text(f"METADATA = {metadata!r}\n", encoding="utf-8")
    (tmp_path / "unavailable.py").write_text(
        "import missing_model_dependency\n"
        "METADATA = {'category': 'navigation', 'tags': ['maze']}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(models, "files", lambda package: tmp_path)
    return metadata


def test_size_and_type_filters_do_not_import_models(preset_catalogue):
    assert [m["name"] for m in list_models(size="small", type="random")] == ["unichain"]
    assert list_models(size="large", type="random") == []
    assert [m["name"] for m in list_models(type="synthetic")] == ["unichain"]
    assert [m["name"] for m in list_models(type="maze")] == ["unavailable"]


@pytest.mark.parametrize("options", [{"size": "huge"}, {"type": "unknown"}, {"type": []}])
def test_bad_filters_fail_before_iteration(preset_catalogue, options):
    with pytest.raises(ValueError):
        load_model(**options)


def test_loader_is_lazy_and_produces_valid_matrices(
    preset_catalogue, isolated_model_cache, monkeypatch
):
    calls = []

    def import_recipe(name):
        calls.append(name)
        return SimpleNamespace(Model=Unichain)

    monkeypatch.setattr(models, "import_module", import_recipe)
    loaded = load_model(size="small", type="random", save=False)
    assert calls == []
    model = next(loaded)
    assert calls == ["mdpforge.models.unichain"]
    assert model.state_dim == 4
    validate_model(model)
    assert all(isinstance(p, csr_matrix) for p in model.transition_matrix)
    # Always stepping left reaches the zero-cost boundary in at most 3 steps.
    np.testing.assert_allclose(model.reward_matrix[:, 0], [0, -1, -1, -1])
    assert model.transition_matrix[0][3, 2] == 1
    assert not persistence.model_cache_path(model.name).exists()
    assert list(loaded) == []


def test_defaults_and_internal_cache_loader_are_preserved(
    preset_catalogue, isolated_model_cache, monkeypatch
):
    monkeypatch.setattr(
        models, "import_module", lambda name: SimpleNamespace(Model=Unichain)
    )
    original = next(load_model(type="random"))
    assert original.state_dim == Unichain().state_dim
    assert persistence.model_cache_path(original.name).exists()
    restored = Unichain()
    assert persistence.load_model(restored) is True
    np.testing.assert_array_equal(restored.reward_matrix, original.reward_matrix)


def test_missing_dependencies_are_reported(preset_catalogue, monkeypatch):
    def fail(name):
        raise ModuleNotFoundError("missing_model_dependency")

    monkeypatch.setattr(models, "import_module", fail)
    with pytest.raises(ModuleNotFoundError, match="missing_model_dependency"):
        next(load_model(type="maze"))


def test_stale_dimension_cache_is_reported(preset_catalogue, isolated_model_cache):
    model = Unichain(state_dim=4)
    model.create_model()
    wrong = MDP.from_matrices("wrong", [np.eye(2)], np.zeros((2, 1)))
    persistence.save_pickle(
        persistence.pack_model(wrong),
        persistence.model_cache_path(model.name),
    )
    with pytest.raises(ValueError, match="expected 4 states, got 2"):
        next(load_model(size="small", type="random"))


def test_custom_track_is_converted_without_mutating_metadata(
    preset_catalogue, isolated_model_cache, monkeypatch
):
    entry = list_models(type="random")[0]
    track = [[1, 1], [1, 0]]
    entry["sizes"]["small"]["parameters"]["custom_track"] = track
    monkeypatch.setattr(models, "list_models", lambda **kwargs: [entry])

    def create_recipe(state_dim, custom_track):
        assert isinstance(custom_track, np.ndarray)
        np.testing.assert_array_equal(custom_track, track)
        custom_track[0, 0] = 0
        return Unichain(state_dim=state_dim)

    monkeypatch.setattr(
        models, "import_module", lambda name: SimpleNamespace(Model=create_recipe)
    )
    model = next(load_model(size="small", save=False))
    assert model.state_dim == 4
    assert track == [[1, 1], [1, 0]]
