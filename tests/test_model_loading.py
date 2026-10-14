from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mdpforge import Benchmark, list_models, models
from mdpforge.core.mdp import MDP
from mdpforge.core.validation import validate_model
from mdpforge.models.unichain import Model as Unichain
from mdpforge.utils import persistence


@pytest.fixture
def preset_catalogue(tmp_path, monkeypatch):
    # Include a missing backend recipe to verify metadata filtering imports only
    # selected models, plus an unavailable large preset to avoid silent fallback.
    metadata = {
        "category": "abstract",
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
    (tmp_path / "unichain.py").write_text(
        f"METADATA = {metadata!r}\n", encoding="utf-8"
    )
    (tmp_path / "unavailable.py").write_text(
        "import missing_model_dependency\nMETADATA = {'category': 'navigation'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(models, "files", lambda package: tmp_path)
    return metadata


def test_size_and_category_filters_do_not_import_models(preset_catalogue):
    assert [m["name"] for m in list_models(size="small", category="abstract")] == [
        "unichain"
    ]
    assert list_models(size="large", category="abstract") == []
    assert [m["name"] for m in list_models(category="abstract")] == ["unichain"]
    assert [m["name"] for m in list_models(category="navigation")] == ["unavailable"]


def test_defaults_and_internal_cache_loader_are_preserved(
    preset_catalogue, isolated_model_cache, monkeypatch
):
    monkeypatch.setattr(
        "mdpforge.core.benchmark.import_module",
        lambda name: SimpleNamespace(Model=Unichain),
    )
    bench = Benchmark().add_mdp(category="abstract")
    original = bench._mdps[0]
    assert original.state_dim == Unichain().state_dim
    assert persistence.model_cache_path(original.name).exists()
    restored = Unichain()
    assert persistence.load_model(restored) is True
    np.testing.assert_array_equal(restored.reward_matrix, original.reward_matrix)


def test_stale_dimension_cache_is_reported(preset_catalogue, isolated_model_cache):
    model = Unichain(state_dim=4)
    model.create_model()
    wrong = MDP.from_matrices("wrong", [np.eye(2)], np.zeros((2, 1)))
    persistence.save_pickle(
        persistence.pack_model(wrong),
        persistence.model_cache_path(model.name),
    )
    bench = Benchmark()
    with pytest.raises(ValueError, match="expected 4 states, got 2"):
        bench.add_mdp(size="small", category="abstract")
    assert bench._mdps == []


def test_custom_track_is_converted_without_mutating_metadata(
    preset_catalogue, isolated_model_cache, monkeypatch
):
    entry = list_models(category="abstract")[0]
    track = [[1, 1], [1, 0]]
    entry["sizes"]["small"]["parameters"]["custom_track"] = track
    monkeypatch.setattr(models, "list_models", lambda **kwargs: [entry])

    def create_recipe(state_dim, custom_track):
        assert isinstance(custom_track, np.ndarray)
        np.testing.assert_array_equal(custom_track, track)
        custom_track[0, 0] = 0
        return Unichain(state_dim=state_dim)

    monkeypatch.setattr(
        "mdpforge.core.benchmark.import_module",
        lambda name: SimpleNamespace(Model=create_recipe),
    )
    bench = Benchmark().add_mdp(size="small")
    model = bench._mdps[0]
    assert model.state_dim == 4
    assert track == [[1, 1], [1, 0]]


@pytest.mark.parametrize(
    "options, expected_states",
    [
        ({"size": "small"}, 4),
        ({"category": "abstract"}, 100),
        ({"size": "small", "category": "abstract"}, 4),
        ({"model": "unichain", "size": "small"}, 4),
        ({"model": "unichain", "category": "abstract"}, 100),
        ({"model": "unichain", "size": "small", "category": "abstract"}, 4),
    ],
)
def test_benchmark_selects_catalogue_models(
    preset_catalogue, isolated_model_cache, monkeypatch, options, expected_states
):
    imports = []

    def import_recipe(name):
        imports.append(name)
        assert name == "mdpforge.models.unichain"
        return SimpleNamespace(Model=Unichain)

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", import_recipe)
    bench = Benchmark()
    assert bench.add_mdp(**options) is bench
    assert imports == ["mdpforge.models.unichain"]
    assert len(bench._mdps) == 1
    model = bench._mdps[0]
    assert model.state_dim == expected_states
    assert model.action_dim == 2
    validate_model(model)
    assert all(isinstance(p, csr_matrix) for p in model.transition_matrix)
    np.testing.assert_allclose(
        model.reward_matrix[:, 0], [0] + [-1] * (expected_states - 1)
    )
    if expected_states == 4:
        results = bench.run(discount=0.9, precision=1e-4, repeats=1)
        assert len(results.rows) == 1
        assert results.rows[0]["status"] == "success", results.rows[0]["error"]
        assert results.rows[0]["error_bound"] <= 1e-4


def test_benchmark_selection_preserves_existing_instances_and_previous_result(
    preset_catalogue, isolated_model_cache, monkeypatch
):
    existing = Unichain(state_dim=4)
    existing.create_model()
    bench = Benchmark().add_mdp(existing)
    result = bench.run(0.9, repeats=1)
    snapshot = deepcopy((result.rows, result.experiment))

    def fail_build(self):
        pytest.fail("Already-registered models must not build or load matrices")

    monkeypatch.setattr(Unichain, "create_model", fail_build)
    for _ in range(2):
        assert bench.add_mdp(size="small", category="abstract") is bench
    assert len(bench._mdps) == 1
    assert bench._mdps[0] is existing
    assert (result.rows, result.experiment) == snapshot
    with pytest.raises(ValueError, match="already registered"):
        bench.add_mdp(existing)
    with pytest.raises(ValueError, match="already registered"):
        bench.add_mdp("unichain", size="small")


def test_benchmark_selection_excludes_unavailable_sizes(preset_catalogue, monkeypatch):
    def fail_import(name):
        pytest.fail("Unavailable size presets must not be built")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    bench = Benchmark()
    assert bench.add_mdp(size="large", category="abstract") is bench
    assert bench._mdps == []


@pytest.mark.parametrize(
    "options, message",
    [
        ({"model": "unichain", "size": "large"}, "preset may be unavailable"),
        ({"model": "unichain", "category": "navigation"}, "does not match"),
        ({"model": "missing", "size": "small"}, "Unknown model 'missing'"),
        ({"model": "unichain", "size": "huge"}, "size must be"),
        ({"model": "unichain", "category": "unknown"}, "Unknown model category"),
        ({"model": " ", "size": "small"}, "nonempty string"),
    ],
)
def test_named_model_filters_fail_before_import(
    preset_catalogue, monkeypatch, options, message
):
    def fail_import(name):
        pytest.fail("Invalid named selections must fail before importing a recipe")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    bench = Benchmark()
    with pytest.raises(ValueError, match=message):
        bench.add_mdp(**options)
    assert bench._mdps == []


@pytest.mark.parametrize(
    "options, expected_states",
    [({}, 100), ({"size": "small", "category": "abstract"}, 4)],
)
def test_named_model_sequences_preserve_order_presets_and_existing_instances(
    preset_catalogue,
    isolated_model_cache,
    tmp_path,
    monkeypatch,
    options,
    expected_states,
):
    (tmp_path / "other.py").write_text(
        f"METADATA = {preset_catalogue!r}\n", encoding="utf-8"
    )

    def import_recipe(path):
        name = path.rsplit(".", 1)[1]

        def build(**parameters):
            model = Unichain(**parameters)
            model.name = f"{name}_{model.name}"
            return model

        return SimpleNamespace(Model=build)

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", import_recipe)
    bench = Benchmark()
    names = ["unichain", "other", "unichain"]
    assert bench.add_mdp(model=names, **options) is bench
    assert names == ["unichain", "other", "unichain"]
    existing = bench._mdps.copy()
    assert [model.name for model in existing] == [
        f"unichain_{expected_states}_2_unichain",
        f"other_{expected_states}_2_unichain",
    ]
    for model in existing:
        assert model.state_dim == expected_states
        validate_model(model)

    def fail_build(self):
        pytest.fail("Already-registered models must not build or load matrices")

    monkeypatch.setattr(Unichain, "create_model", fail_build)
    assert bench.add_mdp(model=tuple(names), **options) is bench
    assert len(bench._mdps) == 2
    assert all(actual is original for actual, original in zip(bench._mdps, existing))
    result = bench.run(0.9, precision=1e-4, repeats=1)
    assert len(result.rows) == 2
    assert all(row["status"] == "success" for row in result.rows)
    assert all(row["error_bound"] <= 1e-4 for row in result.rows)


@pytest.mark.parametrize(
    "names, options, message",
    [
        ([], {}, "nonempty sequence"),
        (["unichain", 1], {}, "nonempty sequence"),
        ([""], {}, "nonempty sequence"),
        ([" "], {}, "nonempty sequence"),
        (["unichain", "missing"], {}, "Unknown model 'missing'"),
        (["unichain", "unavailable"], {"size": "small"}, "does not match"),
        (["unichain"], {"size": "large"}, "preset may be unavailable"),
        (["unichain"], {"category": "navigation"}, "does not match"),
        (["unichain"], {"transitions": [np.eye(2)]}, "cannot be combined"),
    ],
)
def test_named_model_sequences_validate_all_names_before_registration(
    preset_catalogue, monkeypatch, names, options, message
):
    def fail_import(name):
        pytest.fail("Invalid named selections must fail before any recipe is imported")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    bench = Benchmark()
    with pytest.raises(ValueError, match=message):
        bench.add_mdp(model=names, **options)
    assert bench._mdps == []


@pytest.mark.parametrize("add_all", [False, True])
def test_catalogue_additions_check_duplicates_before_building(
    isolated_model_cache, monkeypatch, add_all
):
    existing = Unichain()
    existing.create_model()
    bench = Benchmark().add_mdp(existing)

    def fail_build(self):
        pytest.fail("Already-registered models must not build or load matrices")

    monkeypatch.setattr(Unichain, "create_model", fail_build)
    monkeypatch.setattr(models, "list_models", lambda: [{"name": "unichain"}])
    if add_all:
        assert bench.add_all_models() is bench
    else:
        with pytest.raises(ValueError, match="already registered"):
            bench.add_mdp("unichain")
    assert len(bench._mdps) == 1
    assert bench._mdps[0] is existing


def test_benchmark_selection_adds_multiple_models_and_preserves_previous_result(
    preset_catalogue, chain, tmp_path, monkeypatch
):
    (tmp_path / "other.py").write_text(
        f"METADATA = {preset_catalogue!r}\n", encoding="utf-8"
    )

    def import_recipe(name):
        def build(**options):
            model = Unichain(**options)
            model.name = f"{name.rsplit('.', 1)[1]}_{model.name}"
            return model

        return SimpleNamespace(Model=build)

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", import_recipe)
    bench = Benchmark().add_mdp(chain)
    result = bench.run(0.9, repeats=1)
    snapshot = deepcopy((result.rows, result.experiment))
    bench.add_mdp(size="small", category="abstract")
    assert [model.name for model in bench._mdps] == [
        chain.name,
        "other_4_2_unichain",
        "unichain_4_2_unichain",
    ]
    assert (result.rows, result.experiment) == snapshot
    for model in bench._mdps:
        validate_model(model)


@pytest.mark.parametrize(
    "options", [{"size": "huge"}, {"category": "unknown"}, {"category": []}]
)
def test_benchmark_rejects_bad_filters(preset_catalogue, options, monkeypatch):
    def fail_import(name):
        pytest.fail("Invalid filters must fail before importing any recipe")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    bench = Benchmark()
    with pytest.raises(ValueError):
        bench.add_mdp(**options)
    assert bench._mdps == []


@pytest.mark.parametrize(
    "options",
    [
        {"model": "custom", "transitions": [np.eye(2)], "size": "small"},
        {"transitions": [np.eye(2)], "size": "small"},
        {"reward": np.zeros((2, 1)), "category": "abstract"},
    ],
)
def test_benchmark_rejects_filters_combined_with_matrices(options):
    with pytest.raises(ValueError, match="cannot be combined"):
        Benchmark().add_mdp(**options)


def test_benchmark_rejects_filter_with_built_model(chain):
    with pytest.raises(ValueError, match="cannot be combined"):
        Benchmark().add_mdp(chain, size="small")


def test_benchmark_requires_model_or_filter():
    with pytest.raises(ValueError, match="Provide a model or at least one"):
        Benchmark().add_mdp()


def test_category_selection_has_no_type_alias():
    with pytest.raises(TypeError, match="unexpected keyword argument 'type'"):
        list_models(type="abstract")
    with pytest.raises(TypeError, match="unexpected keyword argument 'type'"):
        Benchmark().add_mdp(type="abstract")


def test_benchmark_selection_warns_about_missing_dependencies(
    preset_catalogue, monkeypatch
):
    def fail_import(name):
        raise ModuleNotFoundError("Missing dependency", name="missing_model_dependency")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    bench = Benchmark()
    with pytest.warns(UserWarning, match="Skipping model 'unavailable'.*missing_model"):
        assert bench.add_mdp(category="navigation") is bench
    assert bench._mdps == []


@pytest.mark.parametrize("options", [{}, {"category": "navigation"}])
def test_named_catalogue_model_reports_missing_dependencies(
    preset_catalogue, monkeypatch, options
):
    def fail_import(name):
        raise ModuleNotFoundError("Missing dependency", name="missing_model_dependency")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    bench = Benchmark()
    with pytest.raises(ModuleNotFoundError, match="Missing dependency"):
        bench.add_mdp("unavailable", **options)
    assert bench._mdps == []


@pytest.mark.parametrize("selection", ["all", "filters", "names"])
def test_collective_additions_continue_after_missing_dependencies(
    chain, monkeypatch, selection
):
    entries = [
        {"name": name, "sizes": {"small": {"state_dim": 3, "parameters": {}}}}
        for name in ("missing", "chain")
    ]
    monkeypatch.setattr(models, "list_models", lambda **kwargs: entries)
    imports = []

    def import_recipe(name):
        imports.append(name)
        if name.endswith(".missing"):
            raise ModuleNotFoundError("Missing dependency", name="missing_backend")
        return SimpleNamespace(Model=lambda: type(chain)(3, 2))

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", import_recipe)
    bench = Benchmark()
    with pytest.warns(UserWarning, match="Skipping model 'missing'.*missing_backend"):
        if selection == "all":
            assert bench.add_all_models() is bench
        elif selection == "names":
            assert bench.add_mdp(model=["missing", "chain"], size="small") is bench
        else:
            assert bench.add_mdp(size="small") is bench
    assert imports == ["mdpforge.models.missing", "mdpforge.models.chain"]
    assert len(bench._mdps) == 1
    model = bench._mdps[0]
    validate_model(model)
    np.testing.assert_allclose(model.reward_matrix, chain.reward_matrix)
    results = bench.run(discount=0.9, precision=1e-4, repeats=1)
    assert len(results.rows) == 1
    assert results.rows[0]["status"] == "success", results.rows[0]["error"]
    assert results.rows[0]["error_bound"] <= 1e-4


@pytest.mark.parametrize("add_all", [False, True])
def test_collective_additions_propagate_construction_errors(
    preset_catalogue, monkeypatch, add_all
):
    class BrokenModel(Unichain):
        def create_model(self):
            raise RuntimeError("Model construction failed")

    monkeypatch.setattr(
        "mdpforge.core.benchmark.import_module",
        lambda name: SimpleNamespace(Model=BrokenModel),
    )
    bench = Benchmark()
    with pytest.raises(RuntimeError, match="Model construction failed"):
        if add_all:
            bench.add_all_models()
        else:
            bench.add_mdp(size="small", category="abstract")
    assert bench._mdps == []
