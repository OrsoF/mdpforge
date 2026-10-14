import pytest

from mdpforge import list_models, models


def test_catalogue_reads_metadata_without_importing_models(tmp_path, monkeypatch):
    (tmp_path / "route.py").write_text(
        "import unavailable_catalogue_dependency\n"
        "METADATA = {'category': 'navigation', 'description': 'Route choice.', "
        "'reference': 'Example reference'}\n"
        "raise RuntimeError('This model must not be imported')\n",
        encoding="utf-8",
    )
    (tmp_path / "queue.py").write_text(
        "METADATA = {'category': 'service_operations', 'description': 'Service choices.'}\n",
        encoding="utf-8",
    )
    (tmp_path / "_helper.py").write_text("invalid Python!", encoding="utf-8")
    monkeypatch.setattr(models, "files", lambda package: tmp_path)

    entries = list_models()
    assert [entry["name"] for entry in entries] == ["queue", "route"]
    assert list_models(category="navigation") == [
        {
            "name": "route",
            "category": "navigation",
            "description": "Route choice.",
            "reference": "Example reference",
            "sizes": {},
        }
    ]
    assert entries[0]["reference"] is None
    assert list_models(category="physical_control") == []


@pytest.mark.parametrize(
    "metadata",
    [
        "",
        "METADATA = {}\n",
        "METADATA = {'category': 'unknown'}\n",
        "METADATA = {'category': ['navigation']}\n",
    ],
)
def test_model_requires_one_known_category(tmp_path, monkeypatch, metadata):
    (tmp_path / "new_recipe.py").write_text(
        metadata + "raise RuntimeError()", encoding="utf-8"
    )
    monkeypatch.setattr(models, "files", lambda package: tmp_path)
    with pytest.raises(ValueError, match="new_recipe: category must be one of"):
        list_models()


@pytest.mark.parametrize("category", ["unknown", "", [], "random", "queueing"])
def test_unknown_categories_fail_without_reading_recipes(monkeypatch, category):
    def fail_read(package):
        pytest.fail("Invalid categories must fail before reading the catalogue")

    monkeypatch.setattr(models, "files", fail_read)
    with pytest.raises(ValueError, match="Unknown model category"):
        list_models(category=category)


def test_catalogue_metadata_is_returned_independently():
    entries = list_models()
    assert entries
    original_category = entries[0]["category"]
    entries[0]["category"] = "changed"
    entries[0]["sizes"]["small"]["requested_state_dim"] = -1
    assert list_models()[0]["category"] == original_category
    assert list_models()[0]["sizes"]["small"]["requested_state_dim"] > 0
    for entry in list_models():
        assert entry["category"] in {
            "abstract",
            "navigation",
            "physical_control",
            "service_operations",
            "asset_management",
            "games_and_puzzles",
        }
        assert entry["description"]
        assert set(entry) == {"name", "category", "description", "reference", "sizes"}
        assert set(entry["sizes"]) == {"small", "medium", "large"}
        for preset in entry["sizes"].values():
            assert "source" not in preset and "state_dim" not in preset
            if preset["parameters"] is None:
                assert preset["requested_state_dim"] is None
                assert preset["reason"]
            else:
                assert type(preset["requested_state_dim"]) is int
                assert preset["requested_state_dim"] > 0
                assert "state_dim" not in preset["parameters"]
        assert entry["reference"] is None or isinstance(entry["reference"], str)


def test_category_boundaries_reflect_the_primary_task():
    categories = {entry["name"]: entry["category"] for entry in list_models()}
    expected = {
        "elevator": "service_operations",
        "taxi": "service_operations",
        "ambulance": "service_operations",
        "car_rental": "asset_management",
        "schoolboy": "navigation",
        "barto": "navigation",
        "sutton": "navigation",
        "swim": "navigation",
        "acrobot": "physical_control",
        "option_pricing": "asset_management",
        "gambler": "games_and_puzzles",
        "blocks_world": "games_and_puzzles",
        "garnet": "abstract",
    }
    assert {name: categories[name] for name in expected} == expected


@pytest.mark.parametrize("value", ["[]", "dict(category='navigation')"])
def test_invalid_metadata_is_reported(tmp_path, monkeypatch, value):
    (tmp_path / "bad.py").write_text(f"METADATA = {value}", encoding="utf-8")
    monkeypatch.setattr(models, "files", lambda package: tmp_path)
    with pytest.raises(TypeError, match="METADATA must be a literal dict"):
        list_models()
