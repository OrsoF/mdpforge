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
        "METADATA = {'category': 'queueing', 'description': 'Service choices.'}\n",
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
            "tags": [],
            "sizes": {},
        }
    ]
    assert entries[0]["reference"] is None
    assert list_models(category="unknown") == []


def test_new_model_without_metadata_is_discoverable(tmp_path, monkeypatch):
    (tmp_path / "new_recipe.py").write_text("raise RuntimeError()", encoding="utf-8")
    monkeypatch.setattr(models, "files", lambda package: tmp_path)
    assert list_models() == [
        {
            "name": "new_recipe",
            "category": None,
            "description": None,
            "reference": None,
            "tags": [],
            "sizes": {},
        }
    ]


def test_catalogue_metadata_is_returned_independently():
    entries = list_models()
    assert entries
    original_category = entries[0]["category"]
    entries[0]["category"] = "changed"
    entries[0]["tags"].append("changed")
    entries[0]["sizes"]["small"]["parameters"]["state_dim"] = -1
    assert list_models()[0]["category"] == original_category
    assert "changed" not in list_models()[0]["tags"]
    assert list_models()[0]["sizes"]["small"]["parameters"]["state_dim"] > 0
    for entry in list_models():
        assert entry["category"]
        assert entry["description"]
        assert entry["tags"]
        assert set(entry["sizes"]) == {"small", "medium", "large"}
        for preset in entry["sizes"].values():
            if preset["parameters"] is None:
                assert preset["state_dim"] is None
                assert preset["source"] == "unavailable" and preset["reason"]
            else:
                assert preset["state_dim"] > 0
                assert preset["source"] in {
                    "measured",
                    "inferred",
                    "configured",
                    "analytical",
                    "parameterized",
                }
        assert entry["reference"] is None or isinstance(entry["reference"], str)


def test_invalid_metadata_is_reported(tmp_path, monkeypatch):
    (tmp_path / "bad.py").write_text("METADATA = []", encoding="utf-8")
    monkeypatch.setattr(models, "files", lambda package: tmp_path)
    with pytest.raises(TypeError, match="METADATA must be a literal dict"):
        list_models()
