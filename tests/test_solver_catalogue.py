import pytest

from mdpforge import list_solvers, solvers


@pytest.fixture
def solver_catalogue(tmp_path, monkeypatch):
    entries = {
        "a_vi": {
            "display_name": "Example VI",
            "family": "vi",
            "backend": "mdpforge",
            "description": "Synchronous example.",
            "update_scheme": "synchronous",
        },
        "b_vi_gs": {
            "family": "vi",
            "backend": "external",
            "update_scheme": "gauss_seidel",
        },
        "c_pi": {
            "family": "pi",
            "backend": "mdpforge",
            "update_scheme": "linear_solve",
        },
    }
    for name, metadata in entries.items():
        (tmp_path / f"{name}.py").write_text(
            f"import unavailable_backend\nMETADATA = {metadata!r}\n"
            "raise RuntimeError('Catalogue modules must not be imported')\n",
            encoding="utf-8",
        )
    (tmp_path / "_helper.py").write_text("invalid Python!", encoding="utf-8")
    (tmp_path / "future.py").write_text("raise RuntimeError()", encoding="utf-8")
    monkeypatch.setattr(solvers, "files", lambda package: tmp_path)
    return tmp_path


def test_solver_catalogue_is_static_and_metadata_is_independent(solver_catalogue):
    entries = list_solvers()
    assert [e["name"] for e in entries] == ["a_vi", "b_vi_gs", "c_pi", "future"]
    assert entries[0]["display_name"] == "Example VI"
    assert entries[-1] == {
        "name": "future",
        "display_name": "future",
        "family": None,
        "backend": None,
        "description": None,
        "update_scheme": None,
    }
    entries[0]["family"] = "changed"
    assert list_solvers()[0]["family"] == "vi"


def test_solver_filters_combine_fields_with_and_and_values_with_or(solver_catalogue):
    assert [e["name"] for e in list_solvers(family="vi")] == ["a_vi", "b_vi_gs"]
    assert [
        e["name"] for e in list_solvers(family=["vi", "pi"], backend="mdpforge")
    ] == ["a_vi", "c_pi"]
    assert [
        e["name"]
        for e in list_solvers(
            family="vi", update_scheme=("synchronous", "gauss_seidel")
        )
    ] == ["a_vi", "b_vi_gs"]
    with pytest.raises(ValueError, match="No catalogue solver matches"):
        list_solvers(family="pi", backend="external")


@pytest.mark.parametrize("field", ["family", "backend", "update_scheme"])
@pytest.mark.parametrize("value", ["typo", [], "", ["vi", 1], 1])
def test_invalid_solver_filters_fail_clearly(solver_catalogue, field, value):
    with pytest.raises(ValueError, match=field):
        list_solvers(**{field: value})


@pytest.mark.parametrize("value", ["[]", "dict(family='vi')"])
def test_solver_metadata_must_be_literal(solver_catalogue, value):
    (solver_catalogue / "bad.py").write_text(f"METADATA = {value}", encoding="utf-8")
    with pytest.raises(TypeError, match="METADATA must be a literal dict"):
        list_solvers()


def test_existing_solver_metadata_covers_the_catalogue():
    entries = list_solvers()
    assert entries
    assert {e["family"] for e in entries} == {"vi", "pi", "mpi", "lp"}
    for entry in entries:
        assert all(isinstance(value, str) and value for value in entry.values())
