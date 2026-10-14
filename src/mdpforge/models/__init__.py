"""Model recipes and their catalogue metadata."""

from importlib.resources import files

from mdpforge.utils._catalogue import _read_metadata

_CATEGORIES = (
    "abstract",
    "navigation",
    "physical_control",
    "service_operations",
    "asset_management",
    "games_and_puzzles",
)


def list_models(category: str | None = None, *, size: str | None = None) -> list[dict]:
    """Read recipe metadata without importing models or constructing matrices.

    Names come from filenames. Models declare one category in a literal
    ``METADATA`` dict containing description, reference and size presets.
    Categories are abstract, navigation, physical_control, service_operations,
    asset_management and games_and_puzzles. Missing sizes are empty; other
    optional fields are None. References are the source attributions declared
    by the model authors.
    Preset state_dim is the expected actual state count, while parameters are
    constructor inputs. Inferred sizes are runtime estimates, not guarantees.
    Category selects a task family; size selects available presets.
    """
    if size is not None and size not in ("small", "medium", "large"):
        raise ValueError("size must be 'small', 'medium', 'large' or None")
    if category is not None and (
        not isinstance(category, str) or category not in _CATEGORIES
    ):
        raise ValueError(
            f"Unknown model category {category!r}. Available categories: "
            + ", ".join(_CATEGORIES)
        )
    entries = []
    for name, metadata in _read_metadata(files(__package__)):
        model_category = metadata.get("category")
        if not isinstance(model_category, str) or model_category not in _CATEGORIES:
            raise ValueError(
                f"{name}: category must be one of {', '.join(_CATEGORIES)}; "
                f"got {model_category!r}"
            )
        entry = {
            "name": name,
            "category": model_category,
            "description": metadata.get("description"),
            "reference": metadata.get("reference"),
            "sizes": metadata.get("sizes", {}),
        }
        entries.append(entry)
    return [
        entry
        for entry in entries
        if (category is None or entry["category"] == category)
        and (size is None or entry["sizes"].get(size, {}).get("parameters") is not None)
    ]
