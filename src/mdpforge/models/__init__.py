"""Model recipes and imported JANI source metadata."""

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


def list_models(
    category: str | None = None,
    *,
    size: str | None = None,
    available: bool | None = None,
) -> list[dict]:
    """Read individual model modules without importing them or building matrices.

    category: abstract, navigation, physical_control, service_operations,
        asset_management, games_and_puzzles or None; selects a task family.
    size: small, medium, large or None; selects an available preset.
    available: True for recipes, False for continuous-time sources, None for both.
    """
    if available is not None and type(available) is not bool:
        raise ValueError("available must be True, False or None")
    if size is not None and size not in ("small", "medium", "large"):
        raise ValueError("size must be 'small', 'medium', 'large' or None")
    if category is not None and (
        not isinstance(category, str) or category not in _CATEGORIES
    ):
        raise ValueError(
            f"Unknown model category {category!r}. Available categories: "
            + ", ".join(_CATEGORIES)
        )
    directory = files(__package__)
    entries = []
    for name, metadata in _read_metadata(directory):
        model_category = metadata.get("category")
        if not isinstance(model_category, str) or model_category not in _CATEGORIES:
            raise ValueError(
                f"{name}: category must be one of {', '.join(_CATEGORIES)}; "
                f"got {model_category!r}"
            )
        entry = dict(metadata)
        entry["name"] = name
        entry.setdefault("sizes", {})
        entry.setdefault("available", True)
        entry.setdefault("reason", None)
        if type(entry["available"]) is not bool:
            raise ValueError(f"{name}: available must be Boolean")
        entries.append(entry)
    return [
        entry
        for entry in sorted(entries, key=lambda item: item["name"])
        if (category is None or entry["category"] == category)
        and (size is None or entry["sizes"].get(size, {}).get("parameters") is not None)
        and (available is None or entry["available"] == available)
    ]
