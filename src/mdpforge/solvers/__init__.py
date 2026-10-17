"""Solver catalogue metadata, readable without importing optional backends."""

from collections.abc import Sequence
from importlib.resources import files

from mdpforge.utils._catalogue import _read_metadata

_METADATA_FIELDS = (
    "display_name",
    "family",
    "backend",
    "description",
)


def list_solvers(
    *,
    family: str | Sequence[str] | None = None,
    backend: str | Sequence[str] | None = None,
) -> list[dict]:
    """Read literal solver metadata without loading backend dependencies.

    Filters accept a string or a nonempty sequence of strings. Fields combine
    with AND; values within a field combine with OR. Unknown values and empty
    intersections raise ValueError. Names come from module filenames; missing
    display names fall back to those names, and other missing fields are None.
    """
    entries = []
    for name, metadata in _read_metadata(files(__package__)):
        entry = {"name": name}
        for field in _METADATA_FIELDS:
            value = metadata.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TypeError(f"{name}.py: {field} must be a nonempty string")
            entry[field] = value
        entry["display_name"] = entry["display_name"] or entry["name"]
        entries.append(entry)

    filters = {}
    for field, value in (
        ("family", family),
        ("backend", backend),
    ):
        if value is None:
            continue
        values = [value] if isinstance(value, str) else value
        if (
            not isinstance(values, Sequence)
            or not values
            or any(not isinstance(item, str) or not item.strip() for item in values)
        ):
            raise ValueError(
                f"{field} must be a string or nonempty sequence of strings"
            )
        available = {entry[field] for entry in entries if entry[field] is not None}
        unknown = set(values) - available
        if unknown:
            raise ValueError(
                f"Unknown solver {field}: {sorted(unknown)}; "
                f"available values: {sorted(available)}"
            )
        filters[field] = set(values)
    selected = [
        entry
        for entry in entries
        if all(entry[field] in values for field, values in filters.items())
    ]
    if filters and not selected:
        raise ValueError("No catalogue solver matches the requested metadata filters")
    return selected
