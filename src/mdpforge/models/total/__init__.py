"""Catalogue of finite total-reward MDPs."""

from importlib.resources import files

from mdpforge.utils.catalogue import _read_metadata


def list_models(category=None, *, size=None, available=None):
    """List total-reward models without importing their implementations."""
    if size is not None and size not in ("tiny", "small", "medium", "large"):
        raise ValueError("size must be 'tiny', 'small', 'medium', 'large' or None")
    if available is not None and type(available) is not bool:
        raise ValueError("available must be True, False or None")
    entries = []
    for name, metadata in _read_metadata(files(__package__)):
        entry = dict(metadata)
        entry["name"] = name
        entry.setdefault("available", True)
        entry.setdefault("reason", None)
        if category is not None and entry.get("category") != category:
            continue
        if (
            size is not None
            and entry.get("sizes", {}).get(size, {}).get("parameters") is None
        ):
            continue
        if available is not None and entry["available"] != available:
            continue
        entries.append(entry)
    return entries
