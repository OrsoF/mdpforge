"""Catalogue of average-reward solver adapters."""

from importlib.resources import files

from mdpforge.utils.catalogue import _read_metadata


def list_solvers(*, family=None, backend=None):
    """List average-reward solvers without importing optional backends."""
    entries = []
    for name, metadata in _read_metadata(files(__package__)):
        entry = {"name": name, **metadata}
        if family is not None and entry.get("family") != family:
            continue
        if backend is not None and entry.get("backend") != backend:
            continue
        entries.append(entry)
    return entries
