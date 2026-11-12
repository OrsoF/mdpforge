"""Catalogue of total-reward solver adapters."""

from importlib.resources import files

from mdpforge.utils.catalogue import _filter_solver_entries, _read_metadata


def list_solvers(*, family=None, backend=None):
    """List total-reward solvers without importing optional backends."""
    entries = []
    for name, metadata in _read_metadata(files(__package__)):
        entry = {"name": name, **metadata}
        entries.append(entry)
    return _filter_solver_entries(entries, family=family, backend=backend)
