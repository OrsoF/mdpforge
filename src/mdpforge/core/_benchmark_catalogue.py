"""Resolve and load catalogue entries for benchmarks."""

import warnings
from importlib import import_module
from importlib.util import find_spec

from mdpforge import models, solvers


def _load_solver(name, *, backend=None):
    """Load a catalogue solver, checking an explicitly selected backend first."""
    dependency = {
        "gurobi": "gurobipy",
        "marmote": "marmote",
        "mdpsolver": "mdpsolver",
        "madupite": "madupite",
        "mdpax": "mdpax",
        "mdptoolbox": "mdptoolbox",
        "quantecon": "quantecon",
    }.get(backend)
    if dependency and find_spec(dependency) is None:
        raise ModuleNotFoundError(
            f"Missing solver dependency: {dependency}", name=dependency
        )
    return import_module(f"{solvers.__name__}.{name}").Solver


def _available_solvers(entries):
    """Yield selected solvers, warning only for missing dependencies."""
    for entry in entries:
        name = entry["name"]
        try:
            constructor = _load_solver(name, backend=entry.get("backend"))
        except ModuleNotFoundError as exc:
            warnings.warn(
                f"Skipping solver {name!r}: missing dependency {exc.name!r}",
                stacklevel=3,
            )
            continue
        yield name, constructor


def _select_models(names=None, *, size=None, category=None):
    """Resolve optional names and filters to ordered model catalogue entries."""
    entries = models.list_models(size=size, category=category)
    if names is None:
        return entries
    selected = {entry["name"]: entry for entry in entries}
    for name in names:
        if name in selected:
            continue
        catalogue = {item["name"] for item in models.list_models()}
        if name not in catalogue:
            raise ValueError(
                f"Unknown model {name!r}. Available models: "
                + ", ".join(sorted(catalogue))
            )
        raise ValueError(
            f"Model {name!r} does not match size={size!r}, "
            f"category={category!r}; "
            "the requested preset may be unavailable."
        )
    return [selected[name] for name in names]


def _load_model(name, *, preset=None, registered=(), grouped=False, verbose=False):
    """Build a catalogue model, optionally skipping duplicates or missing dependencies."""
    try:
        options = dict(preset["parameters"]) if preset is not None else {}
        if preset is not None:
            options["state_dim"] = preset["requested_state_dim"]
        model = import_module(f"{models.__name__}.{name}").Model(**options)
        if any(existing.name == model.name for existing in registered):
            if grouped:
                if verbose:
                    print(
                        f"Skipping MDP {model.name!r}: already registered",
                        flush=True,
                    )
                return None
            raise ValueError(f"An MDP named {model.name!r} is already registered.")
        expected_states, expected_actions = model.state_dim, model.action_dim
        model.create_model()
        if (model.state_dim, model.action_dim) != (
            expected_states,
            expected_actions,
        ):
            raise ValueError(
                f"{name}: expected {expected_states} states, got {model.state_dim}; "
                f"expected {expected_actions} actions, got {model.action_dim}; "
                "check the cached matrices"
            )
        return model
    except ModuleNotFoundError as exc:
        if not grouped:
            raise
        warnings.warn(
            f"Skipping model {name!r}: missing dependency {exc.name!r}",
            stacklevel=4,
        )
        return None
