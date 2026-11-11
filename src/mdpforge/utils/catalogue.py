"""Discover, select and load catalogue models and solvers."""

import ast
import warnings
from importlib import import_module
from importlib.util import find_spec


def _read_metadata(directory):
    """Yield public module names and independently parsed metadata dictionaries."""
    for source in sorted(directory.iterdir(), key=lambda item: item.name):
        if (
            not source.is_file()
            or not source.name.endswith(".py")
            or source.name.startswith("_")
        ):
            continue
        metadata = {}
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "METADATA"
                for target in node.targets
            ):
                try:
                    metadata = ast.literal_eval(node.value)
                except (ValueError, TypeError) as exc:
                    raise TypeError(
                        f"{source.name}: METADATA must be a literal dict"
                    ) from exc
                if not isinstance(metadata, dict):
                    raise TypeError(f"{source.name}: METADATA must be a literal dict")
                break
        yield source.name[:-3], metadata


def _load_solver(name, *, backend=None, catalogue=None):
    """Load a catalogue solver, checking an explicitly selected backend first."""
    if catalogue is None:
        from mdpforge.solvers import discounted as catalogue

    dependencies = {
        "marmote": ("marmote",),
        "mdpsolver": ("mdpsolver",),
        "madupite": ("madupite",),
        "mdpax": ("mdpax",),
        "mdptoolbox": ("mdptoolbox",),
        "pyomo": ("pyomo", "highspy"),
        "quantecon": ("quantecon",),
    }.get(backend, ())
    for dependency in dependencies:
        if find_spec(dependency) is None:
            raise ModuleNotFoundError(
                f"Missing solver dependency: {dependency}", name=dependency
            )
    return import_module(f"{catalogue.__name__}.{name}").Solver


def _available_solvers(entries, *, catalogue=None):
    """Yield selected solvers, warning only for missing dependencies."""
    for entry in entries:
        name = entry["name"]
        try:
            constructor = _load_solver(
                name, backend=entry.get("backend"), catalogue=catalogue
            )
        except ModuleNotFoundError as exc:
            warnings.warn(
                f"Skipping solver {name!r}: missing dependency {exc.name!r}",
                stacklevel=3,
            )
            continue
        yield name, constructor


def _select_models(names=None, *, size=None, category=None, catalogue=None):
    """Resolve optional names and filters to ordered model catalogue entries."""
    if catalogue is None:
        from mdpforge.models import discounted as catalogue

    entries = catalogue.list_models(size=size, category=category)
    if names is None:
        return [entry for entry in entries if entry["available"]]
    selected = {entry["name"]: entry for entry in entries}
    for name in names:
        if name in selected:
            continue
        all_names = {item["name"] for item in catalogue.list_models()}
        if name not in all_names:
            raise ValueError(
                f"Unknown model {name!r}. Available models: "
                + ", ".join(sorted(all_names))
            )
        raise ValueError(
            f"Model {name!r} does not match size={size!r}, "
            f"category={category!r}; "
            "the requested preset may be unavailable."
        )
    return [selected[name] for name in names]


def _load_model(
    entry, *, preset=None, registered=(), grouped=False, verbose=False, catalogue=None
):
    """Build a catalogue model, optionally skipping duplicates or missing dependencies."""
    if catalogue is None:
        from mdpforge.models import discounted as catalogue

    name = entry["name"]
    if not entry["available"]:
        message = f"Model {name!r} is unavailable: {entry['reason']}"
        if grouped:
            warnings.warn(message, stacklevel=4)
            return None
        raise ValueError(message)
    try:
        options = dict(preset["parameters"]) if preset is not None else {}
        if preset is not None:
            options["state_dim"] = preset["requested_state_dim"]
        model = import_module(f"{catalogue.__name__}.{name}").Model(**options)
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
