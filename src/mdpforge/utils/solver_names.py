"""Compact configuration names for catalogue solvers."""

import inspect
import re
import sys
from hashlib import sha256

import numpy as np


def is_catalogue_solver(constructor):
    """Identify solver classes supplied by the catalogue."""
    module = sys.modules.get(getattr(constructor, "__module__", ""))
    return (
        module is not None
        and module.__name__.startswith("mdpforge.solvers.")
        and getattr(module, "Solver", None) is constructor
    )


def solver_names(constructor, options, registered=()):
    """Return a compact unique key and readable label for nondefault options."""
    module = sys.modules[constructor.__module__]
    catalogue_name = module.__name__.rsplit(".", 1)[-1]
    display_name = module.METADATA["display_name"]
    values = []
    labels = []
    aliases = {
        "evaluation_precision": "eval",
        "initial_value": "v0",
        "initial_policy": "pi0",
        "relaxation": "omega",
        "max_projected_iterations": "projected",
        "max_fallback_iterations": "fallback",
        "bellman_iterations": "bellman",
        "aggregation_iterations": "aggregation",
    }
    parameters = inspect.signature(constructor).parameters
    for name in sorted(parameters, key=lambda key: key.endswith("_iterations")):
        parameter = parameters[name]
        if name in {"model", "discount", "final_precision"} or name not in options:
            continue
        value = options[name]
        if value is parameter.default or (
            np.isscalar(value)
            and np.isscalar(parameter.default)
            and value == parameter.default
        ):
            continue
        if isinstance(value, (np.ndarray, list, tuple)):
            array = np.asarray(value)
            digest = sha256(
                str((array.dtype, array.shape)).encode() + array.tobytes()
            ).hexdigest()[:10]
            text = f"array-{digest}"
        else:
            text = str(value)
        values.append(re.sub(r"[^a-zA-Z0-9_.+-]", "-", text))
        if name in {"max_iterations", "max_evaluation_iterations"}:
            labels.append(text)
        else:
            labels.append(f"{aliases.get(name, name)}={text}")
    stem = "_".join([catalogue_name, *values])
    suffix = 1
    while f"{stem}_{suffix}" in registered:
        suffix += 1
    if labels:
        display_name += f" ({', '.join(labels)})"
    return f"{stem}_{suffix}", display_name
