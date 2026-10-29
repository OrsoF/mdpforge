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


def solver_names(constructor, options):
    """Return a configuration key and readable label including default options."""
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
    for name, parameter in parameters.items():
        if name in {"model", "discount", "final_precision"}:
            continue
        value = options.get(name, parameter.default)
        if value is inspect.Parameter.empty:
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
    if labels:
        display_name += f" ({', '.join(labels)})"
    return stem, display_name
