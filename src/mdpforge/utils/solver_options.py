"""Solver option conversions, configuration identifiers and iteration budgets."""

from decimal import Decimal

import numpy as np

from mdpforge.utils.cache_keys import array_digest

_OPTION_CODES = {
    "refinement": {"width": 0, "tiles": 1},
    "norm_method": {
        "region_size": 0,
        "average_reward": 1,
        "average_average_reward": 2,
        "max_reward_region": 3,
    },
    "stop_criterion": {"residual": 0, "envelope": 1},
    "ksp_type": {"gmres": 0, "bcgs": 1, "tfqmr": 2, "richardson": 3},
    "pc_type": {"none": 0, "lu": 1, "svd": 2},
}


def solver_identifier(module: str, **parameters) -> str:
    """Append numeric configuration tokens in the supplied parameter order."""
    tokens = [module.rsplit(".", 1)[-1]]
    for name, value in parameters.items():
        if value is None:
            token = "-1"
        elif isinstance(value, (bool, np.bool_)):
            token = str(int(value))
        elif isinstance(value, str):
            code = _OPTION_CODES.get(name, {}).get(value)
            if code is None:
                code = 10 + int.from_bytes(b"\x01" + value.encode("utf-8"), "big")
            token = str(code)
        elif isinstance(value, (np.ndarray, list, tuple)):
            token = str(int(array_digest(value)[:32], 16))
        else:
            number = Decimal(str(value))
            if not number.is_finite():
                if number.is_nan():
                    token = "-2"
                else:
                    token = "-4" if number.is_signed() else "-3"
            elif number == 0:
                token = "0"
            else:
                token = format(number, "f")
                if "." in token:
                    token = token.rstrip("0").rstrip(".")
        tokens.append(token)
    return "_".join(tokens)


def initial_vectors(model, initial_value=None, initial_policy=None):
    """Copy optional starting vectors."""
    value = (
        None
        if initial_value is None
        else np.array(initial_value, dtype=float, copy=True)
    )
    policy = (
        None
        if initial_policy is None
        else np.array(initial_policy, dtype=int, copy=True)
    )
    return value, policy


def bounded_iterations(max_iterations: int, stage: str):
    """Yield iteration indices and raise if a stage exhausts its budget."""
    for iteration in range(1, max_iterations + 1):
        yield iteration
    raise RuntimeError(f"{stage}: max_iterations ({max_iterations}) exhausted")
