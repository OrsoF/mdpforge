"""Solver option validation, starting-vector conversions and bounded iterations."""

import numpy as np


def validate_discount(discount):
    """Require a finite discount strictly between zero and one."""
    if not np.isfinite(discount) or not 0 < discount < 1:
        raise ValueError("discount must be finite and strictly between 0 and 1")


def validate_aggregation_options(
    *,
    final_precision,
    bellman_iterations,
    max_projected_iter,
    max_agg_iter,
    max_eval_iter,
    switch_to_vi_ratio,
    refinement="width",
    n_tiles=None,
):
    """Validate aggregation tolerances, iteration budgets and partition options."""
    if not np.isfinite(final_precision) or final_precision <= 0:
        raise ValueError("final_precision must be finite and positive")
    for name, value in (
        ("bellman_iterations", bellman_iterations),
        ("max_projected_iter", max_projected_iter),
        ("max_agg_iter", max_agg_iter),
        ("max_eval_iter", max_eval_iter),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(switch_to_vi_ratio) or not 0 < switch_to_vi_ratio <= 1:
        raise ValueError("switch_to_vi_ratio must be finite and in (0, 1]")
    if not isinstance(refinement, str) or refinement not in ("width", "tiles"):
        raise ValueError("refinement must be 'width' or 'tiles'")
    if n_tiles is not None and (type(n_tiles) is not int or n_tiles < 1):
        raise ValueError("n_tiles must be None or a positive integer")


def initial_vectors(initial_value=None, initial_policy=None):
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
