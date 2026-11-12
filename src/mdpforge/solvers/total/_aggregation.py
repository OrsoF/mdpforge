"""Support for undiscounted state aggregation with zero terminal values."""

import numpy as np

from mdpforge.core.operators import bellman_operator
from mdpforge.core.partition import Partition
from mdpforge.core.solve_result import SolveResult


def check_aggregation_options(
    bellman_iterations,
    max_projected_iterations,
    max_fallback_iterations,
    switch_to_vi_ratio,
    refinement="width",
    n_tiles=None,
):
    """Validate aggregation budgets and partition refinement options."""
    for name, value in (
        ("bellman_iterations", bellman_iterations),
        ("max_projected_iterations", max_projected_iterations),
        ("max_fallback_iterations", max_fallback_iterations),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(switch_to_vi_ratio) or not 0 < switch_to_vi_ratio <= 1:
        raise ValueError("switch_to_vi_ratio must be finite and in (0, 1]")
    if refinement not in ("width", "tiles"):
        raise ValueError("refinement must be 'width' or 'tiles'")
    if n_tiles is not None and (type(n_tiles) is not int or n_tiles < 1):
        raise ValueError("n_tiles must be None or a positive integer")


def terminal_partition(model, terminal):
    """Separate absorbing terminals from transient states before aggregation."""
    labels = np.zeros(model.state_dim, dtype=int)
    labels[list(terminal)] = 1
    return Partition(model, labels)


def refine_partition(partition, value, precision, refinement="width", n_tiles=None):
    """Split regions using value widths or equally sized tiles."""
    n_regions = partition.n_regions
    if refinement == "width":
        partition.refine_by_width(value, precision / 2)
    elif np.max(partition.span(value)) > precision / 2:
        tiles = (
            n_tiles
            if n_tiles is not None
            else max(2, int(partition.model.state_dim**0.25))
        )
        partition.refine_by_tiles(value, tiles)
    return partition.n_regions > n_regions


def full_value_iteration(model, value, precision, max_iterations):
    """Finish from the lifted value with undiscounted full-state backups."""
    for _ in range(max_iterations):
        q_value = bellman_operator(model, value, 1.0)
        if not np.isfinite(q_value).all():
            raise RuntimeError("aggregation fallback stopped: non-finite values")
        next_value = q_value.max(axis=1)
        if np.max(np.abs(next_value - value)) <= precision:
            return SolveResult(
                value, q_value.argmax(axis=1), validation_method="residual"
            )
        value = next_value
    raise RuntimeError("aggregation fallback stopped: max_fallback_iterations reached")
