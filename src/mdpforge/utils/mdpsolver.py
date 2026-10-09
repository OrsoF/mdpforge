"""Shared setup and result conversion for the optional MDPSolver backend."""

import numpy as np

from mdpforge.core.conversion import compute_mdpsolver_args
from mdpforge.core.model import MDPProtocol


def create_mdpsolver(model: MDPProtocol, discount: float, backend):
    """Build a backend model without changing the source matrices."""
    transitions, rewards = compute_mdpsolver_args(model)
    mdl = backend.model()
    mdl.mdp(
        discount=discount,
        rewards=rewards,
        tranMatElementwise=transitions,
    )
    return mdl


def solve_mdpsolver(
    mdl,
    *,
    algorithm: str,
    tolerance: float,
    update: str,
    parallel: bool,
    **options,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a discounted MDP and return NumPy values and policy."""
    mdl.solve(
        algorithm=algorithm,
        tolerance=tolerance,
        update=update,
        criterion="discounted",
        parallel=parallel,
        **options,
    )
    return (
        np.asarray(mdl.getValueVector(), dtype=float),
        np.asarray(mdl.getPolicy(), dtype=int),
    )
