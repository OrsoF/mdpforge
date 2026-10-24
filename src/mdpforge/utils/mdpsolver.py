"""Shared setup and result conversion for the optional MDPSolver backend."""

import numpy as np

from mdpforge.core.conversion import compute_mdpsolver_args
from mdpforge.core.mdp import MDP


def create_mdpsolver(model: MDP, discount: float, backend):
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
    initial_value=None,
    initial_policy=None,
    **options,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a discounted MDP and return NumPy values and policy."""
    mdl.solve(
        algorithm=algorithm,
        tolerance=tolerance,
        update=update,
        criterion="discounted",
        parallel=parallel,
        initValueVector=[] if initial_value is None else initial_value.tolist(),
        initPolicy=[] if initial_policy is None else initial_policy.tolist(),
        **options,
    )
    return (
        np.asarray(mdl.getValueVector(), dtype=float),
        np.asarray(mdl.getPolicy(), dtype=int),
    )
