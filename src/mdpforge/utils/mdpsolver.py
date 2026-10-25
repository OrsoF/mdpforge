"""Shared setup and result conversion for the optional MDPSolver backend."""

import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.utils._sparse import _nonzero_entries


def compute_mdpsolver_args(model: "MDP") -> tuple:
    """Convert transition and reward matrices to MDPSolver input lists."""
    transitions = []
    for action in range(model.action_dim):
        for state_from, state_to, value in _nonzero_entries(
            model.transition_matrix[action]
        ):
            transitions.append([state_from, action, state_to, value])

    rewards = [
        [
            float(model.reward_matrix[state, action])
            for action in range(model.action_dim)
        ]
        for state in range(model.state_dim)
    ]

    return transitions, rewards


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
