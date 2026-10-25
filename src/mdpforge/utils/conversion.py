from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from marmote.core import FullMatrix

    from mdpforge.core.mdp import MDP


def _nonzero_entries(matrix: csr_matrix):
    """Traverse stored CSR entries, preserving coordinate sums for duplicates."""
    entries = matrix.tocoo()
    canonical = matrix.has_canonical_format
    for row, column, value in zip(entries.row, entries.col, entries.data):
        if value != 0:
            if not canonical:
                value = matrix[row, column]
            yield int(row), int(column), float(value)


####### MARMOTE #######


def build_marmote_reward_matrix(
    state_dim: int, action_dim: int, reward_matrix: np.ndarray
) -> FullMatrix:
    from marmote.core import FullMatrix

    marmote_reward_matrix = FullMatrix(state_dim, action_dim)
    for a in range(action_dim):
        for s in range(state_dim):
            marmote_reward_matrix.setEntry(s, a, float(reward_matrix[s, a]))
    return marmote_reward_matrix


def build_marmote_transition_list(
    state_dim: int, action_dim: int, transition_matrix: list[csr_matrix]
) -> list:
    from marmote.core import SparseMatrix

    marmote_transitions_list = []
    for action in range(action_dim):
        transition = SparseMatrix(state_dim)
        for state_from, state_to, value in _nonzero_entries(transition_matrix[action]):
            transition.addToEntry(state_from, state_to, value)
        marmote_transitions_list.append(transition)
    return marmote_transitions_list

def compute_marmote_args(model: "MDP") -> tuple:
    """Build native Marmote matrices without changing the CSR model."""
    return (
        build_marmote_transition_list(
            model.state_dim, model.action_dim, model.transition_matrix
        ),
        build_marmote_reward_matrix(
            model.state_dim, model.action_dim, model.reward_matrix
        ),
    )


####### MDPSOLVER #######


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


####### DISPATCH #######


