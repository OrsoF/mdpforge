from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from marmote.core import FullMatrix

    from mdpforge.core.model import GenericModel


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

    marmote_transitions_list = list()
    for aa in range(action_dim):
        P = SparseMatrix(state_dim)

        row_indices, col_indices = transition_matrix[aa].nonzero()
        for i in range(len(row_indices)):
            ss1 = row_indices[i]
            ss2 = col_indices[i]
            val = transition_matrix[aa][ss1, ss2]
            P.addToEntry(int(ss1), int(ss2), val)

        marmote_transitions_list.append(P)
        P = None
    return marmote_transitions_list


def compute_mdpsolver_args(model: "GenericModel") -> tuple:
    """Convert transition and reward matrices to MDPSolver input lists."""
    transitions = []
    for action in range(model.action_dim):
        row_indices, col_indices = model.transition_matrix[action].nonzero()
        for index in range(len(row_indices)):
            state_from = row_indices[index]
            state_to = col_indices[index]
            value = float(model.transition_matrix[action][state_from, state_to])
            transitions.append([int(state_from), action, int(state_to), value])

    rewards = [
        [
            float(model.reward_matrix[state, action])
            for action in range(model.action_dim)
        ]
        for state in range(model.state_dim)
    ]

    return transitions, rewards


####### DISPATCH #######


def compute_marmote_args(model: "GenericModel") -> tuple:
    """Build native Marmote matrices without changing the CSR model."""
    return (
        build_marmote_transition_list(
            model.state_dim, model.action_dim, model.transition_matrix
        ),
        build_marmote_reward_matrix(
            model.state_dim, model.action_dim, model.reward_matrix
        ),
    )
