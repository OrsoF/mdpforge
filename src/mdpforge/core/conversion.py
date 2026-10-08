from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_array

if TYPE_CHECKING:
    from marmote.core import FullMatrix

    from mdpforge.core.model import GenericModel


NUMPY, SPARSE = "numpy", "sparse"


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
    state_dim: int, action_dim: int, transition_matrix: list[np.ndarray]
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


def model_to_numpy(model: "GenericModel") -> None:
    """Convert transition and reward matrices to dense numpy arrays in place."""
    try:
        model.transition_matrix = np.array(
            [matrix.toarray() for matrix in model.transition_matrix]
        )
    except AttributeError:
        pass

    try:
        model.reward_matrix = model.reward_matrix.toarray()
    except AttributeError:
        pass


####### SPARSE #######


def model_to_sparse(model: "GenericModel") -> None:
    """Convert transition matrices to a list of CSR sparse arrays in place."""
    model.transition_matrix = [csr_array(matrix) for matrix in model.transition_matrix]


####### MARMOTE MODEL #######


def model_to_marmote(model: "GenericModel") -> None:
    """Convert transition and reward matrices to Marmote-compatible objects."""
    if not model._is_model_built():
        raise ValueError("Model is not built yet.")

    model.transition_matrix = build_marmote_transition_list(
        model.state_dim,
        model.action_dim,
        model.transition_matrix,
    )
    model.reward_matrix = build_marmote_reward_matrix(
        model.state_dim,
        model.action_dim,
        model.reward_matrix,
    )


####### MDPSOLVER #######


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


def convert_model(model: "GenericModel", mode: str) -> None:
    """Convert model matrices to one of the supported internal formats."""
    if mode == NUMPY:
        model_to_numpy(model)
    elif mode == SPARSE:
        model_to_sparse(model)
    else:
        raise ValueError(f"Unknown model conversion mode: {mode}")
