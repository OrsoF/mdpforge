"""Local matrix conversion for the optional Marmote backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.utils._sparse import _nonzero_entries

if TYPE_CHECKING:
    from marmote.core import FullMatrix

    from mdpforge.core.mdp import MDP


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
