"""Sparse matrix conversion for the optional madupite backend."""

import numpy as np
from scipy.sparse import hstack

from mdpforge.core.mdp import MDP


def madupite_matrices(model: MDP):
    """Return (S,A) rewards and (S*A,S) CSR transitions in state-major order."""
    # Columns of the horizontal stack are ordered (action, next_state).
    # C-order reshape then maps a row to state * n_actions + action.
    transitions = hstack(model.transition_matrix, format="csr")
    transitions = transitions.reshape(
        (model.state_dim * model.action_dim, model.state_dim), order="C"
    ).tocsr()
    return np.asarray(model.reward_matrix, dtype=np.float64), transitions
