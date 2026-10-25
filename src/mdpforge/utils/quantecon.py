"""Local matrix conversion for the optional QuantEcon backend."""

import numpy as np
from quantecon.markov import DiscreteDP
from scipy.sparse import vstack

from mdpforge.core.mdp import MDP


def create_quantecon(model: MDP, discount: float, *, dense: bool = False):
    """Build DiscreteDP without changing the model's rewards or CSR matrices."""
    if dense:
        transitions = np.empty(
            (model.state_dim, model.action_dim, model.state_dim), dtype=float
        )
        for action, matrix in enumerate(model.transition_matrix):
            transitions[:, action, :] = matrix.toarray()
        return DiscreteDP(model.reward_matrix.copy(), transitions, discount)

    transitions = vstack(model.transition_matrix, format="csr")
    states = np.tile(np.arange(model.state_dim), model.action_dim)
    actions = np.repeat(np.arange(model.action_dim), model.state_dim)
    rewards = model.reward_matrix.T.flatten()
    return DiscreteDP(
        rewards, transitions, discount, s_indices=states, a_indices=actions
    )
