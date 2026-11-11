"""An original absorbing chain for undiscounted total reward.

State i records progress; actions move one or two states toward
the terminal state N-1. Each nonterminal action earns -1, and the absorbing
terminal state earns 0. Every policy terminates in at most N-1 steps.
"""

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "abstract",
    "description": (
        "Reach an absorbing goal by advancing one or two states at a cost "
        "of one per move."
    ),
    "reference": None,
    "sizes": {
        "small": {"requested_state_dim": 20, "parameters": {}},
        "medium": {"requested_state_dim": 200, "parameters": {}},
        "large": {"requested_state_dim": 2000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 5}


class Model(MDP):
    """Deterministic finite chain ending at a zero-reward absorbing goal."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 20):
        if type(state_dim) is not int or state_dim < 3:
            raise ValueError("state_dim must be an integer of at least 3")
        super().__init__(state_dim, 2)
        self.name = f"{state_dim}_2_progress_chain"
        self.terminal_states = (state_dim - 1,)
        self.max_steps = state_dim - 1

    def _build_model(self):
        states = np.arange(self.state_dim)
        self.transition_matrix = [
            csr_matrix(
                (
                    np.ones(self.state_dim),
                    (states, np.minimum(states + step, self.state_dim - 1)),
                ),
                shape=(self.state_dim, self.state_dim),
            )
            for step in (1, 2)
        ]
        self.reward_matrix = -np.ones((self.state_dim, self.action_dim))
        self.reward_matrix[-1, :] = 0.0
