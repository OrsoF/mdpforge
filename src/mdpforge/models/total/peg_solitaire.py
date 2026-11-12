"""Reach the catalogue's existing single-peg goal without changing jumps.

State s is a row-major board bit mask, with the top-left cell the most
significant bit. state_dim is a positive Python integer (default 512)
and is the exact S. Board side=max(3, ceil(sqrt((S-1).bit_length()))).
An action selects one of four jump directions and a starting cell, giving
A=4*side**2. Invalid jumps and jumps outside the indexed subset stay in
place. Every valid jump removes one peg. The goal is one peg at [2,2];
requests not containing its bit mask raise ValueError rather than moving
the goal. The goal is already absorbing because no jump is possible.

Transitions are unchanged. Rewards are moved from the goal itself to
nonterminal actions entering it: 1 on those actions, 0 elsewhere and at
the goal. A writable reward array replaces the discounted broadcast view.
Total reward is at most 1 even for policies stuck on invalid actions.
No time limit or seed is added; self-loops prevent the acyclic certificate.
Reference: Winston & Goldberg (2004), Operations Research, 4th ed.; this
retains the catalogue's square-board and indexed-subset adaptation.
"""

import numpy as np

from mdpforge.models.discounted.peg_solitaire import Model as DiscountedPegSolitaire

METADATA = {
    "category": "games_and_puzzles",
    "description": "Jump pegs to reach the existing absorbing single-peg goal.",
    "reference": (
        "Winston, W. L. and Goldberg, J. B. (2004). Operations Research: "
        "Applications and Algorithms, 4th ed."
    ),
    "sizes": {
        "small": {"requested_state_dim": 640, "parameters": {}},
        "medium": {"requested_state_dim": 40000, "parameters": {}},
        "large": {"requested_state_dim": 81920, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 512}


class Model(DiscountedPegSolitaire):
    """Peg-jumping puzzle with finite total reward on entry to the original goal."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 512):
        """Accept positive integer state_dim containing the [2,2] goal (default 512)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        goal = 1 << (self.size * self.size - 1 - (2 * self.size + 2))
        if goal >= self.state_dim:
            raise ValueError("state_dim must include the single-peg goal at [2, 2]")
        self.name += "_total_single_peg_entry_reward_v1"
        self.terminal_states = (goal,)

    def _build_model(self):
        """Keep all jumps and reward only entry into the single-peg goal."""
        super()._build_model()
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))
        goal = self.terminal_states[0]
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = transition[:, goal].toarray().ravel() > 0
        self.reward_matrix[goal, :] = 0.0
