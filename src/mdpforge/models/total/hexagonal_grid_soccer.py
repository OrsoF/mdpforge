"""Score in the right goal of the catalogue's single-agent hexagonal soccer.

State s=(player*cells+ball)*cells+defender records three hex-grid cells.
Actions 0..5 move/kick in the six axial directions. Boundaries and defender
collisions retain the original rules; the deterministic defender approaches
the ball, and capture resets the ball to the left goal. All states with
the ball at the existing right-goal cell remain absorbing success states.

state_dim is a positive Python integer target (default 343). Radius is
the largest r>=1 with (1+3*r*(r+1))**3 <= max(27, state_dim), using r=1
if none fits; cells=1+3*r*(r+1), S=cells**3, A=6. No seed is used.
All transitions are unchanged. Rewards are 1 for nonterminal actions
entering any right-goal state, 0 otherwise and at every terminal. Total
reward is at most 1 even without termination. No time limit is added;
cycles prevent the acyclic certificate. This is the local single-agent
reduction inspired by Uther & Veloso (1998) and Littman (1994).
"""

import numpy as np

from mdpforge.models.discounted.hexagonal_grid_soccer import (
    Model as DiscountedHexagonalGridSoccer,
)

METADATA = {
    "category": "games_and_puzzles",
    "description": "Score in an absorbing right goal against a deterministic defender.",
    "reference": (
        "Continuous U-Tree, Uther & Veloso (1998), enlarged hexagonal-grid soccer; "
        "inspired by Littman (1994) Markov games. Single-agent tabular reduction."
    ),
    "sizes": {
        "tiny": {"requested_state_dim": 685, "parameters": {}},
        "small": {"requested_state_dim": 6859, "parameters": {}},
        "medium": {"requested_state_dim": 405224, "parameters": {}},
        "large": {"requested_state_dim": 753571, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 343}


class Model(DiscountedHexagonalGridSoccer):
    """Single-agent soccer with finite total reward for scoring."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 343):
        """Accept state_dim as a positive Python integer target (default 343)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name = f"{self.state_dim}_{self.action_dim}_hexagonalgridsoccer_total"
        self.right_goal = max(range(self.n_cells), key=lambda k: self.cells[k][0])
        self.terminal_states = tuple(
            self._encode(player, self.right_goal, defender)
            for player in range(self.n_cells)
            for defender in range(self.n_cells)
        )

    def _build_model(self):
        """Keep soccer dynamics and reward entry into any right-goal state."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                np.asarray(transition[:, self.terminal_states].sum(axis=1)).ravel() > 0
            )
        self.reward_matrix[list(self.terminal_states), :] = 0.0
