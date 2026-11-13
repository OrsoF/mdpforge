"""Exit with gold in the catalogue's unchanged fully observed Wumpus World.

States encode (cell, direction, has_gold, wumpus_alive, has_arrow, terminal)
in mixed radix (cells,4,2,2,2,2). Actions 0..5 move forward, turn left,
turn right, grab, shoot and climb. Invalid grabs/shots/climbs keep their
existing state changes; walls clip movement. Pits and the live Wumpus
cause death. Climbing at the start cell terminates, with or without gold.
All terminal-bit states remain absorbing. Only terminal states at the
start with gold are success goals; death and empty-handed exit earn zero.

state_dim is a positive Python integer target (default 1024). Board side=
max(4, round(sqrt(max(16,state_dim//32)))), S=64*side**2, A=6, matching
the existing adjustment. The fixed pit and object locations are unchanged.
Rewards replace all costs and intermediate gold/kill bonuses: 1 only for
nonterminal actions entering a success goal, 0 otherwise and at terminals.
Success entry is deterministic, so total reward is at most 1 even without
termination. No time limit or seed is added; cycles prevent the acyclic
certificate. Reference: Russell & Norvig, Artificial Intelligence: A Modern
Approach; local tabular adaptation used with CAT+RL (Dadvar et al., 2023).
"""

import numpy as np

from mdpforge.models.discounted.wumpus_world import Model as DiscountedWumpusWorld

METADATA = {
    "category": "navigation",
    "description": "Retrieve gold and exit safely, retaining death and empty-handed exit.",
    "reference": (
        "Wumpus World, Russell & Norvig, Artificial Intelligence: A Modern "
        "Approach; used as a discrete-state benchmark in CAT+RL (Dadvar et al., "
        "2023)."
    ),
    "sizes": {
        "tiny": {"requested_state_dim": 680, "parameters": {}},
        "small": {"requested_state_dim": 6800, "parameters": {}},
        "medium": {"requested_state_dim": 54400, "parameters": {}},
        "large": {"requested_state_dim": 217600, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 1024}


class Model(DiscountedWumpusWorld):
    """Wumpus navigation with finite total reward only for exiting with gold."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 1024):
        """Accept state_dim as a positive Python integer target (default 1024)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name = f"{self.state_dim}_{self.action_dim}_wumpusworld_total"
        self.terminal_states = tuple(range(1, self.state_dim, 2))
        start = self._cell(self.size - 1, 0)
        self.goal_states = tuple(
            self._encode(start, direction, True, alive, arrow, True)
            for direction in range(4)
            for alive in (False, True)
            for arrow in (False, True)
        )

    def _build_model(self):
        """Keep Wumpus dynamics and reward only successful gold exits."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                np.asarray(transition[:, self.goal_states].sum(axis=1)).ravel() > 0
            )
        self.reward_matrix[list(self.terminal_states), :] = 0.0
