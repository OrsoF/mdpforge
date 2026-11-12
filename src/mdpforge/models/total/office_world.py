"""Collect mail and coffee and reach the catalogue's absorbing office goal.

State s=4*cell+2*has_mail+has_coffee, with cell=row*width+column, records
position and persistent collection flags. Actions 0..3 are north, south,
east and west. Wall and boundary moves stay in place; entering mail or
coffee cells sets the corresponding flag. Only the office cell with both
flags is terminal, and it remains absorbing.

state_dim is a positive Python integer target (default 100). With
cells=max(25, ceil(state_dim/4)), height=max(5, floor(sqrt(cells))) and
width=max(5, ceil(cells/height)), actual S=4*height*width and A=4. The
wall geometry, plants and item locations reuse the discounted catalogue's
compact abstraction. There is no random generation or seed.

All transitions are unchanged. Rewards replace movement costs, plant
penalties and delivery payoff: 1 only for nonterminal actions reaching
the office with both items, 0 otherwise, including at the goal. Total
reward is at most 1, even for policies that avoid the goal. No time limit
is added; transient cycles prevent the current acyclic certificate.

Reference: Office World, adapted from Icarte et al. (2018), and used as
a tabular/factored benchmark in CAT+RL (Dadvar et al., 2023).
"""

from mdpforge.models.discounted.office_world import Model as DiscountedOfficeWorld

METADATA = {
    "category": "navigation",
    "description": "Collect mail and coffee, rewarding access to the absorbing office.",
    "reference": (
        "Office World, adapted from Icarte et al. (2018) reward-machine benchmark; "
        "used as a tabular/factored benchmark in CAT+RL (Dadvar et al., 2023)."
    ),
    "sizes": {
        "small": {"requested_state_dim": 8000, "parameters": {}},
        "medium": {"requested_state_dim": 32000, "parameters": {}},
        "large": {"requested_state_dim": 256000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 100}


class Model(DiscountedOfficeWorld):
    """Office navigation with finite total reward for completing both collections."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 100):
        """Accept state_dim as a positive Python integer target (default 100)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name += "_total_office_entry_reward_v1"
        office_cell = self._cell(self.height - 2, self.width - 2)
        self.terminal_states = (self._encode(office_cell, True, True),)

    def _build_model(self):
        """Keep office transitions and reward actions entering the completed goal."""
        super()._build_model()
        goal = self.terminal_states[0]
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = transition[:, goal].toarray().ravel() > 0
        self.reward_matrix[goal, :] = 0.0
