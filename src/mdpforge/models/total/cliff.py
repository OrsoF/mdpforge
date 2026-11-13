"""Reach the cliff grid's absorbing exit with the original reset dynamics.

State s=v*width+h encodes a cell. Actions 0..3 increase v, decrease v,
move right and move left; boundaries clip movement. Entering a cliff cell
resets to state 0. The exit at width-1 remains absorbing. state_dim is a
positive Python integer target (default 48); height=max(2, floor(sqrt(
state_dim/3))), width=3*height, S=height*width and A=4.

Transitions are unchanged. Rewards replace all movement and cliff costs:
1 for nonterminal actions entering the exit, 0 otherwise and at the exit.
Total reward is at most 1, including for policies that never exit. No time
limit or random seed is introduced; cycles preclude the acyclic certificate.
Reference: Sutton & Barto (2018), Example 6.6, as adapted by the catalogue.
"""

from mdpforge.models.discounted.cliff import Model as DiscountedCliff

METADATA = {
    "category": "navigation",
    "description": "Reach a cliff-side exit, retaining resets after unsafe moves.",
    "reference": (
        "Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An "
        "Introduction, 2nd ed., Example 6.6."
    ),
    "sizes": {
        "tiny": {"requested_state_dim": 1440, "parameters": {}},
        "small": {"requested_state_dim": 14400, "parameters": {}},
        "medium": {"requested_state_dim": 57600, "parameters": {}},
        "large": {"requested_state_dim": 230400, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 48}


class Model(DiscountedCliff):
    """Cliff navigation with finite total reward for exit access."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 48):
        """Accept state_dim as a positive Python integer target (default 48)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name = "{}_{}_cliff_total".format(
            self.state_dim,
            self.action_dim,
        )
        self.terminal_states = (self.h_dim - 1,)

    def _build_model(self):
        """Keep cliff resets and reward only actions entering the exit."""
        super()._build_model()
        goal = self.terminal_states[0]
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = transition[:, goal].toarray().ravel() > 0
        self.reward_matrix[goal, :] = 0.0
