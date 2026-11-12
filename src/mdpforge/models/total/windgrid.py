"""Reach the existing Windy Gridworld goal without altering its dynamics.

State s=row*width+column encodes a cell. Actions 0..3 are north, south,
east and west; boundaries clip movement before and after upward wind.
Column wind is [0,0,0,1,1,1,2,2,1,0], extended by zeros. The absorbing
goal is (height//2, width-3). state_dim is a positive Python integer target
(default 70): requests up to 70 use 7x10; larger requests use height=
max(7, floor(sqrt(state_dim))), width=max(10, ceil(state_dim/height)).
Actual S=height*width, A=4. All actions are valid, with no random seed.

Transitions are unchanged. Rewards replace movement costs: 1 for a
nonterminal action entering the goal, 0 elsewhere and at the goal. Total
reward is at most 1 even without termination. No time limit is added;
cycles prevent the current acyclic total-reward certificate.
Reference: Sutton & Barto, Example 6.5, with the catalogue's grid scaling.
"""

from mdpforge.models.discounted.windgrid import Model as DiscountedWindgrid

METADATA = {
    "category": "navigation",
    "description": "Reach an absorbing grid goal under column-dependent wind.",
    "reference": (
        "Sutton & Barto, Reinforcement Learning: An Introduction, Example 6.5, "
        "Windy Gridworld."
    ),
    "sizes": {
        "small": {"requested_state_dim": 12000, "parameters": {}},
        "medium": {"requested_state_dim": 48000, "parameters": {}},
        "large": {"requested_state_dim": 192000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 70}


class Model(DiscountedWindgrid):
    """Wind-driven navigation with finite total reward for goal access."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 70):
        """Accept state_dim as a positive Python integer target (default 70)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name += "_total_goal_entry_reward_v1"
        self.terminal_states = (self.goal_state,)

    def _build_model(self):
        """Keep wind transitions and reward actions entering the goal."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.goal_state].toarray().ravel() > 0
            )
        self.reward_matrix[self.goal_state, :] = 0.0
