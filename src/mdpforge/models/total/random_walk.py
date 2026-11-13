"""Reach the right boundary of the catalogue's controlled random walk.

State s is a line position, with absorbing boundaries 0 and S-1. Action 0
moves left with probability 0.8; action 1 moves right with probability 0.8;
the other direction has probability 0.2. Both actions are always valid.
state_dim is a positive Python integer target (default 100); S=max(5,
state_dim), with two actions. There is no random generation or seed.

All transitions are unchanged. The right boundary is the success goal;
the left boundary is failure. Rewards replace the signed discounted entry
rewards: 1 for nonterminal actions with positive probability of entering
the right boundary, 0 otherwise, including both terminals. Each rewarded
action has goal-entry probability at least min(0.8, 1-0.8), giving finite
expected total reward. This is an action reward, not a success probability.
No time limit is added; cycles prevent the current acyclic certificate.

Reference: Sutton & Barto (2018), Reinforcement Learning: An Introduction,
2nd ed., random-walk examples; the catalogue adds directional control.
"""

from mdpforge.models.discounted.random_walk import Model as DiscountedRandomWalk

METADATA = {
    "category": "abstract",
    "description": "Control a random walk toward a successful absorbing right boundary.",
    "reference": (
        "Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An "
        "Introduction, 2nd ed. Random-walk prediction examples."
    ),
    "sizes": {
        "tiny": {"requested_state_dim": 1560, "parameters": {}},
        "small": {"requested_state_dim": 15600, "parameters": {}},
        "medium": {"requested_state_dim": 249600, "parameters": {}},
        "large": {"requested_state_dim": 1996800, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 7}


class Model(DiscountedRandomWalk):
    """Controlled random walk with finite total reward for right-goal access."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 100):
        """Accept state_dim as a positive Python integer target (default 100)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name = f"{self.state_dim}_{self.action_dim}_randomwalk_total"
        self.terminal_states = (0, self.state_dim - 1)

    def _build_model(self):
        """Keep the controlled walk and reward actions with possible goal entry."""
        super()._build_model()
        goal = self.state_dim - 1
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = transition[:, goal].toarray().ravel() > 0
        self.reward_matrix[list(self.terminal_states), :] = 0.0
