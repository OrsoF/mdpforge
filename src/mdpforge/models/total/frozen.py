"""Reach FrozenLake's goal while preserving holes and slippery transitions.

State s=row*side+column encodes a square grid. Actions 0..3 are left,
down, right and up, with equal probabilities for the intended direction
and its two perpendicular directions. Boundary moves stay in place.
state_dim is a positive Python integer target (default 100); side is
floor(sqrt(max(16, state_dim))), giving S=side**2 and four actions. The
4x4 map and larger deterministic hole pattern reuse the discounted model;
there is no random generation or seed. Holes and the goal stay absorbing.

All transitions are unchanged. Rewards are 1 for nonterminal actions with
positive goal-entry probability and 0 elsewhere, including every terminal.
Unlike the discounted model's expected entry reward, this is an action
reward, not an entry probability. Each rewarded action has probability
1/3 of reaching the goal, so expected total reward is bounded by 3, even
for policies that can avoid termination. No time limit is added; transient
cycles prevent the current acyclic total-reward certificate.

Reference: Gymnasium FrozenLake-v1 / OpenAI Gym FrozenLake.
"""

from mdpforge.models.discounted.frozen import Model as DiscountedFrozen

METADATA = {
    "category": "navigation",
    "description": "Slippery lake navigation with holes and reward for goal access.",
    "reference": "Gymnasium FrozenLake-v1 / OpenAI Gym FrozenLake.",
    "sizes": {
        "tiny": {"requested_state_dim": 1400, "parameters": {}},
        "small": {"requested_state_dim": 14000, "parameters": {}},
        "medium": {"requested_state_dim": 224000, "parameters": {}},
        "large": {"requested_state_dim": 896000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 16}


class Model(DiscountedFrozen):
    """Slippery navigation with absorbing success and failure states."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 100):
        """Accept state_dim as a positive Python integer target (default 100)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name = "{}_{}_frozen_total".format(self.state_dim, self.action_dim)
        self.terminal_states = tuple(
            self._state(row, col)
            for row, cells in enumerate(self.desc)
            for col, cell in enumerate(cells)
            if cell in ("H", "G")
        )

    def _build_model(self):
        """Keep slippery transitions and reward actions that can reach the goal."""
        super()._build_model()
        goal = self.state_dim - 1
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = transition[:, goal].toarray().ravel() > 0
        self.reward_matrix[list(self.terminal_states), :] = 0.0
