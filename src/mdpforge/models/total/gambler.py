"""Reach a capital target using the catalogue's unchanged gambling dynamics.

State s is capital in 0..goal, with goal=max(100, state_dim) and S=goal+1.
state_dim is a positive Python integer target (default 200); action_dim is
a positive Python integer (default 10), capped at goal//2. Action a stakes
min(a+1, s, goal-s), preserving the existing clipping of oversized stakes.
Capital increases by the stake with probability 0.4 and decreases otherwise.
Both ruin (0) and the goal remain absorbing. No random seed is used.

Rewards are 1 for nonterminal actions with positive goal-entry probability,
and 0 otherwise, including ruin and the goal. This replaces the discounted
expected entry reward of 0.4 without changing transitions or adding a time
limit. Each rewarded action absorbs at the goal with probability 0.4, so
expected total reward is bounded by 2.5. It is not a success probability.
Transient cycles prevent the current acyclic total-reward certificate.

Reference: Sutton & Barto, Reinforcement Learning: An Introduction,
Example 4.3, Gambler's Problem; the catalogue caps the available stakes.
"""

from mdpforge.models.discounted.gambler import Model as DiscountedGambler

METADATA = {
    "category": "games_and_puzzles",
    "description": "Choose stakes to reach a capital target before absorbing ruin.",
    "reference": (
        "Sutton & Barto, Reinforcement Learning: An Introduction, Example 4.3, "
        "Gambler's Problem."
    ),
    "sizes": {
        "tiny": {"requested_state_dim": 1000, "parameters": {}},
        "small": {"requested_state_dim": 10000, "parameters": {}},
        "medium": {"requested_state_dim": 160000, "parameters": {}},
        "large": {"requested_state_dim": 640000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 100, "action_dim": 2}


class Model(DiscountedGambler):
    """Capital control with absorbing ruin, an absorbing goal and finite return."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 200, action_dim: int = 10):
        """Accept positive Python integers state_dim (200) and action_dim (10)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        if type(action_dim) is not int or action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.name = "{}_{}_gambler_total".format(self.state_dim, self.action_dim)
        self.terminal_states = (0, self.goal)

    def _build_model(self):
        """Keep stake transitions and reward actions that can reach the target."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.goal].toarray().ravel() > 0
            )
        self.reward_matrix[list(self.terminal_states), :] = 0.0
