"""Reach school with the discounted catalogue's unchanged transport dynamics.

State s records progress; S-1 is the absorbing school. Actions 0, 1, 2
select walking, cycling and the bus, respectively; all are valid. Each
action advances by max(1, round((S-1)*15/time)), clipped at school, with
the original travel-time distributions. state_dim defaults to 100 and
must be a positive Python integer; S=max(2, state_dim), with three actions.

Rewards are 1 for nonterminal actions with positive probability of reaching
school and 0 otherwise, including at school. This replaces travel costs
without changing any transitions or adding a time limit. Progress strictly
increases, so every policy terminates within S-1 steps and has finite return.
There is no random generation or seed.

Reference: Hyon, E. (2020). Apprehender le hasard. Tangente Magazine.
"""

from mdpforge.models.discounted.schoolboy import Model as DiscountedSchoolboy

METADATA = {
    "category": "navigation",
    "description": "Reach school using uncertain transport, rewarding access to school.",
    "reference": "Hyon, E. (2020). Apprehender le hasard. Tangente Magazine.",
    "sizes": {
        "small": {"requested_state_dim": 86400, "parameters": {}},
        "medium": {"requested_state_dim": 1500000, "parameters": {}},
        "large": {"requested_state_dim": 2764800, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 10}


class Model(DiscountedSchoolboy):
    """Transport choices with an absorbing school and finite total reward."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 100):
        """Accept state_dim as a positive Python integer target (default 100)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name += "_total_school_entry_reward_v1"
        self.terminal_states = (self.state_dim - 1,)

    def _build_model(self):
        """Reuse transport transitions and reward actions that can reach school."""
        super()._build_model()
        goal = self.terminal_states[0]
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = transition[:, goal].toarray().ravel() > 0
        self.reward_matrix[goal, :] = 0.0
