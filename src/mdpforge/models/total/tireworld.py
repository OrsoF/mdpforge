"""Reach the reduced Tireworld destination without changing road dynamics.

State s=4*(2*column+lane)+2*spare+flat encodes two lanes and tire flags,
plus goal and stranded terminals. Actions 0..4 are forward via fast lane,
forward via depot lane, change lane, load and repair. Existing invalid
actions stay in place or strand a vehicle with no spare in the fast lane;
the last column enters the goal. Both explicit terminals stay absorbing.

state_dim is a positive Python integer target (default 98); length=max(3,
round((state_dim-2)/8)), S=8*length+2, A=5. puncture_probability is in
[0,1] (default 0.15), retaining its probability per driven edge. No seed
is used. Transitions are unchanged; rewards replace all costs and payoffs:
1 for nonterminal goal-entry actions, 0 otherwise and at both terminals.
Goal entry is deterministic, so total reward is at most 1 even without
termination. No time limit is added; cycles prevent the acyclic certificate.
Reference: IPPC 2004 Tireworld (Younes et al., 2005); this is the local
two-lane reduction with renewable depots, not a grounded IPPC instance.
"""

from mdpforge.models.discounted.tireworld import Model as DiscountedTireworld

METADATA = {
    "category": "navigation",
    "description": "Reach a road destination despite punctures and absorbing stranding.",
    "reference": (
        "IPPC 2004 Tireworld (Younes et al., JAIR 2005): "
        "https://ipc04.icaps-conference.org/probabilistic/"
    ),
    "sizes": {
        "small": {"requested_state_dim": 15600, "parameters": {}},
        "medium": {"requested_state_dim": 62400, "parameters": {}},
        "large": {"requested_state_dim": 499200, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 50}


class Model(DiscountedTireworld):
    """Reduced Tireworld with finite total reward for destination access."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 98, puncture_probability: float = 0.15):
        """Use positive integer state_dim and puncture_probability in [0, 1]."""
        super().__init__(
            state_dim=state_dim, puncture_probability=puncture_probability
        )
        self.name += "_total_goal_entry_reward_v1"
        self.terminal_states = (self.goal, self.stranded)

    def _build_model(self):
        """Keep road transitions and reward only actions that reach the goal."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.goal].toarray().ravel() > 0
            )
        self.reward_matrix[list(self.terminal_states), :] = 0.0
