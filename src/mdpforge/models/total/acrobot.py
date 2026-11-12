"""Reach the catalogue Acrobot's height threshold with unchanged dynamics.

States encode angle and velocity bin indices as
((i1*b+i2)*b+iv1)*b+iv2, plus one absorbing height-goal state. Actions
0..2 apply torques -1, 0 and 1. Euler integration (dt=0.2), wrapping,
velocity clipping and nearest-grid projection retain the discounted rules;
the goal is entered when -cos(theta1)-cos(theta1+theta2)>1 after integration.
All actions are valid. No random generation or seed is used.

state_dim is a positive Python integer target (default 256); b=max(2,
round(max(256, state_dim)**0.25)), actual S=b**4+1 and A=3. All transitions
are unchanged. Rewards replace step costs: 1 for nonterminal actions
entering the goal, 0 otherwise and at the goal. Total reward is at most
1 even without termination. No time limit is added; cycles prevent the
acyclic certificate. Reference: Sutton (1996), sparse coarse coding;
this preserves the catalogue's discretized Euler adaptation.
"""

from mdpforge.models.discounted.acrobot import Model as DiscountedAcrobot

METADATA = {
    "category": "navigation",
    "description": "Swing a discretized Acrobot to an absorbing height target.",
    "reference": (
        "Sutton, R. S. (1996). Generalization in Reinforcement Learning: Successful "
        "Examples Using Sparse Coarse Coding. Advances in Neural Information "
        "Processing Systems 8."
    ),
    "sizes": {
        "small": {"requested_state_dim": 10000, "parameters": {}},
        "medium": {"requested_state_dim": 225000, "parameters": {}},
        "large": {"requested_state_dim": 1229314, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 256}


class Model(DiscountedAcrobot):
    """Discretized Acrobot with finite total reward for height-goal access."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 256):
        """Accept state_dim as a positive Python integer target (default 256)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name += "_total_height_entry_reward_v1"
        self.terminal_states = (self.terminal_state,)

    def _build_model(self):
        """Keep Acrobot dynamics and reward actions reaching the height target."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.terminal_state].toarray().ravel() > 0
            )
        self.reward_matrix[self.terminal_state, :] = 0.0
