"""Reward rescue access in the catalogue's unchanged finite helicopter model.

States encode (zone, site status, onboard, urgency) as
((zone*3+status)*2+onboard)*(max_urgency+1)+urgency, plus rescued and lost.
Actions 0..4 are fly next, fly previous, survey, pickup and evacuate/return;
unavailable surveys or pickups retain the original loss and urgency dynamics.
Zone changes forget site knowledge. Rescue and loss are absorbing at zero.

state_dim is a positive Python integer target (default 506); max_urgency
is a positive Python integer (default 3). The number of zones is
max(2, round((state_dim-2)/(6*(max_urgency+1)))-1), giving
S=6*(zones+1)*(max_urgency+1)+2 and five actions. survey_success=0.65,
pickup_success=0.85 and per_step_loss=0.05 are probabilities in [0, 1].
These parameters retain their discounted meanings; no seed is used.

All transitions, including the existing urgency budget, are unchanged.
Rewards replace costs and payoffs: 1 for nonterminal actions with positive
probability of reaching rescued, 0 otherwise. Urgency decreases until
termination, so return is finite within max_urgency+1 steps. No new time
limit is introduced. This remains the reduced Markovian surrogate, not
a faithful grounding of the original IPPC model.

Reference: IPPC 2008 Search-and-Rescue (Teichteil-Konigsbuch).
"""

from mdpforge.models.discounted.search_and_rescue import (
    Model as DiscountedSearchAndRescue,
)

METADATA = {
    "category": "navigation",
    "description": "Rescue survivors before the existing urgency budget expires.",
    "reference": (
        "IPPC 2008 Search-and-Rescue (Teichteil-Konigsbuch): "
        "https://ipc08.icaps-conference.org/probabilistic/wiki/images/0/03/Results.pdf"
    ),
    "sizes": {
        "tiny": {"requested_state_dim": 9120, "parameters": {}},
        "small": {"requested_state_dim": 91200, "parameters": {}},
        "medium": {"requested_state_dim": 364800, "parameters": {}},
        "large": {"requested_state_dim": 2918400, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 74}


class Model(DiscountedSearchAndRescue):
    """Finite helicopter rescue with reward for actions that can reach success."""

    supported_criteria = frozenset({"total"})

    def __init__(
        self,
        state_dim: int = 506,
        max_urgency: int = 3,
        survey_success: float = 0.65,
        pickup_success: float = 0.85,
        per_step_loss: float = 0.05,
    ):
        """Use positive integer dimensions and urgency, and probabilities in [0, 1]."""
        super().__init__(
            state_dim=state_dim,
            max_urgency=max_urgency,
            survey_success=survey_success,
            pickup_success=pickup_success,
            per_step_loss=per_step_loss,
        )
        self.name = (
            f"{self.state_dim}_{self.action_dim}_searchandrescue_total_1"
            f"_{self.zones}_{max_urgency}"
            f"_{survey_success}_{pickup_success}_{per_step_loss}"
        )
        self.terminal_states = (self.rescued, self.lost)

    def _build_model(self):
        """Keep rescue dynamics and reward only actions with possible rescue."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.rescued].toarray().ravel() > 0
            )
        self.reward_matrix[list(self.terminal_states), :] = 0.0
