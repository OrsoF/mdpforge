"""Deliver a passenger using the catalogue's unchanged Taxi transitions.

States enumerate (x, y, passenger_position, destination), followed by one
absorbing delivered state. Passenger positions 0..3 denote corner landmarks
and 4 means onboard; destinations are 0..3. Actions 0..5 are west, east,
north, south, pickup and dropoff. Invalid pickups/dropoffs and boundary
moves stay in place; valid delivery enters the terminal deterministically.

state_dim is a positive Python integer target (default 500). The square
grid side is max(1, floor(sqrt(state_dim//20))), giving S=20*side**2+1
and six actions. Landmarks remain the four corners, even when corners
coincide on a one-cell grid. There is no random generation or seed.

All transitions are unchanged. Rewards are 1 only for actions entering the
delivered state and 0 otherwise. This removes pickup bonuses and the
discounted terminal's continuing reward; the terminal earns 0. Total reward
is at most 1, including under policies that never deliver. No time limit
is added; transient cycles prevent the current acyclic certificate.

Reference: Dietterich (2000), Hierarchical reinforcement learning with the
MAXQ value function decomposition, JAIR 13:227-303; this retains the local
catalogue's open-grid, corner-landmark adaptation.
"""

from mdpforge.models.discounted.taxi import Model as DiscountedTaxi

METADATA = {
    "category": "service_operations",
    "description": "Pick up and deliver a passenger with reward only for final delivery.",
    "reference": (
        "Dietterich, T. G. (2000). Hierarchical reinforcement learning with the "
        "MAXQ value function decomposition. JAIR 13:227-303."
    ),
    "sizes": {
        "small": {"requested_state_dim": 6000, "parameters": {}},
        "medium": {"requested_state_dim": 48000, "parameters": {}},
        "large": {"requested_state_dim": 192000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 80}


class Model(DiscountedTaxi):
    """Taxi delivery with a zero-reward absorbing terminal and finite return."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 500):
        """Accept state_dim as a positive Python integer target (default 500)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name += "_total_delivered_entry_reward_v1"
        self.terminal_states = (self.terminal_state,)

    def _build_model(self):
        """Keep taxi transitions and reward only actions entering delivery."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.terminal_state].toarray().ravel() > 0
            )
        self.reward_matrix[self.terminal_state, :] = 0.0
