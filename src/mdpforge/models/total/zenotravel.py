"""Deliver a passenger with the reduced ZenoTravel's unchanged flight rules.

State s=(city*(fuel_capacity+1)+fuel)*2+onboard, plus delivered and
accident, records one aircraft and passenger on a line of cities. Actions
0..5 board, fly forward, express forward, refuel, deliver and fly backward.
Refuelling stations occur every three cities; express uses three fuel units
to travel two cities. Flights retain delays and accidents, insufficient
fuel causes an accident, and other invalid actions stay in place. Both
explicit terminals remain absorbing; no time limit or seed is added.

state_dim is a positive Python integer target (default 506); fuel_capacity
is a positive Python integer (default 6). cities=max(4, round((state_dim-2)
/(2*(fuel_capacity+1)))), S=2*(fuel_capacity+1)*cities+2, A=6.
flight_failure=0.12 and accident_probability=0.015 are in [0,1].
All transitions are unchanged. Rewards replace costs and payoffs: 1 for
nonterminal actions entering delivered, 0 elsewhere and at both terminals.
Delivery is deterministic, so total reward is at most 1 even for policies
that never terminate. Cycles prevent the current acyclic certificate.
Reference: Younes et al. (2005), IPPC 2004 Zeno Travel; this retains the
catalogue's one-aircraft/passenger reduction, not a PPDDL grounding.
"""

from mdpforge.models.discounted.zenotravel import Model as DiscountedZenotravel

METADATA = {
    "category": "service_operations",
    "description": "Deliver a passenger using fuel-constrained, unreliable flights.",
    "reference": (
        "Younes et al., JAIR 24 (2005), IPPC 2004 Zeno Travel: "
        "https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/younes05a-html/"
    ),
    "sizes": {
        "small": {"requested_state_dim": 12000, "parameters": {}},
        "medium": {"requested_state_dim": 96000, "parameters": {}},
        "large": {"requested_state_dim": 768000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 58}


class Model(DiscountedZenotravel):
    """Reduced air transport with finite total reward for passenger delivery."""

    supported_criteria = frozenset({"total"})

    def __init__(
        self,
        state_dim: int = 506,
        fuel_capacity: int = 6,
        flight_failure: float = 0.12,
        accident_probability: float = 0.015,
    ):
        """Use positive integer sizes and flight_failure/accident_probability in [0,1]."""
        super().__init__(
            state_dim=state_dim,
            fuel_capacity=fuel_capacity,
            flight_failure=flight_failure,
            accident_probability=accident_probability,
        )
        self.name += "_total_delivered_entry_reward_v1"
        self.terminal_states = (self.delivered, self.accident)

    def _build_model(self):
        """Keep flight transitions and reward only passenger delivery actions."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.delivered].toarray().ravel() > 0
            )
        self.reward_matrix[list(self.terminal_states), :] = 0.0
