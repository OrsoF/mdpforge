"""Deliver the reduced Boxworld crate using unchanged transport transitions.

State s=(truck*locations+crate)*2+loaded records positions and loading,
plus delivered and damaged terminals. Actions 0..4 drive next, drive
previous, express next-two, pick up and drop. Delivery requires dropping
while loaded at the final location. Invalid actions stay in place; driving
retains boundary clipping, delays and cargo damage. Both terminals absorb.

state_dim is a positive Python integer target (default 514); locations=
max(3, round(sqrt(max(0,state_dim-2)/2))), S=2*locations**2+2, A=5.
traffic_delay is in [0,1] (default 0.2); express adds 0.15 capped at 1.
cargo_damage is in [0,0.5] (default 0.025), since express doubles it.
The narrower domain rejects negative probabilities admitted by the
discounted constructor; valid transition definitions are unchanged.

Rewards are 1 for nonterminal actions entering delivered and 0 otherwise,
including both terminals, replacing all costs and payoffs. Delivery is
deterministic, so total reward is at most 1 even without termination.
No time limit or seed is added; cycles prevent the acyclic certificate.
Reference: Younes et al. (2005), IPPC 2004 Boxworld; this remains the
catalogue's one-truck, one-crate road reduction, not a PPDDL grounding.
"""

from mdpforge.models.discounted.boxworld import Model as DiscountedBoxworld

METADATA = {
    "category": "service_operations",
    "description": "Deliver a crate with traffic delays and absorbing cargo damage.",
    "reference": (
        "Younes et al., JAIR 24 (2005), IPPC 2004 Boxworld: "
        "https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/younes05a-html/"
    ),
    "sizes": {
        "small": {"requested_state_dim": 20000, "parameters": {}},
        "medium": {"requested_state_dim": 80000, "parameters": {}},
        "large": {"requested_state_dim": 320000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 52}


class Model(DiscountedBoxworld):
    """Reduced crate transport with finite total reward for final delivery."""

    supported_criteria = frozenset({"total"})

    def __init__(
        self,
        state_dim: int = 514,
        traffic_delay: float = 0.2,
        cargo_damage: float = 0.025,
    ):
        """Use positive state_dim, traffic_delay in [0,1], cargo_damage in [0,0.5]."""
        if not 0 <= cargo_damage <= 0.5:
            raise ValueError("cargo_damage must be in [0, 0.5] for express transport")
        super().__init__(
            state_dim=state_dim,
            traffic_delay=traffic_delay,
            cargo_damage=cargo_damage,
        )
        self.name += "_total_delivered_entry_reward_v1"
        self.terminal_states = (self.delivered, self.damaged)

    def _build_model(self):
        """Keep cargo transitions and reward only successful delivery actions."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.delivered].toarray().ravel() > 0
            )
        self.reward_matrix[list(self.terminal_states), :] = 0.0
