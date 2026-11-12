"""Reach the drone destination while preserving battery and wind dynamics.

State s=(location*(battery_capacity+1)+battery)*2+wind, plus landed and
crashed, records route position, battery and a wind bit. Actions 0..3 fly
forward, fly backward, recharge and wait. Flying spends one battery unit;
flight at zero battery crashes. Recharge adds two units only at locations
divisible by four; otherwise it behaves as waiting. Delays, clipping and
wind resampling retain the catalogue rules. Both terminals stay absorbing.

state_dim is a positive Python integer target (default 506), and
battery_capacity is a positive Python integer (default 5). locations=
max(4, round((state_dim-2)/(2*(battery_capacity+1)))), giving
S=2*(battery_capacity+1)*locations+2 and A=4. wind_probability=0.25 and
stall_probability=0.15 are in [0,1]; stall probability without wind is 0.05.

Transitions are unchanged. Rewards replace costs and payoffs: 1 for
nonterminal actions with positive probability of entering landed, 0
otherwise and at both terminals. This is an action reward, not a success
probability. For each finite instance, the minimum positive goal-entry
probability bounds the expected number of rewarded actions, giving finite
return even without termination. No time limit or seed is added; cycles
prevent the acyclic certificate. Reference: ElSayed-Aly, Parker & Feng,
NFM 2024 Energy; this is the local reduced model, not the PRISM benchmark.
"""

from mdpforge.models.discounted.drone_energy import Model as DiscountedDroneEnergy

METADATA = {
    "category": "navigation",
    "description": "Reach an absorbing drone destination with battery and uncertain wind.",
    "reference": (
        "ElSayed-Aly, Parker & Feng, Distributional Probabilistic Model Checking "
        "(NFM 2024), Energy drone example: "
        "https://www.prismmodelchecker.org/files/nfm24dpmc/"
    ),
    "sizes": {
        "small": {"requested_state_dim": 20000, "parameters": {"battery_capacity": 5}},
        "medium": {
            "requested_state_dim": 160000,
            "parameters": {"battery_capacity": 5},
        },
        "large": {"requested_state_dim": 640000, "parameters": {"battery_capacity": 5}},
    },
}

TEST_PARAMETERS = {"state_dim": 50, "battery_capacity": 5}


class Model(DiscountedDroneEnergy):
    """Battery-constrained drone with finite total reward for destination access."""

    supported_criteria = frozenset({"total"})

    def __init__(
        self,
        state_dim: int = 506,
        battery_capacity: int = 5,
        wind_probability: float = 0.25,
        stall_probability: float = 0.15,
    ):
        """Use positive integer sizes and wind_probability/stall_probability in [0,1]."""
        super().__init__(
            state_dim=state_dim,
            battery_capacity=battery_capacity,
            wind_probability=wind_probability,
            stall_probability=stall_probability,
        )
        self.name += "_total_landed_entry_reward_v1"
        self.terminal_states = (self.landed, self.crashed)

    def _build_model(self):
        """Keep drone dynamics and reward only actions with possible landing."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.landed].toarray().ravel() > 0
            )
        self.reward_matrix[list(self.terminal_states), :] = 0.0
