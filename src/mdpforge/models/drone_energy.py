"""Finite battery-and-wind drone MDP, inspired by PRISM NFM'24 Energy.

States are (location, battery, wind), plus landed/crashed terminals.
Actions: fly forward, fly backward, recharge, wait.  Wind is resampled at
all nonterminal steps; flight can be delayed and uses one battery unit.
These transitions/rewards are a reduced discounted formulation, not the
published PRISM benchmark or a conversion of its properties.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "navigation",
    "description": "Battery-constrained drone route with uncertain wind and charging pads.",
    "reference": (
        "ElSayed-Aly, Parker & Feng, Distributional Probabilistic Model Checking "
        "(NFM 2024), Energy drone example: "
        "https://www.prismmodelchecker.org/files/nfm24dpmc/"
    ),
    "sizes": {
        "small": {"requested_state_dim": 20000, "parameters": {"battery_capacity": 5}},
        "medium": {"requested_state_dim": 506, "parameters": {"battery_capacity": 5}},
        "large": {"requested_state_dim": 5006, "parameters": {"battery_capacity": 5}},
    },
}
TEST_PARAMETERS = {"state_dim": 50, "battery_capacity": 5}


class Model(MDP):
    """Absorbing delivery (success) and crash (failure) states."""

    def __init__(
        self,
        state_dim: int = 506,
        battery_capacity: int = 5,
        wind_probability: float = 0.25,
        stall_probability: float = 0.15,
    ):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if type(battery_capacity) is not int or battery_capacity < 1:
            raise ValueError("battery_capacity must be a positive integer")
        if not 0 <= wind_probability <= 1 or not 0 <= stall_probability <= 1:
            raise ValueError("wind_probability and stall_probability must be in [0, 1]")
        self.battery_capacity = battery_capacity
        self.wind_probability = float(wind_probability)
        self.stall_probability = float(stall_probability)
        self.locations = max(4, round((state_dim - 2) / (2 * (battery_capacity + 1))))
        super().__init__(2 * (battery_capacity + 1) * self.locations + 2, 4)
        self.requested_state_dim = state_dim
        self.landed, self.crashed = self.state_dim - 2, self.state_dim - 1
        self.name = (
            f"{self.state_dim}_{self.action_dim}_drone_energy_v1"
            f"_n{self.locations}_battery{battery_capacity}"
            f"_wind{self.wind_probability}_stall{self.stall_probability}"
        )

    def _encode(self, location: int, battery: int, wind: int) -> int:
        return (location * (self.battery_capacity + 1) + battery) * 2 + wind

    def _build_model(self):
        t = TransitionBuilder(self.state_dim, self.action_dim)
        for location in range(self.locations):
            for battery in range(self.battery_capacity + 1):
                for wind in (0, 1):
                    s = self._encode(location, battery, wind)
                    for action in range(4):
                        t.rewards[s, action] = -0.5
                        if action in (0, 1):
                            if battery == 0:
                                t.add(action, s, self.crashed, 1)
                                t.rewards[s, action] -= 25
                                continue
                            direction = 1 if action == 0 else -1
                            destination = max(0, location + direction)
                            destination = min(self.locations - 1, destination)
                            stalled = self.stall_probability if wind else 0.05
                            t.rewards[s, action] -= 0.5
                            for new_location, p_move in (
                                (destination, 1 - stalled),
                                (location, stalled),
                            ):
                                if new_location == self.locations - 1:
                                    t.add(action, s, self.landed, p_move)
                                    t.rewards[s, action] += 35 * p_move
                                else:
                                    for next_wind, p_wind in (
                                        (0, 1 - self.wind_probability),
                                        (1, self.wind_probability),
                                    ):
                                        t.add(
                                            action,
                                            s,
                                            self._encode(
                                                new_location, battery - 1, next_wind
                                            ),
                                            p_move * p_wind,
                                        )
                        else:
                            charging = action == 2 and location % 4 == 0
                            next_battery = (
                                min(self.battery_capacity, battery + 2)
                                if charging
                                else battery
                            )
                            t.rewards[s, action] -= 0.3 if charging else 1.0
                            for next_wind, p in (
                                (0, 1 - self.wind_probability),
                                (1, self.wind_probability),
                            ):
                                t.add(
                                    action,
                                    s,
                                    self._encode(location, next_battery, next_wind),
                                    p,
                                )
        t.absorbing(self.landed)
        t.absorbing(self.crashed)
        self.transition_matrix, self.reward_matrix = t.finish(), t.rewards
