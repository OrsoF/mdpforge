"""Finite PPDDL-inspired ZenoTravel with fuel and unreliable flights.

State (city, fuel, passenger-onboard), plus delivered/accident terminals.
Actions: board, fly forward, express forward, refuel, deliver, fly backward.
Simplifies IPPC 2004 PPDDL (one aircraft/passenger; linear city topology).
All rewards represent one-stage *expected* discounted rewards.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "service_operations",
    "description": "Single-aircraft passenger delivery with fuel, refuelling and flight failure.",
    "reference": (
        "Younes et al., JAIR 24 (2005), IPPC 2004 Zeno Travel: "
        "https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/younes05a-html/"
    ),
    "sizes": {
        "small": {"requested_state_dim": 58, "parameters": {}},
        "medium": {"requested_state_dim": 506, "parameters": {}},
        "large": {"requested_state_dim": 5014, "parameters": {}},
    },
}
TEST_PARAMETERS = {"state_dim": 58}


class Model(MDP):
    """Stations every three cities, fuel=0..capacity and onboard bit."""

    def __init__(
        self,
        state_dim: int = 506,
        fuel_capacity: int = 6,
        flight_failure: float = 0.12,
        accident_probability: float = 0.015,
    ):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if type(fuel_capacity) is not int or fuel_capacity < 1:
            raise ValueError("fuel_capacity must be a positive integer")
        if not 0 <= flight_failure <= 1 or not 0 <= accident_probability <= 1:
            raise ValueError("probabilities must be in [0, 1]")
        self.fuel_capacity = fuel_capacity
        self.flight_failure = float(flight_failure)
        self.accident_probability = float(accident_probability)
        self.cities = max(4, round((state_dim - 2) / (2 * (fuel_capacity + 1))))
        super().__init__(2 * (fuel_capacity + 1) * self.cities + 2, 6)
        self.requested_state_dim = state_dim
        self.delivered, self.accident = self.state_dim - 2, self.state_dim - 1
        self.name = (
            f"{self.state_dim}_{self.action_dim}_zenotravel_v1"
            f"_n{self.cities}_fuel{fuel_capacity}"
            f"_failure{flight_failure}_accident{accident_probability}"
        )

    def _encode(self, city, fuel, boarded):
        return (city * (self.fuel_capacity + 1) + fuel) * 2 + boarded

    def _build_model(self):
        t = TransitionBuilder(self.state_dim, self.action_dim)
        for city in range(self.cities):
            for fuel in range(self.fuel_capacity + 1):
                for boarded in (0, 1):
                    s = self._encode(city, fuel, boarded)
                    for action in range(6):
                        t.rewards[s, action] = -0.8
                        if action == 0 and city == 0 and not boarded:
                            t.add(action, s, self._encode(city, fuel, 1), 1)
                        elif (
                            action == 3 and city % 3 == 0 and fuel < self.fuel_capacity
                        ):
                            t.add(
                                action,
                                s,
                                self._encode(city, self.fuel_capacity, boarded),
                                1,
                            )
                            t.rewards[s, action] -= 2.0
                        elif action == 4 and boarded and city == self.cities - 1:
                            t.add(action, s, self.delivered, 1)
                            t.rewards[s, action] += 80
                        elif action in (1, 2, 5):
                            distance = 2 if action == 2 else 1
                            required = 3 if action == 2 else 1
                            direction = -1 if action == 5 else 1
                            target = city + direction * distance
                            if 0 <= target < self.cities and fuel >= required:
                                p_acc = self.accident_probability
                                t.add(action, s, self.accident, p_acc)
                                t.add(
                                    action,
                                    s,
                                    self._encode(target, fuel - required, boarded),
                                    (1 - p_acc) * (1 - self.flight_failure),
                                )
                                t.add(
                                    action,
                                    s,
                                    self._encode(city, fuel - required, boarded),
                                    (1 - p_acc) * self.flight_failure,
                                )
                                t.rewards[s, action] -= 3 * required + 35 * p_acc
                            elif (
                                target >= 0 and target < self.cities and fuel < required
                            ):
                                t.add(action, s, self.accident, 1)
                                t.rewards[s, action] = -35
                            else:
                                t.add(action, s, s, 1)
                                t.rewards[s, action] = -3
                        else:
                            t.add(action, s, s, 1)
                            t.rewards[s, action] = -3
        t.absorbing(self.delivered)
        t.absorbing(self.accident)
        self.transition_matrix, self.reward_matrix = t.finish(), t.rewards
