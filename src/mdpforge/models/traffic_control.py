"""Two-approach stochastic traffic-light MDP inspired by IPPC 2011 Traffic.

States (queue_N, queue_E, green_phase). Actions hold/switch the phase.
Switching incurs an all-red time slot; traffic arrivals are Bernoulli and
arrivals beyond queue capacity are dropped with expected overflow penalty.
A deliberately small, infinite-horizon discounted surrogate, not IPPC RDDL.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "service_operations",
    "description": "Two queues with independent arrivals and costly traffic-light switches.",
    "reference": (
        "IPPC 2011 Traffic Control RDDL benchmark (inspiration): "
        "https://users.cecs.anu.edu.au/~ssanner/IPPC_2011/"
    ),
    "sizes": {
        "small": {"requested_state_dim": 50, "parameters": {}},
        "medium": {"requested_state_dim": 512, "parameters": {}},
        "large": {"requested_state_dim": 5000, "parameters": {}},
    },
}
TEST_PARAMETERS = {"state_dim": 50}


class Model(MDP):
    """Action 0 maintains green; action 1 switches (one all-red slot)."""

    def __init__(
        self,
        state_dim: int = 512,
        arrival_north: float = 0.45,
        arrival_east: float = 0.35,
    ):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if not 0 <= arrival_north <= 1 or not 0 <= arrival_east <= 1:
            raise ValueError("arrival probabilities must be in [0, 1]")
        self.capacity = max(2, round((state_dim / 2) ** 0.5) - 1)
        self.arrival_north, self.arrival_east = (
            float(arrival_north),
            float(arrival_east),
        )
        super().__init__(2 * (self.capacity + 1) ** 2, 2)
        self.requested_state_dim = state_dim
        self.name = (
            f"traffic_control_v1_q{self.capacity}_n{self.arrival_north:.17g}"
            f"_e{self.arrival_east:.17g}"
        )

    def _encode(self, north, east, phase):
        return (north * (self.capacity + 1) + east) * 2 + phase

    def _build_model(self):
        t = TransitionBuilder(self.state_dim, self.action_dim)
        for north in range(self.capacity + 1):
            for east in range(self.capacity + 1):
                for phase in (0, 1):
                    s = self._encode(north, east, phase)
                    for action in (0, 1):
                        switching = action == 1
                        after_n = max(0, north - int(not switching and phase == 0))
                        after_e = max(0, east - int(not switching and phase == 1))
                        next_phase = phase ^ switching
                        t.rewards[s, action] = -after_n - after_e - 0.75 * switching
                        for n_arrival, pn in (
                            (0, 1 - self.arrival_north),
                            (1, self.arrival_north),
                        ):
                            for e_arrival, pe in (
                                (0, 1 - self.arrival_east),
                                (1, self.arrival_east),
                            ):
                                p = pn * pe
                                overflow = (after_n + n_arrival > self.capacity) + (
                                    after_e + e_arrival > self.capacity
                                )
                                t.rewards[s, action] -= 3 * overflow * p
                                t.add(
                                    action,
                                    s,
                                    self._encode(
                                        min(self.capacity, after_n + n_arrival),
                                        min(self.capacity, after_e + e_arrival),
                                        next_phase,
                                    ),
                                    p,
                                )
        self.transition_matrix, self.reward_matrix = t.finish(), t.rewards
