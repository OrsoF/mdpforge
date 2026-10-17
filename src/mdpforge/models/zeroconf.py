"""Reduced IPv4 Zeroconf address-probing MDP, not the PRISM timed automaton."""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "service_operations",
    "description": (
        "Reduced Zeroconf: probe losses, occupied addresses and retries; "
        "discounted rewards."
    ),
    "reference": (
        "PRISM Benchmark Suite, IPv4 Zeroconf (KNPS06): "
        "https://www.prismmodelchecker.org/benchmarks/models.php#mdps"
    ),
    "sizes": {
        "small": {"requested_state_dim": 52, "parameters": {"probes": 4}},
        "medium": {"requested_state_dim": 502, "parameters": {"probes": 4}},
        "large": {"requested_state_dim": 5002, "parameters": {"probes": 4}},
    },
}

TEST_PARAMETERS = {"state_dim": 42, "probes": 2, "existing_hosts": 13004}


class Model(MDP):
    """State (attempt, candidate_occupied, probes_sent) plus success/failure."""

    def __init__(
        self,
        state_dim: int = 102,
        probes: int = 4,
        existing_hosts: int = 1000,
        loss_probability: float = 0.1,
    ):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if type(probes) is not int or probes < 1:
            raise ValueError("probes must be a positive integer")
        if type(existing_hosts) is not int or not 0 <= existing_hosts <= 65024:
            raise ValueError("existing_hosts must be in [0, 65024]")
        if not 0 <= loss_probability <= 1:
            raise ValueError("loss_probability must be in [0, 1]")
        self.probes = probes
        self.existing_hosts = existing_hosts
        self.loss_probability = float(loss_probability)
        self.occupied_probability = existing_hosts / 65024
        self.attempts = max(1, round((state_dim - 2) / (2 * (probes + 1))))
        super().__init__(2 * (probes + 1) * self.attempts + 2, 2)
        self.requested_state_dim = state_dim
        self.success = self.state_dim - 2
        self.failure = self.state_dim - 1
        self.name = (
            f"{self.state_dim}_{self.action_dim}_zeroconf_v1"
            f"_k{probes}_attempts{self.attempts}_hosts{existing_hosts}"
            f"_loss{self.loss_probability}"
        )

    def _encode(self, attempt, occupied, sent):
        return (attempt * 2 + occupied) * (self.probes + 1) + sent

    def _candidate(self, transitions, action, state, attempt, probability):
        if attempt >= self.attempts:
            transitions.add(action, state, self.failure, probability)
        else:
            p = self.occupied_probability
            transitions.add(
                action, state, self._encode(attempt, 0, 0), probability * (1 - p)
            )
            transitions.add(action, state, self._encode(attempt, 1, 0), probability * p)

    def _build_model(self):
        builder = TransitionBuilder(self.state_dim, self.action_dim)
        for attempt in range(self.attempts):
            for occupied in (0, 1):
                for sent in range(self.probes + 1):
                    s = self._encode(attempt, occupied, sent)
                    # Action 1 restarts with a newly sampled candidate IP.
                    builder.rewards[s, 1] = -2.0
                    self._candidate(builder, 1, s, attempt + 1, 1.0)
                    # Action 0 sends the next probe; collisions can be missed.
                    builder.rewards[s, 0] = -1.0
                    if sent == self.probes:
                        final = self.failure if occupied else self.success
                        builder.add(0, s, final, 1.0)
                        builder.rewards[s, 0] += -20 if occupied else 20
                    elif occupied:
                        detected = 1 - self.loss_probability
                        self._candidate(builder, 0, s, attempt + 1, detected)
                        builder.add(
                            0, s, self._encode(attempt, 1, sent + 1), 1 - detected
                        )
                    else:
                        builder.add(0, s, self._encode(attempt, 0, sent + 1), 1.0)
        builder.absorbing(self.success)
        builder.absorbing(self.failure)
        self.transition_matrix = builder.finish()
        self.reward_matrix = builder.rewards
