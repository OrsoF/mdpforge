"""Reduced randomized mutual exclusion, not the original Rabin protocol."""

import math

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "service_operations",
    "description": (
        "Reduced randomized contention with scheduled processes and "
        "exclusive critical section."
    ),
    "reference": (
        "Rabin (1982); PRISM mutual exclusion model: "
        "https://www.prismmodelchecker.org/casestudies/rabin.php"
    ),
    "sizes": {
        "small": {"requested_state_dim": 1800, "parameters": {}},
        "medium": {"requested_state_dim": 1024, "parameters": {}},
        "large": {"requested_state_dim": 4096, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 64}


class Model(MDP):
    """Per-process idle, waiting, candidate or critical state; scheduler actions."""

    def __init__(self, state_dim: int = 256):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        self.processes = max(2, min(7, round(math.log(state_dim, 4))))
        super().__init__(4**self.processes, self.processes)
        self.requested_state_dim = state_dim
        self.name = (
            f"{self.state_dim}_{self.action_dim}_rabin_v1"
            f"_n{self.processes}"
        )

    def _encode(self, statuses):
        result = 0
        for status in statuses:
            result = result * 4 + status
        return result

    def _decode(self, state):
        statuses = [0] * self.processes
        for i in range(self.processes - 1, -1, -1):
            statuses[i] = state % 4
            state //= 4
        return statuses

    def _build_model(self):
        builder = TransitionBuilder(self.state_dim, self.action_dim)
        for s in range(self.state_dim):
            statuses = self._decode(s)
            for a in range(self.processes):
                status = statuses[a]
                competing = sum(value in (1, 2) for value in statuses)
                has_claim = any(
                    value in (2, 3) for i, value in enumerate(statuses) if i != a
                )
                builder.rewards[s, a] = -0.1
                if status == 0:
                    updates = ((1, 1),)  # request entry
                elif status == 1:
                    if has_claim:
                        updates = ((1, 1),)
                    else:
                        win = 1 / max(1, competing)
                        updates = ((2, win), (1, 1 - win))
                elif status == 2:
                    updates = ((1 if has_claim else 3, 1),)
                else:
                    updates = ((0, 1),)
                    builder.rewards[s, a] = 8  # completed critical section
                for next_status, probability in updates:
                    nxt = statuses.copy()
                    nxt[a] = next_status
                    builder.add(a, s, self._encode(nxt), probability)
        self.transition_matrix = builder.finish()
        self.reward_matrix = builder.rewards
