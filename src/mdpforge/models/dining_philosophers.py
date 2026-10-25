"""Reduced scheduled Lehmann--Rabin dining-philosophers MDP."""

import math

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "service_operations",
    "description": (
        "Reduced random-first-fork philosophers with shared forks and "
        "scheduler actions."
    ),
    "reference": (
        "Lehmann and Rabin (1981); PRISM example: "
        "https://www.prismmodelchecker.org/casestudies/phil.php"
    ),
    "sizes": {
        "small": {"requested_state_dim": 7400, "parameters": {}},
        "medium": {"requested_state_dim": 29600, "parameters": {}},
        "large": {"requested_state_dim": 118400, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 125}


class Model(MDP):
    """Five statuses per philosopher: thinking, hungry, left, right, eating."""

    def __init__(self, state_dim: int = 125):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        self.philosophers = max(2, min(7, round(math.log(state_dim, 5))))
        super().__init__(5**self.philosophers, self.philosophers)
        self.requested_state_dim = state_dim
        self.name = (
            f"{self.state_dim}_{self.action_dim}_dining_philosophers_1"
            f"_{self.philosophers}"
        )

    def _encode(self, statuses):
        result = 0
        for status in statuses:
            result = result * 5 + status
        return result

    def _decode(self, state):
        statuses = [0] * self.philosophers
        for i in range(self.philosophers - 1, -1, -1):
            statuses[i] = state % 5
            state //= 5
        return statuses

    def _forks_available(self, status, index):
        # Philosopher i's left fork is shared with i-1's right fork.
        left = status[(index - 1) % self.philosophers] not in (3, 4)
        right = status[(index + 1) % self.philosophers] not in (2, 4)
        return left, right

    def _build_model(self):
        builder = TransitionBuilder(self.state_dim, self.action_dim)
        for s in range(self.state_dim):
            statuses = self._decode(s)
            for a in range(self.philosophers):
                status = statuses[a]
                left_free, right_free = self._forks_available(statuses, a)
                builder.rewards[s, a] = -0.1
                changes = []
                if status == 0:
                    changes = [(1, 1)]  # thinking -> hungry
                elif status == 1:
                    # Random first fork, dropping infeasible choices.
                    changes = [
                        (2 if left_free else 1, 0.5),
                        (3 if right_free else 1, 0.5),
                    ]
                elif status == 2:
                    changes = [(4 if right_free else 1, 1)]
                elif status == 3:
                    changes = [(4 if left_free else 1, 1)]
                else:
                    changes = [(0, 1)]  # eat -> thinking, put down both
                    builder.rewards[s, a] = 5
                for new_status, probability in changes:
                    next_statuses = statuses.copy()
                    next_statuses[a] = new_status
                    builder.add(a, s, self._encode(next_statuses), probability)
        self.transition_matrix = builder.finish()
        self.reward_matrix = builder.rewards
