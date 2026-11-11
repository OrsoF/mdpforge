# Source:
# Sutton, R. S. and Barto, A. G. (2018).
# Reinforcement Learning: An Introduction, 2nd ed., Example 10.2.
#
# Access-control queueing task. A state is (number of free servers,
# priority class of the arriving customer). The agent accepts or rejects.
# This is naturally an average-reward continuing MDP.

import numpy as np
from scipy.sparse import dok_array
from scipy.stats import binom

from mdpforge.core.mdp import MDP

METADATA = {
    "sizes": {
        "small": {
            "requested_state_dim": 500,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 4000,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 8000,
            "parameters": {},
        },
    },
    "category": "service_operations",
    "description": "Accept or reject priority requests with limited free servers.",
    "reference": (
        "Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An "
        "Introduction, 2nd ed., Example 10.2."
    ),
}


class Model(MDP):
    REJECT = 0
    ACCEPT = 1

    def __init__(self, state_dim: int = 400, action_dim: int = 10):
        self.requested_state_dim = int(state_dim)
        del action_dim
        requested = max(20, int(state_dim))
        self.num_priorities = 4
        self.num_servers = max(1, requested // self.num_priorities - 1)

        self.priority_rewards = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        self.free_probability = 0.06
        self.action_dim = 2
        self.state_dim = (self.num_servers + 1) * self.num_priorities
        self.name = f"{self.state_dim}_{self.action_dim}_access_control"

    def _encode(self, free_servers: int, priority: int) -> int:
        return free_servers * self.num_priorities + priority

    def _build_model(self) -> None:
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim),
            dtype=np.float64,
        )
        matrices = [
            dok_array((self.state_dim, self.state_dim), dtype=np.float64)
            for _ in range(self.action_dim)
        ]

        priority_probability = 1.0 / self.num_priorities

        for free_servers in range(self.num_servers + 1):
            busy_servers = self.num_servers - free_servers
            release_distributions = {
                busy: binom.pmf(np.arange(busy + 1), busy, self.free_probability)
                for busy in (busy_servers, busy_servers + 1)
                if busy <= self.num_servers
            }

            for priority in range(self.num_priorities):
                state = self._encode(free_servers, priority)

                for action in range(self.action_dim):
                    accepted = action == self.ACCEPT and free_servers > 0
                    free_after_decision = free_servers - int(accepted)
                    busy_after_decision = busy_servers + int(accepted)

                    self.reward_matrix[state, action] = (
                        self.priority_rewards[priority] if accepted else 0.0
                    )

                    # During the step, each busy server independently becomes
                    # free with probability free_probability.
                    for released, release_probability in enumerate(
                        release_distributions[busy_after_decision]
                    ):
                        next_free = min(
                            self.num_servers,
                            free_after_decision + released,
                        )

                        for next_priority in range(self.num_priorities):
                            next_state = self._encode(next_free, next_priority)
                            matrices[action][state, next_state] += (
                                release_probability * priority_probability
                            )

        self.transition_matrix = []
        for matrix in matrices:
            matrix = matrix.tocsr()
            row_sums = np.asarray(matrix.sum(axis=1)).ravel()
            if np.any(row_sums <= 0):
                raise ValueError("Transition matrix contains an empty row.")

            # PyMDPToolbox requires row sums accurate to ~2e-15.
            # Normalize in place without allocating an array of size nnz.
            for state, total in enumerate(row_sums):
                start, end = matrix.indptr[state : state + 2]
                matrix.data[start:end] /= total

            # Compensate any residual accumulation error in CSR row sums.
            residuals = 1.0 - np.asarray(matrix.sum(axis=1)).ravel()
            for state, residual in enumerate(residuals):
                if residual:
                    start, end = matrix.indptr[state : state + 2]
                    largest = start + np.argmax(matrix.data[start:end])
                    matrix.data[largest] += residual

            self.transition_matrix.append(matrix)
