# Source: discretized scalar linear-quadratic regulator.

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "tags": ["control", "real-world"],
    "sizes": {
        "small": {
            "state_dim": 100,
            "parameters": {"state_dim": 100},
            "source": "measured",
        },
        "medium": {
            "state_dim": 1000,
            "parameters": {"state_dim": 1000},
            "source": "measured",
        },
        "large": {
            "state_dim": 4600,
            "parameters": {"state_dim": 4600},
            "source": "inferred",
        },
    },
    "category": "control",
    "description": (
        "Discretized scalar dynamics with quadratic state and control costs."
    ),
    "reference": (
        "Family reference: Kalman (1960). Contributions to the Theory "
        "of Optimal Control."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        self.state_dim = max(5, int(state_dim))
        self.action_dim = max(3, int(action_dim))

        self.dynamics_a = 0.9
        self.dynamics_b = 0.8
        self.noise_probability = 0.1

        self.name = "{}_{}_lqr".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.states = np.linspace(-2.0, 2.0, self.state_dim)
        self.actions = np.linspace(-1.0, 1.0, self.action_dim)

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        for state_index, state in enumerate(self.states):
            for action_index, action in enumerate(self.actions):
                mean_next = self.dynamics_a * state + self.dynamics_b * action
                center = self._closest_state_index(mean_next)
                left = max(0, center - 1)
                right = min(self.state_dim - 1, center + 1)

                self.transition_matrix[action_index][state_index, center] += (
                    1.0 - self.noise_probability
                )
                self.transition_matrix[action_index][state_index, left] += (
                    self.noise_probability / 2.0
                )
                self.transition_matrix[action_index][state_index, right] += (
                    self.noise_probability / 2.0
                )

                self.reward_matrix[state_index, action_index] = -(
                    state**2 + 0.1 * action**2
                )

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]

    def _closest_state_index(self, state: float) -> int:
        return int(np.argmin(np.abs(self.states - state)))
