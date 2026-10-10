# Source:
# Mason, W. A. and Watts, D. J. (2012).
# Collaborative learning in networks.
# PNAS 109(3): 764–769.

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "tags": ["resource-management", "real-world"],
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
            "state_dim": 4800,
            "parameters": {"state_dim": 4800},
            "source": "inferred",
        },
    },
    "category": "resource_management",
    "description": (
        "Survey-location decisions with distance-dependent exploration rewards."
    ),
    "reference": (
        "Mason, W. A. and Watts, D. J. (2012). Collaborative learning in networks. "
        "PNAS 109(3): 764–769."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        if state_dim < 2:
            raise ValueError("state_dim must be at least 2.")

        if action_dim < 2:
            raise ValueError("action_dim must be at least 2.")

        self.state_dim = state_dim
        self.action_dim = action_dim

        self._reward_type = "laplace"

        self.l = 1.0
        self.c = 0.75

        self.name = f"{self.state_dim}_{self.action_dim}_oil"

    def _laplace_survey_function(self, state: float) -> float:
        return np.exp(-self.l * abs(state - self.c))

    def _quadratic_survey_function(self, state: float) -> float:
        return 1.0 - self.l * (state - self.c) ** 2

    def _reward_function(self, state: int, action: int) -> float:
        normalized_state = state / (self.state_dim - 1)
        normalized_action = action / (self.action_dim - 1)

        if self._reward_type == "laplace":
            survey_value = self._laplace_survey_function(normalized_state)
        elif self._reward_type == "quadratic":
            survey_value = self._quadratic_survey_function(normalized_state)
        else:
            raise ValueError(f"Unknown reward type: {self._reward_type}")

        return max(
            0.0,
            survey_value - abs(normalized_state - normalized_action),
        )

    @staticmethod
    def _normalize_transition_matrix(matrix: csr_matrix) -> csr_matrix:
        matrix = matrix.astype(np.float64).tocsr()

        row_sums = np.asarray(matrix.sum(axis=1)).ravel()

        if np.any(row_sums <= 0.0):
            invalid_rows = np.flatnonzero(row_sums <= 0.0)
            raise ValueError(
                f"Transition matrix contains empty rows: {invalid_rows.tolist()}"
            )

        inverse_row_sums = 1.0 / row_sums

        return matrix.multiply(inverse_row_sums[:, None]).tocsr()

    def _build_model(self) -> None:
        transition_matrices = [
            dok_matrix(
                (self.state_dim, self.state_dim),
                dtype=np.float64,
            )
            for _ in range(self.action_dim)
        ]

        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim),
            dtype=np.float64,
        )

        for state in range(self.state_dim):
            for action in range(self.action_dim):
                # The action determines the next state. This requires
                # action_dim <= state_dim.
                if action >= self.state_dim:
                    raise ValueError(
                        "The oil-discovery transition rule uses the action "
                        "as the next-state index, so action_dim cannot exceed "
                        "state_dim."
                    )

                transition_matrices[action][state, action] = 1.0
                self.reward_matrix[state, action] = self._reward_function(
                    state,
                    action,
                )

        self.transition_matrix = [
            self._normalize_transition_matrix(matrix.tocsr())
            for matrix in transition_matrices
        ]
