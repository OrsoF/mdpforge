# Source:
# Sinclair, Banerjee & Yu (2019),
# Adaptive Discretization for Episodic Reinforcement Learning.
#
# Oil discovery is continuous on [0, 1].
# This implementation provides a uniform tabular discretization.

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.model import GenericModel


class Model(GenericModel):
    """Discretized oil-discovery benchmark.

    States and actions are uniformly discretized on [0, 1].

    The action represents a drilling or exploration location. The reward is
    maximal when the selected location is close to the latent reservoir and
    includes a fixed drilling cost and a distance-dependent movement cost.

    The latent reservoir position evolves according to a Gaussian transition
    kernel whose mean drifts toward the selected action.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = max(50, int(state_dim))
        self.action_dim = max(10, int(action_dim))

        self.sigma = 0.08
        self.drift = 0.15
        self.transition_std = 0.02
        self.kernel_tolerance = 1e-6

        self.name = f"{self.state_dim}_{self.action_dim}_oil_discovery"

    def _build_model(self) -> None:
        self.state_grid = np.linspace(
            0.0,
            1.0,
            self.state_dim,
            dtype=np.float64,
        )
        self.action_grid = np.linspace(
            0.0,
            1.0,
            self.action_dim,
            dtype=np.float64,
        )

        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim),
            dtype=np.float64,
        )

        transition_matrices = [
            lil_matrix(
                (self.state_dim, self.state_dim),
                dtype=np.float64,
            )
            for _ in range(self.action_dim)
        ]

        for state, reservoir_location in enumerate(self.state_grid):
            for action, drilling_location in enumerate(self.action_grid):
                self.reward_matrix[state, action] = self._reward_function(
                    reservoir_location=reservoir_location,
                    drilling_location=drilling_location,
                )

                mean_next = np.clip(
                    (1.0 - self.drift) * reservoir_location
                    + self.drift * drilling_location,
                    0.0,
                    1.0,
                )

                self._add_local_kernel(
                    matrix=transition_matrices[action],
                    state=state,
                    mean=mean_next,
                    std=self.transition_std,
                )

        self.transition_matrix = [matrix.tocsr() for matrix in transition_matrices]

        self._validate_transition_matrices()

    def _reward_function(
        self,
        reservoir_location: float,
        drilling_location: float,
    ) -> float:
        distance = drilling_location - reservoir_location

        payoff = np.exp(-0.5 * (distance / self.sigma) ** 2)
        cost = 0.05 + 0.10 * abs(distance)

        return float(10.0 * payoff - cost)

    def _add_local_kernel(
        self,
        matrix: lil_matrix,
        state: int,
        mean: float,
        std: float,
    ) -> None:
        if std <= 0.0:
            raise ValueError(f"std must be positive, received {std}.")

        distances = (self.state_grid - mean) / std
        weights = np.exp(-0.5 * distances**2)

        retained_indices = np.flatnonzero(weights > self.kernel_tolerance)

        if retained_indices.size == 0:
            nearest_state = int(np.argmin(np.abs(self.state_grid - mean)))
            matrix[state, nearest_state] = 1.0
            return

        retained_weights = weights[retained_indices]
        retained_sum = retained_weights.sum(dtype=np.float64)

        if not np.isfinite(retained_sum) or retained_sum <= 0.0:
            nearest_state = int(np.argmin(np.abs(self.state_grid - mean)))
            matrix[state, nearest_state] = 1.0
            return

        # Normalize after removing small entries. Normalizing before truncation
        # would make the resulting sparse row sum to slightly less than one.
        retained_weights = retained_weights / retained_sum

        # Absorb any final floating-point residual into the largest entry so
        # that the stored sparse row sums to one up to machine precision.
        residual = 1.0 - retained_weights.sum(dtype=np.float64)
        largest_index = int(np.argmax(retained_weights))
        retained_weights[largest_index] += residual

        matrix[state, retained_indices] = retained_weights

    def _validate_transition_matrices(self) -> None:
        for action, matrix in enumerate(self.transition_matrix):
            row_sums = np.asarray(
                matrix.sum(axis=1),
                dtype=np.float64,
            ).ravel()

            valid_rows = np.isclose(
                row_sums,
                1.0,
                rtol=0.0,
                atol=1e-12,
            )

            if not np.all(valid_rows):
                invalid_rows = np.flatnonzero(~valid_rows)
                first_invalid = int(invalid_rows[0])

                raise ValueError(
                    f"Transition matrix for action {action} is not stochastic: "
                    f"row {first_invalid} sums to "
                    f"{row_sums[first_invalid]:.17g}."
                )
