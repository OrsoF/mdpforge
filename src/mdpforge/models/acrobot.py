# Source:
# Sutton, R. S. (1996).
# Generalization in Reinforcement Learning: Successful Examples Using Sparse
# Coarse Coding. Advances in Neural Information Processing Systems 8.
#
# Uniformly discretized Acrobot. The continuous equations follow the standard
# two-link underactuated system used by Gymnasium. Each sparse transition row
# is deterministic under Euler integration. A terminal state is entered when
# the end effector reaches the target height.

import numpy as np
from scipy.sparse import dok_array

from mdpforge.core.model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        del action_dim

        requested = max(256, int(state_dim))
        bins_per_dimension = max(2, int(round(requested**0.25)))

        self.angle_bins = bins_per_dimension
        self.velocity_bins = bins_per_dimension
        self.torques = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        self.action_dim = len(self.torques)

        self.dt = 0.2
        self.max_velocity_1 = 4.0 * np.pi
        self.max_velocity_2 = 9.0 * np.pi

        self.theta_1_grid = np.linspace(-np.pi, np.pi, self.angle_bins, endpoint=False)
        self.theta_2_grid = np.linspace(-np.pi, np.pi, self.angle_bins, endpoint=False)
        self.velocity_1_grid = np.linspace(
            -self.max_velocity_1,
            self.max_velocity_1,
            self.velocity_bins,
        )
        self.velocity_2_grid = np.linspace(
            -self.max_velocity_2,
            self.max_velocity_2,
            self.velocity_bins,
        )

        self.nonterminal_state_dim = (
            self.angle_bins * self.angle_bins * self.velocity_bins * self.velocity_bins
        )
        self.terminal_state = self.nonterminal_state_dim
        self.state_dim = self.nonterminal_state_dim + 1
        self.name = f"{self.state_dim}_{self.action_dim}_acrobot"

    def _encode(self, i1: int, i2: int, iv1: int, iv2: int) -> int:
        return (
            (i1 * self.angle_bins + i2) * self.velocity_bins + iv1
        ) * self.velocity_bins + iv2

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return ((angle + np.pi) % (2.0 * np.pi)) - np.pi

    @staticmethod
    def _nearest(grid: np.ndarray, value: float) -> int:
        return int(np.argmin(np.abs(grid - value)))

    @staticmethod
    def _terminal(theta_1: float, theta_2: float) -> bool:
        # Height of the tip relative to the base.
        return -np.cos(theta_1) - np.cos(theta_1 + theta_2) > 1.0

    def _dynamics(
        self,
        theta_1: float,
        theta_2: float,
        dtheta_1: float,
        dtheta_2: float,
        torque: float,
    ) -> tuple[float, float, float, float]:
        # Standard Acrobot parameters.
        m1 = m2 = 1.0
        l1 = lc1 = lc2 = 1.0
        i1 = i2 = 1.0
        g = 9.8

        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * np.cos(theta_2))
            + i1
            + i2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta_2)) + i2

        phi2 = m2 * lc2 * g * np.cos(theta_1 + theta_2 - np.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta_2**2 * np.sin(theta_2)
            - 2.0 * m2 * l1 * lc2 * dtheta_2 * dtheta_1 * np.sin(theta_2)
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta_1 - np.pi / 2.0)
            + phi2
        )

        ddtheta_2 = (
            torque
            + d2 / d1 * phi1
            - m2 * l1 * lc2 * dtheta_1**2 * np.sin(theta_2)
            - phi2
        ) / (m2 * lc2**2 + i2 - d2**2 / d1)
        ddtheta_1 = -(d2 * ddtheta_2 + phi1) / d1

        next_dtheta_1 = float(
            np.clip(
                dtheta_1 + self.dt * ddtheta_1,
                -self.max_velocity_1,
                self.max_velocity_1,
            )
        )
        next_dtheta_2 = float(
            np.clip(
                dtheta_2 + self.dt * ddtheta_2,
                -self.max_velocity_2,
                self.max_velocity_2,
            )
        )
        next_theta_1 = self._wrap_angle(theta_1 + self.dt * next_dtheta_1)
        next_theta_2 = self._wrap_angle(theta_2 + self.dt * next_dtheta_2)

        return next_theta_1, next_theta_2, next_dtheta_1, next_dtheta_2

    def _build_model(self) -> None:
        matrices = [
            dok_array((self.state_dim, self.state_dim), dtype=np.float64)
            for _ in range(self.action_dim)
        ]
        self.reward_matrix = -np.ones(
            (self.state_dim, self.action_dim),
            dtype=np.float64,
        )
        self.reward_matrix[self.terminal_state, :] = 0.0

        for action in range(self.action_dim):
            matrices[action][self.terminal_state, self.terminal_state] = 1.0

        for i1, theta_1 in enumerate(self.theta_1_grid):
            for i2, theta_2 in enumerate(self.theta_2_grid):
                for iv1, dtheta_1 in enumerate(self.velocity_1_grid):
                    for iv2, dtheta_2 in enumerate(self.velocity_2_grid):
                        state = self._encode(i1, i2, iv1, iv2)

                        for action, torque in enumerate(self.torques):
                            next_values = self._dynamics(
                                theta_1,
                                theta_2,
                                dtheta_1,
                                dtheta_2,
                                torque,
                            )

                            if self._terminal(next_values[0], next_values[1]):
                                next_state = self.terminal_state
                                self.reward_matrix[state, action] = 0.0
                            else:
                                next_state = self._encode(
                                    self._nearest(self.theta_1_grid, next_values[0]),
                                    self._nearest(self.theta_2_grid, next_values[1]),
                                    self._nearest(self.velocity_1_grid, next_values[2]),
                                    self._nearest(self.velocity_2_grid, next_values[3]),
                                )

                            matrices[action][state, next_state] = 1.0

        self.transition_matrix = [matrix.tocsr() for matrix in matrices]
