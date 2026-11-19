"""Finite-state discretization of a single-arm pendulum stabilization problem."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

TEST_PARAMETERS = {"state_dim": 36}
METADATA = {
    "category": "navigation",
    "description": "Discretized pendulum with torque control, damping, and stochastic actuation.",
    "reference": "Wingate and Seppi (2005), Prioritization methods for accelerating MDP solvers, JMLR 6; independent discretization.",
    "sizes": {
        "tiny": {"requested_state_dim": 36, "parameters": {}},
        "small": {"requested_state_dim": 144, "parameters": {}},
        "medium": {"requested_state_dim": 576, "parameters": {}},
        "large": {"requested_state_dim": 2304, "parameters": {}},
    },
}


class Model(MDP):
    def __init__(
        self,
        state_dim: int = 144,
        action_dim: int = 3,
        *,
        time_step: float = 0.12,
        torque: float = 2.0,
        damping: float = 0.15,
        actuation_noise: float = 0.15,
    ):
        if (
            isinstance(state_dim, bool)
            or not isinstance(state_dim, int)
            or state_dim < 16
        ):
            raise ValueError("state_dim must be integer >=16")
        if action_dim != 3:
            raise ValueError("pendulum has three torque actions")
        if (
            any(
                not np.isfinite(x)
                for x in (time_step, torque, damping, actuation_noise)
            )
            or time_step <= 0
            or torque <= 0
            or damping < 0
            or not 0 <= actuation_noise <= 1
        ):
            raise ValueError("invalid pendulum physics parameters")
        self.requested_state_dim = state_dim
        self.grid_side = max(4, int(np.ceil(np.sqrt(state_dim))))
        self.state_dim = self.grid_side**2
        self.action_dim = 3
        self.time_step = float(time_step)
        self.torque = float(torque)
        self.damping = float(damping)
        self.actuation_noise = float(actuation_noise)
        self.name = f"{self.state_dim}_3_singlependulum_{time_step}_{torque}_{damping}_{actuation_noise}"

    def _build_model(self):
        m, n = self.grid_side, self.state_dim
        angles = -np.pi + (np.arange(m) + 0.5) * 2 * np.pi / m
        velocities = np.linspace(-5, 5, m)
        rewards = np.empty((n, 3))
        rows = [[] for _ in range(3)]
        cols = [[] for _ in range(3)]
        vals = [[] for _ in range(3)]
        for index in range(n):
            ai, wi = divmod(index, m)
            theta, omega = angles[ai], velocities[wi]
            for action, torque in enumerate((-self.torque, 0, self.torque)):
                rewards[index, action] = -(theta**2 + 0.1 * omega**2 + 0.01 * torque**2)
                next_probs = {}
                for disturbance, probability in (
                    (0.0, 1 - self.actuation_noise),
                    (-self.torque, self.actuation_noise / 2),
                    (self.torque, self.actuation_noise / 2),
                ):
                    if probability == 0:
                        continue
                    new_omega = np.clip(
                        omega
                        + self.time_step
                        * (
                            9.81 * np.sin(theta)
                            - self.damping * omega
                            + torque
                            + disturbance
                        ),
                        -5,
                        5,
                    )
                    new_theta = (
                        (theta + self.time_step * new_omega + np.pi) % (2 * np.pi)
                    ) - np.pi
                    j = int(np.floor((new_theta + np.pi) / (2 * np.pi) * m)) % m
                    k = int(np.argmin(np.abs(velocities - new_omega)))
                    new_index = j * m + k
                    next_probs[new_index] = next_probs.get(new_index, 0.0) + probability
                for target, probability in next_probs.items():
                    rows[action].append(index)
                    cols[action].append(target)
                    vals[action].append(probability)
        self.reward_matrix = rewards
        self.transition_matrix = [
            csr_matrix((vals[a], (rows[a], cols[a])), shape=(n, n)) for a in range(3)
        ]
