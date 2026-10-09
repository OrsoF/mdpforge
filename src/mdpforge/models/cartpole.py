# Source: inspired by the classical CartPole model
# Barto, Sutton and Anderson (1983), and by discretized control benchmarks.

import itertools
from typing import Tuple

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP


def index_closest_value(value: float, list_of_values) -> Tuple[int, float]:
    distances = [abs(elem - value) for elem in list_of_values]
    index = int(np.argmin(distances))
    return index, list_of_values[index]


class Model(MDP):
    def __init__(self, state_dim: int = 256, action_dim: int = 10) -> None:
        """
        Discretized CartPole finite MDP.

        State variables:
        - cart position x
        - cart velocity x_dot
        - pole angle theta
        - pole angular velocity theta_dot

        Actions:
        - 0: push left
        - 1: push right
        """

        state_dim = max(state_dim, 100)

        # 4D discretization.
        n = int(np.floor(state_dim**0.25))
        n = max(n, 4)

        self.x_states_number = n
        self.x_dot_states_number = n
        self.theta_states_number = n
        self.theta_dot_states_number = n

        self.state_dim = (
            self.x_states_number
            * self.x_dot_states_number
            * self.theta_states_number
            * self.theta_dot_states_number
        )

        self.action_dim = 2

        self.name = "{}_{}_cartpole_{}_{}_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.x_states_number,
            self.x_dot_states_number,
            self.theta_states_number,
            self.theta_dot_states_number,
        )

    def _build_model(self):
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        # Standard CartPole termination thresholds.
        self.x_threshold = 2.4
        self.theta_threshold = 12.0 * 2.0 * np.pi / 360.0

        # Discretization ranges.
        self.x_state_list = np.linspace(
            -self.x_threshold,
            self.x_threshold,
            self.x_states_number,
        )
        self.x_dot_state_list = np.linspace(
            -3.0,
            3.0,
            self.x_dot_states_number,
        )
        self.theta_state_list = np.linspace(
            -self.theta_threshold,
            self.theta_threshold,
            self.theta_states_number,
        )
        self.theta_dot_state_list = np.linspace(
            -3.5,
            3.5,
            self.theta_dot_states_number,
        )

        self.state_space = {
            state: ss
            for ss, state in enumerate(
                itertools.product(
                    self.x_state_list,
                    self.x_dot_state_list,
                    self.theta_state_list,
                    self.theta_dot_state_list,
                )
            )
        }

        self.state_list = list(self.state_space.keys())

        _, x0 = index_closest_value(0.0, self.x_state_list)
        _, x_dot0 = index_closest_value(0.0, self.x_dot_state_list)
        _, theta0 = index_closest_value(0.0, self.theta_state_list)
        _, theta_dot0 = index_closest_value(0.0, self.theta_dot_state_list)

        self._start_states = [
            (x, x_dot, theta, theta_dot)
            for (x, x_dot, theta, theta_dot) in self.state_list
            if (
                abs(x - x0) <= 1e-12
                and abs(x_dot - x_dot0) <= 1e-12
                and abs(theta - theta0) <= 1e-12
                and abs(theta_dot - theta_dot0) <= 1e-12
            )
        ]

        assert len(self._start_states), (
            "There is no start state possible in those conditions. {}".format(
                self.state_dim
            )
        )

        for aa in range(self.action_dim):
            for ss1 in range(self.state_dim):
                x, x_dot, theta, theta_dot = self.state_list[ss1]

                if self._is_terminal(x, theta):
                    self.reward_matrix[ss1, aa] = 0.0

                    for start_state in self._start_states:
                        ss2 = self.state_space[start_state]
                        self.transition_matrix[aa][ss1, ss2] = 1.0 / len(
                            self._start_states
                        )

                else:
                    next_state = self._next_state(x, x_dot, theta, theta_dot, aa)
                    next_discrete_state = self._closest_state(next_state)
                    ss2 = self.state_space[next_discrete_state]

                    self.reward_matrix[ss1, aa] = -1.0
                    self.transition_matrix[aa][ss1, ss2] = 1.0

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]

    def _is_terminal(self, x: float, theta: float) -> bool:
        return (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )

    def _next_state(
        self,
        x: float,
        x_dot: float,
        theta: float,
        theta_dot: float,
        action: int,
    ):
        assert action in [0, 1]

        gravity = 9.8
        mass_cart = 1.0
        mass_pole = 0.1
        total_mass = mass_cart + mass_pole
        length = 0.5
        polemass_length = mass_pole * length
        force_mag = 10.0
        tau = 0.02

        force = -force_mag if action == 0 else force_mag

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass

        theta_acc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - mass_pole * costheta**2 / total_mass)
        )

        x_acc = temp - polemass_length * theta_acc * costheta / total_mass

        next_x = x + tau * x_dot
        next_x_dot = x_dot + tau * x_acc
        next_theta = theta + tau * theta_dot
        next_theta_dot = theta_dot + tau * theta_acc

        return next_x, next_x_dot, next_theta, next_theta_dot

    def _closest_state(self, state):
        x, x_dot, theta, theta_dot = state

        _, x_discrete = index_closest_value(x, self.x_state_list)
        _, x_dot_discrete = index_closest_value(x_dot, self.x_dot_state_list)
        _, theta_discrete = index_closest_value(theta, self.theta_state_list)
        _, theta_dot_discrete = index_closest_value(
            theta_dot,
            self.theta_dot_state_list,
        )

        return (
            x_discrete,
            x_dot_discrete,
            theta_discrete,
            theta_dot_discrete,
        )
