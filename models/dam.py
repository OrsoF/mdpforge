# Source: Carpentier, P., Chancelier, J.-P., Leclere, V. and Pacaud, F. (2018). Stochastic decomposition applied to large-scale hydro valleys management. EJOR 270(3):1086-1098.
"""
Model from "Stochastic decomposition applied to large-scale hydro valleys management", Carpentier, Chancelier, Leclere, Pacaud
"""

from itertools import product
from typing import Tuple

import numpy as np
from scipy.sparse import lil_matrix

from core.model import GenericModel

seed = 0
rng_gen = np.random.default_rng(seed)


def update_first_dam(
    current_volume: int,
    natural_flow: int,
    turbinating_water: int,
    maximum_dam_capacity: int,
    maximum_turbinating_water: int,
    maximum_spilling_water: int,
) -> Tuple:
    incoming_volume = current_volume + natural_flow
    if incoming_volume <= turbinating_water:
        next_volume = 0
        next_turbinating_water = incoming_volume
        next_spilling_water = 0
    elif incoming_volume - turbinating_water > maximum_dam_capacity:
        next_volume = maximum_dam_capacity
        next_turbinating_water = turbinating_water
        next_spilling_water = min(
            incoming_volume - turbinating_water - maximum_dam_capacity,
            maximum_spilling_water,
        )
    else:
        next_volume = incoming_volume - turbinating_water
        next_turbinating_water = turbinating_water
        next_spilling_water = 0

    return next_volume, next_turbinating_water, next_spilling_water


def update_middle_dam(
    current_volume: int,
    natural_flow: int,
    upcoming_water: int,
    turbinating_water: int,
    maximum_dam_capacity: int,
    maximum_turbinating_water: int,
    maximum_spilling_water: int,
) -> Tuple:
    return update_first_dam(
        current_volume=current_volume,
        natural_flow=natural_flow + upcoming_water,
        turbinating_water=turbinating_water,
        maximum_dam_capacity=maximum_dam_capacity,
        maximum_turbinating_water=maximum_turbinating_water,
        maximum_spilling_water=maximum_spilling_water,
    )


def find_closest_indices(
    state_dim_aimed: int, max_capa: int = 10, max_dam: int = 10
) -> Tuple:
    state_dim_calculus = lambda capa, n_dam: (capa + 1) ** (2 * n_dam)
    best_tuple = (1, 1)
    best_distance = abs(state_dim_aimed - state_dim_calculus(*best_tuple))
    for capa in range(1, max_capa):
        for n_dam in range(1, max_dam):
            distance = abs(state_dim_calculus(capa, n_dam) - state_dim_aimed)
            if distance < best_distance:
                best_tuple = (capa, n_dam)
                best_distance = distance
    return best_tuple


class Model(GenericModel):
    def __init__(self, state_dim=0, action_dim=0):
        self._max_dam_capa, self._n_dam = find_closest_indices(state_dim)
        self._build_action_space()
        self._build_state_space()
        self.name = "{}_{}_dam".format(self.state_dim, self.action_dim)

    def _build_action_space(self):
        # Action décrite par la turbinated water sur chaque dam
        self.action_list = list(
            product(range(self._max_dam_capa + 1), repeat=self._n_dam)
        )
        self.action_dim = (self._max_dam_capa + 1) ** self._n_dam  # max_dam_capa**n_dam

    def _build_state_space(self):
        """
        state : (current_water_volume, evacuated_water)
        state_dim = capa**(2*n_dam)
        """
        self.state_list = list(
            product(range(self._max_dam_capa + 1), repeat=self._n_dam * 2)
        )
        self.state_dict = {
            state: state_index for (state_index, state) in enumerate(self.state_list)
        }
        self.state_dim = len(self.state_list)

    def _build_model(self):
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        self.transition_matrix_normalization = np.zeros(
            (self.state_dim, self.action_dim)
        )

        for aa, action in enumerate(self.action_list):
            turbinating_water = action

            for ss in range(self.state_dim):
                state = self.state_list[ss]
                dam_current_volumes = state[: self._n_dam]
                upcoming_water_all_dam = state[self._n_dam : 2 * self._n_dam]
                total_transition = 0

                for natural_flow in range(0, self._max_dam_capa + 1):
                    next_volumes, next_upcoming_water = [], []

                    for dam_index in range(self._n_dam):
                        if dam_index == 0:
                            current_volume = dam_current_volumes[dam_index]
                            current_turbinating_water = turbinating_water[dam_index]
                            next_volume, next_turbinating_water, next_spilling_water = (
                                update_first_dam(
                                    current_volume,
                                    natural_flow,
                                    current_turbinating_water,
                                    self._max_dam_capa,
                                    self._max_dam_capa,
                                    self._max_dam_capa,
                                )
                            )
                        else:
                            current_volume = dam_current_volumes[dam_index]
                            current_turbinating_water = turbinating_water[dam_index]
                            upcoming_water = upcoming_water_all_dam[dam_index - 1]
                            next_volume, next_turbinating_water, next_spilling_water = (
                                update_middle_dam(
                                    current_volume,
                                    natural_flow,
                                    upcoming_water,
                                    current_turbinating_water,
                                    self._max_dam_capa,
                                    self._max_dam_capa,
                                    self._max_dam_capa,
                                )
                            )

                        next_volumes.append(next_volume)  # Nouveau
                        next_upcoming_water.append(
                            min(
                                (next_turbinating_water + next_spilling_water),
                                self._max_dam_capa,
                            )
                        )

                    next_state = tuple(int(elem) for elem in next_volumes) + tuple(
                        int(elem) for elem in next_upcoming_water
                    )

                    next_state_index = self.state_dict[next_state]
                    self.transition_matrix[aa][ss, next_state_index] += 1.0
                    total_transition += 1.0

                self.transition_matrix_normalization[ss, aa] = total_transition

            # Transition normalization
            row_indices, col_indices = self.transition_matrix[aa].nonzero()
            for ss1, ss2 in zip(row_indices, col_indices):
                self.transition_matrix[aa][ss1, ss2] = (
                    self.transition_matrix[aa][ss1, ss2]
                    / self.transition_matrix_normalization[ss1, aa]
                )

            # Reward definition
            turbinated_water = sum(self.action_list[aa])
            self.reward_matrix[:, aa] = turbinated_water * rng_gen.random(
                size=self.state_dim
            )

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
