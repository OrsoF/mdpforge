# Source: Carpentier, P., Chancelier, J.-P., Leclere, V. and Pacaud, F. (2018). Stochastic decomposition applied to large-scale hydro valleys management. EJOR 270(3):1086-1098.
"""
Model from "Stochastic decomposition applied to large-scale hydro valleys management", Carpentier, Chancelier, Leclere, Pacaud
"""

from itertools import product
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

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


METADATA = {
    "sizes": {
        "small": {
            "requested_state_dim": 900,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 15000,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 50000,
            "parameters": {},
        },
    },
    "category": "asset_management",
    "description": "Reservoir release decisions with stochastic inflow.",
    "reference": (
        "Carpentier, P., Chancelier, J.-P., Leclere, V. and Pacaud, F. (2018). "
        "Stochastic decomposition applied to large-scale hydro valleys management. "
        "EJOR 270(3):1086-1098."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10, *, seed: int = 0):
        """Use seed (nonnegative integer, default 0) for reproducible rewards."""
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        self.seed = int(seed)
        self.requested_state_dim = int(state_dim)
        self._max_dam_capa, self._n_dam = find_closest_indices(state_dim)
        self._build_action_space()
        self._build_state_space()
        self.name = f"{self.state_dim}_{self.action_dim}_dam_v2_seed{self.seed}"

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
        rng = np.random.default_rng(self.seed)
        base = self._max_dam_capa + 1
        volumes = np.empty((base, base, 2 * base - 1), dtype=np.intp)
        outgoing = np.empty_like(volumes)
        for volume, release, inflow in product(
            range(base), range(base), range(2 * base - 1)
        ):
            next_volume, turbined, spilled = update_first_dam(
                volume,
                inflow,
                release,
                self._max_dam_capa,
                self._max_dam_capa,
                self._max_dam_capa,
            )
            volumes[volume, release, inflow] = next_volume
            outgoing[volume, release, inflow] = min(
                turbined + spilled, self._max_dam_capa
            )

        states = np.asarray(self.state_list, dtype=np.intp)
        rows = np.arange(self.state_dim)
        self.transition_matrix = []
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))
        self.transition_matrix_normalization = np.full(
            (self.state_dim, self.action_dim), float(base)
        )
        for aa, action in enumerate(self.action_list):
            matrix = csr_matrix((self.state_dim, self.state_dim))
            for natural_flow in range(base):
                next_volumes = np.zeros(self.state_dim, dtype=np.intp)
                next_outgoing = np.zeros(self.state_dim, dtype=np.intp)
                for dam_index, release in enumerate(action):
                    inflow = natural_flow
                    if dam_index:
                        inflow = inflow + states[:, self._n_dam + dam_index - 1]
                    current = states[:, dam_index]
                    next_volumes = (
                        base * next_volumes + volumes[current, release, inflow]
                    )
                    next_outgoing = (
                        base * next_outgoing + outgoing[current, release, inflow]
                    )
                targets = next_volumes * base**self._n_dam + next_outgoing
                matrix += csr_matrix(
                    (np.ones(self.state_dim), (rows, targets)), shape=matrix.shape
                )
            matrix.data /= base
            self.transition_matrix.append(matrix)

            # Keep the historical action/state draw order within each instance.
            turbinated_water = sum(self.action_list[aa])
            self.reward_matrix[:, aa] = turbinated_water * rng.random(
                size=self.state_dim
            )
