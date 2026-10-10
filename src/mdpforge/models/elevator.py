from array import array
from itertools import combinations
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

TEST_PARAMETERS = {"state_dim": 176, "action_dim": 5}

METADATA = {
    "tags": ["control", "real-world"],
    "sizes": {
        "small": {
            "state_dim": 176,
            "parameters": {"state_dim": 176},
            "source": "analytical",
        },
        "medium": {
            "state_dim": 1664,
            "parameters": {"state_dim": 1664},
            "source": "analytical",
        },
        "large": {
            "state_dim": 14592,
            "parameters": {"state_dim": 14592},
            "source": "analytical",
        },
    },
    "category": "control",
    "description": (
        "Elevator movement and pickup decisions with passenger destination masks."
    ),
    "reference": (
        "Tartan, E. O. and Ciflikli, C. (2023). Sequential Decision Making for "
        "Elevator Control."
    ),
}



class Model(MDP):
    """
    Elevator control finite MDP.

    Source:
        Tartan, E. O. and Ciflikli, C. (2023).
        Sequential Decision Making for Elevator Control.

    State encoding:
        state = (
            floor,
            direction,
            up_car_mask,
            down_car_mask,
            up_hall_mask,
            down_hall_mask,
        )

        floor:
            0, ..., n_floors - 1

        direction:
            -1 = down
             0 = idle
             1 = up

        up_car_mask:
            destinations above entrance floor.
            bit k corresponds to floor k + 1.

        down_car_mask:
            destinations below top floor.
            bit k corresponds to floor k.

        up_hall_mask:
            up hall calls.
            bit k corresponds to floor k, for k = 0, ..., n_floors - 2.

        down_hall_mask:
            down hall calls.
            bit k corresponds to floor k + 1.

    Actions:
        0 = move up
        1 = move down
        2 = pick up up-passenger
        3 = pick up down-passenger
        4 = wait

    Notes:
        This is a benchmark, not a full traffic simulator: calls are not created
        after initialization. state_dim is the EXACT number of states, including
        for sizes that do not correspond to a complete floor configuration.

        Floors are chosen to accommodate the requested size. States are selected
        by increasing number of pending requests, so the smallest models capture
        simpler traffic first. For a partial state space, an outcome that leaves
        the selected states is converted to a penalized self-loop, preserving
        stochastic row sums. Thus non-complete sizes define finite, truncated
        MDPs, NOT exact versions of the full-floor model.

        Complete state spaces (with no truncated transitions) have sizes
        176, 1664, 14592, 122880, ... for 3, 4, 5, 6 floors, respectively.
        Storage of the transition matrices is sparse, O(state_dim * n_floors),
        rather than quadratic in state_dim.
    """

    MOVE_UP = 0
    MOVE_DOWN = 1
    PICK_UP = 2
    PICK_DOWN = 3
    WAIT = 4

    DIR_DOWN = -1
    DIR_IDLE = 0
    DIR_UP = 1

    @staticmethod
    def _full_state_count(n_floors: int) -> int:
        """Valid states, excluding meaningless moving directions with no riders."""
        if n_floors < 1:
            raise ValueError("n_floors must be positive")
        # Each of 2(n-1) hall bits is independent. For each floor: one idle
        # state, and one state per NONEMPTY subset of upward/downward car calls.
        return (1 << (2 * (n_floors - 1))) * (
            (1 << (n_floors + 1)) - n_floors - 2
        )

    def __init__(
        self,
        state_dim: int = 2304,
        action_dim: int = 5,
        n_floors: Optional[int] = None,
    ):
        if isinstance(state_dim, (bool, np.bool_)) or not isinstance(
            state_dim, (int, np.integer)
        ) or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if action_dim != 5:
            raise ValueError("The elevator has exactly 5 actions (action_dim=5)")
        if n_floors is None:
            # At least two floors unless a one-state model is requested.
            n_floors = 1 if state_dim == 1 else 2
            while self._full_state_count(n_floors) < state_dim:
                n_floors += 1
        elif (
            isinstance(n_floors, (bool, np.bool_))
            or not isinstance(n_floors, (int, np.integer))
            or n_floors < 1
        ):
            raise ValueError("n_floors must be a positive integer")

        self.n_floors = int(n_floors)
        if state_dim > self._full_state_count(self.n_floors):
            raise ValueError(
                f"{self.n_floors} floors have only "
                f"{self._full_state_count(self.n_floors)} valid states"
            )

        self.state_dim = int(state_dim)
        self.action_dim = 5
        self.dt = 1.0
        self.invalid_penalty = 10.0
        self.car_wait_penalty = 2.0
        self.idle_move_penalty = 0.2

        self.states = []
        self.state_to_id = {}
        self._enumerate_states()
        assert len(self.states) == self.state_dim
        self.name = f"{self.state_dim}_{self.action_dim}_elevator"

    def _build_model(self):
        """Build one CSR matrix at a time, never a dense S x S matrix.

        Missing next states can occur ONLY when the requested state_dim is
        smaller than the complete space of the chosen number of floors.
        Their probability is redirected to the current state with a penalty.
        """
        n = self.state_dim
        self.reward_matrix = np.empty((n, self.action_dim), dtype=np.float64)
        self.transition_matrix = []

        for action in range(self.action_dim):
            indptr = np.empty(n + 1, dtype=np.int64)
            indptr[0] = 0
            indices = array("q")  # 64-bit index, even for large state spaces.
            probabilities = array("d")

            for sid, state in enumerate(self.states):
                row = {}
                expected_reward = 0.0
                for next_state, prob, reward in self._transitions(state, action):
                    next_id = self.state_to_id.get(next_state)
                    if next_id is None:
                        next_id = sid
                        reward -= self.invalid_penalty
                    row[next_id] = row.get(next_id, 0.0) + prob
                    expected_reward += prob * reward

                # Aggregate duplicate destinations before constructing CSR.
                for next_id, prob in sorted(row.items()):
                    indices.append(next_id)
                    probabilities.append(prob)
                indptr[sid + 1] = len(indices)
                self.reward_matrix[sid, action] = expected_reward

            self.transition_matrix.append(
                csr_matrix(
                    (
                        np.asarray(probabilities),
                        np.asarray(indices),
                        indptr,
                    ),
                    shape=(n, n),
                )
            )

    @staticmethod
    def _mask_combinations(positions, count):
        """Enumerate masks with exactly `count` set bits at given positions."""
        for selected in combinations(positions, count):
            mask = 0
            for bit in selected:
                mask |= 1 << bit
            yield mask

    def _enumerate_states(self):
        """Generate only requested states, never a Cartesian mask product.

        All states have consistent direction and onboard passenger requests.
        Ordering by pending-call count provides useful low-load states even
        when state_dim is much smaller than a full n-floor configuration.
        """
        n = self.n_floors
        hall_bits = n - 1
        hall_positions = range(2 * hall_bits)
        hall_split_mask = (1 << hall_bits) - 1

        def add(state):
            self.state_to_id[state] = len(self.states)
            self.states.append(state)
            return len(self.states) == self.state_dim

        # At most 2(n-1) hall requests plus n-1 car destinations.
        for n_calls in range(3 * hall_bits + 1):
            for floor in range(n):
                # Idle: no onboard passengers. Prioritize all floors first.
                if n_calls <= 2 * hall_bits:
                    for hall_mask in self._mask_combinations(
                        hall_positions, n_calls
                    ):
                        if add((floor, self.DIR_IDLE, 0, 0,
                                hall_mask & hall_split_mask,
                                hall_mask >> hall_bits)):
                            return

                # Moving: at least one onboard destination in the indicated
                # direction, and every destination must be ahead of the car.
                for direction, car_positions in (
                    (self.DIR_UP, range(floor, hall_bits)),
                    (self.DIR_DOWN, range(floor)),
                ):
                    for n_car in range(1, min(n_calls, len(car_positions)) + 1):
                        n_hall = n_calls - n_car
                        if n_hall > 2 * hall_bits:
                            continue
                        for car_mask in self._mask_combinations(
                            car_positions, n_car
                        ):
                            for hall_mask in self._mask_combinations(
                                hall_positions, n_hall
                            ):
                                if add((
                                    floor,
                                    direction,
                                    car_mask if direction == self.DIR_UP else 0,
                                    car_mask if direction == self.DIR_DOWN else 0,
                                    hall_mask & hall_split_mask,
                                    hall_mask >> hall_bits,
                                )):
                                    return

        raise RuntimeError("State enumeration ended before state_dim was reached")

    def _is_valid_state(self, state):
        floor, direction, up_car, down_car, up_hall, down_hall = state
        n = self.n_floors
        mask_limit = 1 << (n - 1)
        if not (0 <= floor < n):
            return False
        if any(mask < 0 or mask >= mask_limit for mask in (
            up_car, down_car, up_hall, down_hall
        )):
            return False
        if direction == self.DIR_IDLE:
            return up_car == down_car == 0
        if direction == self.DIR_UP:
            # No downward riders, no upward destinations at/below floor.
            return down_car == 0 and up_car != 0 and up_car % (1 << floor) == 0
        if direction == self.DIR_DOWN:
            # No upward riders, no downward destinations at/above floor.
            return up_car == 0 and down_car != 0 and down_car < (1 << floor)
        return False

    def _transitions(self, state, action):
        if action == self.MOVE_UP:
            return self._move_up(state)

        if action == self.MOVE_DOWN:
            return self._move_down(state)

        if action == self.PICK_UP:
            return self._pick_up_passenger(state)

        if action == self.PICK_DOWN:
            return self._pick_down_passenger(state)

        if action == self.WAIT:
            return self._wait(state)

        raise ValueError("Invalid action.")

    def _move_up(self, state):
        floor, direction, up_car, down_car, up_hall, down_hall = state

        if floor == self.n_floors - 1:
            return self._invalid_action(state)

        # Do not allow movement opposite to passengers already inside.
        if direction == self.DIR_DOWN:
            return self._invalid_action(state)

        next_floor = floor + 1
        next_direction = self.DIR_UP

        # Serve up car call at the arrived floor.
        up_car = self._clear_up_car_call(up_car, next_floor)

        # If no more passengers in the car, become idle.
        if up_car == 0:
            next_direction = self.DIR_IDLE

        next_state = self._normalize_state(
            (
                next_floor,
                next_direction,
                up_car,
                0,
                up_hall,
                down_hall,
            )
        )

        reward = self._reward(state, self.MOVE_UP)
        return [(next_state, 1.0, reward)]

    def _move_down(self, state):
        floor, direction, up_car, down_car, up_hall, down_hall = state

        if floor == 0:
            return self._invalid_action(state)

        # Do not allow movement opposite to passengers already inside.
        if direction == self.DIR_UP:
            return self._invalid_action(state)

        next_floor = floor - 1
        next_direction = self.DIR_DOWN

        # Serve down car call at the arrived floor.
        down_car = self._clear_down_car_call(down_car, next_floor)

        # If no more passengers in the car, become idle.
        if down_car == 0:
            next_direction = self.DIR_IDLE

        next_state = self._normalize_state(
            (
                next_floor,
                next_direction,
                0,
                down_car,
                up_hall,
                down_hall,
            )
        )

        reward = self._reward(state, self.MOVE_DOWN)
        return [(next_state, 1.0, reward)]

    def _pick_up_passenger(self, state):
        floor, direction, up_car, down_car, up_hall, down_hall = state

        if floor == self.n_floors - 1:
            return self._invalid_action(state)

        if not self._has_up_hall_call(up_hall, floor):
            return self._invalid_action(state)

        if direction == self.DIR_DOWN:
            return self._invalid_action(state)

        up_hall = self._clear_up_hall_call(up_hall, floor)

        possible_destinations = list(range(floor + 1, self.n_floors))
        prob = 1.0 / len(possible_destinations)

        transitions = []
        for dest in possible_destinations:
            new_up_car = self._set_up_car_call(up_car, dest)

            next_state = self._normalize_state(
                (
                    floor,
                    self.DIR_UP,
                    new_up_car,
                    0,
                    up_hall,
                    down_hall,
                )
            )

            reward = self._reward(state, self.PICK_UP)
            transitions.append((next_state, prob, reward))

        return transitions

    def _pick_down_passenger(self, state):
        floor, direction, up_car, down_car, up_hall, down_hall = state

        if floor == 0:
            return self._invalid_action(state)

        if not self._has_down_hall_call(down_hall, floor):
            return self._invalid_action(state)

        if direction == self.DIR_UP:
            return self._invalid_action(state)

        down_hall = self._clear_down_hall_call(down_hall, floor)

        possible_destinations = list(range(0, floor))
        prob = 1.0 / len(possible_destinations)

        transitions = []
        for dest in possible_destinations:
            new_down_car = self._set_down_car_call(down_car, dest)

            next_state = self._normalize_state(
                (
                    floor,
                    self.DIR_DOWN,
                    0,
                    new_down_car,
                    up_hall,
                    down_hall,
                )
            )

            reward = self._reward(state, self.PICK_DOWN)
            transitions.append((next_state, prob, reward))

        return transitions

    def _wait(self, state):
        reward = self._reward(state, self.WAIT)
        return [(state, 1.0, reward)]

    def _invalid_action(self, state):
        reward = self._reward(state, None) - self.invalid_penalty
        return [(state, 1.0, reward)]

    def _reward(self, state, action):
        floor, direction, up_car, down_car, up_hall, down_hall = state

        n_hall_calls = self._bit_count(up_hall) + self._bit_count(down_hall)
        n_car_calls = self._bit_count(up_car) + self._bit_count(down_car)

        reward = -n_hall_calls * self.dt

        # Waiting while passengers are inside is bad.
        if action == self.WAIT and n_car_calls > 0:
            reward -= self.car_wait_penalty * n_car_calls * self.dt

        # Moving with no pending call is mildly discouraged.
        if action in (self.MOVE_UP, self.MOVE_DOWN):
            if n_hall_calls == 0 and n_car_calls == 0:
                reward -= self.idle_move_penalty

        return reward

    def _normalize_state(self, state):
        floor, direction, up_car, down_car, up_hall, down_hall = state
        if up_car == 0 and down_car == 0:
            direction = self.DIR_IDLE
        normalized = (
            floor, direction, up_car, down_car, up_hall, down_hall
        )
        if not self._is_valid_state(normalized):
            raise ValueError(f"Invalid transition target: {normalized}")
        return normalized

    def _up_car_destinations(self, up_car):
        # bit k means destination floor k + 1
        return [bit + 1 for bit in range(self.n_floors - 1) if up_car & (1 << bit)]

    def _down_car_destinations(self, down_car):
        # bit k means destination floor k
        return [bit for bit in range(self.n_floors - 1) if down_car & (1 << bit)]

    def _set_up_car_call(self, up_car, floor):
        assert 1 <= floor <= self.n_floors - 1
        return up_car | (1 << (floor - 1))

    def _clear_up_car_call(self, up_car, floor):
        if floor == 0:
            return up_car
        return up_car & ~(1 << (floor - 1))

    def _set_down_car_call(self, down_car, floor):
        assert 0 <= floor <= self.n_floors - 2
        return down_car | (1 << floor)

    def _clear_down_car_call(self, down_car, floor):
        if floor == self.n_floors - 1:
            return down_car
        return down_car & ~(1 << floor)

    def _has_up_hall_call(self, up_hall, floor):
        if floor >= self.n_floors - 1:
            return False
        return bool(up_hall & (1 << floor))

    def _clear_up_hall_call(self, up_hall, floor):
        if floor >= self.n_floors - 1:
            return up_hall
        return up_hall & ~(1 << floor)

    def _has_down_hall_call(self, down_hall, floor):
        if floor == 0:
            return False
        return bool(down_hall & (1 << (floor - 1)))

    def _clear_down_hall_call(self, down_hall, floor):
        if floor == 0:
            return down_hall
        return down_hall & ~(1 << (floor - 1))

    def _bit_count(self, x):
        return int(x).bit_count()
