import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP


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
        This is a compact benchmark implementation, not a full traffic simulator.
        It includes stochastic destination choice when a passenger is picked up,
        but does not inject new exogenous hall calls after initialization.
    """

    MOVE_UP = 0
    MOVE_DOWN = 1
    PICK_UP = 2
    PICK_DOWN = 3
    WAIT = 4

    DIR_DOWN = -1
    DIR_IDLE = 0
    DIR_UP = 1

    def __init__(self, state_dim: int = 2304, action_dim: int = 10):
        # The full 6-floor model in the paper is very large.
        # For benchmark use, choose 3 floors by default.
        #
        # n=3 gives at most: 3 * 3 * 2^(4*(3-1)) = 2304 raw states.
        # n=4 gives at most: 4 * 3 * 2^(12) = 49152 raw states.
        if state_dim >= 20000:
            self.n_floors = 4
        else:
            self.n_floors = 3

        self.action_dim = 5

        self.dt = 1.0
        self.invalid_penalty = 10.0
        self.car_wait_penalty = 2.0
        self.idle_move_penalty = 0.2

        self.states = []
        self.state_to_id = {}

        self._enumerate_states()

        self.state_dim = len(self.states)
        self.name = "{}_{}_elevator".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        for s_id, state in enumerate(self.states):
            for action in range(self.action_dim):
                transitions = self._transitions(state, action)

                expected_reward = 0.0
                for next_state, prob, reward in transitions:
                    next_id = self.state_to_id[next_state]
                    self.transition_matrix[action][s_id, next_id] += prob
                    expected_reward += prob * reward

                self.reward_matrix[s_id, action] = expected_reward

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _enumerate_states(self):
        n = self.n_floors
        mask_size = n - 1
        max_mask = 1 << mask_size

        for floor in range(n):
            for direction in (self.DIR_DOWN, self.DIR_IDLE, self.DIR_UP):
                for up_car in range(max_mask):
                    for down_car in range(max_mask):
                        for up_hall in range(max_mask):
                            for down_hall in range(max_mask):
                                state = (
                                    floor,
                                    direction,
                                    up_car,
                                    down_car,
                                    up_hall,
                                    down_hall,
                                )

                                if self._is_valid_state(state):
                                    self.state_to_id[state] = len(self.states)
                                    self.states.append(state)

    def _is_valid_state(self, state):
        floor, direction, up_car, down_car, up_hall, down_hall = state

        if not (0 <= floor < self.n_floors):
            return False

        if direction not in (self.DIR_DOWN, self.DIR_IDLE, self.DIR_UP):
            return False

        # Idle car should not contain passengers / car calls.
        if direction == self.DIR_IDLE and (up_car != 0 or down_car != 0):
            return False

        # Moving up cannot have down-car passengers.
        if direction == self.DIR_UP and down_car != 0:
            return False

        # Moving down cannot have up-car passengers.
        if direction == self.DIR_DOWN and up_car != 0:
            return False

        # If moving up, only destinations above current floor are admissible.
        if direction == self.DIR_UP:
            for dest in self._up_car_destinations(up_car):
                if dest <= floor:
                    return False

        # If moving down, only destinations below current floor are admissible.
        if direction == self.DIR_DOWN:
            for dest in self._down_car_destinations(down_car):
                if dest >= floor:
                    return False

        return True

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

        if direction == self.DIR_IDLE:
            up_car = 0
            down_car = 0

        normalized = (
            floor,
            direction,
            up_car,
            down_car,
            up_hall,
            down_hall,
        )

        if normalized not in self.state_to_id:
            # Conservative fallback: make the car idle and drop invalid car calls.
            normalized = (
                floor,
                self.DIR_IDLE,
                0,
                0,
                up_hall,
                down_hall,
            )

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
