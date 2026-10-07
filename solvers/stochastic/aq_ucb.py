"""
AQ-UCB: Aggregated Q-learning with Upper Confidence Bounds.

Reference target: Dong, Van Roy & Zhou (2020), Algorithm 1.
This is a finite-horizon, fixed-aggregation implementation compatible with the
project's finite GenericModel interface.

Default aggregation: contiguous state regions, but actions are NOT merged. A
cell is therefore (region(s), action), which matches the paper's state-action
aggregation convention while remaining compatible with the existing Partition
utility. You can pass a custom state partition by changing `initial_regions` or
replacing `_build_partition`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from solvers_agg.solvers.common import greedy_episode_reward, require_params

import utils.simulation as simulation
from core.model import GenericModel
from core.partition import Partition
from core.solver import GenericSolver
from utils.exact_value_function import distance_to_optimal
from utils.simulation import generate_sars

REQUIRED_RUN_PARAMS = (
    "initial_regions",
    "horizon",
    "use_discount",
    "delta",
    "epsilon_aggregation_error",
    "optimism_scale",
    "initial_state",
    "random_initial_state",
    "tie_break_random",
    "num_episodes",
    "episode_length",
    "log_every",
    "reward_ma_window",
    "td_ma_window",
    "random_seed",
    "verbose",
)


class Solver(GenericSolver):
    def __init__(self, model: GenericModel, discount: float, final_precision: float):
        self.model = model
        self.model._normalize_reward_matrix()
        self.discount = discount
        self.final_precision = final_precision
        self.name = "AQ-UCB Aggregated Q-Learning"

    def _contiguous_regions(
        self, states: np.ndarray, n_regions: int
    ) -> list[list[int]]:
        n_regions = int(np.clip(n_regions, 1, max(1, len(states))))
        return [
            tile.tolist() for tile in np.array_split(states, n_regions) if len(tile)
        ]

    def _build_partition(self, run_params: dict) -> Partition:
        partition = Partition(self.model)
        initial_regions = int(run_params["initial_regions"])
        partition.states_in_region = self._contiguous_regions(
            np.arange(self.model.state_dim),
            initial_regions,
        )
        return partition

    def _state_to_region(self, partition: Partition) -> np.ndarray:
        return np.asarray(partition.state_to_region, dtype=int)

    def _cell(self, state: int, action: int) -> int:
        region = int(self.state_to_region[state])
        return region * self.model.action_dim + int(action)

    def _state_action_values_at_stage(self, h: int, state: int) -> np.ndarray:
        return np.asarray(
            [
                self.q_hat[h, self._cell(state, action)]
                for action in range(self.model.action_dim)
            ],
            dtype=float,
        )

    def _choose_greedy_action(self, h: int, state: int, run_params: dict) -> int:
        values = self._state_action_values_at_stage(h, state)
        best = np.flatnonzero(values == values.max())
        if run_params["tie_break_random"]:
            return int(simulation.rng.choice(best))
        return int(best[0])

    def _initial_state(self, run_params: dict) -> int:
        if run_params["initial_state"] is not None:
            return int(run_params["initial_state"])
        if run_params["random_initial_state"]:
            return int(simulation.rng.integers(self.model.state_dim))
        return 0

    def _rollout(self, run_params: dict) -> list[tuple[int, int, float, int]]:
        state = self._initial_state(run_params)
        trajectory = []
        for h in range(self.horizon):
            action = self._choose_greedy_action(h, state, run_params)
            next_state, reward = generate_sars(self.model, state, action)
            trajectory.append((state, action, float(reward), int(next_state)))
            state = int(next_state)
        return trajectory

    def _beta(self, n: int, run_params: dict) -> float:
        delta = min(max(float(run_params["delta"]), 1e-12), 1.0)
        epsilon = float(run_params["epsilon_aggregation_error"])
        scale = float(run_params["optimism_scale"])
        log_term = np.log(
            max(2.0, self.horizon * max(1, run_params["num_episodes"]) / delta)
        )
        return scale * (
            2.0 * (self.horizon**1.5) * np.sqrt(log_term) + epsilon * np.sqrt(max(1, n))
        )

    def _alpha(self, n: int) -> float:
        return (self.horizon + 1.0) / (self.horizon + float(n))

    def _discount_factor(self, run_params: dict) -> float:
        if run_params["use_discount"]:
            return float(self.discount)
        return 1.0

    def _stage_value_upper_bound(self, h: int, run_params: dict) -> float:
        remaining = max(0, self.horizon - int(h))
        gamma = self._discount_factor(run_params)
        if np.isclose(gamma, 1.0):
            return float(remaining)
        return float((1.0 - gamma**remaining) / (1.0 - gamma))

    def _update_from_trajectory(
        self,
        trajectory: list[tuple[int, int, float, int]],
        run_params: dict,
    ) -> tuple[float, int]:
        td_abs_sum = 0.0
        td_count = 0

        # Scan forward, as in Algorithm 1. q_hat[h+1] is defined for h=H-1.
        for h, (state, action, reward, next_state) in enumerate(trajectory):
            cell = self._cell(state, action)
            self.visit_counts[h, cell] += 1
            n = int(self.visit_counts[h, cell])
            alpha = self._alpha(n)
            next_value = 0.0
            if h + 1 < self.horizon:
                next_value = float(
                    self._state_action_values_at_stage(h + 1, next_state).max()
                )
            bonus = self._beta(n, run_params) / np.sqrt(float(n))
            target = reward + self._discount_factor(run_params) * next_value + bonus
            old_value = float(self.q_hat[h, cell])
            new_value = (1.0 - alpha) * old_value + alpha * target
            self.q_hat[h, cell] = min(
                new_value, self._stage_value_upper_bound(h, run_params)
            )

            td_abs_sum += abs(target - old_value)
            td_count += 1

        return td_abs_sum, td_count

    def _full_q_value(self, stage: int = 0) -> np.ndarray:
        q_value = np.zeros((self.model.state_dim, self.model.action_dim), dtype=float)
        stage = int(np.clip(stage, 0, self.horizon - 1))
        for state in range(self.model.state_dim):
            for action in range(self.model.action_dim):
                q_value[state, action] = self.q_hat[stage, self._cell(state, action)]
        return q_value

    def _moving_average(self, key: str, window_size: int) -> float:
        values = self.infos[key][-window_size:]
        if not values:
            return float("nan")
        return float(np.mean(values))

    def _record_progress(self, step: int, run_params: dict) -> None:
        q_value = self._full_q_value(stage=0)
        error = distance_to_optimal(q_value, self.model, self.discount, norm_method=2)
        self.infos["error_to_optimal"].append(float(error))
        reward_ma = self._moving_average("rewards", run_params["reward_ma_window"])
        td_ma = self._moving_average("td_errors", run_params["td_ma_window"])
        progress = 100.0 * (step + 1) / run_params["num_episodes"]
        print(
            f"[{step + 1:8,d}/{run_params['num_episodes']:,d} | {progress:6.2f}%] "
            f"regions={self.partition.n_regions:6d} | "
            f"cells={self.n_cells:6d} | "
            f"err_opt={error:10.6f} | "
            f"reward_ma{run_params['reward_ma_window']}={reward_ma:8.4f} | "
            f"td_ma{run_params['td_ma_window']}={td_ma:10.6f}"
        )

    def run(self, run_params: dict) -> None:
        params = require_params(run_params, REQUIRED_RUN_PARAMS)

        simulation.rng = np.random.default_rng(params["random_seed"])
        eval_rng = np.random.default_rng(int(params["random_seed"]) + 1_000_003)
        self.horizon = int(params["horizon"] or params["episode_length"])

        self.partition = self._build_partition(params)
        self.state_to_region = self._state_to_region(self.partition)
        self.n_cells = self.partition.n_regions * self.model.action_dim
        self.n_regions = self.partition.n_regions

        # q_hat[h, m], h in 0..H-1. The terminal value is implicit zero.
        optimistic_bounds = np.asarray(
            [self._stage_value_upper_bound(h, params) for h in range(self.horizon)],
            dtype=float,
        )
        self.q_hat = np.repeat(optimistic_bounds[:, None], self.n_cells, axis=1)
        self.visit_counts = np.zeros((self.horizon, self.n_cells), dtype=np.int64)

        self.infos: dict[str, list[Any]] = {
            "environment_steps": [],
            "error_to_optimal": [],
            "rewards": [],
            "greedy_rewards": [],
            "td_errors": [],
            "number_of_regions": [],
            "number_of_cells": [],
        }
        self.reward_history = self.infos["rewards"]
        self.td_error_history = self.infos["td_errors"]

        start_time = time.time()

        for step in range(params["num_episodes"]):
            trajectory = self._rollout(params)
            avg_reward = float(np.mean([reward for _, _, reward, _ in trajectory]))
            self.infos["rewards"].append(avg_reward)
            self.infos["environment_steps"].append((step + 1) * self.horizon)
            self.infos["number_of_regions"].append(self.partition.n_regions)
            self.infos["number_of_cells"].append(self.n_cells)

            td_abs_sum, td_count = self._update_from_trajectory(trajectory, params)
            self.infos["td_errors"].append(td_abs_sum / max(1, td_count))
            self.infos["greedy_rewards"].append(
                greedy_episode_reward(
                    self.model,
                    self._full_q_value(stage=0),
                    params["episode_length"],
                    eval_rng,
                )
            )

            if params["verbose"] and (
                step % params["log_every"] == 0 or step + 1 == params["num_episodes"]
            ):
                self._record_progress(step, params)

        self.q_value = self._full_q_value(stage=0)
        self.contracted_q_value = self.q_hat.copy()
        self.value = self.q_value.max(axis=1)
        self.policy = self.q_value.argmax(axis=1)
        self.n_regions = self.partition.n_regions
        self.runtime = time.time() - start_time
