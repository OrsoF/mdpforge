import time

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
    "refine_interval",
    "max_regions",
    "split_threshold",
    "min_visits_before_split",
    "require_all_actions_for_split",
    "hold_action_until_region_exit",
    "num_episodes",
    "episode_length",
    "num_replay_passes",
    "exploration_prob",
    "learning_rate",
    "decay_rate",
    "lr_decay_scale",
    "log_every",
    "reward_ma_window",
    "td_ma_window",
    "random_seed",
    "verbose",
)


class Solver(GenericSolver):
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.model = model
        self.model._normalize_reward_matrix()

        self.discount = discount
        self.final_precision = final_precision
        self.name = "Reynolds Q-Learning"

        self.steps_done_learning = 0

    def current_alpha(self, run_params: dict) -> float:
        return (
            run_params["learning_rate"]
            / (1.0 + run_params["lr_decay_scale"] * self.steps_done_learning)
            ** run_params["decay_rate"]
        )

    def next_alpha(self, run_params: dict) -> float:
        self.steps_done_learning += 1
        return self.current_alpha(run_params)

    def _moving_average(self, info_key: str, window_size: int) -> float:
        values = self.infos[info_key][-window_size:]
        if len(values) == 0:
            return float("nan")
        return float(np.mean(values))

    def _state_to_region(self, partition: Partition) -> np.ndarray:
        state_to_region = np.empty(self.model.state_dim, dtype=int)

        for region_index, states in enumerate(partition.states_in_region):
            state_to_region[np.asarray(states, dtype=int)] = region_index

        return state_to_region

    def _contiguous_regions(
        self,
        states: np.ndarray,
        n_regions: int,
    ) -> list[list[int]]:
        n_regions = int(np.clip(n_regions, 1, max(1, len(states))))
        return [
            tile.tolist() for tile in np.array_split(states, n_regions) if len(tile) > 0
        ]

    def _build_partition(self, run_params: dict) -> Partition:
        partition = Partition(self.model)
        initial_regions = int(run_params["initial_regions"])

        states = np.arange(self.model.state_dim)

        if self.discount >= 1.0:
            absorbing = np.array(
                [
                    s
                    for s in range(self.model.state_dim)
                    if min(
                        self.model.transition_matrix[a][s, s]
                        for a in range(self.model.action_dim)
                    )
                    >= 1.0
                ],
                dtype=int,
            )

            if 0 < len(absorbing) < self.model.state_dim:
                non_absorbing = np.setdiff1d(states, absorbing)
                partition.states_in_region = [
                    absorbing.tolist(),
                    *self._contiguous_regions(non_absorbing, initial_regions),
                ]
                return partition

        partition.states_in_region = self._contiguous_regions(states, initial_regions)
        return partition

    def _split_region(self, region: list[int]) -> list[list[int]]:
        if len(region) <= 1:
            return [region]

        left, right = np.array_split(np.asarray(region, dtype=int), 2)
        return [left.tolist(), right.tolist()]

    def _choose_action(
        self,
        region_index: int,
        contracted_q_value: np.ndarray,
        run_params: dict,
    ) -> int:
        if simulation.rng.random() < run_params["exploration_prob"]:
            return int(simulation.rng.integers(self.model.action_dim))

        return int(contracted_q_value[region_index].argmax())

    def _generate_region_trajectory(
        self,
        state_to_region: np.ndarray,
        contracted_q_value: np.ndarray,
        run_params: dict,
    ) -> list[tuple[int, int, float, int]]:
        trajectory = []

        state = int(simulation.rng.integers(self.model.state_dim))
        current_region = -1
        current_action = None

        for _ in range(run_params["episode_length"]):
            region_index = int(state_to_region[state])

            must_resample_action = (
                current_action is None
                or not run_params["hold_action_until_region_exit"]
                or region_index != current_region
            )

            if must_resample_action:
                current_region = region_index
                current_action = self._choose_action(
                    region_index,
                    contracted_q_value,
                    run_params,
                )

            next_state, reward = generate_sars(self.model, state, current_action)
            trajectory.append((state, current_action, reward, next_state))

            state = int(next_state)

        return trajectory

    def _decision_boundary_score(
        self,
        region_index: int,
        neighbor_index: int,
        contracted_q_value: np.ndarray,
        visit_counts: np.ndarray,
        run_params: dict,
    ) -> float:
        action_i = int(contracted_q_value[region_index].argmax())
        action_j = int(contracted_q_value[neighbor_index].argmax())

        if action_i == action_j:
            return 0.0

        loss_i = (
            contracted_q_value[region_index, action_i]
            - contracted_q_value[region_index, action_j]
        )
        loss_j = (
            contracted_q_value[neighbor_index, action_j]
            - contracted_q_value[neighbor_index, action_i]
        )

        score = float(max(loss_i, loss_j))
        if score < float(run_params["split_threshold"]):
            return 0.0

        min_visits = int(run_params["min_visits_before_split"])

        if run_params["require_all_actions_for_split"]:
            ready = np.all(
                visit_counts[[region_index, neighbor_index], :] >= min_visits
            )
        else:
            ready = all(
                visit_counts[region, action] >= min_visits
                for region in (region_index, neighbor_index)
                for action in (action_i, action_j)
            )

        if not ready:
            return 0.0

        return score

    def _refine_partition(
        self,
        partition: Partition,
        contracted_q_value: np.ndarray,
        visit_counts: np.ndarray,
        adjacent_regions: set[tuple[int, int]],
        run_params: dict,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        max_regions = run_params["max_regions"]

        if max_regions is not None and partition.n_regions >= int(max_regions):
            return contracted_q_value, visit_counts, False

        split_scores = np.zeros(partition.n_regions, dtype=float)

        for region_index, neighbor_index in adjacent_regions:
            if region_index == neighbor_index:
                continue
            if (
                region_index >= partition.n_regions
                or neighbor_index >= partition.n_regions
            ):
                continue

            score = self._decision_boundary_score(
                region_index,
                neighbor_index,
                contracted_q_value,
                visit_counts,
                run_params,
            )

            if score > 0.0:
                split_scores[region_index] = max(split_scores[region_index], score)
                split_scores[neighbor_index] = max(split_scores[neighbor_index], score)

        candidates = sorted(
            (
                (float(score), region_index)
                for region_index, score in enumerate(split_scores)
                if score > 0.0 and len(partition.states_in_region[region_index]) > 1
            ),
            reverse=True,
        )

        if not candidates:
            return contracted_q_value, visit_counts, False

        split_budget = len(candidates)

        if max_regions is not None:
            split_budget = min(
                split_budget,
                max(0, int(max_regions) - partition.n_regions),
            )

        if split_budget <= 0:
            return contracted_q_value, visit_counts, False

        regions_to_split = {
            region_index for _, region_index in candidates[:split_budget]
        }

        new_regions: list[list[int]] = []
        parent_indices: list[int] = []
        new_visit_counts: list[np.ndarray] = []

        for region_index, region in enumerate(partition.states_in_region):
            if region_index not in regions_to_split:
                new_regions.append(region)
                parent_indices.append(region_index)
                new_visit_counts.append(visit_counts[region_index].copy())
                continue

            for child in self._split_region(region):
                if len(child) == 0:
                    continue

                new_regions.append(child)
                parent_indices.append(region_index)

                # Reynolds: new children inherit Q-values but are marked unvisited.
                new_visit_counts.append(
                    np.zeros(self.model.action_dim, dtype=visit_counts.dtype)
                )

        partition.states_in_region = new_regions

        parent = np.asarray(parent_indices, dtype=int)
        new_contracted_q_value = contracted_q_value[parent].copy()
        new_visit_counts_array = np.vstack(new_visit_counts)

        self.infos["refinement_steps"].append(len(self.infos["rewards"]))

        return new_contracted_q_value, new_visit_counts_array, True

    def _record_progress(
        self,
        q_value: np.ndarray,
        step: int,
        run_params: dict,
        partition: Partition,
    ) -> None:
        error = distance_to_optimal(
            q_value,
            self.model,
            self.discount,
            norm_method=2,
        )

        reward_ma = self._moving_average("rewards", run_params["reward_ma_window"])
        td_ma = self._moving_average("td_errors", run_params["td_ma_window"])
        progress = 100.0 * (step + 1) / run_params["num_episodes"]

        print(
            f"[{step + 1:8,d}/{run_params['num_episodes']:,d} | {progress:6.2f}%] "
            f"regions={partition.n_regions:6d} | "
            f"err_opt={error:10.6f} | "
            f"alpha={self.current_alpha(run_params):9.6f} | "
            f"reward_ma{run_params['reward_ma_window']}={reward_ma:8.4f} | "
            f"td_ma{run_params['td_ma_window']}={td_ma:10.6f}"
        )

        self.infos["error_to_optimal"].append(float(error))

    def run(self, run_params: dict) -> None:
        params = require_params(run_params, REQUIRED_RUN_PARAMS)

        simulation.rng = np.random.default_rng(params["random_seed"])
        eval_rng = np.random.default_rng(int(params["random_seed"]) + 1_000_003)
        self.steps_done_learning = 0

        partition = self._build_partition(params)
        state_to_region = self._state_to_region(partition)

        contracted_q_value = np.zeros(
            (partition.n_regions, self.model.action_dim),
            dtype=float,
        )
        visit_counts = np.zeros_like(contracted_q_value, dtype=int)
        adjacent_regions: set[tuple[int, int]] = set()

        self.infos = {
            "environment_steps": [],
            "error_to_optimal": [],
            "rewards": [],
            "greedy_rewards": [],
            "td_errors": [],
            "number_of_regions": [],
            "refinement_steps": [],
        }

        self.reward_history = self.infos["rewards"]
        self.td_error_history = self.infos["td_errors"]

        start_time = time.time()

        for step in range(params["num_episodes"]):
            trajectory = self._generate_region_trajectory(
                state_to_region,
                contracted_q_value,
                params,
            )

            avg_reward = float(np.mean([reward for _, _, reward, _ in trajectory]))
            self.infos["rewards"].append(avg_reward)
            self.infos["environment_steps"].append(
                (step + 1) * params["episode_length"]
            )

            # Count only real environment transitions, not replay passes.
            for state, action, _, next_state in trajectory:
                region = int(state_to_region[state])
                next_region = int(state_to_region[next_state])

                visit_counts[region, action] += 1

                if region != next_region:
                    adjacent_regions.add((region, next_region))
                    adjacent_regions.add((next_region, region))

            td_abs_sum = 0.0
            td_count = 0

            for _ in range(params["num_replay_passes"]):
                for state, action, reward, next_state in reversed(trajectory):
                    region = int(state_to_region[state])
                    next_region = int(state_to_region[next_state])

                    target = (
                        reward + self.discount * contracted_q_value[next_region].max()
                    )
                    delta = target - contracted_q_value[region, action]

                    contracted_q_value[region, action] += (
                        self.next_alpha(params) * delta
                    )

                    td_abs_sum += abs(float(delta))
                    td_count += 1

            avg_td_error = td_abs_sum / max(1, td_count)
            self.infos["td_errors"].append(float(avg_td_error))
            self.infos["number_of_regions"].append(partition.n_regions)

            if params["verbose"] and (
                step % params["log_every"] == 0 or step + 1 == params["num_episodes"]
            ):
                q_value = np.asarray(partition.phi.dot(contracted_q_value))
                self._record_progress(q_value, step, params, partition)

            should_refine = (
                params["refine_interval"] > 0
                and (step + 1) % params["refine_interval"] == 0
                and step + 1 < params["num_episodes"]
            )

            if should_refine:
                contracted_q_value, visit_counts, refined = self._refine_partition(
                    partition,
                    contracted_q_value,
                    visit_counts,
                    adjacent_regions,
                    params,
                )

                if refined:
                    state_to_region = self._state_to_region(partition)
                    adjacent_regions = set()

            self.infos["greedy_rewards"].append(
                greedy_episode_reward(
                    self.model,
                    np.asarray(partition.phi.dot(contracted_q_value)),
                    params["episode_length"],
                    eval_rng,
                )
            )

        self.partition = partition

        q_value = np.asarray(partition.phi.dot(contracted_q_value))
        self.q_value = q_value
        self.contracted_q_value = contracted_q_value
        self.value = q_value.max(axis=1)
        self.policy = q_value.argmax(axis=1)
        self.n_regions = partition.n_regions
        self.runtime = time.time() - start_time
