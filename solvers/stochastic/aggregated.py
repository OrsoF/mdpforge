import time
from collections import deque

import numpy as np
from solvers_agg.solvers.common import greedy_episode_reward, require_params

import utils.simulation as simulation
from core.model import GenericModel
from core.partition import Partition
from core.solver import GenericSolver
from utils.exact_value_function import distance_to_optimal
from utils.simulation import generate_sars

REQUIRED_RUN_PARAMS = (
    "refine_interval",
    "initial_regions",
    "initial_bins_per_variable",
    "initial_q",
    "model_init_bellman_steps",
    "init_q_from_rewards",
    "max_regions",
    "split_threshold",
    "split_confidence_delta",
    "split_confidence_scale",
    "split_variance_floor",
    "split_method",
    "split_excess_threshold",
    "min_visits_before_split",
    "min_ready_states_per_region",
    "min_split_candidates_per_region",
    "split_require_fraction",
    "min_region_size_to_split",
    "max_splits_per_refine",
    "split_score_quantile",
    "replay_buffer_size",
    "post_refine_replay_passes",
    "region_lr_scaling",
    "region_lr_reference_size",
    "region_lr_min_multiplier",
    "region_lr_max_multiplier",
    "final_exploration_prob",
    "exploration_decay_episodes",
    "random_tie_breaking",
    "debug_splits",
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
    "state_shape",
)


class Solver(GenericSolver):
    """
    Confidence-adaptive state-aggregation Q-learning.

    The solver alternates between:
      1. Q-learning on the current aggregated partition.
      2. Collecting per-state/action TD-target statistics.
      3. Refining regions whose states have significantly different TD targets.

    Compared with the simpler version:
      - split tests use empirical variance;
      - regional means are visitation-weighted;
      - replay statistics are not artificially inflated by replay passes;
      - old samples are reused after a split to adapt inherited child Q-values.
    """

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
        self.name = "Confidence Adaptive Aggregated Q-Learning"

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

    def _region_lr_reference_size(self, run_params: dict) -> float:
        reference_size = run_params["region_lr_reference_size"]
        if reference_size is not None:
            return max(1.0, float(reference_size))

        max_regions = run_params["max_regions"]
        if max_regions is not None:
            return max(1.0, self.model.state_dim / max(1, int(max_regions)))

        initial_regions = int(run_params["initial_regions"])
        return max(1.0, self.model.state_dim / max(1, initial_regions))

    def region_lr_multiplier(self, region_size: int, run_params: dict) -> float:
        scaling = run_params["region_lr_scaling"]
        if scaling in (None, "none", False):
            raw_multiplier = 1.0
        else:
            size_ratio = max(1.0, float(region_size)) / self._region_lr_reference_size(
                run_params
            )
            if scaling == "sqrt":
                raw_multiplier = np.sqrt(size_ratio)
            elif scaling == "linear":
                raw_multiplier = size_ratio
            else:
                raise ValueError("region_lr_scaling must be one of: none, sqrt, linear")

        return float(
            np.clip(
                raw_multiplier,
                float(run_params["region_lr_min_multiplier"]),
                float(run_params["region_lr_max_multiplier"]),
            )
        )

    def next_region_alpha(self, region_size: int, run_params: dict) -> float:
        return self.next_alpha(run_params) * self.region_lr_multiplier(
            region_size,
            run_params,
        )

    def _state_to_region(self, partition: Partition) -> np.ndarray:
        return np.asarray(partition.state_to_region, dtype=int)

    def _moving_average(self, info_key: str, window_size: int) -> float:
        values = self.infos[info_key][-window_size:]
        if len(values) == 0:
            return float("nan")
        return float(np.mean(values))

    def _record_progress(
        self,
        q_value: np.ndarray,
        step: int,
        avg_reward: float,
        avg_td_error: float,
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

        _max_split_score = (
            self.infos["max_split_score"][-1]
            if len(self.infos["max_split_score"]) > 0
            else 0.0
        )

        print(
            f"[{step + 1:8,d}/{run_params['num_episodes']:,d} | {progress:6.2f}%] "
            f"regions={partition.n_regions:6d} | "
            f"err_opt={error:10.6f} | "
            f"alpha={self.current_alpha(run_params):9.6f} | "
            f"reward_ma{run_params['reward_ma_window']}={reward_ma:8.4f} | "
            f"td_ma{run_params['td_ma_window']}={td_ma:10.6f}"
        )

        self.infos["error_to_optimal"].append(error)

    def _contiguous_regions(
        self, states: np.ndarray, n_regions: int
    ) -> list[list[int]]:
        n_regions = int(np.clip(n_regions, 1, max(1, len(states))))
        return [
            tile.tolist() for tile in np.array_split(states, n_regions) if len(tile)
        ]

    def _feature_grid_regions(self, run_params: dict) -> list[list[int]] | None:
        state_shape = run_params["state_shape"]
        bins = run_params["initial_bins_per_variable"]
        if state_shape is None or bins is None:
            return None

        shape = tuple(int(x) for x in state_shape)
        if int(np.prod(shape)) != self.model.state_dim:
            return None

        if np.isscalar(bins):
            bins = (int(bins),) * len(shape)
        else:
            bins = tuple(int(x) for x in bins)
            if len(bins) != len(shape):
                raise ValueError("initial_bins_per_variable must match state_shape.")

        axes = [
            np.array_split(np.arange(n), min(max(1, b), n)) for n, b in zip(shape, bins)
        ]
        regions = []
        for blocks in (
            np.array(
                np.meshgrid(*[np.arange(len(axis)) for axis in axes], indexing="ij")
            )
            .reshape(len(shape), -1)
            .T
        ):
            masks = np.ix_(*[axes[d][int(blocks[d])] for d in range(len(shape))])
            states = np.ravel_multi_index(masks, shape).ravel()
            if len(states):
                regions.append(states.astype(int).tolist())
        return regions

    def _build_partition(self, run_params: dict) -> Partition:
        partition = Partition(self.model)
        initial_regions = int(run_params["initial_regions"])

        feature_regions = self._feature_grid_regions(run_params)
        if feature_regions is not None:
            partition.states_in_region = feature_regions
        elif initial_regions > 1:
            states = np.arange(self.model.state_dim)
            partition.states_in_region = self._contiguous_regions(
                states,
                initial_regions,
            )

        if self.discount < 1.0:
            return partition

        absorbing = [
            s
            for s in range(self.model.state_dim)
            if min(
                self.model.transition_matrix[a][s, s]
                for a in range(self.model.action_dim)
            )
            >= 1.0
        ]

        if absorbing and len(absorbing) < self.model.state_dim:
            other = np.setdiff1d(np.arange(self.model.state_dim), absorbing)
            partition.states_in_region = [
                absorbing,
                *self._contiguous_regions(other, initial_regions),
            ]

        return partition

    def _generate_trajectory(
        self,
        state_to_region: np.ndarray,
        contracted_q_value: np.ndarray,
        run_params: dict,
        exploration_prob: float,
    ) -> list[tuple[int, int, float, int]]:
        trajectory = []
        state = int(simulation.rng.integers(self.model.state_dim))

        for _ in range(run_params["episode_length"]):
            state_region = state_to_region[state]

            if simulation.rng.random() < exploration_prob:
                action = int(simulation.rng.integers(self.model.action_dim))
            elif run_params["random_tie_breaking"]:
                q = contracted_q_value[state_region]
                action = int(simulation.rng.choice(np.flatnonzero(q == q.max())))
            else:
                action = int(contracted_q_value[state_region].argmax())

            next_state, reward = generate_sars(self.model, state, action)
            trajectory.append((state, action, reward, next_state))

            state = next_state

        return trajectory

    def _exploration_prob(self, run_params: dict, step: int) -> float:
        final = run_params["final_exploration_prob"]
        if final is None:
            return float(run_params["exploration_prob"])

        decay = int(run_params["exploration_decay_episodes"])
        if decay <= 0:
            return float(final)

        t = min(1.0, float(step) / float(decay))
        start = float(run_params["exploration_prob"])
        return (1.0 - t) * start + t * float(final)

    def _td_target_moments(
        self,
        td_target_sums: np.ndarray,
        td_target_sq_sums: np.ndarray,
        td_target_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = np.divide(
            td_target_sums,
            td_target_counts,
            out=np.zeros_like(td_target_sums, dtype=float),
            where=td_target_counts > 0,
        )

        second_moment = np.divide(
            td_target_sq_sums,
            td_target_counts,
            out=np.zeros_like(td_target_sq_sums, dtype=float),
            where=td_target_counts > 0,
        )

        variance = np.maximum(0.0, second_moment - mean * mean)

        return mean, variance

    def _confidence_radius(
        self,
        variance: np.ndarray,
        counts: np.ndarray,
        log_term: float,
        scale: float,
        variance_floor: float,
    ) -> np.ndarray:
        safe_counts = np.maximum(counts.astype(float), 1.0)
        safe_variance = np.maximum(variance, variance_floor)

        radius = scale * np.sqrt(2.0 * safe_variance * log_term / safe_counts)
        radius[counts <= 0] = np.inf

        return radius

    def _refine_by_midpoint(
        self,
        partition: Partition,
        split_scores: np.ndarray,
        threshold: float,
        max_regions: int | None,
        run_params: dict,
    ) -> np.ndarray:
        regions = partition.states_in_region
        min_region_size = max(2, int(run_params["min_region_size_to_split"]))
        split_order = sorted(
            (
                (float(np.max(split_scores[region])), index)
                for index, region in enumerate(regions)
                if len(region) >= min_region_size
                and np.max(split_scores[region]) > threshold
            ),
            reverse=True,
        )

        score_quantile = float(run_params["split_score_quantile"])
        if split_order and score_quantile > 0.0:
            score_quantile = min(score_quantile, 1.0)
            scores = np.asarray([score for score, _ in split_order], dtype=float)
            score_floor = float(np.quantile(scores, score_quantile))
            split_order = [
                (score, index) for score, index in split_order if score >= score_floor
            ]

        remaining_splits = len(split_order)
        if max_regions is not None:
            remaining_splits = min(
                remaining_splits,
                max(0, int(max_regions) - partition.n_regions),
            )
        max_splits_per_refine = run_params["max_splits_per_refine"]
        if max_splits_per_refine is not None:
            remaining_splits = min(remaining_splits, int(max_splits_per_refine))

        regions_to_split = {index for _, index in split_order[:remaining_splits]}

        if not regions_to_split:
            return np.arange(partition.n_regions)

        new_regions = []
        parent = []
        for index, region in enumerate(regions):
            if index not in regions_to_split:
                new_regions.append(region)
                parent.append(index)
                continue

            for child in np.array_split(np.asarray(region, dtype=int), 2):
                if len(child) == 0:
                    continue
                new_regions.append(child.tolist())
                parent.append(index)

        partition.states_in_region = new_regions
        return np.asarray(parent, dtype=int)

    def _initial_q_value(self, partition: Partition, run_params: dict) -> np.ndarray:
        q_value = np.full(
            (partition.n_regions, self.model.action_dim),
            float(run_params["initial_q"]),
            dtype=float,
        )

        init_steps = int(run_params["model_init_bellman_steps"])
        if init_steps > 0:
            q_full = np.maximum(
                self.model.reward_matrix.astype(float, copy=True),
                float(run_params["initial_q"]),
            )
            for _ in range(init_steps):
                value = q_full.max(axis=1)
                for action in range(self.model.action_dim):
                    q_full[:, action] = self.model.reward_matrix[
                        :, action
                    ] + self.discount * self.model.transition_matrix[action].dot(value)

            for region_index, region in enumerate(partition.states_in_region):
                q_value[region_index] = q_full[region].mean(axis=0)
            return q_value

        if not run_params["init_q_from_rewards"]:
            return q_value

        for region_index, region in enumerate(partition.states_in_region):
            q_value[region_index] = np.maximum(
                q_value[region_index],
                self.model.reward_matrix[region].mean(axis=0),
            )

        return q_value

    def _compute_split_scores(
        self,
        partition: Partition,
        td_target_sums: np.ndarray,
        td_target_sq_sums: np.ndarray,
        td_target_counts: np.ndarray,
        run_params: dict,
    ) -> np.ndarray:
        state_dim = self.model.state_dim
        action_dim = self.model.action_dim

        mean_target, variance_target = self._td_target_moments(
            td_target_sums,
            td_target_sq_sums,
            td_target_counts,
        )

        delta = float(run_params["split_confidence_delta"])
        delta = min(max(delta, 1e-12), 1.0)

        log_term = np.log(
            max(
                2.0,
                2.0 * state_dim * action_dim * max(1, partition.n_regions) / delta,
            )
        )

        confidence_scale = float(run_params["split_confidence_scale"])
        variance_floor = float(run_params["split_variance_floor"])

        radius = self._confidence_radius(
            variance=variance_target,
            counts=td_target_counts,
            log_term=log_term,
            scale=confidence_scale,
            variance_floor=variance_floor,
        )

        min_visits = int(run_params["min_visits_before_split"])
        min_ready_states = int(run_params["min_ready_states_per_region"])
        min_split_candidates = int(run_params["min_split_candidates_per_region"])
        split_fraction = float(run_params["split_require_fraction"])
        split_threshold = float(run_params["split_threshold"])
        split_excess_threshold = float(run_params["split_excess_threshold"])

        split_scores = np.zeros(state_dim, dtype=float)
        ready = td_target_counts >= min_visits

        debug_max_raw = 0.0
        debug_max_excess = 0.0
        debug_ready_state_actions = int(np.sum(ready))

        for region_list in partition.states_in_region:
            if len(region_list) <= 1:
                continue

            region = np.asarray(region_list, dtype=int)
            region_scores = np.zeros(len(region), dtype=float)

            for action in range(action_dim):
                ready_positions = np.flatnonzero(ready[region, action])
                action_states = region[ready_positions]

                if len(action_states) < min_ready_states:
                    continue

                counts = td_target_counts[action_states, action].astype(float)
                weights = counts / np.sum(counts)

                regional_mean = float(
                    np.dot(weights, mean_target[action_states, action])
                )

                regional_se = np.sqrt(
                    np.sum(
                        (weights**2)
                        * np.maximum(
                            variance_target[action_states, action],
                            variance_floor,
                        )
                        / counts
                    )
                )

                regional_radius = (
                    confidence_scale * np.sqrt(2.0 * log_term) * regional_se
                )

                raw_disagreement = np.abs(
                    mean_target[action_states, action] - regional_mean
                )

                total_radius = radius[action_states, action] + regional_radius
                statistical_excess = raw_disagreement - total_radius

                debug_max_raw = max(debug_max_raw, float(np.max(raw_disagreement)))
                debug_max_excess = max(
                    debug_max_excess,
                    float(np.max(statistical_excess)),
                )

                significant = (raw_disagreement > split_threshold) & (
                    statistical_excess > split_excess_threshold
                )

                np.maximum.at(
                    region_scores,
                    ready_positions[significant],
                    raw_disagreement[significant],
                )

            candidate_count = int(np.sum(region_scores > split_threshold))

            required_candidates = max(
                min_split_candidates,
                int(np.ceil(split_fraction * len(region))),
            )

            if candidate_count >= required_candidates:
                split_scores[region] = region_scores

        if run_params["debug_splits"]:
            print(
                "split debug | "
                f"regions={partition.n_regions} | "
                f"ready_state_actions={debug_ready_state_actions} | "
                f"max_raw={debug_max_raw:.6g} | "
                f"max_excess={debug_max_excess:.6g} | "
                f"threshold={split_threshold:.6g} | "
                f"candidates={int(np.sum(split_scores > split_threshold))}"
            )

        return split_scores

    def _refine_partition(
        self,
        partition: Partition,
        contracted_q_value: np.ndarray,
        td_target_sums: np.ndarray,
        td_target_sq_sums: np.ndarray,
        td_target_counts: np.ndarray,
        run_params: dict,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        max_regions = run_params["max_regions"]

        if max_regions is not None and partition.n_regions >= int(max_regions):
            return contracted_q_value, np.arange(partition.n_regions), False

        old_region_count = partition.n_regions

        split_scores = self._compute_split_scores(
            partition=partition,
            td_target_sums=td_target_sums,
            td_target_sq_sums=td_target_sq_sums,
            td_target_counts=td_target_counts,
            run_params=run_params,
        )

        max_score = float(np.max(split_scores))

        self.infos["max_split_score"].append(max_score)
        self.infos["split_candidate_count"].append(
            int(np.sum(split_scores > run_params["split_threshold"]))
        )

        if max_score <= run_params["split_threshold"]:
            return contracted_q_value, np.arange(partition.n_regions), False

        if run_params["split_method"] == "midpoint":
            parent = self._refine_by_midpoint(
                partition=partition,
                split_scores=split_scores,
                threshold=run_params["split_threshold"],
                max_regions=max_regions,
                run_params=run_params,
            )
        else:
            parent = partition.refine_by_width(
                split_scores,
                run_params["split_threshold"],
            )

        refined = partition.n_regions != old_region_count

        if refined:
            contracted_q_value = contracted_q_value[parent]
            self.infos["refinement_steps"].append(len(self.infos["rewards"]))

        return contracted_q_value, parent, refined

    def _replay_transitions(
        self,
        transitions: list[tuple[int, int, float, int]],
        state_to_region: np.ndarray,
        contracted_q_value: np.ndarray,
        td_target_sums: np.ndarray,
        td_target_sq_sums: np.ndarray,
        td_target_counts: np.ndarray,
        run_params: dict,
        record_split_stats: bool,
    ) -> tuple[float, int]:
        td_abs_sum = 0.0
        td_count = 0
        region_sizes = np.bincount(
            state_to_region,
            minlength=contracted_q_value.shape[0],
        )

        for state, action, reward, next_state in reversed(transitions):
            state_region = state_to_region[state]
            next_state_region = state_to_region[next_state]

            target = (
                reward + self.discount * contracted_q_value[next_state_region].max()
            )

            delta = target - contracted_q_value[state_region, action]

            contracted_q_value[state_region, action] += (
                self.next_region_alpha(region_sizes[state_region], run_params) * delta
            )

            if record_split_stats:
                td_target_sums[state, action] += target
                td_target_sq_sums[state, action] += target * target
                td_target_counts[state, action] += 1

            td_abs_sum += abs(delta)
            td_count += 1

        return td_abs_sum, td_count

    def run(
        self,
        run_params: dict,
    ) -> None:
        params = require_params(run_params, REQUIRED_RUN_PARAMS)

        simulation.rng = np.random.default_rng(params["random_seed"])
        eval_rng = np.random.default_rng(int(params["random_seed"]) + 1_000_003)
        self.steps_done_learning = 0

        partition = self._build_partition(params)
        state_to_region = self._state_to_region(partition)

        contracted_q_value = self._initial_q_value(partition, params)

        td_target_sums = np.zeros(
            (self.model.state_dim, self.model.action_dim),
            dtype=float,
        )

        td_target_sq_sums = np.zeros(
            (self.model.state_dim, self.model.action_dim),
            dtype=float,
        )

        td_target_counts = np.zeros(
            (self.model.state_dim, self.model.action_dim),
            dtype=np.int64,
        )

        replay_buffer = deque(maxlen=int(params["replay_buffer_size"]))

        self.infos = {
            "environment_steps": [],
            "error_to_optimal": [],
            "rewards": [],
            "greedy_rewards": [],
            "td_errors": [],
            "number_of_regions": [],
            "refinement_steps": [],
            "max_split_score": [],
            "split_candidate_count": [],
        }

        self.reward_history = self.infos["rewards"]
        self.td_error_history = self.infos["td_errors"]

        start_time = time.time()

        for step in range(params["num_episodes"]):
            trajectory = self._generate_trajectory(
                state_to_region=state_to_region,
                contracted_q_value=contracted_q_value,
                run_params=params,
                exploration_prob=self._exploration_prob(params, step),
            )

            replay_buffer.extend(trajectory)

            avg_reward = float(np.mean([reward for _, _, reward, _ in trajectory]))

            self.infos["rewards"].append(avg_reward)
            self.infos["environment_steps"].append(
                (step + 1) * params["episode_length"]
            )

            td_abs_sum = 0.0
            td_count = 0

            for replay_pass in range(params["num_replay_passes"]):
                record_split_stats = replay_pass == 0

                pass_td_abs_sum, pass_td_count = self._replay_transitions(
                    transitions=trajectory,
                    state_to_region=state_to_region,
                    contracted_q_value=contracted_q_value,
                    td_target_sums=td_target_sums,
                    td_target_sq_sums=td_target_sq_sums,
                    td_target_counts=td_target_counts,
                    run_params=params,
                    record_split_stats=record_split_stats,
                )

                td_abs_sum += pass_td_abs_sum
                td_count += pass_td_count

            avg_td_error = td_abs_sum / max(1, td_count)

            self.infos["td_errors"].append(avg_td_error)
            self.infos["number_of_regions"].append(partition.n_regions)

            if params["verbose"] and (
                step % params["log_every"] == 0 or step + 1 == params["num_episodes"]
            ):
                q_value = partition.phi.dot(contracted_q_value)

                self._record_progress(
                    q_value=q_value,
                    step=step,
                    avg_reward=avg_reward,
                    avg_td_error=avg_td_error,
                    run_params=params,
                    partition=partition,
                )

            should_refine = (
                params["refine_interval"] > 0
                and (step + 1) % params["refine_interval"] == 0
                and step + 1 < params["num_episodes"]
            )

            if should_refine:
                _old_region_count = partition.n_regions

                contracted_q_value, parent, refined = self._refine_partition(
                    partition=partition,
                    contracted_q_value=contracted_q_value,
                    td_target_sums=td_target_sums,
                    td_target_sq_sums=td_target_sq_sums,
                    td_target_counts=td_target_counts,
                    run_params=params,
                )

                if refined:
                    state_to_region = self._state_to_region(partition)

                    td_target_sums.fill(0.0)
                    td_target_sq_sums.fill(0.0)
                    td_target_counts.fill(0)

                    for _ in range(int(params["post_refine_replay_passes"])):
                        self._replay_transitions(
                            transitions=list(replay_buffer),
                            state_to_region=state_to_region,
                            contracted_q_value=contracted_q_value,
                            td_target_sums=td_target_sums,
                            td_target_sq_sums=td_target_sq_sums,
                            td_target_counts=td_target_counts,
                            run_params=params,
                            record_split_stats=False,
                        )

                    # if params["verbose"]:
                    #     print(
                    #         f"refined partition: "
                    #         f"{_old_region_count} -> {partition.n_regions} regions"
                    #     )

            self.infos["greedy_rewards"].append(
                greedy_episode_reward(
                    self.model,
                    partition.phi.dot(contracted_q_value),
                    params["episode_length"],
                    eval_rng,
                )
            )

        self.partition = partition

        q_value = partition.phi.dot(contracted_q_value)

        self.q_value = q_value
        self.contracted_q_value = contracted_q_value
        self.value = q_value.max(axis=1)
        self.policy = q_value.argmax(axis=1)
        self.n_regions = partition.n_regions
        self.runtime = time.time() - start_time
