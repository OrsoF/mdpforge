"""
CAT+RL-style Conditional Abstraction Tree Q-learning.

Reference target: Dadvar, Nayyar & Srivastava (2023).
This implementation is a compact, project-compatible version:
  - Q-learning is run on abstract leaves.
  - Evaluation logs TD-error dispersion per abstract state.
  - Refinement selects top unstable leaves and splits the feature/variable that
    best explains TD-error variation.
  - The tree is conditional because each split is local to a leaf; the same
    variable may have different refinements under different ancestor conditions.

For meaningful conditional abstractions, pass run_params["state_shape"]. Without
it, the only variable is the integer state id, so refinement is just interval-like.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
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
    "state_shape",
    "initial_bins_per_variable",
    "max_leaves",
    "refine_check_interval",
    "force_refinement",
    "success_threshold",
    "success_reward_threshold",
    "evaluation_episodes",
    "top_k_unstable",
    "min_td_errors_for_refine",
    "td_std_threshold",
    "split_balance_min_fraction",
    "extended_actions",
    "max_primitive_steps_per_abstract_action",
    "inherit_q_on_split",
    "verbose_splits",
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


@dataclass
class CATNode:
    node_id: int
    states: list[int]
    depth: int = 0
    parent: int | None = None
    split_variable: int | None = None
    split_threshold: float | None = None
    left: "CATNode | None" = None
    right: "CATNode | None" = None
    q_values: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class Solver(GenericSolver):
    def __init__(self, model: GenericModel, discount: float, final_precision: float):
        self.model = model
        self.model._normalize_reward_matrix()
        self.discount = discount
        self.final_precision = final_precision
        self.name = "CAT+RL Q-Learning"
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

    def _build_features(self, run_params: dict) -> np.ndarray:
        state_shape = run_params["state_shape"]
        if state_shape is None:
            return np.arange(self.model.state_dim, dtype=float).reshape(-1, 1)

        shape = tuple(int(x) for x in state_shape)
        n = int(np.prod(shape))
        if n not in (self.model.state_dim, self.model.state_dim - 1):
            raise ValueError(
                "run_params['state_shape'] must multiply to model.state_dim "
                "or model.state_dim - 1."
            )

        features = np.zeros((self.model.state_dim, len(shape)), dtype=float)
        for state in range(n):
            features[state] = np.asarray(np.unravel_index(state, shape), dtype=float)
        if n == self.model.state_dim - 1:
            features[-1] = np.asarray(shape, dtype=float)
        return features

    def _new_node(
        self, states: list[int], depth: int, parent: int | None = None
    ) -> CATNode:
        node = CATNode(
            node_id=len(self.nodes),
            states=list(states),
            depth=depth,
            parent=parent,
            q_values=np.zeros(self.model.action_dim, dtype=float),
        )
        self.nodes.append(node)
        return node

    def _build_initial_subtree(
        self,
        states: list[int],
        feature: int,
        groups: list[np.ndarray],
        depth: int,
        parent: int | None,
        run_params: dict,
    ) -> CATNode:
        node = self._new_node(states, depth=depth, parent=parent)
        if len(groups) == 1:
            return self._build_initial_tree(
                states, feature + 1, depth, parent, run_params, node
            )

        mid = len(groups) // 2
        threshold = float(groups[mid - 1][-1])
        x = self.features[np.asarray(states, dtype=int), feature]
        left_states = np.asarray(states, dtype=int)[x <= threshold].tolist()
        right_states = np.asarray(states, dtype=int)[x > threshold].tolist()
        if not left_states or not right_states:
            return node

        node.split_variable = feature
        node.split_threshold = threshold
        node.left = self._build_initial_subtree(
            left_states,
            feature,
            groups[:mid],
            depth + 1,
            node.node_id,
            run_params,
        )
        node.right = self._build_initial_subtree(
            right_states,
            feature,
            groups[mid:],
            depth + 1,
            node.node_id,
            run_params,
        )
        return node

    def _build_initial_tree(
        self,
        states: list[int],
        feature: int,
        depth: int,
        parent: int | None,
        run_params: dict,
        node: CATNode | None = None,
    ) -> CATNode:
        bins = int(run_params["initial_bins_per_variable"])
        if bins <= 1 or feature >= self.features.shape[1] or len(states) <= 1:
            return node if node is not None else self._new_node(states, depth, parent)

        states_array = np.asarray(states, dtype=int)
        values = np.unique(self.features[states_array, feature])
        if len(values) <= 1:
            return self._build_initial_tree(
                states, feature + 1, depth, parent, run_params, node
            )

        groups = [
            group
            for group in np.array_split(values, min(bins, len(values)))
            if len(group)
        ]
        if len(groups) <= 1:
            return self._build_initial_tree(
                states, feature + 1, depth, parent, run_params, node
            )

        if node is None:
            return self._build_initial_subtree(
                states, feature, groups, depth, parent, run_params
            )

        mid = len(groups) // 2
        threshold = float(groups[mid - 1][-1])
        x = self.features[states_array, feature]
        left_states = states_array[x <= threshold].tolist()
        right_states = states_array[x > threshold].tolist()
        if not left_states or not right_states:
            return node

        node.split_variable = feature
        node.split_threshold = threshold
        node.left = self._build_initial_subtree(
            left_states,
            feature,
            groups[:mid],
            depth + 1,
            node.node_id,
            run_params,
        )
        node.right = self._build_initial_subtree(
            right_states,
            feature,
            groups[mid:],
            depth + 1,
            node.node_id,
            run_params,
        )
        return node

    def _leaf_for_state(self, state: int) -> CATNode:
        node = self.root
        x = self.features[state]
        while not node.is_leaf:
            assert node.split_variable is not None
            assert node.split_threshold is not None
            if x[node.split_variable] <= node.split_threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node

    def _refresh_leaf_cache(self) -> None:
        leaves = [node for node in self.nodes if node.is_leaf]
        leaves.sort(key=lambda n: n.node_id)
        self.leaves = leaves
        self.leaf_id_to_index = {node.node_id: i for i, node in enumerate(leaves)}
        self.state_to_leaf = np.empty(self.model.state_dim, dtype=int)
        for state in range(self.model.state_dim):
            leaf = self._leaf_for_state(state)
            self.state_to_leaf[state] = self.leaf_id_to_index[leaf.node_id]

    def _choose_action(
        self, leaf_index: int, run_params: dict, greedy: bool = False
    ) -> int:
        if (not greedy) and simulation.rng.random() < run_params["exploration_prob"]:
            return int(simulation.rng.integers(self.model.action_dim))
        q = self.leaves[leaf_index].q_values
        return int(simulation.rng.choice(np.flatnonzero(q == q.max())))

    def _abstract_step(
        self,
        state: int,
        action: int,
        run_params: dict,
    ) -> tuple[int, float, int]:
        start_leaf = int(self.state_to_leaf[state])
        total_reward = 0.0
        discount_power = 1.0
        current_state = int(state)
        steps = 0

        if not run_params["extended_actions"]:
            next_state, reward = generate_sars(self.model, current_state, action)
            return int(next_state), float(reward), 1

        for _ in range(int(run_params["max_primitive_steps_per_abstract_action"])):
            next_state, reward = generate_sars(self.model, current_state, action)
            total_reward += discount_power * float(reward)
            discount_power *= self.discount
            steps += 1
            current_state = int(next_state)
            if int(self.state_to_leaf[current_state]) != start_leaf:
                break
        return current_state, float(total_reward), steps

    def _run_episode(
        self,
        run_params: dict,
        update_q: bool,
        collect_td: bool,
        greedy: bool = False,
    ) -> tuple[float, float, int, dict[int, list[float]]]:
        state = int(simulation.rng.integers(self.model.state_dim))
        total_reward = 0.0
        td_abs_sum = 0.0
        td_count = 0
        environment_steps = 0
        eval_td: dict[int, list[float]] = defaultdict(list)

        for _ in range(run_params["episode_length"]):
            leaf_index = int(self.state_to_leaf[state])
            action = self._choose_action(leaf_index, run_params, greedy=greedy)
            next_state, reward, primitive_steps = self._abstract_step(
                state, action, run_params
            )
            next_leaf_index = int(self.state_to_leaf[next_state])

            q = self.leaves[leaf_index].q_values
            target = (
                reward
                + (self.discount**primitive_steps)
                * self.leaves[next_leaf_index].q_values.max()
            )
            delta = target - q[action]

            if update_q:
                q[action] += self.next_alpha(run_params) * delta

            if collect_td:
                eval_td[leaf_index].append(float(delta))

            total_reward += float(reward)
            td_abs_sum += abs(float(delta))
            td_count += 1
            environment_steps += int(primitive_steps)
            state = int(next_state)

        return total_reward, td_abs_sum / max(1, td_count), environment_steps, eval_td

    def _evaluate_abstraction(
        self, run_params: dict
    ) -> tuple[dict[int, list[float]], int]:
        saved_exploration = run_params["exploration_prob"]
        eval_params = run_params.copy()
        eval_params["exploration_prob"] = min(0.05, float(saved_exploration))
        aggregated: dict[int, list[float]] = defaultdict(list)
        environment_steps = 0
        for _ in range(int(run_params["evaluation_episodes"])):
            _, _, episode_steps, td_errors = self._run_episode(
                eval_params,
                update_q=False,
                collect_td=True,
                greedy=False,
            )
            environment_steps += episode_steps
            for leaf, values in td_errors.items():
                aggregated[leaf].extend(values)
        return aggregated, environment_steps

    def _success_rate(self, run_params: dict) -> float:
        window = self.infos["rewards"][-int(run_params["refine_check_interval"]) :]
        if not window:
            return 0.0
        threshold = float(run_params["success_reward_threshold"])
        return float(np.mean([reward >= threshold for reward in window]))

    def _needs_refinement(self, step: int, run_params: dict) -> bool:
        if len(self.leaves) >= int(run_params["max_leaves"]):
            return False
        interval = int(run_params["refine_check_interval"])
        if interval <= 0 or (step + 1) % interval != 0:
            return False
        if run_params["force_refinement"]:
            return True
        return self._success_rate(run_params) < float(run_params["success_threshold"])

    def _split_quality(
        self,
        node: CATNode,
        variable: int,
        threshold: float,
        concrete_td: dict[int, list[float]],
        run_params: dict,
    ) -> float:
        states = np.asarray(node.states, dtype=int)
        left_mask = self.features[states, variable] <= threshold
        if not np.any(left_mask) or not np.any(~left_mask):
            return -np.inf

        min_fraction = float(run_params["split_balance_min_fraction"])
        if min(np.mean(left_mask), np.mean(~left_mask)) < min_fraction:
            return -np.inf

        left_values: list[float] = []
        right_values: list[float] = []
        for state in states[left_mask]:
            left_values.extend(concrete_td.get(int(state), []))
        for state in states[~left_mask]:
            right_values.extend(concrete_td.get(int(state), []))

        min_samples = int(run_params["min_td_errors_for_refine"])
        if len(left_values) < min_samples or len(right_values) < min_samples:
            return -np.inf

        left_mean = float(np.mean(left_values))
        right_mean = float(np.mean(right_values))
        pooled_std = float(np.std(left_values + right_values)) + 1e-12
        return abs(left_mean - right_mean) / pooled_std

    def _collect_concrete_td_for_leaf(
        self,
        node: CATNode,
        run_params: dict,
        episodes: int,
    ) -> tuple[dict[int, list[float]], int]:
        eval_params = run_params.copy()
        eval_params["exploration_prob"] = min(
            0.05, float(run_params["exploration_prob"])
        )
        concrete_td: dict[int, list[float]] = defaultdict(list)
        environment_steps = 0

        for _ in range(episodes):
            state = int(simulation.rng.integers(self.model.state_dim))
            for _ in range(run_params["episode_length"]):
                leaf_index = int(self.state_to_leaf[state])
                action = self._choose_action(leaf_index, eval_params, greedy=False)
                next_state, reward, primitive_steps = self._abstract_step(
                    state, action, eval_params
                )
                next_leaf_index = int(self.state_to_leaf[next_state])
                q = self.leaves[leaf_index].q_values
                target = (
                    reward
                    + (self.discount**primitive_steps)
                    * self.leaves[next_leaf_index].q_values.max()
                )
                delta = float(target - q[action])
                if leaf_index == self.leaf_id_to_index[node.node_id]:
                    concrete_td[state].append(delta)
                environment_steps += int(primitive_steps)
                state = int(next_state)
        return concrete_td, environment_steps

    def _best_refinement_for_leaf(
        self,
        node: CATNode,
        run_params: dict,
    ) -> tuple[tuple[int, float, float] | None, int]:
        if len(node.states) <= 1:
            return None, 0

        concrete_td, environment_steps = self._collect_concrete_td_for_leaf(
            node,
            run_params,
            episodes=max(1, int(run_params["evaluation_episodes"]) // 2),
        )

        best: tuple[float, int, float] | None = None
        states = np.asarray(node.states, dtype=int)
        for variable in range(self.features.shape[1]):
            values = np.unique(self.features[states, variable])
            if len(values) <= 1:
                continue
            thresholds = sorted(
                set(float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75))
            )
            for threshold in thresholds:
                quality = self._split_quality(
                    node, variable, threshold, concrete_td, run_params
                )
                if best is None or quality > best[0]:
                    best = (quality, variable, threshold)

        if best is None or not np.isfinite(best[0]):
            return None, environment_steps
        return (int(best[1]), float(best[2]), float(best[0])), environment_steps

    def _refine_from_dispersion(
        self,
        td_dispersion: dict[int, list[float]],
        run_params: dict,
    ) -> tuple[bool, int]:
        min_samples = int(run_params["min_td_errors_for_refine"])
        candidates = []
        for leaf_index, values in td_dispersion.items():
            if leaf_index >= len(self.leaves):
                continue
            node = self.leaves[leaf_index]
            if len(node.states) <= 1 or len(values) < min_samples:
                continue
            std = float(np.std(values))
            if std >= float(run_params["td_std_threshold"]):
                candidates.append((std, node))

        if not candidates:
            return False, 0

        candidates.sort(key=lambda x: x[0], reverse=True)
        split_budget = min(
            int(run_params["top_k_unstable"]),
            max(0, int(run_params["max_leaves"]) - len(self.leaves)),
            len(candidates),
        )
        if split_budget <= 0:
            return False, 0

        refined = False
        environment_steps = 0
        for _, node in candidates[:split_budget]:
            refinement, refinement_steps = self._best_refinement_for_leaf(
                node, run_params
            )
            environment_steps += refinement_steps
            if refinement is None:
                continue
            variable, threshold, quality = refinement
            states = np.asarray(node.states, dtype=int)
            left_states = states[self.features[states, variable] <= threshold].tolist()
            right_states = states[self.features[states, variable] > threshold].tolist()
            if not left_states or not right_states:
                continue

            node.split_variable = variable
            node.split_threshold = threshold
            node.left = self._new_node(left_states, node.depth + 1, parent=node.node_id)
            node.right = self._new_node(
                right_states, node.depth + 1, parent=node.node_id
            )
            if run_params["inherit_q_on_split"]:
                node.left.q_values = node.q_values.copy()
                node.right.q_values = node.q_values.copy()

            refined = True
            self.infos["refinement_steps"].append(len(self.infos["rewards"]))
            self.infos["split_scores"].append(float(quality))

            if run_params["verbose_splits"]:
                print(
                    f"CAT split leaf={node.node_id} variable={variable} "
                    f"threshold={threshold:.6g} quality={quality:.6g}"
                )

        if refined:
            self._refresh_leaf_cache()
        return refined, environment_steps

    def _partition_from_tree(self) -> Partition:
        partition = Partition(self.model)
        partition.states_in_region = [list(node.states) for node in self.leaves]
        return partition

    def _q_array(self) -> np.ndarray:
        q = np.zeros((len(self.leaves), self.model.action_dim), dtype=float)
        for i, node in enumerate(self.leaves):
            q[i] = node.q_values
        return q

    def _full_q_value(self) -> np.ndarray:
        q_value = np.zeros((self.model.state_dim, self.model.action_dim), dtype=float)
        for state in range(self.model.state_dim):
            q_value[state] = self.leaves[int(self.state_to_leaf[state])].q_values
        return q_value

    def _moving_average(self, key: str, window_size: int) -> float:
        values = self.infos[key][-window_size:]
        if not values:
            return float("nan")
        return float(np.mean(values))

    def _record_progress(self, step: int, run_params: dict) -> None:
        q_value = self._full_q_value()
        error = distance_to_optimal(q_value, self.model, self.discount, norm_method=2)
        self.infos["error_to_optimal"].append(float(error))
        reward_ma = self._moving_average("rewards", run_params["reward_ma_window"])
        td_ma = self._moving_average("td_errors", run_params["td_ma_window"])
        progress = 100.0 * (step + 1) / run_params["num_episodes"]
        print(
            f"[{step + 1:8,d}/{run_params['num_episodes']:,d} | {progress:6.2f}%] "
            f"leaves={len(self.leaves):6d} | "
            f"err_opt={error:10.6f} | "
            f"alpha={self.current_alpha(run_params):9.6f} | "
            f"reward_ma{run_params['reward_ma_window']}={reward_ma:8.4f} | "
            f"td_ma{run_params['td_ma_window']}={td_ma:10.6f}"
        )

    def run(self, run_params: dict) -> None:
        params = require_params(run_params, REQUIRED_RUN_PARAMS)

        simulation.rng = np.random.default_rng(params["random_seed"])
        eval_rng = np.random.default_rng(int(params["random_seed"]) + 1_000_003)
        self.steps_done_learning = 0
        self.features = self._build_features(params)
        self.nodes: list[CATNode] = []
        self.root = self._build_initial_tree(
            list(range(self.model.state_dim)),
            feature=0,
            depth=0,
            parent=None,
            run_params=params,
        )
        self._refresh_leaf_cache()
        self._environment_steps = 0

        self.infos: dict[str, list[Any]] = {
            "environment_steps": [],
            "error_to_optimal": [],
            "rewards": [],
            "greedy_rewards": [],
            "td_errors": [],
            "number_of_regions": [],
            "refinement_steps": [],
            "split_scores": [],
        }
        self.reward_history = self.infos["rewards"]
        self.td_error_history = self.infos["td_errors"]

        start_time = time.time()

        for step in range(params["num_episodes"]):
            total_reward, avg_td_error, environment_steps, _ = self._run_episode(
                params,
                update_q=True,
                collect_td=False,
                greedy=False,
            )
            self._environment_steps += environment_steps
            training_environment_steps = self._environment_steps
            self.infos["rewards"].append(
                float(total_reward / max(1, params["episode_length"]))
            )
            self.infos["td_errors"].append(float(avg_td_error))

            if params["verbose"] and (
                step % params["log_every"] == 0 or step + 1 == params["num_episodes"]
            ):
                self._record_progress(step, params)

            if (
                self._needs_refinement(step, params)
                and step + 1 < params["num_episodes"]
            ):
                td_dispersion, evaluation_steps = self._evaluate_abstraction(params)
                self._environment_steps += evaluation_steps
                _, refinement_steps = self._refine_from_dispersion(
                    td_dispersion, params
                )
                self._environment_steps += refinement_steps

            self.infos["environment_steps"].append(training_environment_steps)
            self.infos["number_of_regions"].append(len(self.leaves))
            self.infos["greedy_rewards"].append(
                greedy_episode_reward(
                    self.model,
                    self._full_q_value(),
                    params["episode_length"],
                    eval_rng,
                )
            )

        self.partition = self._partition_from_tree()
        self.q_value = self._full_q_value()
        self.contracted_q_value = self._q_array()
        self.value = self.q_value.max(axis=1)
        self.policy = self.q_value.argmax(axis=1)
        self.n_regions = self.partition.n_regions
        self.runtime = time.time() - start_time
