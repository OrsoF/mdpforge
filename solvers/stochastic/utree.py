"""
U-Tree-style adaptive state abstraction for primitive-action Q-learning.

Reference target: Jonsson & Barto (2000), adapting McCallum's U-Tree.
This implementation is a flat, primitive-action version designed to match the
same Solver interface as the existing Q-learning solvers in this project.

Key choices for compatibility with the current codebase:
  - The generic model exposes finite integer states, not named features.
  - If run_params["state_shape"] is given, state ids are decoded into a factored
    feature vector with np.unravel_index-compatible mixed radix semantics.
  - Otherwise the sole feature is the integer state id. In that fallback, U-Tree
    still runs, but splits are just thresholds over state indices.
  - The tree uses binary threshold distinctions rather than full McCallum-style
    fringe subtrees over observation/history dimensions. This keeps the code
    self-contained and comparable to the attached aggregated-Q baselines.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
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
    "initial_q",
    "model_sweeps_per_episode",
    "model_sweeps_after_refine",
    "refine_interval",
    "max_leaves",
    "ks_threshold",
    "min_leaf_action_samples",
    "min_child_samples",
    "candidate_quantiles",
    "max_instances_per_leaf",
    "max_split_instances_per_leaf",
    "state_shape",
    "initial_bins_per_variable",
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
class TransitionInstance:
    state: int
    action: int
    reward: float
    next_state: int
    leaf: int
    next_leaf: int
    td_target: float


@dataclass
class UTreeNode:
    node_id: int
    states: list[int]
    depth: int = 0
    split_feature: int | None = None
    split_threshold: float | None = None
    left: "UTreeNode | None" = None
    right: "UTreeNode | None" = None
    parent: int | None = None
    q_values: np.ndarray | None = None
    instances: list[TransitionInstance] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class Solver(GenericSolver):
    def __init__(self, model: GenericModel, discount: float, final_precision: float):
        self.model = model
        self.model._normalize_reward_matrix()
        self.discount = discount
        self.final_precision = final_precision
        self.name = "U-Tree Q-Learning"
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
    ) -> UTreeNode:
        node = UTreeNode(
            node_id=len(self.nodes),
            states=list(states),
            depth=depth,
            parent=parent,
            q_values=np.full(self.model.action_dim, self.initial_q, dtype=float),
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
    ) -> UTreeNode:
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

        node.split_feature = feature
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
        node: UTreeNode | None = None,
    ) -> UTreeNode:
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

        node.split_feature = feature
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

    def _leaf_for_state(self, state: int) -> UTreeNode:
        node = self.root
        x = self.features[state]
        while not node.is_leaf:
            assert node.split_feature is not None
            assert node.split_threshold is not None
            if x[node.split_feature] <= node.split_threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node

    def _refresh_leaf_cache(self) -> None:
        leaves = [node for node in self.nodes if node.is_leaf]
        leaves.sort(key=lambda node: node.node_id)
        self.leaves = leaves
        self.leaf_id_to_index = {node.node_id: i for i, node in enumerate(leaves)}
        self.state_to_leaf = np.empty(self.model.state_dim, dtype=int)
        for state in range(self.model.state_dim):
            leaf = self._leaf_for_state(state)
            self.state_to_leaf[state] = self.leaf_id_to_index[leaf.node_id]

    def _leaf_q_array(self) -> np.ndarray:
        q = np.zeros((len(self.leaves), self.model.action_dim), dtype=float)
        for i, node in enumerate(self.leaves):
            q[i] = node.q_values
        return q

    def _choose_action(self, state: int, run_params: dict) -> int:
        leaf_index = int(self.state_to_leaf[state])
        if simulation.rng.random() < run_params["exploration_prob"]:
            return int(simulation.rng.integers(self.model.action_dim))
        q = self.leaves[leaf_index].q_values
        return int(simulation.rng.choice(np.flatnonzero(q == q.max())))

    def _generate_trajectory(
        self, run_params: dict
    ) -> list[tuple[int, int, float, int]]:
        state = int(simulation.rng.integers(self.model.state_dim))
        trajectory = []
        for _ in range(run_params["episode_length"]):
            action = self._choose_action(state, run_params)
            next_state, reward = generate_sars(self.model, state, action)
            trajectory.append((state, action, float(reward), int(next_state)))
            state = int(next_state)
        return trajectory

    @staticmethod
    def _ks_statistic(
        x: list[float] | np.ndarray, y: list[float] | np.ndarray
    ) -> float:
        x = np.sort(np.asarray(x, dtype=float))
        y = np.sort(np.asarray(y, dtype=float))
        if len(x) == 0 or len(y) == 0:
            return 0.0
        values = np.sort(np.unique(np.concatenate([x, y])))
        cdf_x = np.searchsorted(x, values, side="right") / len(x)
        cdf_y = np.searchsorted(y, values, side="right") / len(y)
        return float(np.max(np.abs(cdf_x - cdf_y)))

    def _model_value_iteration_sweep(self) -> None:
        q_array = self._leaf_q_array()
        value = q_array.max(axis=1)

        transition_counts: dict[tuple[int, int], dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        reward_sums: dict[tuple[int, int], float] = defaultdict(float)
        visit_counts: dict[tuple[int, int], int] = defaultdict(int)

        for leaf_index, node in enumerate(self.leaves):
            for inst in node.instances:
                key = (leaf_index, inst.action)
                transition_counts[key][inst.next_leaf] += 1
                reward_sums[key] += inst.reward
                visit_counts[key] += 1

        for leaf_index, node in enumerate(self.leaves):
            for action in range(self.model.action_dim):
                key = (leaf_index, action)
                n = visit_counts[key]
                if n <= 0:
                    continue
                expected_reward = reward_sums[key] / n
                expected_next_value = 0.0
                for next_leaf, count in transition_counts[key].items():
                    expected_next_value += (count / n) * value[next_leaf]
                node.q_values[action] = (
                    expected_reward + self.discount * expected_next_value
                )

    def _trim_leaf_instances(self, node: UTreeNode, run_params: dict) -> None:
        max_instances = int(run_params["max_instances_per_leaf"])
        if max_instances > 0 and len(node.instances) > max_instances:
            del node.instances[: len(node.instances) - max_instances]

    def _split_instances(
        self, node: UTreeNode, run_params: dict
    ) -> list[TransitionInstance]:
        max_instances = int(run_params["max_split_instances_per_leaf"])
        if max_instances > 0 and len(node.instances) > max_instances:
            return node.instances[-max_instances:]
        return node.instances

    def _q_learning_update(
        self,
        trajectory: list[tuple[int, int, float, int]],
        run_params: dict,
        record_instances: bool,
    ) -> tuple[float, int]:
        td_abs_sum = 0.0
        td_count = 0

        for state, action, reward, next_state in reversed(trajectory):
            leaf = self.leaves[int(self.state_to_leaf[state])]
            next_leaf_index = int(self.state_to_leaf[next_state])
            next_value = float(self.leaves[next_leaf_index].q_values.max())
            target = reward + self.discount * next_value
            delta = target - leaf.q_values[action]
            leaf.q_values[action] += self.next_alpha(run_params) * delta

            if record_instances:
                inst = TransitionInstance(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    leaf=int(self.state_to_leaf[state]),
                    next_leaf=next_leaf_index,
                    td_target=float(target),
                )
                leaf.instances.append(inst)
                self._trim_leaf_instances(leaf, run_params)

            td_abs_sum += abs(float(delta))
            td_count += 1

        return td_abs_sum, td_count

    def _candidate_splits(
        self, states: list[int], run_params: dict
    ) -> list[tuple[int, float]]:
        states_array = np.asarray(states, dtype=int)
        candidates: list[tuple[int, float]] = []
        quantiles = tuple(float(q) for q in run_params["candidate_quantiles"])

        for feature in range(self.features.shape[1]):
            values = self.features[states_array, feature]
            unique_values = np.unique(values)
            if len(unique_values) <= 1:
                continue
            for quantile in quantiles:
                threshold = float(np.quantile(values, quantile))
                left = values <= threshold
                if np.any(left) and np.any(~left):
                    candidates.append((feature, threshold))
        return candidates

    def _split_score(
        self,
        node: UTreeNode,
        feature: int,
        threshold: float,
        run_params: dict,
    ) -> float:
        min_leaf_action_samples = int(run_params["min_leaf_action_samples"])
        min_child_samples = int(run_params["min_child_samples"])
        score = 0.0

        node_instances = self._split_instances(node, run_params)
        if len(node_instances) < min_leaf_action_samples:
            return 0.0

        for action in range(self.model.action_dim):
            action_instances = [
                inst for inst in node_instances if inst.action == action
            ]
            if len(action_instances) < min_leaf_action_samples:
                continue

            parent_targets = [inst.td_target for inst in action_instances]
            left_targets = []
            right_targets = []
            for inst in action_instances:
                if self.features[inst.state, feature] <= threshold:
                    left_targets.append(inst.td_target)
                else:
                    right_targets.append(inst.td_target)

            if (
                len(left_targets) < min_child_samples
                or len(right_targets) < min_child_samples
            ):
                continue

            score += self._ks_statistic(parent_targets, left_targets)
            score += self._ks_statistic(parent_targets, right_targets)

        return float(score)

    def _refine_tree(self, run_params: dict) -> bool:
        max_leaves = run_params["max_leaves"]
        if max_leaves is not None and len(self.leaves) >= int(max_leaves):
            return False

        best: tuple[float, UTreeNode, int, float] | None = None
        for node in self.leaves:
            if len(node.states) <= 1:
                continue
            for feature, threshold in self._candidate_splits(node.states, run_params):
                score = self._split_score(node, feature, threshold, run_params)
                if best is None or score > best[0]:
                    best = (score, node, feature, threshold)

        if best is None:
            return False

        score, node, feature, threshold = best
        if score <= float(run_params["ks_threshold"]):
            return False

        states = np.asarray(node.states, dtype=int)
        left_states = states[self.features[states, feature] <= threshold].tolist()
        right_states = states[self.features[states, feature] > threshold].tolist()
        if not left_states or not right_states:
            return False

        node.split_feature = int(feature)
        node.split_threshold = float(threshold)
        node.left = self._new_node(left_states, node.depth + 1, parent=node.node_id)
        node.right = self._new_node(right_states, node.depth + 1, parent=node.node_id)
        node.left.q_values = node.q_values.copy()
        node.right.q_values = node.q_values.copy()

        # Redistribute stored instances to keep the local empirical model useful.
        old_instances = node.instances
        node.instances = []
        for inst in old_instances:
            child = (
                node.left
                if self.features[inst.state, feature] <= threshold
                else node.right
            )
            child.instances.append(inst)

        self._trim_leaf_instances(node.left, run_params)
        self._trim_leaf_instances(node.right, run_params)

        self._refresh_leaf_cache()
        self.infos["refinement_steps"].append(len(self.infos["rewards"]))
        self.infos["split_scores"].append(float(score))

        for _ in range(int(run_params["model_sweeps_after_refine"])):
            self._model_value_iteration_sweep()

        if run_params["verbose_splits"]:
            print(
                f"U-Tree split leaf={node.node_id} feature={feature} "
                f"threshold={threshold:.6g} score={score:.6g} "
                f"leaves={len(self.leaves)}"
            )
        return True

    def _partition_from_tree(self) -> Partition:
        partition = Partition(self.model)
        partition.states_in_region = [list(node.states) for node in self.leaves]
        return partition

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
        self.initial_q = float(params["initial_q"])
        self.features = self._build_features(params)
        self.nodes: list[UTreeNode] = []
        self.root = self._build_initial_tree(
            list(range(self.model.state_dim)),
            feature=0,
            depth=0,
            parent=None,
            run_params=params,
        )
        self._refresh_leaf_cache()

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
            trajectory = self._generate_trajectory(params)
            avg_reward = float(np.mean([reward for _, _, reward, _ in trajectory]))
            self.infos["rewards"].append(avg_reward)
            self.infos["environment_steps"].append(
                (step + 1) * params["episode_length"]
            )

            td_abs_sum = 0.0
            td_count = 0
            for replay_pass in range(params["num_replay_passes"]):
                pass_td_sum, pass_td_count = self._q_learning_update(
                    trajectory,
                    params,
                    record_instances=replay_pass == 0,
                )
                td_abs_sum += pass_td_sum
                td_count += pass_td_count

            for _ in range(int(params["model_sweeps_per_episode"])):
                self._model_value_iteration_sweep()

            self.infos["td_errors"].append(td_abs_sum / max(1, td_count))
            self.infos["number_of_regions"].append(len(self.leaves))

            if params["verbose"] and (
                step % params["log_every"] == 0 or step + 1 == params["num_episodes"]
            ):
                self._record_progress(step, params)

            if (
                params["refine_interval"] > 0
                and (step + 1) % params["refine_interval"] == 0
                and step + 1 < params["num_episodes"]
            ):
                self._refine_tree(params)

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
        self.contracted_q_value = self._leaf_q_array()
        self.value = self.q_value.max(axis=1)
        self.policy = self.q_value.argmax(axis=1)
        self.n_regions = self.partition.n_regions
        self.runtime = time.time() - start_time
