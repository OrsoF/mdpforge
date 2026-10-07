import time

import numpy as np
from solvers_agg.solvers.common import (
    greedy_episode_reward,
    require_params,
    seed_rng,
    standard_infos,
    state_shape_features,
)

import utils.simulation as simulation
from core.model import NUMPY, GenericModel
from core.partition import Partition

REQUIRED_RUN_PARAMS = (
    "num_episodes",
    "episode_length",
    "random_seed",
    "state_shape",
    "learning_rate",
    "exploration_prob",
    "replay_capacity",
    "batch_size",
    "target_update",
)


class Solver:
    def __init__(self, model: GenericModel, discount: float, final_precision: float):
        self.model = model
        self.discount = discount
        self.final_precision = final_precision
        self.model._convert_model(NUMPY)
        self.model._normalize_reward_matrix()
        self.name = "DQN"

    def _features(self, state_shape):
        x = state_shape_features(self.model.state_dim, state_shape)
        x = x / np.maximum(x.max(axis=0, keepdims=True), 1.0)
        return np.c_[x, np.ones(self.model.state_dim)]

    def _sample(self, state: int, action: int):
        p = self.model.transition_matrix[action, state]
        next_state = int(simulation.rng.choice(self.model.state_dim, p=p))
        return next_state, float(self.model.reward_matrix[state, action])

    @staticmethod
    def _mlp(x, theta):
        w1, b1, w2, b2 = theta
        z = x @ w1 + b1
        h = np.maximum(z, 0.0)
        return h @ w2 + b2, z, h

    def run(self, run_params: dict):
        start_time = time.time()
        params = require_params(run_params, REQUIRED_RUN_PARAMS)
        seed_rng(params["random_seed"])
        eval_rng = np.random.default_rng(int(params["random_seed"]) + 1_000_003)

        x = self._features(params["state_shape"])
        hidden_dim = int(params.get("hidden_dim", 0))
        if hidden_dim:
            theta = [
                simulation.rng.normal(
                    0.0,
                    np.sqrt(2.0 / x.shape[1]),
                    (x.shape[1], hidden_dim),
                ),
                np.zeros(hidden_dim),
                simulation.rng.normal(
                    0.0,
                    0.01,
                    (hidden_dim, self.model.action_dim),
                ),
                np.zeros(self.model.action_dim),
            ]
            target = [p.copy() for p in theta]
            m = [np.zeros_like(p) for p in theta]
            v = [np.zeros_like(p) for p in theta]
        else:
            w = np.zeros((x.shape[1], self.model.action_dim))
            target_w = w.copy()
        replay = []
        pos = 0
        updates = 0

        self.infos = standard_infos()
        self.reward_history = self.infos["rewards"]
        self.td_error_history = self.infos["td_errors"]

        for ep in range(params["num_episodes"]):
            state = int(simulation.rng.integers(self.model.state_dim))
            rewards = []
            td = []

            for _ in range(params["episode_length"]):
                q = self._mlp(x[state], theta)[0] if hidden_dim else x[state] @ w
                action = (
                    int(simulation.rng.integers(self.model.action_dim))
                    if simulation.rng.random() < params["exploration_prob"]
                    else int(q.argmax())
                )
                next_state, reward = self._sample(state, action)
                item = (state, action, reward, next_state)
                if len(replay) < params["replay_capacity"]:
                    replay.append(item)
                else:
                    replay[pos] = item
                    pos = (pos + 1) % params["replay_capacity"]
                rewards.append(reward)
                state = next_state

                if len(replay) < params["batch_size"]:
                    continue

                batch = simulation.rng.choice(
                    len(replay), size=params["batch_size"], replace=False
                )
                s = np.asarray([replay[i][0] for i in batch], dtype=int)
                a = np.asarray([replay[i][1] for i in batch], dtype=int)
                r = np.asarray([replay[i][2] for i in batch], dtype=float)
                ns = np.asarray([replay[i][3] for i in batch], dtype=int)
                if hidden_dim:
                    next_q = self._mlp(x[ns], target)[0]
                    pred_q, z, h = self._mlp(x[s], theta)
                else:
                    next_q = x[ns] @ target_w
                    pred_q = x[s] @ w
                y = r + self.discount * next_q.max(axis=1)
                pred = pred_q[np.arange(len(batch)), a]
                err = y - pred
                if hidden_dim:
                    delta = np.zeros_like(pred_q)
                    delta[np.arange(len(batch)), a] = np.clip(err, -1.0, 1.0) / len(
                        batch
                    )
                    dz = (delta @ theta[2].T) * (z > 0.0)
                    grad = [
                        x[s].T @ dz,
                        dz.sum(axis=0),
                        h.T @ delta,
                        delta.sum(axis=0),
                    ]
                    beta1, beta2 = params["adam_betas"]
                    for i in range(len(theta)):
                        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i]
                        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] ** 2
                        mh = m[i] / (1.0 - beta1 ** (updates + 1))
                        vh = v[i] / (1.0 - beta2 ** (updates + 1))
                        theta[i] += (
                            params["learning_rate"]
                            * mh
                            / (np.sqrt(vh) + params["adam_eps"])
                        )
                else:
                    for aa in range(self.model.action_dim):
                        mask = a == aa
                        if np.any(mask):
                            w[:, aa] += (
                                params["learning_rate"]
                                * (x[s[mask]].T @ err[mask])
                                / len(batch)
                            )
                td.append(float(np.abs(err).mean()))
                updates += 1
                if updates % params["target_update"] == 0:
                    if hidden_dim:
                        target = [p.copy() for p in theta]
                    else:
                        target_w = w.copy()

            self.reward_history.append(float(np.mean(rewards)))
            self.td_error_history.append(float(np.mean(td)) if td else 0.0)
            self.infos["environment_steps"].append((ep + 1) * params["episode_length"])
            self.infos["number_of_regions"].append(self.model.state_dim)
            q_value = self._mlp(x, theta)[0] if hidden_dim else x @ w
            self.infos["greedy_rewards"].append(
                greedy_episode_reward(
                    self.model,
                    q_value,
                    params["episode_length"],
                    eval_rng,
                )
            )

        q = self._mlp(x, theta)[0] if hidden_dim else x @ w
        self.q_value = q
        self.contracted_q_value = q
        self.value = q.max(axis=1)
        self.policy = q.argmax(axis=1)
        self.partition = Partition.from_regions(
            self.model,
            [[s] for s in range(self.model.state_dim)],
        )
        self.n_regions = self.partition.n_regions
        self.runtime = time.time() - start_time
