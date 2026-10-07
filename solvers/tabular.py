import time

import numpy as np
from solvers_agg.solvers.common import greedy_episode_reward, require_params

import utils.simulation as simulation
from core.model import GenericModel
from core.partition import Partition
from core.solver import GenericSolver
from utils.exact_value_function import distance_to_optimal

REQUIRED_RUN_PARAMS = (
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
    "final_exploration_prob",
    "exploration_decay_episodes",
)


class Solver(GenericSolver):
    def __init__(self, model: GenericModel, discount: float, final_precision: float):
        self.model = model
        self.discount = discount
        self.final_precision = final_precision
        self.model._normalize_reward_matrix()
        self.name = "Personal Q-Learning"
        self.steps_done_learning = 0

    def current_alpha(self, run_params: dict) -> float:
        return (
            run_params["learning_rate"]
            / (1 + run_params["lr_decay_scale"] * self.steps_done_learning)
            ** run_params["decay_rate"]
        )

    def next_alpha(self, run_params: dict) -> float:
        self.steps_done_learning += 1
        return self.current_alpha(run_params)

    def _moving_average(self, info_key: str, window_size: int) -> float:
        values = self.infos[info_key][-window_size:]
        return float(np.mean(values))

    def _record_progress(
        self,
        q_value: np.ndarray,
        step: int,
        avg_reward: float,
        avg_td_error: float,
        run_params: dict,
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
            f"regions={self.model.state_dim:6d} | "
            f"err_opt={error:10.6f} | "
            f"alpha={self.current_alpha(run_params):9.6f} | "
            f"reward_ma{run_params['reward_ma_window']}={reward_ma:8.4f} | "
            f"td_ma{run_params['td_ma_window']}={td_ma:10.6f} | "
        )
        self.infos["error_to_optimal"].append(error)

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

    def run(self, run_params: dict):
        start_time = time.time()
        params = require_params(run_params, REQUIRED_RUN_PARAMS)
        self.steps_done_learning = 0
        simulation.rng = np.random.default_rng(params["random_seed"])
        eval_rng = np.random.default_rng(int(params["random_seed"]) + 1_000_003)
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
        q_value = np.zeros((self.model.state_dim, self.model.action_dim))

        for step in range(params["num_episodes"]):
            trajectory = simulation.generate_trajectory(
                self.model,
                params["episode_length"],
                self._exploration_prob(params, step),
                q_value,
            )

            avg_reward = np.mean([x[2] for x in trajectory])
            self.reward_history.append(float(avg_reward))
            self.infos["environment_steps"].append(
                (step + 1) * params["episode_length"]
            )
            self.infos["number_of_regions"].append(self.model.state_dim)

            td_abs_sum = 0.0
            td_abs_max = 0.0
            td_count = 0
            discount = self.discount
            q = q_value

            for _ in range(params["num_replay_passes"]):
                for current_state, action, reward, next_state in reversed(trajectory):
                    delta_sa = (
                        reward
                        + discount * q[next_state].max()
                        - q[current_state, action]
                    )
                    q[current_state, action] += self.next_alpha(params) * delta_sa

                    abs_td = abs(delta_sa)
                    td_abs_sum += abs_td
                    td_abs_max = max(td_abs_max, abs_td)
                    td_count += 1

            avg_td_error = td_abs_sum / max(1, td_count)
            self.td_error_history.append(float(avg_td_error))
            self.infos["greedy_rewards"].append(
                greedy_episode_reward(
                    self.model,
                    q_value,
                    params["episode_length"],
                    eval_rng,
                )
            )

            if params["verbose"] and (
                step % params["log_every"] == 0 or step + 1 == params["num_episodes"]
            ):
                self._record_progress(
                    q,
                    step,
                    float(avg_reward),
                    float(avg_td_error),
                    params,
                )

        self.value = q_value.max(axis=1)
        self.policy = q_value.argmax(axis=1)
        self.q_value = q_value
        self.contracted_q_value = q_value
        self.partition = Partition.from_regions(
            self.model,
            [[state] for state in range(self.model.state_dim)],
        )
        self.n_regions = self.partition.n_regions
        self.runtime = time.time() - start_time
