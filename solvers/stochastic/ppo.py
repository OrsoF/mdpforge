from solvers_agg.solvers.policy_baselines import (
    PPO_RUN_PARAMS,
    ActorCriticBase,
    PPOMixin,
    run_policy_gradient,
)


class Solver(PPOMixin, ActorCriticBase):
    name = "PPO"

    def run(self, run_params: dict):
        run_policy_gradient(self, run_params, self._update, PPO_RUN_PARAMS)
