from solvers_agg.solvers.policy_baselines import (
    A2CMixin,
    ActorCriticBase,
    run_policy_gradient,
)


class Solver(A2CMixin, ActorCriticBase):
    name = "A2C"

    def run(self, run_params: dict):
        run_policy_gradient(self, run_params, self._update)
