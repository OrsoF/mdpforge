from abc import ABC, abstractmethod

import numpy as np

from mdpforge.core.conversion import NUMPY, SPARSE
from mdpforge.utils.persistence import (
    get_cached_value_function,
    load_model,
    model_cache_path,
    save_model,
)


class GenericModel(ABC):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = f"{state_dim}_{action_dim}_{self.__class__.__name__.lower()}"

        self.transition_matrix: list | np.ndarray
        self.reward_matrix: np.ndarray
        self.params: dict = {}

    def create_model(
        self,
        check_transition: bool = False,
        normalize_reward: bool = False,
        save: bool = True,
    ):
        """
        Function that create the reward and transition matrices.
        """
        self.pickle_file_name = "{}.pkl".format(self.name)
        pickle_file_path = model_cache_path(self.name)

        if hasattr(self, "transition_matrix") and hasattr(self, "reward_matrix"):
            return
        elif not pickle_file_path.exists():
            self._build_model()
            if normalize_reward:
                self._normalize_reward_matrix()
            if check_transition:
                self.test_model()
                print("Transition is stochastic.")

            if save:
                save_model(self)

        else:
            load_model(self)

    @abstractmethod
    def _build_model(self):
        # Define it in the specific model.
        raise NotImplementedError(
            "Subclasses of GenericModel must implement _build_model."
        )

    def lighten_model(self):
        """
        Use it to remove heavy matrices during computations.
        """
        try:
            del self.transition_matrix
            del self.reward_matrix
        except AttributeError:
            pass

    def test_model(self):
        from mdpforge.core.validation import validate_model

        validate_model(self)

    def _normalize_reward_matrix(self):
        """
        Transform the reward matrix of the model :
        0 <= self.reward_matrix <= 1 after application.
        """
        min_reward = self.reward_matrix.min()
        max_reward = self.reward_matrix.max()
        reward_range = max_reward - min_reward

        if np.isclose(reward_range, 0.0):
            self.reward_matrix = np.zeros_like(self.reward_matrix)
            return

        self.reward_matrix = (self.reward_matrix - min_reward) / reward_range

    def randomize_states(
        self, seed: int | None = None, permutation: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Randomly relabel states in-place.

        The returned permutation maps each new state index to the original state
        index used at that position. Transition rows and columns are permuted so
        the model remains equivalent up to state relabeling.
        """
        if not self._is_model_built():
            raise ValueError("Model is not built yet.")

        if permutation is None:
            rng = np.random.default_rng(seed)
            permutation = rng.permutation(self.state_dim)
        else:
            permutation = np.asarray(permutation)

        if permutation.shape != (self.state_dim,):
            raise ValueError(
                f"Permutation must have shape ({self.state_dim},), "
                f"got {permutation.shape}."
            )

        if not np.issubdtype(permutation.dtype, np.integer):
            raise ValueError("Permutation must contain integer state indices.")

        expected_states = np.arange(self.state_dim)
        if not np.array_equal(np.sort(permutation), expected_states):
            raise ValueError("Permutation must contain each state exactly once.")

        if isinstance(self.transition_matrix, list):
            self.transition_matrix = [
                matrix[permutation, :][:, permutation]
                for matrix in self.transition_matrix
            ]
        else:
            self.transition_matrix = self.transition_matrix[:, permutation, :][
                :, :, permutation
            ]

        self.reward_matrix = self.reward_matrix[permutation, :]
        return permutation

    def get_transition_density(self) -> float:
        """Return the density of the transition matrix: |T|/S^2/A"""
        return (
            sum(len(matrix.data) for matrix in self.transition_matrix)
            / self.state_dim**2
            / self.action_dim
        )

    def get_reward_density(self) -> float:
        return np.count_nonzero(self.reward_matrix) / self.state_dim / self.action_dim

    def get_model_type(self) -> str:
        if isinstance(self.transition_matrix, list):
            return SPARSE
        else:
            return NUMPY

    def _model_to_numpy(self):
        """Convert the transition and reward matrices to numpy arrays."""
        from mdpforge.core.conversion import model_to_numpy

        model_to_numpy(self)

    def _model_to_sparse(self):
        from mdpforge.core.conversion import model_to_sparse

        model_to_sparse(self)

    def _model_to_marmote(self):
        """Convert the transition and reward matrices to marmote format."""
        from mdpforge.core.conversion import model_to_marmote

        model_to_marmote(self)

    def _compute_mdpsolver_args(self) -> tuple:
        """Convert the model transition and reward to value made for MDPSolver."""
        from mdpforge.core.conversion import compute_mdpsolver_args

        return compute_mdpsolver_args(self)

    def _is_model_built(self) -> bool:
        return hasattr(self, "transition_matrix") and hasattr(self, "reward_matrix")

    def _convert_model(self, mode: str):
        from mdpforge.core.conversion import convert_model

        convert_model(self, mode)

    def get_model_main_parameters(self) -> list:
        # State dim, action dim, transition density, transition average, transition std, reward density, reward average, reward std
        result = [self.state_dim, self.action_dim, self.get_transition_density()]
        trans_avg = np.mean([matrix.mean() for matrix in self.transition_matrix])
        trans_std = np.std(
            np.concatenate([matrix.data for matrix in self.transition_matrix])
        )
        reward_density = self.get_reward_density()
        reward_avg = np.mean(self.reward_matrix)
        reward_std = np.std(self.reward_matrix)
        result += [trans_avg, trans_std, reward_density, reward_avg, reward_std]
        return result

    ###### UTILITY METHODS ######

    def plot_optimal_value(self, discount: float):
        """Plot the optimal value function of the model."""
        import matplotlib.pyplot as plt

        optimal_value = self.optimal_value_function(discount)
        square_size = int(np.sqrt(self.state_dim))
        if square_size**2 == self.state_dim:
            plt.imshow(optimal_value.reshape((square_size, square_size)), aspect="auto")
            plt.colorbar()
            plt.title("Optimal value function")
            plt.show()
        else:
            plt.plot(optimal_value)
            plt.title("Optimal value function")
            plt.xlabel("States")
            plt.ylabel("Value")
            plt.legend()
            plt.show()

    def transition_reward_policy(self, policy: np.ndarray) -> tuple:
        """Compute the transition and reward policy for a given policy."""
        from mdpforge.core.operators import compute_transition_reward_policy

        return compute_transition_reward_policy(self, policy)

    def optimal_value_function(self, discount: float) -> np.ndarray:
        """Load cached values or solve a built model with VI at precision 1e-3.

        Total reward (discount=1) requires a model with convergent value iteration.
        """

        def compute_value():
            from mdpforge.solvers.personal_vi import Solver

            solver = Solver(
                self, discount, final_precision=1e-3, mode=self.get_model_type()
            )
            solver.run()
            return solver.value

        return get_cached_value_function(
            f"{discount}_{self.name}",
            compute_value,
        )
