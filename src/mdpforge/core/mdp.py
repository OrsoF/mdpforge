"""Finite MDPs built by a model recipe or supplied through matrices."""

import numpy as np
from scipy.sparse import csr_matrix, issparse

from mdpforge.utils.persistence import (
    get_cached_value_function,
    load_model,
    save_model,
)


class MDP:
    """Finite MDP with shared construction, validation and cache utilities.

    Generated models subclass MDP and implement _build_model(); call
    create_model() to build or load their matrices. Use from_matrices() when
    transitions and rewards are already available.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = f"{state_dim}_{action_dim}_{self.__class__.__name__.lower()}"

        self.transition_matrix: list[csr_matrix]
        self.reward_matrix: np.ndarray

    @classmethod
    def from_matrices(cls, name: str, transition_matrix, reward_matrix) -> "MDP":
        """Create a validated MDP with CSR transitions and NumPy rewards.

        Transitions are ordered (action, current_state, next_state), with one
        (S, S) matrix per action. Rewards have shape (S, A) and determine the
        dimensions. The returned model is ready for solvers and benchmarks.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MDP name must be a nonempty string.")

        rewards = (
            reward_matrix.toarray()
            if issparse(reward_matrix)
            else np.asarray(reward_matrix)
        )
        rewards = np.asarray(rewards, dtype=float)
        if rewards.ndim != 2:
            raise ValueError("reward_matrix must have shape (state_dim, action_dim).")

        state_dim, action_dim = rewards.shape
        model = cls(state_dim=state_dim, action_dim=action_dim)
        model.name = name
        model.reward_matrix = rewards
        model.transition_matrix = [
            csr_matrix(matrix, dtype=float) for matrix in transition_matrix
        ]
        model.test_model()
        return model

    def create_model(
        self,
        check_transition: bool = False,
        normalize_reward: bool = False,
        save: bool = True,
    ):
        """Load cached matrices or build, optionally validate, normalize and save."""
        already_built = self._is_model_built() or load_model(self)
        if not already_built:
            self._build_model()
        self.transition_matrix = [
            csr_matrix(matrix) for matrix in self.transition_matrix
        ]
        if already_built:
            return
        if normalize_reward:
            self._normalize_reward_matrix()
        if check_transition:
            self.test_model()
            print("Transition is stochastic.")
        if save:
            save_model(self)

    def _build_model(self):
        raise NotImplementedError(
            "Use MDP.from_matrices() or implement _build_model() in a subclass."
        )

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

        self.transition_matrix = [
            matrix[permutation, :][:, permutation] for matrix in self.transition_matrix
        ]

        self.reward_matrix = self.reward_matrix[permutation, :]
        return permutation

    def get_transition_density(self) -> float:
        """Return the density of the transition matrix: |T|/S^2/A"""
        return (
            sum(matrix.nnz for matrix in self.transition_matrix)
            / self.state_dim**2
            / self.action_dim
        )

    def get_reward_density(self) -> float:
        return np.count_nonzero(self.reward_matrix) / self.state_dim / self.action_dim

    def _is_model_built(self) -> bool:
        return hasattr(self, "transition_matrix") and hasattr(self, "reward_matrix")

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

    def optimal_value_function(self, discount: float) -> np.ndarray:
        """Load cached discounted values or solve with VI at precision 1e-3."""
        from mdpforge.utils.exact_value_function import compute_value_function

        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        return get_cached_value_function(
            f"{discount}_{self.name}",
            lambda: compute_value_function(self, discount, 1e-3),
        )
