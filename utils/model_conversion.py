import gymnasium as gym
import numpy as np

from core.model import GenericModel


class ModelToGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        model: GenericModel,
        max_step: int = 100,
        random_seed: int = 0,
        use_state_list: bool = False,
    ):
        super().__init__()
        self.model = model
        self.max_step = max_step
        self.rng = np.random.default_rng(seed=random_seed)
        self.use_state_list = use_state_list

        self.action_space = gym.spaces.Discrete(self.model.action_dim)
        self.observation_space = gym.spaces.Discrete(self.model.state_dim)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps_done = 0
        self.state = int(self.rng.integers(self.model.state_dim))
        obs = self.get_observation(self.state)
        info = {}
        return obs, info

    def step(self, action):
        assert hasattr(self, "state"), "Use env.reset before env.step."
        self.steps_done += 1
        transitions = self._transition_probabilities(int(action))
        reward = float(self.model.reward_matrix[self.state, int(action)])
        self.state = int(self.rng.choice(self.model.state_dim, p=transitions))
        obs = self.get_observation(self.state)
        terminated = False
        truncated = self.steps_done >= self.max_step
        info = {}
        return obs, reward, terminated, truncated, info

    def _transition_probabilities(self, action: int) -> np.ndarray:
        try:
            transitions = self.model.transition_matrix[action].getrow(self.state)
        except AttributeError:
            transitions = self.model.transition_matrix[action][self.state]

        if hasattr(transitions, "toarray"):
            transitions = transitions.toarray()

        return np.asarray(transitions, dtype=float).reshape(-1)

    def get_observation(self, state: int) -> int | tuple:
        if self.use_state_list and hasattr(self.model, "state_list"):
            return self.model.state_list[state]
        return int(state)
