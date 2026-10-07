import numpy as np

from core.model import GenericModel


class ToyModel(GenericModel):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__(state_dim, action_dim)
        self.name = "toy_model"
        self.state_dim = 1
        self.action_dim = 1
        self.transition_matrix = [np.ones((1, 1))]
        self.reward_matrix = np.zeros((1, 1))

    def _build_model(self):
        return None
