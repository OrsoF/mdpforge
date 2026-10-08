import numpy as np
import pytest
from scipy.sparse import csr_array

from mdpforge.core.model import GenericModel
from mdpforge.models.rooms import Model as Rooms
from mdpforge.models.schoolboy import Model as Schoolboy
from mdpforge.models.swim import Model as Swim
from mdpforge.utils import persistence


class FourStateModel(GenericModel):
    """Small synthetic MDP for validation and partition invariants."""

    def _build_model(self):
        self.transition_matrix = [
            csr_array(np.eye(4)),
            csr_array(np.roll(np.eye(4), 1, axis=1)),
        ]
        self.reward_matrix = np.array([[1, 0.5], [1, 1], [2, 1.5], [3, 2]])


@pytest.fixture
def isolated_model_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "SAVED_MODELS_PATH", tmp_path / "models")


@pytest.fixture
def small_model(isolated_model_cache):
    model = FourStateModel(4, 2)
    model.create_model(save=False)
    return model


@pytest.fixture(
    params=[
        pytest.param((Rooms, 101, 4, (100, 4)), id="rooms"),
        pytest.param((Schoolboy, 7, 4, (7, 3)), id="schoolboy"),
        pytest.param((Swim, 7, 4, (7, 2)), id="swim"),
    ]
)
def model_spec(request):
    return request.param
