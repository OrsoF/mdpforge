import pytest

from models.forest import Model as Forest
from models.random_walk import Model as RandomWalk
from models.rooms import Model as Rooms
from utils import persistence


@pytest.fixture
def isolated_model_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "SAVED_MODELS_PATH", tmp_path / "models")


@pytest.fixture
def forest(isolated_model_cache):
    model = Forest(4, 2)
    model.create_model(check_transition=True, save=False)
    return model


@pytest.fixture(
    params=[
        pytest.param((Rooms, 101, 4, (100, 4)), id="rooms"),
        pytest.param((Forest, 4, 4, (4, 2)), id="forest"),
        pytest.param((RandomWalk, 7, 4, (7, 2)), id="random_walk"),
    ]
)
def model_spec(request):
    return request.param
