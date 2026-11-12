import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP
from mdpforge.solvers import discounted
from mdpforge.utils import persistence
from tests._contracts import sample_models
from tests._solver_support import solver_names


class FourStateModel(MDP):
    """Small synthetic MDP for validation and partition invariants."""

    def _build_model(self):
        self.transition_matrix = [
            csr_matrix(np.eye(4)),
            csr_matrix(np.roll(np.eye(4), 1, axis=1)),
        ]
        self.reward_matrix = np.array([[1, 0.5], [1, 1], [2, 1.5], [3, 2]])


class ChainModel(MDP):
    """Transient rewards with known optimal value [2 * discount, 2, 0]."""

    def _build_model(self):
        self.transition_matrix = [
            csr_matrix([[0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=float),
            csr_matrix(np.eye(3)),
        ]
        self.reward_matrix = np.array([[0, 0], [2, 0], [0, 0]], dtype=float)


@pytest.fixture
def isolated_model_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "SAVED_MODELS_PATH", tmp_path / "models")


@pytest.fixture
def small_model(isolated_model_cache):
    model = FourStateModel(4, 2)
    model.create_model(cache="readonly")
    return model


@pytest.fixture(
    params=[entry for entry in solver_names() if entry.id != "mdptoolbox_vigs"]
)
def solver_module(request):
    return f"{discounted.__name__}.{request.param}"


@pytest.fixture
def chain(tmp_path, monkeypatch, isolated_model_cache):
    monkeypatch.setattr(persistence, "SAVED_VALUE_FUNCTIONS_PATH", tmp_path / "values")
    model = ChainModel(3, 2)
    model.create_model(cache="readonly")
    return model


@pytest.fixture(
    params=[
        pytest.param(
            (entry.values[0], case, discount),
            marks=entry.marks,
            id=f"{entry.id}-{case}-{discount}",
        )
        for entry in solver_names()
        for case in ("ties_and_negative_rewards", "stochastic_switches", "generated")
        for discount in (0.5, 0.99)
        if entry.id != "mdptoolbox_vigs"
        and (entry.id != "mdptoolbox_vi" or case == "ties_and_negative_rewards")
        and (
            entry.id != "mdptoolbox_mpi"
            or (case == "ties_and_negative_rewards" and discount == 0.5)
        )
    ]
)
def solver_case(request):
    """Return a retained solver, small model, and discount combination."""
    module, name, discount = request.param
    model = next(case for case in sample_models() if case.name == name)
    return f"{discounted.__name__}.{module}", model, discount
