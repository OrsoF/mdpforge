import random
from types import SimpleNamespace

import numpy as np
import pytest

import benchmark_sizes
from mdpforge.models.unichain import Model as Unichain


@pytest.fixture
def constructor_request(monkeypatch):
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    monkeypatch.setattr(
        benchmark_sizes, "import_module", lambda name: SimpleNamespace(Model=Unichain)
    )
    request = {
        "model": "unichain",
        "model_seed": 0,
        "constructor_options": '{"state_dim": 4}',
        "expected_state_dim": 4,
        "max_states": 100,
        "dimensions_only": True,
    }
    try:
        yield request
    finally:
        np.random.set_state(numpy_state)
        random.setstate(random_state)


def test_dimension_audit_never_builds_matrices(constructor_request, monkeypatch):
    def forbidden_build(self, **kwargs):
        pytest.fail("Constructor audit must not build or read matrix caches")

    monkeypatch.setattr(Unichain, "create_model", forbidden_build)
    result = benchmark_sizes.build_model(constructor_request)
    assert result["status"] == "success"
    assert result["state_dim"] == 4 and result["action_dim"] == 2


def test_preset_mismatch_fails_before_matrix_construction(constructor_request):
    # Full mode must reject the dimension mismatch before trying to access a
    # model_path or building any matrices.
    constructor_request.update(expected_state_dim=5, dimensions_only=False)
    result = benchmark_sizes.build_model(constructor_request)
    assert result["status"] == "dimension_mismatch"
    assert "expects 5 states" in result["error"]


def test_size_limit_applies_to_constructor_audit(constructor_request):
    constructor_request["max_states"] = 3
    result = benchmark_sizes.build_model(constructor_request)
    assert result["status"] == "size_limit"
