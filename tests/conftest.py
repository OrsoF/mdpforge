import ast
import inspect
import sys
from pathlib import Path
from pkgutil import iter_modules

import numpy as np
import pytest
from scipy.sparse import csr_array

from mdpforge import models, solvers
from mdpforge.core.model import GenericModel
from mdpforge.utils import persistence


class FourStateModel(GenericModel):
    """Small synthetic MDP for validation and partition invariants."""

    def _build_model(self):
        self.transition_matrix = [
            csr_array(np.eye(4)),
            csr_array(np.roll(np.eye(4), 1, axis=1)),
        ]
        self.reward_matrix = np.array([[1, 0.5], [1, 1], [2, 1.5], [3, 2]])


class ChainModel(GenericModel):
    """Transient rewards with known optimal value [2 * discount, 2, 0]."""

    def _build_model(self):
        self.transition_matrix = np.array(
            [[[0, 1, 0], [0, 0, 1], [0, 0, 1]], np.eye(3)], dtype=float
        )
        self.reward_matrix = np.array([[0, 0], [2, 0], [0, 0]], dtype=float)


def module_names(package, solver_type=None):
    # Subpackages contain specialized workflows with their own configurations.
    parameters = []
    for module in sorted(iter_modules(package.__path__), key=lambda item: item.name):
        if module.ispkg or module.name.startswith("_"):
            continue
        source = Path(package.__file__).with_name(f"{module.name}.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        if solver_type is not None and not any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "solver_type"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value == solver_type
            for node in ast.walk(tree)
        ):
            continue
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imports.add(node.module.split(".")[0])
        dependencies = (
            imports - sys.stdlib_module_names - {"mdpforge", "numpy", "scipy"}
        )
        marks = (
            [pytest.mark.optional(dependencies=sorted(dependencies))]
            if dependencies
            else []
        )
        parameters.append(pytest.param(module.name, marks=marks, id=module.name))
    return parameters


def pytest_addoption(parser):
    parser.addoption(
        "--optional", action="store_true", help="Test optional integrations"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--optional"):
        return
    for item in items:
        if item.get_closest_marker("optional"):
            item.add_marker(
                pytest.mark.skip(reason="Optional integration; use --optional")
            )


@pytest.fixture
def isolated_model_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "SAVED_MODELS_PATH", tmp_path / "models")


@pytest.fixture
def small_model(isolated_model_cache):
    model = FourStateModel(4, 2)
    model.create_model(save=False)
    return model


@pytest.fixture(params=module_names(models))
def model_module(request):
    return f"{models.__name__}.{request.param}"


@pytest.fixture(params=module_names(solvers))
def solver_module(request):
    return f"{solvers.__name__}.{request.param}"


@pytest.fixture(params=module_names(solvers, solver_type="vi"))
def vi_module(request):
    return f"{solvers.__name__}.{request.param}"


@pytest.fixture
def chain(tmp_path, monkeypatch, isolated_model_cache):
    monkeypatch.setattr(persistence, "SAVED_VALUE_FUNCTIONS_PATH", tmp_path / "values")
    model = ChainModel(3, 2)
    model.create_model(save=False)
    return model


@pytest.fixture
def model_script():
    # Reuse the synthetic model in isolated interpreters without importing tests.
    return (
        "import numpy as np\nfrom mdpforge.core.model import GenericModel\n"
        + inspect.getsource(ChainModel)
    )
