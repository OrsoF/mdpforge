import subprocess
import sys


def test_installed_namespace_and_workflow(tmp_path):
    # Run outside the checkout, without PYTHONPATH or the current directory.
    script = """
from importlib import import_module
from pathlib import Path
import sys

import mdpforge
from mdpforge.models.rooms import Model
from mdpforge.solvers.personal_vi import Solver
from mdpforge.utils.paths import ARTIFACTS_PATH, MODEL_PATH, SOLVER_PATH

assert mdpforge.__file__
assert ARTIFACTS_PATH == Path.cwd() / 'artifacts'
assert MODEL_PATH == Path(mdpforge.__file__).parent / 'models'
assert SOLVER_PATH == Path(mdpforge.__file__).parent / 'solvers'
for name in ('rooms', 'total.rooms_total'):
    model = import_module(f'mdpforge.models.{name}').Model(100, 4)
    model.create_model(save=False)
    solver = Solver(model, discount=0.9)
    solver.run()
    assert solver.bellman_residual() < 1e-4

model = Model(100, 4)
model.create_model()
assert (ARTIFACTS_PATH / 'saved_models' / f'{model.name}.pkl').is_file()
model.optimal_value_function(0.9)
assert (ARTIFACTS_PATH / 'saved_value_functions' / f'0.9_{model.name}.pkl').is_file()
assert not {'core', 'models', 'solvers', 'utils'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
