import subprocess
import sys


def test_installed_package_works_outside_checkout(tmp_path):
    # Run outside the checkout, without PYTHONPATH or the current directory.
    script = """
from pathlib import Path

from mdpforge.models.rooms import Model
from mdpforge.solvers.personal_vi import Solver
from mdpforge.utils.paths import ARTIFACTS_PATH

assert ARTIFACTS_PATH == Path.cwd() / 'artifacts'
model = Model(100, 4)
model.create_model()
solver = Solver(model, discount=0.9)
solver.run()
assert solver.bellman_residual() < 1e-4
assert (ARTIFACTS_PATH / 'saved_models' / f'{model.name}.pkl').is_file()
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
