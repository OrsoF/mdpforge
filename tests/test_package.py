import subprocess
import sys


def test_installed_package_works_outside_checkout(tmp_path, model_script):
    # Run outside the checkout, without PYTHONPATH or the current directory.
    script = (
        model_script
        + """
from pathlib import Path

from mdpforge.utils.paths import ARTIFACTS_PATH

assert ARTIFACTS_PATH == Path.cwd() / 'artifacts'
model = ChainModel(3, 2)
model.create_model()
np.testing.assert_allclose(model.optimal_value_function(0.9), [1.8, 2, 0])
assert (ARTIFACTS_PATH / 'saved_models' / f'{model.name}.pkl').is_file()
"""
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
