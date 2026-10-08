import subprocess
import sys


def test_core_workflow_without_optional_dependencies(tmp_path, model_script):
    # A fresh interpreter also catches optional imports hidden by test collection.
    script = (
        """
import sys
from pathlib import Path

class BlockOptionalImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mdptoolbox', 'matplotlib', 'mazelib'}:
            raise ModuleNotFoundError(f'Optional dependency blocked: {fullname}')

sys.meta_path.insert(0, BlockOptionalImports())
"""
        + model_script
        + """
from mdpforge.utils import persistence
from mdpforge.core.validation import validate_model
persistence.SAVED_MODELS_PATH = Path(sys.argv[1]) / 'models'
persistence.SAVED_VALUE_FUNCTIONS_PATH = Path(sys.argv[1]) / 'values'
model = ChainModel(3, 2)
model.create_model(save=False)
validate_model(model)
np.testing.assert_allclose(model.optimal_value_function(0.9), [1.8, 2, 0])
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
