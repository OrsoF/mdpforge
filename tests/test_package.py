import subprocess
import sys


def test_installed_core_without_external_dependencies(tmp_path, model_script):
    # Run outside the checkout, without PYTHONPATH or external backend imports.
    script = (
        """
import sys

class BlockExternalImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mdptoolbox', 'matplotlib', 'mazelib'}:
            raise ModuleNotFoundError(f'External dependency blocked: {fullname}')

sys.meta_path.insert(0, BlockExternalImports())
"""
        + model_script
        + """
from pathlib import Path

from mdpforge import Benchmark
from mdpforge.utils.paths import ARTIFACTS_PATH
from mdpforge.core.validation import validate_model

assert ARTIFACTS_PATH == Path.cwd() / 'artifacts'
model = ChainModel(3, 2)
model.create_model()
validate_model(model)
np.testing.assert_allclose(model.optimal_value_function(0.9), [1.8, 2, 0])
assert (ARTIFACTS_PATH / 'saved_models' / f'{model.name}.pkl').is_file()
matrix_model = MDP.from_matrices('matrix_chain', model.transition_matrix, model.reward_matrix)
assert isinstance(model, MDP) and isinstance(matrix_model, MDP)
np.testing.assert_allclose(matrix_model.optimal_value_function(0.9), [1.8, 2, 0])
bench = Benchmark().add_mdp('chain', transitions=model.transition_matrix, reward=model.reward_matrix)
results = bench.run(discount=0.9, repeats=1)
assert len(results) == 1 and all(row['status'] == 'success' for row in results)
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
