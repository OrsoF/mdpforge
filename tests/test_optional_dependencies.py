import subprocess
import sys


def test_core_workflow_without_optional_dependencies(tmp_path):
    # A fresh interpreter also catches optional imports hidden by test collection.
    script = """
import importlib
import sys
from pathlib import Path

class BlockOptionalImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'gurobipy', 'mdpsolver', 'mdptoolbox', 'marmote', 'stable_baselines3',
            'torch', 'gymnasium', 'mazelib', 'matplotlib', 'pandas', 'seaborn',
            'openpyxl', 'IPython', 'tqdm', 'solvers_agg',
        }:
            raise ModuleNotFoundError(f'Optional dependency blocked: {fullname}')

sys.meta_path.insert(0, BlockOptionalImports())
for name in ('mdpforge.core.model', 'mdpforge.core.solver', 'mdpforge.core.partition', 'mdpforge.core.operators',
             'mdpforge.core.conversion', 'mdpforge.core.validation', 'mdpforge.utils.bellman',
             'mdpforge.utils.projected_bellman', 'mdpforge.utils.exact_value_function',
             'mdpforge.utils.data_management', 'main', 'main_total'):
    importlib.import_module(name)
import main
for module_name, _ in main.SOLVERS.values():
    importlib.import_module(f'mdpforge.solvers.{module_name}')

from mdpforge.models.forest import Model
from mdpforge.solvers.personal_vi import Solver
from mdpforge.utils import persistence
from mdpforge.core.validation import validate_model
persistence.SAVED_MODELS_PATH = Path(sys.argv[1]) / 'models'
model = Model(4, 2)
model.create_model(save=False)
validate_model(model)
solver = Solver(model, discount=0.9, final_precision=1e-4)
solver.run()
assert solver.bellman_residual() < 1e-4
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
