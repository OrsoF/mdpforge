from pathlib import Path

# Keep generated research data in the working directory, including wheel installs.
PROJECT_PATH = Path.cwd()
PACKAGE_PATH = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = PROJECT_PATH / "artifacts"
RESULTS_PATH = ARTIFACTS_PATH / "results"
FIGURES_PATH = ARTIFACTS_PATH / "figures"
SAVED_MODELS_PATH = ARTIFACTS_PATH / "saved_models"
SAVED_VALUE_FUNCTIONS_PATH = ARTIFACTS_PATH / "saved_value_functions"
EXPS_ARTIFACTS_PATH = ARTIFACTS_PATH / "exps"
SOLVER_PATH = PACKAGE_PATH / "solvers"
MODEL_PATH = PACKAGE_PATH / "models"
