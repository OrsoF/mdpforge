from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = PROJECT_PATH / "artifacts"
RESULTS_PATH = ARTIFACTS_PATH / "results"
FIGURES_PATH = ARTIFACTS_PATH / "figures"
SAVED_MODELS_PATH = ARTIFACTS_PATH / "saved_models"
SAVED_VALUE_FUNCTIONS_PATH = ARTIFACTS_PATH / "saved_value_functions"
EXPS_ARTIFACTS_PATH = ARTIFACTS_PATH / "exps"
SOLVER_PATH = PROJECT_PATH / "solvers"
SOLVERS_AGG_PATH = PROJECT_PATH / "solvers_agg"
MODEL_PATH = PROJECT_PATH / "models"
