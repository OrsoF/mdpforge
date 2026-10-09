from pathlib import Path

# Keep generated research data in the working directory, including wheel installs.
ARTIFACTS_PATH = Path.cwd() / "artifacts"
SAVED_MODELS_PATH = ARTIFACTS_PATH / "saved_models"
SAVED_VALUE_FUNCTIONS_PATH = ARTIFACTS_PATH / "saved_value_functions"
