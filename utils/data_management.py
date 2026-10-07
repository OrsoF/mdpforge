import os
import platform
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from typing import Tuple

import numpy as np

from core.model import GenericModel
from core.solver import GenericSolver
from utils.paths import (
    MODEL_PATH,
    SAVED_MODELS_PATH,
    SAVED_VALUE_FUNCTIONS_PATH,
    SOLVER_PATH,
)
from utils.toy_model import ToyModel

TOY_DISCOUNT = 0.8
TOY_PRECISION = 1e-2


def import_solver_from_file(file_name: str) -> GenericSolver:
    """
    Given a file (as "q_learning.py")
    returns the associated_solver.
    """
    if not file_name.endswith(".py"):
        file_name += ".py"
    spec = spec_from_file_location("__temp_module__", SOLVER_PATH / file_name)
    module = module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise Exception(f"Solver not found: {file_name}")
    return module.Solver(ToyModel(0, 0), TOY_DISCOUNT, TOY_PRECISION)


def import_models_from_file(file_name: str):
    """For a given model name (example: rooms), import the model itself from the string. Returns Model not built."""
    if not file_name.endswith(".py"):
        file_name += ".py"
    spec = spec_from_file_location("__temp_module__", MODEL_PATH / file_name)
    assert spec is not None, "File not found {}".format(file_name)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def get_models_from_model_file(model_file: str) -> list[GenericModel]:
    model, param_list = import_models_from_file(file_name=model_file)
    return [model(params) for params in param_list]


def reset():
    """
    Function that reset the experience environment.
    """
    for folder in [SAVED_MODELS_PATH, SAVED_VALUE_FUNCTIONS_PATH]:
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass


def gurobi_license() -> bool:
    """
    To check if the Gurobi license exists or not.
    Output : True if license, False otherwise.
    """
    gurobi_log_string = "gurobi.log"
    import gurobipy as grb

    grb.Env(gurobi_log_string)
    f = open(gurobi_log_string, "r")
    log = f.read()
    f.close()
    os.remove(os.path.join(os.getcwd(), gurobi_log_string))
    return "Restricted" not in log


def linux() -> bool:
    """
    To check if the system runs on Linux.
    Output : True if Linux, False otherwise
    """
    return platform.system() == "Linux"


def remove_unused_solver_files(solver_name_list: list) -> list:
    for file_name in ["__pycache__", "_pyMarmoteMDP.so", "pyMarmoteMDP.py"]:
        try:
            solver_name_list.remove(file_name)
        except ValueError:
            pass
    return solver_name_list


def solve(
    model_name: str,
    solver_name: str,
    state_dim: int,
    action_dim: int = 6,
    discount: float = 0.9,
    precision: float = 1e-2,
    repeat: int = 1,
) -> Tuple[GenericModel, GenericSolver, float, float]:
    """
    Instantiate a model-solver pair and benchmark solver runtime.

    Parameters
    ----------
    model_name : str
        Identifier used to dynamically import the model class.
    solver_name : str
        Identifier used to dynamically import the solver class.
    state_dim : int
        Dimensionality of the state space.
    action_dim : int, default=6
        Dimensionality of the action space.
    discount : float, default=0.9
        Discount factor γ ∈ (0, 1) applied to future rewards.
    precision : float, default=1e-2
        Convergence tolerance (e.g., ‖V_{k+1} - V_k‖ ≤ precision).
    repeat : int, default=1
        Number of independent solver runs for runtime statistics.

    Returns
    -------
    model : GenericModel
        Instantiated and initialized model.
    solver : GenericSolver
        Configured solver after execution.
    mean : float
        Mean runtime across runs.
    std : float
        Standard deviation of runtime (0 if repeat == 1).
    """

    # Model creation
    model = import_models_from_file(model_name)
    model: GenericModel = model(state_dim, action_dim)
    model.create_model()
    solver = import_solver_from_file(solver_name)
    solver.__init__(model, discount, precision)

    if repeat == 1:
        solver.run()
        # Save the runtime result to a file

        mean = solver.runtime
        std = 0.0

    else:
        runtimes = []
        for _ in range(repeat):
            solver.run()
            runtimes.append(solver.runtime)

        mean = np.mean(runtimes)
        std = np.std(runtimes)

    return model, solver, mean, std


def load_and_build_model(
    model_name: str, state_dim: int, action_dim: int
) -> GenericModel:
    model = import_models_from_file(model_name)
    model: GenericModel = model(state_dim, action_dim)
    model.create_model()
    return model


def write_text(filename: str, text: str, show: bool = False):
    if show:
        print(text)
    with open(filename, "a") as file:
        # Add a newline before the text if the file already has content
        if file.tell() > 0:
            file.write("\n")
        file.write(text + "\n")


def gurobi_model_creation():
    from gurobipy import GRB, Model

    model = Model("MDP")
    model.setParam("OutputFlag", 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam("LogToConsole", 0)
    model.setParam("MemLimit", 16)
    return model
