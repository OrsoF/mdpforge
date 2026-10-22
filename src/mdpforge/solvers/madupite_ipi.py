"""Inexact Policy Iteration using the optional madupite/PETSc backend."""

from pathlib import Path
from tempfile import TemporaryDirectory

import madupite as md
import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.madupite import madupite_matrices
from mdpforge.utils.solver_options import positive_integer

METADATA = {
    "display_name": "iPI · madupite",
    "family": "pi",
    "backend": "madupite",
    "description": (
        "madupite distributed inexact policy iteration, running on one MPI rank "
        "with PETSc file conversion and independent residual certification."
    ),
}


class Solver(GenericSolver):
    """Adapt madupite's MAXREWARD iPI to a single-rank mdpforge trial."""

    solver_type = "pi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 1000,
        max_evaluation_iterations: int = 1000,
        ksp_type: str = "gmres",
        pc_type: str = "none",
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; outer iteration budget; default 1000.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            1000.
        ksp_type: nonempty PETSc Krylov solver name; default "gmres".
        pc_type: nonempty PETSc preconditioner name; default "none".
        """
        super().__init__(model, discount, final_precision)
        if not ksp_type or not pc_type:
            raise ValueError("PETSc solver names must be nonempty")
        self.max_iterations = positive_integer(max_iterations, "max_iterations")
        self.max_evaluation_iterations = positive_integer(
            max_evaluation_iterations, "max_evaluation_iterations"
        )
        self.ksp_type = ksp_type
        self.pc_type = pc_type
        self.name = "madupite iPI"

    def _solve(self) -> SolveResult:
        # mdpforge independently manages processes and checks one complete
        # vector on each trial. An external mpirun needs a separate runner.
        _rank, size = md.mpi_rank_size()
        if size != 1:
            raise RuntimeError(
                "madupite adapter supports a single MPI rank; "
                "use madupite directly for distributed runs"
            )
        rewards, transitions = madupite_matrices(self.model)
        with TemporaryDirectory(prefix="mdpforge-madupite-") as directory:
            root = Path(directory)
            reward_path, transition_path = root / "rewards.bin", root / "P.bin"
            value_path, policy_path = root / "values.txt", root / "policy.txt"
            md.writePETScBinary(rewards, str(reward_path))
            md.writePETScBinary(transitions, str(transition_path))
            mdp = md.MDP()
            mdp.setStageCostMatrix(
                md.Matrix.fromFile(
                    filename=str(reward_path),
                    category=md.MatrixCategory.Cost,
                    type=md.MatrixType.Dense,
                )
            )
            mdp.setTransitionProbabilityTensor(
                md.Matrix.fromFile(
                    filename=str(transition_path),
                    category=md.MatrixCategory.Dynamics,
                    type=md.MatrixType.Sparse,
                )
            )
            # These options are backed by a PETSc *global* database in madupite 1.0.
            for key, option in (
                ("-mode", "MAXREWARD"),
                ("-discount_factor", str(self.discount)),
                ("-atol_pi", str(self.final_precision * (1 - self.discount))),
                ("-max_iter_pi", str(self.max_iterations)),
                ("-max_iter_ksp", str(self.max_evaluation_iterations)),
                ("-ksp_max_it", str(self.max_evaluation_iterations)),
                ("-ksp_type", self.ksp_type),
                ("-pc_type", self.pc_type),
                ("-file_cost", str(value_path)),
                ("-file_policy", str(policy_path)),
                ("-overwrite", "true"),
                ("-verbose", "false"),
            ):
                mdp.setOption(key, option)
            mdp.solve()
            if not value_path.is_file() or not policy_path.is_file():
                raise RuntimeError("madupite did not write its value/policy files")
            values = np.atleast_1d(np.loadtxt(value_path, dtype=float)).ravel()
            raw_policy = np.atleast_1d(np.loadtxt(policy_path, dtype=float)).ravel()

        if not np.all(np.isfinite(raw_policy)) or not np.all(
            raw_policy == np.rint(raw_policy)
        ):
            raise RuntimeError("madupite returned nonintegral policy actions")
        policy = raw_policy.astype(int)
        if values.shape != (self.model.state_dim,):
            raise RuntimeError("madupite returned an invalid value shape")
        if policy.shape != (self.model.state_dim,) or np.any(
            (policy < 0) | (policy >= self.model.action_dim)
        ):
            raise RuntimeError("madupite returned an invalid policy")
        value = certify_value(self.model, values, self.discount, self.final_precision)
        return self._result(value, policy)
