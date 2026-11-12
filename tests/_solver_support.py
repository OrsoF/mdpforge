"""Solver discovery and process isolation shared by solver tests."""

import ast
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from pkgutil import iter_modules

import pytest

from mdpforge.solvers import discounted

OPTIONAL_BACKENDS = {
    "madupite": ("madupite",),
    "mdpax": ("mdpax",),
    "marmote": ("marmote",),
    "mdpsolver": ("mdpsolver",),
    "mdptoolbox": ("mdptoolbox",),
    "quantecon": ("quantecon",),
    "cmaes": ("cma",),
    "pyomo": ("pyomo", "highspy"),
}


def solver_names(catalogue=discounted):
    """Return solver cases with optional-dependency markers without importing backends."""
    parameters = []
    for module in sorted(iter_modules(catalogue.__path__), key=lambda item: item.name):
        if module.ispkg or module.name.startswith("_"):
            continue
        source = Path(catalogue.__file__).with_name(f"{module.name}.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imports.add(node.module.split(".")[0])
        dependencies = (
            imports - sys.stdlib_module_names - {"mdpforge", "numpy", "scipy"}
        )
        backend = module.name.split("_", 1)[0]
        if backend in OPTIONAL_BACKENDS:
            dependencies.update(OPTIONAL_BACKENDS[backend])
        marks = (
            [pytest.mark.external(dependencies=sorted(dependencies))]
            if dependencies
            else []
        )
        parameters.append(pytest.param(module.name, marks=marks, id=module.name))
    return parameters


def run_external_test(request, tmp_path):
    """Isolate installed native backends that may own global process state."""
    marker = request.node.get_closest_marker("external")
    if marker is None or os.environ.get("MDPFORGE_TEST_CHILD") == "1":
        return False
    for dependency in marker.kwargs["dependencies"]:
        if find_spec(dependency) is None:
            pytest.skip(f"Dependency not installed: {dependency}")
    path, test_name = request.node.nodeid.split("::", 1)
    nodeid = f"{request.config.rootpath / path}::{test_name}"
    result = subprocess.run(
        [sys.executable, "-I", "-m", "pytest", nodeid, "-q", "--tb=short"],
        cwd=tmp_path,
        env={**os.environ, "MDPFORGE_TEST_CHILD": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if "1 xfailed" in result.stdout:
        pytest.xfail(result.stdout[-4000:].strip())
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return True
