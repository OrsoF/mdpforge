"""List benchmarks that do not yet have ten runs per solver."""

import csv
from pathlib import Path

from main import MODELS as DISCOUNTED_MODELS
from main import SOLVERS as DISCOUNTED_SOLVERS
from main_total import MODELS as TOTAL_MODELS
from main_total import SOLVERS as TOTAL_SOLVERS


def show(title, directory, models, solvers):
    print(f"{title}:")
    remaining = False
    for model in models:
        path = directory / f"{model}_runs.csv"
        runs = {solver: set() for solver in solvers}
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["solver"] in runs:
                        runs[row["solver"]].add(int(row["run"]))
        missing = [
            f"{solver} {len(ids)}/10"
            for solver, ids in runs.items()
            if ids != set(range(1, 11))
        ]
        if missing:
            remaining = True
            print(f"  {model}: {', '.join(missing)}")
    if not remaining:
        print("  aucun")


root = Path(__file__).parent / "artifacts" / "results"
show("Discounted", root / "article_runtimes", DISCOUNTED_MODELS, DISCOUNTED_SOLVERS)
show("Total", root / "article_runtimes_total", TOTAL_MODELS, TOTAL_SOLVERS)
