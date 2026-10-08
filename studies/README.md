# Studies

This folder replaces the old `exps/` and `notebooks/` trees with theme-oriented notebooks.
Notebook outputs were stripped during consolidation; run only the sections needed for a given analysis.
Some sections preserve old command-line scripts as notebook cells; set the parameter block manually before running those cells.

These notebooks preserve historical research, not the current supported benchmark
suite. The retained discounted models are Rooms, Schoolboy, and RiverSwim; the
retained solvers are VI and MPI. Cells using other models, aggregation solvers, or
external solver adapters require those sources to be added back first. Installing
an extra supplies dependencies, not removed code. The supported starting workflow
is documented in [the README](../README.md).

The old benchmark scripts and the `data_management`/`toy_model` helpers were
removed. Cells using those helpers require direct imports and fresh solver
construction for each repetition. Notebook contents remain historical material.

Install analysis dependencies from the repository root with
`python -m pip install -e ".[notebooks]"`. Add the relevant backend extra for each
study (for example, `.[notebooks,gurobi]` for LP comparisons). Notebook execution
also needs a separately installed Jupyter frontend/kernel. Historical RL cells
that import the absent `solvers_agg` package remain unavailable even with `deep`.

Cells import the installed `mdpforge` package; they no longer add the checkout to
`sys.path`. The setup cell selects the repository root as the working directory
so research outputs and caches stay under its `artifacts/` folder. Historical
references to removed modules (such as `utils.generic_model`) and old solver
locations still need adaptation before those sections can run.

## Notebooks

- `01_tutorial_and_data_extraction.ipynb`: tutorial and data extraction notes.
- `02_solver_comparison_smoke_tests.ipynb`: legacy solver comparison and smoke-test recipes.
- `03_manuscript_runtime_tables.ipynb`: manuscript, JMLR, MTG, discounted, and total-reward tables.
- `04_lp_vs_dp_scaling.ipynb`: LP-vs-DP state/action scaling studies.
- `05_abstraction_and_rl_studies.ipynb`: abstraction, aggregation, and RL study material.
