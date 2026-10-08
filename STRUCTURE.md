# Structure du dépôt

Carte compacte du code actuellement présent. Le parcours d’installation et de démonstration du `README.md` utilise `main.py` et `main_total.py`. Les fichiers `agg_*.py` et `solvers_agg/` sont absents ; certains modules RL historiques dépendent encore de ce dernier et ne font pas partie de ce parcours.

La bibliothèque est installée sous le namespace `mdpforge` depuis `src/mdpforge/`.
Les scripts de benchmark restent dans le dépôt. Les sorties et caches sont placés
sous `artifacts/` dans le répertoire de travail au moment de l’import.

## Points d’entrée

- `main.py` : benchmark des solveurs actualisés. Charge un ou plusieurs modèles, construit les solveurs, répète les runs et écrit CSV/LaTeX.
- `main_total.py` : équivalent pour le critère de récompense totale (`discount=1`).
- `reset.py` : nettoyage des modèles et fonctions de valeur mis en cache.

Flux principal : `main*.py` → `src/mdpforge/models/<env>.py` → `src/mdpforge/solvers/<solver>.py` → résultats sous `artifacts/`.

## `src/mdpforge/core/` — primitives MDP

- `model.py` : `GenericModel`, construction/conversion des matrices, interface commune des environnements.
- `solver.py` : interface minimale `GenericSolver`.
- `operators.py` : opérateurs de Bellman et construction efficace de `(P^π, r^π)`.
- `partition.py` : partitions d’états, matrices projection/extension, agrégation et raffinement.
- `conversion.py` : conversions NumPy/creux.
- `validation.py` : validation des dimensions, valeurs et transitions stochastiques.

Convention modèle : `state_dim`, `action_dim`, `transition_matrix[action]`, `reward_matrix[state, action]`, puis `create_model()`.

## `src/mdpforge/models/` — environnements

Chaque fichier expose généralement une classe `Model(GenericModel)`.

- Modèles centraux des benchmarks : `rooms.py`, `ninerooms.py`, `taxi.py`, `mountain.py`, `sysadmin.py`, `tandem.py`, `dam.py`, `inventory.py`, `impatience.py`, `garnet.py`, `block.py`.
- Autres MDP tabulaires : `forest.py`, `cliff.py`, `windgrid.py`, `random_walk.py`, `queue_network.py`, `maintenance.py`, etc.
- `src/mdpforge/models/total/` : variantes à récompense totale, notamment `rooms_total.py`, `taxi_total.py`, `mountain_total.py`, `barto_total.py` et les trois `maze_*_total.py`.
- `cliff_total.py` est à la racine de `src/mdpforge/models/`, hors registre actuel de `main_total.py`.

## `src/mdpforge/solvers/` — algorithmes

Interface usuelle : `Solver(model, discount, final_precision, ...)`, puis `run()` renseigne `value`, `policy`/`q_value` et `runtime`.

### Solveurs principaux actualisés

- `personal_vi.py`, `personal_pim.py`, `personal_qvi.py`, `personal_pi.py` : références tabulaires VI/MPI/QVI/PI.
- `aggregated_vi.py`, `aggregated_qvi.py`, `aggregated_pim.py` : PDVI, PDQVI et PDPI avec partitions adaptatives.
- `chen_td.py` : alternance Bellman/TD agrégé de Chen.
- `bertsekas_pi.py`, `abate.py`, `deanlin.py` : méthodes d’agrégation/adaptation issues de la littérature.

### Critère total

`src/mdpforge/solvers/total/` contient les homologues :

- `personal_vi_total.py`, `personal_pim_total.py` ;
- `aggregated_vi_total.py`, `aggregated_qvi_total.py`, `aggregated_pim_total.py` ;
- adaptateurs Gurobi, Marmote et MDPtoolbox suffixés `_total.py`.

### Autres familles

- `src/mdpforge/solvers/industrial/` : adaptateurs vers Gurobi, Marmote, mdpsolver et MDPtoolbox.
- `src/mdpforge/solvers/average/` : critère moyen, principalement adaptateurs externes.
- `src/mdpforge/solvers/stochastic/` : RL échantillonné (`aggregated.py`, `aq_ucb.py`, `cat_rl.py`, `utree.py`, `dqn.py`, `a2c.py`, `ppo.py`).
- `tabular.py`, `reynolds.py` : apprentissage tabulaire et abstraction de Reynolds.

## `src/mdpforge/utils/` — fonctions partagées

- `bellman.py` : opérateurs Bellman historiques, résidus, politiques gloutonnes et évaluations.
- `projected_bellman.py` : opérateurs Bellman projetés pour PDVI/PDQVI/PDPI.
- `data_management.py` : import dynamique des modèles/solveurs et helpers de benchmark.
- `exact_value_function.py` : calcul/chargement de valeurs exactes et distances à l’optimum.
- `paths.py`, `persistence.py` : chemins et persistance.
- `simulation.py`, `q_learning.py`, `toy_model.py` : simulation, Q-learning et petit modèle de test.
- `model_conversion.py` : anciennes fonctions de conversion de modèles.

## Données et documentation

- `artifacts/results/` : résultats de référence destinés à être conservés.
- `artifacts/tmp/` : profils, essais et benchmarks temporaires.
- `artifacts/figures/`, `artifacts/exps/` : figures et anciennes expériences.
- `studies/` : notebooks de tutoriel, comparaisons, tableaux d’article et études de passage à l’échelle.
- `tests/` : tests unitaires des invariants numériques et tests d’intégration des solveurs et points d’entrée.
- `pyproject.toml` : installation du projet, dépendances essentielles et optionnelles, configuration pytest/Ruff.
- `.github/workflows/ci.yml` : installation `.[dev]`, lint, format et tests sur Python 3.11/3.12 pour chaque push/PR.
- `AGENTS.md` : conventions de développement et de benchmark.
- `MODELS.md` : documentation des modèles.

## Repères pour modifier le code

- Performance Bellman/politique : regarder d’abord `src/mdpforge/core/operators.py`, `src/mdpforge/utils/bellman.py` et `src/mdpforge/utils/projected_bellman.py`.
- Agrégation/raffinement : `src/mdpforge/core/partition.py`, puis le solveur `aggregated_*.py` concerné.
- Configuration d’un benchmark : `main.py` ou `main_total.py`; garder les sorties exploratoires dans `artifacts/tmp/`.
- Cas actualisé et total sont séparés : ne pas appliquer une accélération contenant `1 / (1-discount)` lorsque `discount=1`.
