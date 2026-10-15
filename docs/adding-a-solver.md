# Ajouter un solveur

Contraintes d'intégration au catalogue, en complément des [consignes du dépôt](../AGENTS.md).

## Interface

- Créer `src/mdpforge/solvers/<nom>.py` (minuscules et underscores, sans underscore initial), exposant une classe `Solver`. La découverte est automatique.
- Accepter `Solver(model, discount, final_precision=1e-3, ...)` : options explicites au constructeur, puis `run()` sans argument. Vérifier par assertion `0 < discount < 1` et accepter uniquement une précision finie strictement positive.
- Utiliser les données du modèle construit par `create_model()` : `state_dim`, `action_dim`, `transition_matrix[a]` (CSR SciPy de forme `(S, S)`) et `reward_matrix[s, a]` (NumPy de forme `(S, A)`). Ne pas exiger de méthodes du modèle.
- Préserver les entrées, y compris leur représentation. Garder toute conversion propre au backend dans le solveur ; éviter les matrices denses inutiles.
- Après `run()`, exposer `name`, `value` (tableau NumPy fini de forme `(S,)`), `policy` (`None` ou tableau NumPy d'entiers de forme `(S,)`, actions dans `[0, A)`) et `runtime` (secondes, fini et positif ou nul).

Base recommandée : [`GenericSolver`](../src/mdpforge/core/solver.py). Appeler son constructeur, définir un nom lisible et implémenter `_solve()` qui renvoie la valeur et peut renseigner `self.policy` ; `run()` stocke et chronomètre le résultat. Voir [`mdpforge_vi.py`](../src/mdpforge/solvers/mdpforge_vi.py).

## Catalogue et dépendances

Définir un dictionnaire **littéral** `METADATA`, lu sans importer le backend :

| Champ           | Contenu                                                                                      |
| --------------- | -------------------------------------------------------------------------------------------- |
| `display_name`  | Nom lisible du solveur                                                                       |
| `family`        | Famille : actuellement `vi`, `pi`, `mpi`, `lp`                                               |
| `backend`       | Bibliothèque : actuellement `mdpforge`, `mdptoolbox`, `mdpsolver`, `marmote`, `gurobi`, `cma` |
| `description`   | Algorithme et particularités en une phrase                                                   |
| `update_scheme` | Actuellement `synchronous`, `gauss_seidel`, `sor`, `linear_solve`, `mixed`, `not_applicable` |

Utiliser des chaînes non vides et les valeurs existantes quand elles conviennent. Vérifier la découverte avec `list_solvers()` et l'ajout avec `bench.add_solver("<nom>")`. Pour une VI, déclarer aussi `solver_type = "vi"` dans la classe afin d'inclure les tests de précision dédiés.

Déclarer toute dépendance dans `pyproject.toml`, avec un extra pour un backend optionnel. Son absence doit laisser le cœur NumPy/SciPy et le catalogue utilisables.

## Contrat scientifique

- Documenter l'algorithme, sa référence éventuelle, les adaptations, l'initialisation et le critère d'arrêt. Docstrings courtes ; pour chaque option publique, préciser les valeurs acceptées et leur sens. Toute source aléatoire doit être reproductible.
- `final_precision` borne l'erreur absolue sur la valeur. Pour la valeur retournée, vérifier `max(abs(T(V) - V)) / (1 - discount) <= final_precision`, avec l'opérateur de Bellman optimal complet. Un critère fondé sur le span seul ne suffit pas.
- Ne pas modifier implicitement récompenses, actualisation, conventions numériques, comportement aléatoire des solveurs existants ou cache de référence optimale par VI.
- Signaler une précision non atteinte ou une limite d'itérations par un échec explicite. Le benchmark mesure constructeur et `run()`, conversions et corrections incluses.

## Validation

Les tests génériques de [`tests/test_workflows.py`](../tests/test_workflows.py) découvrent les solveurs : interface, entrées préservées, reproductibilité, résidu, valeur et qualité de politique. Ajouter de petits tests déterministes pour les options et comportements spécifiques, avec une référence analytique ou indépendante.

Utiliser Python 3.11+ dans Conda `benchmark`. **Ne lancer pytest ou Ruff que sur demande explicite** ; alors exécuter la suite pytest complète, tous backends installés inclus (ignorer uniquement les dépendances absentes). Commandes dans [AGENTS.md](../AGENTS.md). Pas de benchmark lourd automatique. Rapporter les changements, contrôles réellement exécutés, résultats et limites restantes.
