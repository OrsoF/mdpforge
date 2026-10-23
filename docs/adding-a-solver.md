# Ajouter un solveur

Contraintes d'intégration au catalogue, en complément des [consignes du dépôt](../AGENTS.md).

## Interface

- Créer `src/mdpforge/solvers/<nom>.py` (minuscules et underscores, sans underscore initial), exposant une classe `Solver`. Le nom du fichier sans `.py` est l'identifiant utilisé par `list_solvers()` et `bench.add_solver("<nom>")` ; la découverte est automatique.
- Accepter `Solver(model, discount, ...)` avec des options explicites au constructeur, puis `run()` sans argument. Les solveurs itératifs exposent `final_precision`, finie et strictement positive ; `mdpforge_pi`, `mdptoolbox_pi`, `gurobi_primal` et `gurobi_dual` ne prennent pas cet argument inutilisé. Le benchmark transmet la précision uniquement aux constructeurs qui l'acceptent et contrôle la qualité de toutes les valeurs retournées. Vérifier par assertion `0 < discount < 1`.
- Utiliser les données du modèle construit par `create_model()` : `state_dim`, `action_dim`, `transition_matrix[a]` (CSR SciPy de forme `(S, S)`) et `reward_matrix[s, a]` (NumPy de forme `(S, A)`). Ne pas exiger de méthodes du modèle.
- Préserver les entrées, y compris leur représentation. Garder toute conversion propre au backend dans le solveur ; éviter les matrices denses inutiles.
- Appeler les API publiques des backends sans remplacer leur initialisation ni modifier leurs attributs internes pour contourner les contrôles, bornes ou critères d'arrêt. Les erreurs natives doivent être remontées et enregistrées par le benchmark.
- `run()` retourne un [`SolveResult`](../src/mdpforge/core/solve_result.py) contenant `value` (tableau NumPy fini de forme `(S,)`), `policy` (`None` ou tableau NumPy d'entiers de forme `(S,)`, actions dans `[0, A)`), `runtime` (secondes, fini et positif ou nul), `iterations` (entier ou `None`) et `stop_reason` (texte ou `None`). Exposer aussi `name`, `value`, `policy` et `runtime` sur le solveur pour le benchmark.

Base recommandée : [`GenericSolver`](../src/mdpforge/core/solver.py). Appeler son constructeur, définir `self.name` comme libellé lisible et implémenter `_solve()`, qui applique le critère d'arrêt, renvoie la valeur et peut renseigner `self.policy`. Son `run()` stocke la valeur dans `self.value`, mesure `self.runtime` et retourne `SolveResult`. Pour un `run()` propre au backend, retourner `SolveResult.from_solver(self)` après avoir renseigné ces attributs. Voir [`mdpforge_vi.py`](../src/mdpforge/solvers/mdpforge_vi.py).

## Catalogue et dépendances

Définir un dictionnaire **littéral** `METADATA`, lu sans importer le backend :

| Champ           | Contenu                                                                                      |
| --------------- | -------------------------------------------------------------------------------------------- |
| `display_name`  | Libellé du catalogue, distinct de l'identifiant du fichier                                    |
| `family`        | Famille : `vi`, `pi`, `mpi`, `lp` ou `experimental`                                          |
| `backend`       | Bibliothèque : actuellement `mdpforge`, `mdptoolbox`, `mdpsolver`, `marmote`, `madupite`, `mdpax`, `gurobi`, `cma` |
| `description`   | Algorithme et particularités en une phrase                                                   |

Utiliser des chaînes non vides et les valeurs existantes quand elles conviennent. Vérifier la découverte avec `list_solvers()` et l'ajout avec `bench.add_solver("<nom>")`. Pour une VI, déclarer aussi `solver_type = "vi"` dans la classe afin d'inclure les tests de précision dédiés.

Déclarer toute dépendance dans `pyproject.toml`, avec un extra pour un backend optionnel. Son absence doit laisser le cœur NumPy/SciPy et le catalogue utilisables.

## Contrat scientifique

- Documenter l'algorithme, sa référence éventuelle, les adaptations, l'initialisation et le critère d'arrêt. Docstrings courtes ; pour chaque option publique, préciser les valeurs acceptées et leur sens. Toute source aléatoire doit être reproductible après réinitialisation des seeds NumPy et Python, y compris les générateurs privés.
- `final_precision` borne l'erreur absolue sur la valeur. Pour la valeur retournée, vérifier `max(abs(T(V) - V)) / (1 - discount) <= final_precision`, avec l'opérateur de Bellman optimal complet. Un critère fondé sur le span seul ne suffit pas.
- Ne pas modifier implicitement récompenses, actualisation, conventions numériques, comportement aléatoire des solveurs existants ou cache de référence optimale par VI.
- Signaler une précision non atteinte ou une limite d'itérations par un échec explicite. Le benchmark mesure constructeur et `run()`, conversions et corrections incluses.

Les budgets du constructeur sont des entiers strictement positifs : `max_iterations` pour la boucle principale, `max_evaluation_iterations` pour l'évaluation d'une politique, `max_projected_iterations` pour un lot projeté et `max_fallback_iterations` pour le repli en espace complet. `bellman_iterations` et `aggregation_iterations` fixent le nombre de mises à jour par lot. Consulter les signatures pour les options disponibles et leurs valeurs par défaut ; les profils définis par le nom du solveur conservent leurs choix algorithmiques. Les constructeurs n'exposent plus `verbose` ; les backends restent silencieux et `Benchmark.run(verbose=...)` contrôle uniquement l'affichage du benchmark.

Pour `mdptoolbox_vi` et `mdptoolbox_vigs`, `max_iterations` est transmis au paramètre public `max_iter` ; MDPToolbox peut le remplacer par sa borne calculée lors de l'initialisation. Cette borne est conservée. `mdptoolbox_vi` utilise directement `ValueIteration`, avec ses contrôles natifs et ses limites, notamment les échecs d'initialisation sur certaines récompenses constantes ou certaines transitions mélangeantes dans pymdptoolbox 4.0b3.

`mdpforge_vi`, `mdpforge_vi_reward_init`, `mdpforge_mpi` et `mdpforge_mpi_reward_init` utilisent `GenericSolver._check_iteration()` : `iterations` compte les itérations, et `stop_reason` indique `converged`, `max_iterations`, `non_finite` ou `stagnation`. Un échec lève une `RuntimeError` ; une convergence obtenue à la dernière itération autorisée est acceptée. Les budgets MDPSolver concernent l'évaluation interne ; l'API installée ne fournit pas de plafond externe. Les variantes MPIGS/MPISOR, avec ou sans initialisation explicite, utilisent désormais toutes `max_evaluation_iterations=10` par défaut.

Les initialisations configurables se nomment `initial_value` et `initial_policy` ; les profils `mdpsolver_mpigs_init` et `mdpsolver_mpisor_init` conservent une initialisation nulle fixe. La seed privée de CMA-ES se nomme `seed`. Les méthodes de raffinement utilisent `refinement`. Ces noms remplacent les anciennes options sans alias ; adapter les appels qui utilisaient `split_method`, `projected_steps`, `bellman_updates`, `iter_agg`, `iter_bellman`, `random_seed`, `max_policy_iterations`, `max_iter_pi`, `max_iter_ksp`, `max_eval_iter` ou `evaluation_sweeps`.

## Validation

Les tests génériques de [`tests/test_workflows.py`](../tests/test_workflows.py) découvrent automatiquement les fichiers lors de la collecte et vérifient l'interface, la préservation des entrées, la reproductibilité, le résidu, la valeur et la qualité de politique. Ils utilisent les options par défaut, avec notamment `discount=0.999` et `final_precision=1e-4` ; deux exécutions avec les mêmes seeds doivent reproduire les valeurs et les valeurs de politique à `1e-12` près. Ajouter de petits tests déterministes pour les options et comportements spécifiques, avec une référence analytique ou indépendante.

Utiliser Python 3.11+ dans Conda `benchmark`. **Ne lancer pytest ou Ruff que sur demande explicite** ; alors exécuter la suite pytest complète, tous backends installés inclus (ignorer uniquement les dépendances absentes). Commandes dans [AGENTS.md](../AGENTS.md). Pas de benchmark lourd automatique. Rapporter les changements, contrôles réellement exécutés, résultats et limites restantes.
