# Ajouter un solveur

Contrat d'intégration au catalogue mdpforge ; voir aussi [AGENTS.md](../AGENTS.md).

## 1. Interface

Créer `src/mdpforge/solvers/<nom>.py` (minuscules, underscores, sans underscore initial), exposant `Solver`. Découverte automatique : `<nom>` est l'identifiant de `list_solvers()` et `bench.add_solver("<nom>")`.

- API : `Solver(model, discount, ...)` avec options explicites, `assert 0 < discount < 1`, puis `run()` sans argument. Pour un solveur itératif : `final_precision` finie, > 0 ; **ne pas** la proposer si inutilisée (`mdpforge_pi`, `mdptoolbox_pi`, `gurobi_primal`, `gurobi_dual`). Le benchmark la transmet seulement si acceptée.
- Données disponibles : `state_dim=S`, `action_dim=A`, `transition_matrix[a]` = SciPy CSR `(S,S)`, `reward_matrix` = NumPy `(S,A)`. Aucune méthode supplémentaire du modèle n'est requise. **Ne pas modifier les entrées**, ni leur format ; convertir uniquement dans le solveur, sans dense inutile.
- `run()` → [`SolveResult`](../src/mdpforge/core/solve_result.py) : `value` NumPy fini `(S,)`, `policy` = `None` ou entiers `(S,)` dans `[0,A)`, `runtime` fini ≥ 0 (s), `iterations` entier/`None`, `stop_reason` texte/`None`. Exposer aussi `name`, `value`, `policy`, `runtime` sur l'instance.

**Préférer [`GenericSolver`](../src/mdpforge/core/solver.py)** : appeler son constructeur, définir `self.name`, implémenter `_solve()` (valeurs et éventuellement politique). `run()` mesure le temps et construit le résultat. Pour un `run()` spécifique, renseigner les attributs et retourner `SolveResult.from_solver(self)`. Exemple : [`mdpforge_vi.py`](../src/mdpforge/solvers/mdpforge_vi.py).

## 2. Catalogue

Déclarer `METADATA` **littéral**, lisible sans importer le backend, avec quatre chaînes non vides :

- `display_name` (libellé), `description` (une phrase) ;
- `family` : `vi`, `pi`, `mpi`, `lp`, `experimental` ;
- `backend` : `mdpforge`, `mdptoolbox`, `mdpsolver`, `marmote`, `madupite`, `mdpax`, `gurobi`, `cma`.

Pour une VI, ajouter `solver_type = "vi"` (tests dédiés). Vérifier `list_solvers()` et `bench.add_solver("<nom>")`. Déclarer les dépendances dans `pyproject.toml` ; backend optionnel = extra, sans bloquer le cœur NumPy/SciPy ni le catalogue.

## 3. Précision et comportement

- Documenter algorithme/référence, adaptations, initialisation, arrêt et options (valeurs admises). Aléatoire reproductible après reset des seeds NumPy/Python, y compris pour les RNG privés.
- **Condition de précision** : `||T(V) − V||∞ / (1 − discount) <= final_precision` avec **Bellman optimal complet** ; ni variation entre itérations ni span seuls ne suffisent.
- Précision non atteinte, limite dépassée, résultat non fini ⇒ **échec explicite**. Le benchmark mesure constructeur + `run()`, conversions/corrections incluses.
- Backends : utiliser leurs **API publiques**, ne pas contourner initialisation/contrôles/bornes ; laisser remonter les erreurs natives. Ne pas changer implicitement récompenses, discount, conventions numériques, solveurs existants ni cache VI de référence.

**Options spécifiques** : budgets entiers > 0 : `max_iterations`, `max_evaluation_iterations`, `max_projected_iterations`, `max_fallback_iterations` ; mises à jour par lot : `bellman_iterations`, `aggregation_iterations`. Respecter les signatures/profils existants. Noms actuels : `initial_value`, `initial_policy`, `seed` (CMA-ES), `refinement` ; **aucun alias historique**. Pas de `verbose` au constructeur : affichage via `Benchmark.run(verbose=...)`.

**Exceptions existantes à respecter** : `mdptoolbox_vi`/`mdptoolbox_vigs` passent `max_iterations` à `max_iter`, sans neutraliser les limites natives (`mdptoolbox_vi` utilise `ValueIteration`). MDPSolver ne borne que l'évaluation interne. MPIGS/MPISOR ont `max_evaluation_iterations=10` par défaut, et les profils `*_init` concernés gardent leur initialisation nulle. Les VI/MPI mdpforge appellent `GenericSolver._check_iteration()` : `stop_reason` = `converged`, `max_iterations`, `non_finite` ou `stagnation` ; échec → `RuntimeError`, convergence au dernier tour acceptée.

## 4. Validation

[`tests/test_workflows.py`](../tests/test_workflows.py) découvre et contrôle interface, entrées intactes, reproductibilité, résidu, valeurs et politiques (`discount=0.999`, `final_precision=1e-4`, tolérance de reproductibilité `1e-12` sur valeurs et valeurs de politique). Ajouter des tests déterministes des options spécifiques avec référence indépendante/analytique.

Python 3.11+, Conda `benchmark`. **Ne lancer pytest/Ruff que sur demande explicite** ; si demandés, suite complète, tous backends installés (ignorer seulement les dépendances absentes). Pas de benchmark lourd automatique. Documenter changements, vérifications réelles, résultats et limites ; commandes dans [AGENTS.md](../AGENTS.md).
