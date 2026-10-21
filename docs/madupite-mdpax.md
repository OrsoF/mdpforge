# Solveurs madupite et MDPax

Ces adaptateurs facultatifs acceptent les mêmes MDP finis actualisés que les solveurs mdpforge (`transition_matrix`: liste de CSR de forme `(S,S)`, `reward_matrix`: tableau `(S,A)`). Ils conservent les récompenses, maximisent le retour actualisé et vérifient indépendamment le résidu de Bellman après résolution. Les tableaux d'entrée ne sont pas modifiés.

## Installation

- **MDPax 0.2.2+** : `pip install 'mdpforge[mdpax]'`. Installe JAX ; pour une exécution sur GPU, suivre également [les instructions officielles JAX](https://docs.jax.dev/en/latest/installation.html). Le backend active `jax_enable_x64=True` afin de préserver la précision des matrices converties. Son premier lancement peut coûter cher à cause de la compilation JIT.
- **madupite 1.x** : les dépendances C++/PETSc/MPI nécessitent [l'environnement et la procédure d'installation officiels](https://github.com/madupite/madupite#installation) (notamment Linux) ; ce backend n'a pas d'extra pip autonome. Installer ensuite mdpforge dans le même environnement. Le wrapper ne lance pas `mpirun` et **ne prend en charge qu'un rang MPI** ; utiliser directement madupite pour des benchmarks distribués.

L'absence de ces backends ne gêne ni `import mdpforge`, ni `list_solvers()`. Lors d’une sélection groupée par filtre (`benchmark.add_solver(backend="mdpax")`), une dépendance manquante provoque un avertissement et le solveur est ignoré ; un `benchmark.add_solver("mdpax_vi")` explicite, comme un import direct, nécessite effectivement sa dépendance.

## Utilisation

```python
from mdpforge import Benchmark, list_solvers
from mdpforge.core.mdp import MDP

mdp = MDP.from_matrices(
    "chain_example",
    [
        [[0, 1, 0], [0, 0, 1], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
    [[0, 0], [2, 0], [0, 0]],
)

print(list_solvers(backend=["madupite", "mdpax"]))
benchmark = Benchmark()
benchmark.add_mdp(mdp)
benchmark.add_solver("madupite_ipi")
benchmark.add_solver("madupite_vi")
benchmark.add_solver("madupite_pi")
benchmark.add_solver("madupite_opi", max_evaluation_iterations=20)
benchmark.add_solver("madupite_ipi_bicgstab")
benchmark.add_solver("madupite_ipi_tfqmr")
benchmark.add_solver("madupite_ipi_gmres")
benchmark.add_solver("mdpax_vi", max_iterations=20_000)
benchmark.add_solver("mdpax_pi", max_evaluation_iterations=10_000)
results = benchmark.run(discount=0.9, precision=1e-4, repeats=2)
```

On peut aussi instancier directement `Solver(mdp, discount=0.9, final_precision=1e-4)` dans les modules `mdpforge.solvers.madupite_vi`, `mdpforge.solvers.madupite_pi`, `mdpforge.solvers.madupite_opi`, `mdpforge.solvers.madupite_ipi`, `mdpforge.solvers.madupite_ipi_bicgstab`, `mdpforge.solvers.madupite_ipi_tfqmr`, `mdpforge.solvers.madupite_ipi_gmres`, `mdpforge.solvers.mdpax_vi` ou `mdpforge.solvers.mdpax_pi`, puis appeler `run()` et lire `value`, `policy`, `runtime`.

## Conversion et coût scientifique

| Backend | Algorithme | Adaptation des données | Limite spécifique |
| --- | --- | --- | --- |
| `madupite_ipi` | Inexact Policy Iteration / PETSc | CSR `(A,S,S)` vers CSR `(S*A,S)` avec lignes ordonnées `s*A+a`, écriture PETSc temporaire, lecture ASCII de la valeur et politique | Un rang MPI ; options PETSc globales dans madupite 1.0 |
| `madupite_vi` | Value Iteration / Richardson (`ksp_type=richardson`, `pc_type=none`) | Même conversion PETSc que `madupite_ipi` | Un pas KSP par mise à jour (`-max_iter_ksp=1` et `-ksp_max_it=1`) |
| `madupite_pi` | Policy Iteration / Richardson préconditionné par LU | Même conversion PETSc que `madupite_ipi` | LU nécessite une factorisation directe disponible, à utiliser sur petits MDP en séquentiel |
| `madupite_opi` | Optimistic/Modified Policy Iteration / Richardson | Même conversion PETSc que `madupite_ipi` | `max_evaluation_iterations=20` par défaut, contrôlé par `-ksp_max_it` |
| `madupite_ipi_bicgstab` | Inexact PI / PETSc `bcgs` (BiCGStab) | Même conversion PETSc que `madupite_ipi` | `max_evaluation_iterations=1000` par défaut |
| `madupite_ipi_tfqmr` | Inexact PI / PETSc `tfqmr` | Même conversion PETSc que `madupite_ipi` | `max_evaluation_iterations=1000` par défaut |
| `madupite_ipi_gmres` | Inexact PI / PETSc `gmres` | Même conversion PETSc que `madupite_ipi` | Identique à l'option par défaut de `madupite_ipi`, mais identifiant distinct |
| `mdpax_vi` | Value Iteration / JAX | Chaque ligne CSR est transformée en liste d'événements (successeur, probabilité) avec padding JAX | `max_event_entries=2_000_000` par défaut |
| `mdpax_pi` | Policy Iteration / JAX avec évaluation itérative | Même conversion par événements, plutôt que matrice dense `(A,S,S)` | `max_evaluation_iterations=10_000` par défaut |

**Précision** : chaque solveur utilise une tolérance interne `final_precision * (1-discount)`, et la valeur finale est centrée puis vérifiée à partir de l'opérateur optimal de mdpforge. Une convergence insuffisante, une limite d'itérations ou une valeur non finie doivent aboutir à un échec explicite plutôt qu'à une réussite du benchmark. Cette vérification en double précision flottante n'est pas une preuve formelle tenant compte des erreurs d'arrondi.

**Comparaisons** : le temps du benchmark inclut la conversion, les écritures/lectures temporaires pour madupite et la compilation JAX pour MDPax lorsque celle-ci se produit. Les essais ne mesurent donc pas seulement le noyau algorithme. Le nombre d'événements JAX est le nombre maximal de successeurs non nuls d'une ligne CSR, répété pour toutes les lignes ; les problèmes très denses peuvent coûter beaucoup de mémoire JAX et sont refusés au-delà de `max_event_entries`. Cette représentation sert à comparer les *mêmes MDP explicites* ; elle ne reproduit pas l'avantage principal de MDPax sur des MDP structurels définis directement par fonctions.

**Profils madupite** : toutes les variantes partagent l'adaptateur `madupite_ipi`, qui écrit désormais explicitement `-max_iter_ksp` **et** l'option PETSc `-ksp_max_it` pour fixer le nombre d'itérations internes. La variante PI utilise `ksp_type=richardson`, `pc_type=lu` et un seul pas Richardson (une application du préconditionneur LU) : c'est une évaluation directe en arithmétique flottante, pas une garantie d'exactitude symbolique. Les variantes VI et OPI utilisent Richardson sans préconditionneur avec respectivement un pas et au plus `max_evaluation_iterations` pas. Les profils BiCGStab/TFQMR/GMRES utilisent respectivement `bcgs`, `tfqmr`, `gmres`. Le profil GMRES est un alias de configuration, pas un algorithme additionnel par rapport à `madupite_ipi` par défaut. Toutes les variantes vérifient indépendamment la précision retournée ; certaines peuvent échouer par non-convergence ou incompatibilité PETSc.

**madupite** : les chemins de fichiers sont isolés dans un dossier temporaire par exécution, mais la base d'options PETSc reste globale dans madupite 1.0. Éviter des exécutions madupite concurrentes au sein du même interpréteur ; les processus isolés de Benchmark sont préférables si nécessaire.

## Références des API

- [madupite : tutoriels et format](https://madupite.github.io/tutorial.html), [options](https://madupite.github.io/options.html).
- [MDPax : problème personnalisé](https://mdpax.readthedocs.io/en/latest/notebooks/create_custom_problem.html), [code source](https://github.com/joefarrington/mdpax).
- [PETSc KSP : solveurs et limites d’itérations](https://petsc.org/release/manual/ksp/).
