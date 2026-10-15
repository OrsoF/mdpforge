# Expérience : évaluation des politiques par CMA-ES

Le solveur `mdpforge_cmaes_pi` alterne une évaluation approchée par CMA-ES et une amélioration gloutonne de politique. Il utilise les matrices du modèle ; aucune simulation Monte Carlo, résolution linéaire ou itération de valeur de secours n'est ajoutée.

## Objectif et arrêt

Pour chaque politique déterministe `π`, CMA-ES minimise `Fπ(v) = 0.5 * ||v - γPπv - rπ||₂²`, en repartant de la dernière valeur calculée. Pour `0 < γ < 1`, cette quadratique est strictement convexe et son minimiseur est `Vπ`. L'objectif est calculé depuis les matrices creuses ; son évaluation est déterministe, sa minimisation reste aléatoire et approchée.

L'évaluation interne exige `||v - γPπv - rπ||∞ <= ε(1 - γ)/4`. Ensuite, l'amélioration utilise `argmax_a [r(s,a) + γP[a]v]`. Une politique stable n'est pas un critère d'arrêt suffisant.

Avant de retourner, le solveur exige `(||T*v - v||∞ + ||Tπv - v||∞) / (1 - γ) <= ε`. Cette condition borne à la fois `||v - V*||∞` et, par inégalité triangulaire, `||Vπ - V*||∞`. Une limite de budget ou un arrêt de CMA-ES avant certification provoque une erreur explicite.

Cette expérience ne revendique ni amélioration monotone entre politiques, ni convergence globale de la boucle approchée, ni avantage de vitesse sur PI/MPI. La convexité de chaque sous-problème ne suffit pas à établir ces propriétés. La covariance dense de CMA-ES demande une mémoire quadratique en nombre d'états ; commencer avec de petites instances. Les modèles à un seul état sont refusés, car pycma ne prend pas officiellement en charge l'optimisation en dimension un.

## Utilisation

Installer l'extra `cma` dans l'environnement Conda `benchmark` :

```text
conda run -n benchmark python -m pip install -e ".[cma]"
```

```python
from mdpforge import Benchmark
from mdpforge.models.chain_walk import Model

model = Model(state_dim=8)
model.create_model()
bench = Benchmark(default_solvers=False)
bench.add_mdp(model)
bench.add_solver("mdpforge_pi")
bench.add_solver("mdpforge_cmaes_pi")
result = bench.run(discount=0.9, precision=1e-3, repeats=3, seed=17)
```

Le catalogue classe cette variante dans `family="mpi"`, `backend="cma"`, `update_scheme="mixed"`. Le chargement collectif ignore le solveur si `cma` manque ; un ajout explicite remonte cette absence. Le cœur NumPy/SciPy et la consultation du catalogue restent indépendants de cet extra.

| Option du constructeur | Valeurs acceptées et sens |
|---|---|
| `final_precision` | Réel fini `> 0`, défaut `1e-3` : borne finale `ε` |
| `initial_policy` | `None` (action zéro partout) ou tableau d'entiers `(S,)` dans `[0, A)` |
| `initial_value` | `None` (zéros) ou tableau fini `(S,)` : première moyenne de recherche |
| `sigma0` | Réel fini `> 0`, défaut `1` : écart-type initial à chaque évaluation |
| `max_generations` | Entier `>= 1`, défaut `2000` : générations CMA-ES par politique |
| `max_policy_iterations` | Entier `>= 1`, défaut `100` : évaluations de politique au maximum |
| `random_seed` | `None` ou entier dans `[0, 2**32)` : seed privée ; `None` la dérive du flux NumPy initialisé par le benchmark |

Passer les options par `bench.add_solver(..., sigma0=..., max_generations=...)` ; le benchmark fournit lui-même `final_precision`. Laisser `random_seed=None` pour obtenir les seeds des répétitions du benchmark. `policy_iterations`, `generations` et `objective_evaluations` sont disponibles sur une instance directe du solveur. Le temps mesuré inclut la recherche et les contrôles de précision.

Références : [Hansen, tutoriel CMA-ES](https://arxiv.org/abs/1604.00772) ; [interface pycma](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html). Il s'agit d'une combinaison expérimentale, pas de l'implémentation revendiquée d'un résultat de convergence de ces sources.
