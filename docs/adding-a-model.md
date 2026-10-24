# Ajouter un modèle MDP

Contrat d'intégration au catalogue mdpforge, à lire avec [AGENTS.md](../AGENTS.md). **DOIT** = obligatoire ; **DEVRAIT** = recommandé (justifier toute exception).

## 1. Définition et interface

Créer `src/mdpforge/models/<nom>.py` (minuscules, underscores, sans underscore initial). La découverte est automatique. Exposer :

- `Model(MDP)` avec `__init__(state_dim, ...)` et `_build_model()` ; laisser `create_model()` gérer construction, cache et conversion CSR.
- Dès la fin du constructeur : `state_dim` et `action_dim` = **dimensions réelles**, entiers Python > 0 ; `requested_state_dim` = cible reçue avant adaptation ; paramètres invalides = erreur explicite. Documenter l'adaptation des dimensions. Garder le constructeur léger.
- `name = "<state_dim>_<action_dim>_<nom>[_paramètres]"` : inclure **toute** option qui change les matrices (dynamique, récompenses, seed, géométrie), et versionner le nom quand leur définition change. Pour les flottants, préférer `str(value)` sans arrondi supplémentaire.
- Un `TEST_PARAMETERS` littéral décrivant une instance petite et rapide.

Après `create_model()`, les données doivent respecter :

| Attribut | Type et sens |
| --- | --- |
| `transition_matrix` | Liste de `A` matrices SciPy **CSR** `(S, S)` ; `[a][s, t] = P(t \| s, a)` |
| `reward_matrix` | Tableau NumPy `(S, A)` ; `[s, a] = E[r \| s, a]` |

`S = state_dim`, `A = action_dim`. Toutes les valeurs sont finies ; `P >= 0` et chaque ligne somme à 1 (tolérance actuelle : `1e-6`). Calculer l'espérance si la récompense dépend de l'état suivant. Préférer les structures creuses ; ne pas convertir le modèle spécifiquement pour un backend.

**Documentation scientifique obligatoire** : encodage des états, actions valides/invalides, transitions, récompenses (signe/échelle), terminaux/absorbants, paramètres (défauts/domaines), référence et adaptations éventuelles. Pour un modèle original : `reference=None`, objectif et hypothèses explicites. MDP fini actualisé (`0 < discount < 1`), indépendant des solveurs, sans normalisation implicite des récompenses.

## 2. Métadonnées et tailles

Déclarer `METADATA` comme **dictionnaire littéral** (lu avec `ast.literal_eval`, donc sans calculs ni références à des variables), avec :

- `category` : **une** catégorie parmi `abstract`, `navigation`, `physical_control`, `service_operations`, `asset_management`, `games_and_puzzles` ; choisir d'après la tâche principale, pas le décor ni le mécanisme secondaire.
- `description` : tâche et particularités du modèle.
- `reference` : source bibliographique ou `None` pour une création originale.
- `sizes` : **exactement** les entrées `small`, `medium`, `large`.

Pour chaque taille disponible : `{"requested_state_dim": <entier positif>, "parameters": {<autres arguments>}}` (`parameters` peut être `{}` et ne contient jamais `state_dim`). La dimension effective vient **uniquement** du constructeur : ne pas la répéter dans le preset. Pour une taille indisponible : `{"requested_state_dim": None, "parameters": None, "reason": "<motif>"}`. Ne pas déguiser une même configuration en plusieurs tailles ; l'adaptation doit préserver la structure du modèle. Fournir au moins une petite instance exploitable pour les tests, même si certains presets sont indisponibles.

`list_models()` doit lire les métadonnées sans importer/construire le modèle. Vérifier `list_models(size="small", category="abstract")` (filtres par intersection) et `bench.add_mdp("<nom>", size="small")`. Un preset explicitement indisponible doit échouer, **sans** repli silencieux ; les ajouts collectifs ignorent seulement les dépendances optionnelles manquantes (avec avertissement). Aucun fichier de `artifacts/` ne doit être requis à l'import du paquet.

## 3. Reproductibilité et cache

- Génération aléatoire : documenter la seed, garantir les mêmes matrices pour les mêmes paramètres ; préférer un générateur local `np.random.default_rng(seed)`.
- `model.name` identifie le cache des matrices ; le cache de valeur optimale utilise `<discount>_<model.name>` (référence VI à `1e-3`). Changer l'identité/version si la dynamique ou les récompenses changent, pas pour une simple retouche documentaire.
- Isoler le cache dans les tests (`isolated_model_cache`) : `create_model(save=False)` empêche l'écriture, **pas** la lecture d'un cache existant.

## 4. Tests et calibration

**Validation minimale :**

1. `mdpforge.core.validation.validate_model(model)` et découverte automatique par `tests/test_workflows.py` via `TEST_PARAMETERS`.
2. Au moins un test **scientifique** indépendant des formes de tableaux (transition/récompense connue, terminal, valeur ou politique). Couvrir aussi les paramètres invalides et les adaptations de taille ; pour l'aléatoire, tester la reproductibilité cache isolé.
3. Si un solveur sert de contrôle, vérifier sa **précision réelle** : `||T(V) - V||∞ / (1 - discount) <= précision demandée` (Bellman optimal complet). Un échec/timeout n'est jamais une durée de résolution valide.

**Calibration** : objectifs indicatifs `small / medium / large ≈ 1 / 5 / 30 s` par couple modèle-solveur, avec `discount=0.99`, erreur absolue `1e-3`, un thread. Mesurer préparation + exécution du solveur ; rapporter séparément construction du modèle et mémoire. Conserver les CSV et la provenance : paramètres, dimensions demandées/réelles, seeds, solveurs/versions, précision, machine, répétitions, timeouts et éventuelles extrapolations (méthode, hypothèses, solveurs couverts). Les tailles peuvent rester indisponibles avec motif.

Depuis la racine, dans Conda `benchmark` (Python 3.11+) :

```bash
# Vérifier successivement dimensions, construction et résolution
conda run -n benchmark python check_model_sizes.py --dimensions-only --models <nom>
conda run -n benchmark python check_model_sizes.py --build-only --sizes small medium large --models <nom>
conda run -n benchmark python check_model_sizes.py --sizes large --models <nom>

# Calibrer uniquement le nouveau modèle si demandé
conda run -n benchmark python benchmark_sizes.py --models <nom> --repeats 3 --output artifacts/results/<nom>_sizes.csv
```

`benchmark_sizes.py` ne mesure que `small` et `medium` avec ses propres `STATE_DIM_REQUESTS` (pas les presets `METADATA`). `check_model_sizes.py` exécute chaque cas avec timeout, reprend ses résultats existants et renvoie un code non nul en cas d'indisponibilité, échec ou dépassement : changer `--output` pour forcer une nouvelle mesure. Scripts temporaires : `artifacts/tmp/` ; résultats conservés : `artifacts/results/`. Ne pas lancer de mesure lourde automatiquement.

Toute dépendance va dans `pyproject.toml` ; pour une dépendance optionnelle, utiliser un extra sans empêcher l'import du cœur ou l'accès au catalogue. **Ne lancer pytest/Ruff que sur demande explicite** (voir [AGENTS.md](../AGENTS.md)). Toujours indiquer les contrôles réellement exécutés, les résultats et les limites restantes.

## Vérification avant intégration

- [ ] Contrat scientifique, dimensions et matrices valides.
- [ ] `METADATA`, trois tailles (ou motifs), `TEST_PARAMETERS` et découverte fonctionnels.
- [ ] Seed et `name` garantissent reproductibilité et identité du cache.
- [ ] Tests scientifiques et validation numérique prévus/exécutés selon la demande.
- [ ] Provenance de calibration conservée si mesure réalisée ; dépendances déclarées.
