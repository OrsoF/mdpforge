# Cahier des charges pour ajouter un modèle

Ce guide définit le contrat d'intégration d'un nouveau modèle dans le catalogue
mdpforge. **DOIT** indique une exigence d'intégration ; **DEVRAIT** indique une
recommandation dont une exception doit être expliquée. Il complète les
[consignes du dépôt](../AGENTS.md).

Un modèle peut être accepté avec certaines tailles indisponibles. Il doit alors
expliquer cette limitation et fournir au moins une petite instance exploitable
pour les tests. Les budgets de temps sont des objectifs de calibration, pas des
garanties indépendantes de la machine et du solveur.

## 1. Définition scientifique

Le module DOIT documenter :

- la signification des états et leur encodage ;
- les actions, leur disponibilité et le traitement des actions invalides ;
- la dynamique et les probabilités de transition ;
- les récompenses ou coûts, leur signe et leur échelle ;
- les états terminaux et leur représentation, notamment les états absorbants ;
- les paramètres, leurs valeurs par défaut et leurs domaines de validité.

Une référence bibliographique DOIT être fournie pour un modèle repris d'une
publication ou d'un environnement existant. Toute adaptation DOIT être explicite :
discrétisation, réduction à un seul agent, changement de récompense, simplification
de la dynamique ou suppression d'états. Une création originale peut déclarer
`reference=None`, en indiquant son objectif et ses hypothèses.

Le modèle DOIT être un MDP fini utilisable avec un critère actualisé
`0 < discount < 1`. Il ne doit pas dépendre d'un solveur particulier. La
normalisation des récompenses ne doit pas être appliquée implicitement.

## 2. Interface et construction

Le fichier DOIT être placé dans `src/mdpforge/models/<nom>.py`, avec un nom en
minuscules et des underscores, sans underscore initial. Le catalogue découvre
automatiquement ces fichiers : aucune liste centrale de noms n'est à compléter.

Le module DOIT exposer une classe `Model(MDP)` avec un constructeur et une méthode
`_build_model()`. Dès la fin du constructeur :

- `state_dim` et `action_dim` DOIVENT être des entiers Python strictement positifs,
  décrivant les dimensions réelles ;
- `name` DOIT être une chaîne non vide identifiant la configuration ;
- les paramètres invalides DOIVENT provoquer une erreur explicite ;
- le lien entre dimensions demandées et dimensions réelles DOIT être documenté.

Si le modèle impose une grille, un nombre fixe d'actions ou des tailles discrètes,
le constructeur DOIT exposer les dimensions obtenues après adaptation. Il ne doit
pas attendre `_build_model()` pour les définir. Le constructeur DEVRAIT rester
léger et laisser la construction des matrices à `_build_model()`.

L'utilisateur appelle `create_model()`, qui gère la construction, le cache et la
conversion finale en CSR. Une recette ne DOIT pas remplacer ce mécanisme pour
effectuer une conversion propre à un backend de solveur.

Après `create_model()`, le modèle DOIT exposer :

| Attribut | Contrat |
|---|---|
| `transition_matrix` | Liste de `action_dim` matrices SciPy `csr_matrix`, chacune de forme `(state_dim, state_dim)` |
| `transition_matrix[a][s, t]` | Probabilité de passer de l'état `s` à l'état `t` sous l'action `a` |
| `reward_matrix` | Tableau NumPy de forme `(state_dim, action_dim)` |
| `reward_matrix[s, a]` | Récompense immédiate espérée pour l'état `s` et l'action `a` |

Les probabilités DOIVENT être positives ou nulles et chaque ligne DOIT sommer à
1 à la tolérance du validateur du dépôt, actuellement `1e-6`. Toutes les valeurs
DOIVENT être finies. Une récompense dépendant de l'état suivant doit être ramenée
à son espérance sous la transition correspondante.

La construction DEVRAIT utiliser directement des structures creuses lorsque les
transitions sont creuses, sans créer inutilement des tableaux denses `(A, S, S)`.

### Exemple de recette minimale

Cet exemple illustre l'interface. Pour en faire un module du catalogue, compléter
également les métadonnées décrites dans la section suivante.

```python
import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

TEST_PARAMETERS = {"state_dim": 4, "action_dim": 2}


class Model(MDP):
    """Chaîne contrôlée : coût -1 par étape, état 0 absorbant de coût nul."""

    def __init__(self, state_dim: int = 32, action_dim: int = 2):
        if not isinstance(state_dim, int) or state_dim < 2:
            raise ValueError("state_dim must be an integer >= 2")
        if action_dim != 2:
            raise ValueError("This model has exactly two actions")
        super().__init__(state_dim, 2)
        self.name = f"{self.state_dim}_2_example_chain_v1"

    def _build_model(self):
        states = np.arange(self.state_dim)
        left = np.maximum(states - 1, 0)
        right = np.minimum(states + 1, self.state_dim - 1)
        right[0] = 0
        self.transition_matrix = [
            csr_matrix(
                (np.ones(self.state_dim), (states, destination)),
                shape=(self.state_dim, self.state_dim),
            )
            for destination in (left, right)
        ]
        self.reward_matrix = -np.ones((self.state_dim, self.action_dim))
        self.reward_matrix[0, :] = 0
```

## 3. Métadonnées, tags et tailles

Chaque nouveau module DOIT définir un dictionnaire littéral `METADATA`, avec
`category`, `description`, `reference`, `tags` et `sizes`. Il est lu par
`ast.literal_eval` : pas d'appel de fonction, de tableau NumPy, de compréhension
ou de référence à une variable dans ce dictionnaire.

Les catégories actuelles sont `synthetic`, `navigation`, `control`, `queueing`,
`resource_management` et `games`. Le contributeur DEVRAIT réutiliser les tags
existants, accessibles via `list_models()`, et justifier l'ajout d'un nouveau tag.
Plusieurs tags peuvent décrire un modèle :

- `random` : la génération des paramètres, récompenses ou transitions est
  aléatoire ; avoir des transitions stochastiques ne suffit pas ;
- `maze` : navigation dans un labyrinthe ;
- `real-world` : application simulée inspirée d'un système réel ;
- `grid-world`, `racetrack`, `puzzle` : descriptions complémentaires du problème.

`sizes` DOIT déclarer les trois labels `small`, `medium` et `large`. Pour chaque
taille disponible :

| Champ | Signification |
|---|---|
| `state_dim` | Nombre réel d'états attendu après construction |
| `parameters` | Arguments exacts du constructeur pour obtenir cette configuration |
| `source` | `measured` si la configuration a été mesurée, `inferred` si elle résulte d'une extrapolation |

Une configuration qui n'a pas encore été mesurée peut utiliser `configured`
(choix explicite), `analytical` (dimensions calculées analytiquement) ou
`parameterized` (extension par paramètres). Ces origines ne constituent pas une
validation des budgets de temps et doivent être expliquées dans la recette.

Une taille indisponible DOIT utiliser `state_dim=None`, `parameters=None`,
`source="unavailable"` et une `reason` non vide. Il ne faut pas dupliquer une même
configuration sous plusieurs tailles pour donner l'impression que le modèle
est extensible.

Exemple pour une recette originale dont les tailles restent à calibrer :

```python
METADATA = {
    "category": "synthetic",
    "description": "Controlled chain with a zero-cost absorbing boundary.",
    "reference": None,
    "tags": ["synthetic"],
    "sizes": {
        "small": {
            "state_dim": None,
            "parameters": None,
            "source": "unavailable",
            "reason": "Calibration pending.",
        },
        "medium": {
            "state_dim": None,
            "parameters": None,
            "source": "unavailable",
            "reason": "Calibration pending.",
        },
        "large": {
            "state_dim": None,
            "parameters": None,
            "source": "unavailable",
            "reason": "Calibration pending.",
        },
    },
}
```

La découverte via `list_models()` DOIT fonctionner sans importer la recette ni
construire ses matrices. `list_models(size="small", type="random")` filtre les
métadonnées ; `load_model(size="small", type="random")` retourne un itérateur de
modèles construits progressivement. `type` correspond à une catégorie ou un tag.
Une taille indisponible est exclue de cette sélection ; les erreurs d'import ou
de construction sont remontées pendant l'itération.

Le chargement du paquet installé ne DOIT pas dépendre d'un fichier de calibration
dans `artifacts/`. Les configurations utilisables doivent être dans le module.

## 4. Reproductibilité et identité du cache

Une génération aléatoire DOIT documenter sa seed et reproduire les mêmes données
avec les mêmes paramètres et la même seed. Les nouveaux modèles DEVRAIENT exposer
un paramètre `seed` et utiliser un générateur local, par exemple
`np.random.default_rng(seed)`, sans modifier l'état aléatoire global.

`name` DOIT distinguer les configurations susceptibles de produire des matrices
différentes : dimensions effectives, paramètres de dynamique et de récompense,
seed et géométrie personnalisée si applicable. Pour une géométrie complexe, une
empreinte stable peut être utilisée. Deux configurations ayant le même nombre
d'états ne sont pas nécessairement le même MDP.

Le cache de matrices utilise `model.name` ; le cache de valeurs optimales conserve
la clé `<discount>_<model.name>` et une référence calculée par VI à précision
`1e-3`. Une modification de la dynamique ou des récompenses DOIT traiter les
caches devenus obsolètes, par exemple avec une nouvelle version dans le nom.
Changer uniquement les descriptions ou les tags ne nécessite pas une nouvelle
identité numérique.

Les tests de génération DOIVENT isoler le cache. `create_model(save=False)`
empêche l'écriture de nouvelles matrices mais peut encore lire un cache existant.
La fixture `isolated_model_cache` est disponible dans les tests du dépôt.

## 5. Validation et calibration

Le module DOIT fournir un `TEST_PARAMETERS` littéral pour une instance petite et
rapide. La suite découvre automatiquement les recettes et utilise ces paramètres
pour le contrôle générique dans `tests/test_workflows.py`.

Des tests spécifiques DOIVENT vérifier au moins une propriété scientifique qui
ne découle pas simplement de la forme des tableaux : transition ou récompense
connue, dynamique d'un état terminal, valeur analytique ou qualité d'une politique.
Ils DEVRAIENT couvrir les frontières, les paramètres invalides et les arrondis de
dimension pertinents. Un modèle aléatoire DOIT avoir un test de reproductibilité
avec un cache isolé. Les tests supplémentaires peuvent être placés dans
`tests/test_<nom>_model.py`.

La validité numérique DOIT être vérifiée avec
`mdpforge.core.validation.validate_model(model)`. Pour un contrôle par solveur,
vérifier aussi la qualité de la solution : le résidu de Bellman complet divisé
par `1 - discount` borne l'erreur absolue sur la valeur. Un solveur rapide qui
manque la précision demandée ne fournit pas un temps de calibration valide.

Les objectifs de calibration sont environ **1 s / 5 s / 30 s** par couple
modèle-solveur pour `small` / `medium` / `large`, à `discount=0.99`, précision
absolue `1e-3` et un thread. Les temps concernent la préparation et l'exécution du
solveur ; le temps de construction du modèle et la mémoire doivent être rapportés
séparément. Un échec ou un timeout DOIT être conservé comme tel, sans être traité
comme un temps de résolution réussi.

Le contributeur DOIT conserver la provenance des tailles disponibles : résultats
CSV et, pour une extrapolation, méthode et hypothèses dans un rapport associé.
Rapporter les paramètres, dimensions réelles, seeds, solveurs, versions, précision,
machine, répétitions et limites de temps. Plusieurs répétitions sont recommandées.
Une taille `inferred` ne garantit pas un temps réel de résolution ; préciser les
solveurs couverts et exclure les mesures échouées du calcul.

Le [script de calibration](../benchmark_sizes.py) mesure uniquement `small` et
`medium`. Il utilise ses propres `STATE_DIM_REQUESTS` et ses valeurs par défaut,
pas `METADATA["sizes"]`. Adapter les demandes si la nouvelle recette en a besoin.
Lorsqu'une calibration est demandée, limiter le lancement au modèle ajouté :

```text
conda run -n benchmark python benchmark_sizes.py --models my_model --repeats 3 --output artifacts/results/my_model_sizes.csv
```

Les mesures lourdes ne font pas partie de l'ajout automatique d'un modèle. Les
scripts temporaires vont dans `artifacts/tmp/` ; les résultats et rapports de
référence sont conservés dans `artifacts/results/`.

Pour vérifier les paramètres déjà enregistrés dans `METADATA`, utiliser
`check_model_sizes.py`, qui lit les configurations actuelles :

```text
conda run -n benchmark python check_model_sizes.py --dimensions-only --models my_model
conda run -n benchmark python check_model_sizes.py --build-only --sizes small medium large --models my_model
conda run -n benchmark python check_model_sizes.py --sizes large --models my_model
```

Le premier contrôle vérifie les constructeurs, le deuxième construit et valide
les matrices, le troisième mesure aussi les solveurs et vérifie la précision.
Chaque étape est isolée dans un processus avec un timeout. Les résultats sont
enregistrés progressivement dans des CSV distincts par mode, avec la dimension
attendue, la dimension obtenue, la qualité de solution et le respect du budget.
Le script retourne un code non nul si une configuration est indisponible, échoue
ou dépasse son budget. Il reprend les cas déjà terminés ; choisir un autre
`--output` pour refaire les mêmes mesures.

## 6. Dépendances et vérifications

Toute nouvelle dépendance DOIT être déclarée dans `pyproject.toml`. Une intégration
optionnelle DOIT utiliser un extra approprié et ne pas empêcher l'import du cœur
NumPy/SciPy ou la consultation du catalogue. Les tests des dépendances optionnelles
sont ignorés uniquement lorsqu'elles ne sont pas installées.

Utiliser Python 3.11+ dans l'environnement Conda `benchmark`. Conformément à
AGENTS.md, ne pas lancer pytest ou Ruff automatiquement : les exécuter lorsqu'ils
sont explicitement demandés. Les commandes complètes depuis la racine sont :

```text
conda run -n benchmark python -m pytest
conda run -n benchmark python -m ruff check .
conda run -n benchmark python -m ruff format --check .
```

Rapporter précisément les contrôles exécutés, leurs résultats et ceux qui restent
à faire. Distinguer les contrôles locaux de la CI et les échecs existants des
régressions introduites.

## Checklist d'intégration

- [ ] États, actions, dynamique, récompenses et états terminaux documentés.
- [ ] Référence ou origine du modèle précisée ; adaptations expliquées.
- [ ] `state_dim`, `action_dim` et `name` définis dès le constructeur.
- [ ] Dimensions réelles, paramètres invalides et paliers documentés.
- [ ] Matrices CSR et récompenses NumPy conformes au validateur.
- [ ] `METADATA` littéral, catégorie, tags et trois labels de taille déclarés.
- [ ] Tailles disponibles reproductibles ; tailles indisponibles justifiées.
- [ ] Calibration mesurée ou inférée accompagnée de sa provenance, si disponible.
- [ ] Génération aléatoire reproductible et identité de cache suffisante.
- [ ] `TEST_PARAMETERS` fourni et tests scientifiques ajoutés.
- [ ] Dépendances déclarées et intégrations optionnelles isolées.
- [ ] Vérifications exécutées et limitations restantes rapportées explicitement.
