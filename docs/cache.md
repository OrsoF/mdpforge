# Cache des modèles et des références numériques

Le cache v2 privilégie la validité des expériences à la réutilisation agressive. Il ne lit **jamais automatiquement les anciens fichiers pickle** et n'utilise jamais le seul `model.name` pour identifier des matrices ou une fonction de valeur.

## Politiques de cache

`model.create_model(cache="readwrite")` utilise le cache s'il existe, sinon construit et enregistre le modèle. C'est la valeur par défaut. `cache="readonly"` lit seulement une entrée existante et construit le modèle en cas d'absence, sans rien écrire. Une entrée corrompue produit une exception `CacheCorruptError` en mode `readonly`. `cache="refresh"` sur une nouvelle instance non construite reconstruit les matrices et publie un nouvel artefact (une divergence de contenu pour une même recette est signalée). `cache="off"` ne lit et n'écrit aucun fichier de cache de modèle.

La compatibilité `save=True` / `save=False` reste assurée lorsque `cache` est omis : `save=False` équivaut à `cache="readonly"`, **pas** à `cache="off"`. Ne fournissez pas `save` et `cache` simultanément.

## Identité d'une recette

Les modèles **déterministes et reproductibles** doivent opter explicitement pour le cache, par exemple :

```python
class Model(MDP):
    CACHE_VERSION = 1

    def get_cache_config(self):
        return {
            "states": self.state_dim,
            "actions": self.action_dim,
            "my_parameter": self.my_parameter,
        }
```

La configuration est capturée avant la construction des matrices. La clé SHA-256 inclut la classe, `CACHE_VERSION`, une empreinte du module source, les paramètres canoniques et la normalisation des récompenses. Le `get_cache_config()` par défaut inspecte les attributs existants avant construction ; un type non sérialisable entraîne l'absence de mise en cache plutôt qu'une clé imprécise. **Les recettes aléatoires sans graine explicitement reproductible doivent laisser `CACHE_VERSION = None`**, valeur par défaut. Les modèles dont l'état non matriciel est modifié pendant la construction doivent également désactiver le cache ou restaurer eux-mêmes cet état de manière fiable.

Les modèles du catalogue dont la génération est déterministe et dont les attributs additionnels restent stables lors de la construction ont été déclarés compatibles ; les autres restent volontairement sans cache de matrices, y compris les modèles déterministes qui peuplent des tables annexes dans `_build_model()`. Les recettes aléatoires de `garnet`, `block`, `dam` et les générateurs de labyrinthes restent non cacheables tant que leurs graines et leur état de génération ne sont pas explicités.

## Références de valeur

`model.optimal_value_function(discount, cache="readwrite")` et `get_exact_value(model, discount, cache="readwrite")` identifient les références par l'empreinte SHA-256 des matrices CSR et des récompenses **effectivement utilisées**, le facteur d'actualisation, la précision exigée et le code du solveur de référence VI. La lecture vérifie l'intégrité des données, la dimension, l'absence de valeurs non finies et la borne de résidu de Bellman. Le modèle doit être construit : un nom ne prouve jamais l'identité numérique. Les références restent respectivement à précision absolue `1e-3` et `1e-6`.

Les matrices et les valeurs sont enregistrées en NPZ compressé avec métadonnées JSON, sans `allow_pickle`. Les écritures sont atomiques et synchronisées par entrée. Les chemins sont `artifacts/saved_models/v2/` et `artifacts/saved_value_functions/v2/`.

## Migration des caches v1

Les anciens `*.pkl` sont laissés en place et ne sont **pas** lus automatiquement. Le moyen recommandé est de reconstruire les modèles avec `cache="refresh"`. Pour des données de provenance vérifiée seulement, `persistence.import_legacy_model_cache(model, trusted=True)` permet d'importer un cache pickle après contrôle des matrices. **`trusted=True` implique d'accepter le risque d'exécution de code lié à la désérialisation pickle.** La concordance des paramètres de génération ne peut pas être prouvée à partir de l'ancien nom de fichier : vérifiez manuellement que les données correspondent à la recette souhaitée.

En cas de modification des dépendances de génération ou d'un algorithme, augmentez explicitement `CACHE_VERSION` pour invalider les anciennes recettes si nécessaire.
