# Vérification des problèmes des presets large — 10 octobre 2026

Le CSV initial `model_size_runtimes.csv` est conservé. Les nouvelles mesures sont
dans `model_size_recheck.csv` : **12 essais réussis**, tous sous 30 secondes, avec
une borne d'erreur absolue calculée sur le résidu de Bellman au plus égale à 10⁻³.
Conditions : environnement Conda `benchmark`, γ = 0,99, précision 10⁻³, seed 0,
un thread, un essai par couple. Les durées excluent la construction du modèle,
sa copie et le contrôle indépendant du résidu.

## Corrections

Les six erreurs initiales concernent exclusivement `mdptoolbox_vi` :

- Access Control et Impatience : validation upstream des sommes de lignes trop
  stricte pour l'arrondi flottant de ces modèles.
- Ambulance et Ambulance Relocation : logarithme indéfini dans le calcul
  automatique d'une borne d'itérations.
- Blocks World : comparaison des matrices creuses à zéro pouvant allouer une
  matrice dense de 68,7 GiB.
- Cartpole : rejet explicite des récompenses maximales constantes dans
  l'adaptateur, ajouté pour contourner la borne d'itérations upstream.

Le même calcul de borne, quadratique en nombre d'états, ralentissait Elevator
et provoquait le timeout de Peg Solitaire. Cette borne était déjà remplacée par
la limite explicite de l'adaptateur après l'initialisation.

L'adaptateur initialise désormais le backend après validation native des CSR.
La boucle VI de pymdptoolbox 4.0b3, son opérateur de Bellman, son seuil de span,
l'initialisation à zéro et la certification du résidu restent inchangés. Les
probabilités, récompenses et tailles de ces huit modèles ne sont pas modifiées.

## Nouvelles mesures

| Modèle | États | Solveur | Durée (s) |
|---|---:|---|---:|
| access_control | 1 300 | mdptoolbox_vi | 0,007 |
| ambulance | 400 | mdptoolbox_vi | 0,006 |
| ambulance_relocation | 360 | mdptoolbox_vi | 0,006 |
| blocks_world | 96 000 | mdptoolbox_vi | 7,954 |
| cartpole | 4 096 | mdptoolbox_vi | 0,001 |
| elevator | 14 592 | mdptoolbox_vi | 0,010 |
| impatience | 1 800 | mdptoolbox_vi | 0,061 |
| peg_solitaire | 8 192 | mdptoolbox_vi | 6,247 |
| hexagonal_grid_soccer | 50 653 | mdpforge_vi | 3,173 |
| hexagonal_grid_soccer | 50 653 | mdptoolbox_vi | 4,212 |
| hexagonal_grid_soccer | 50 653 | marmote_vi | 6,122 |
| hexagonal_grid_soccer | 50 653 | mdpsolver_vi | 8,609 |

Hexagonal Grid Soccer dispose désormais d'un preset large mesuré : rayon 3,
37 cellules et 37³ = 50 653 états. Il respecte également la marge de 15 secondes
visée lors de l'extrapolation initiale.

Les 189 succès du CSV initial ne sont pas tous remesurés. Les nouvelles mesures
portent sur les huit essais mdptoolbox problématiques et les quatre solveurs du
nouveau preset Hexagonal Grid Soccer. Les tailles small/medium ne sont pas
recalibrées ici. Des tests de régression couvrent les récompenses constantes,
les transitions mélangeantes, l'arrondi, l'absence de conversion dense et la
préservation du modèle à plusieurs discounts ; pytest et Ruff n'ont pas été lancés.

Pour une nouvelle campagne complète, utiliser un autre CSV afin de ne pas
reprendre les anciennes erreurs déjà enregistrées :

```text
conda run -n benchmark python check_model_sizes.py --sizes large --output artifacts/results/model_size_runtimes_v2.csv
```
