# Étude Thermodynamique de Systèmes de Spins Dilués

Ce projet regroupe plusieurs scripts Python permettant de modéliser et d'analyser les propriétés thermodynamiques de réseaux de spins S = 1/2 avec dilution, à température finie. Les géométries considérées sont les réseaux carrés, triangulaires et chaînes linéaires. Ce travail a été réalisé dans le cadre d'un stage de troisième année de licence de physique à Sorbonne Université, sous la supervision de Laura Messio chercheuse au Laboratoire de Physique Théorique de la Matière Condensée (LPTMC).

## Objectifs
- Générer tous les clusters connexes (à translation, rotation et symétrie près) sur différents réseaux discrets
- Calculer les grandeurs thermodynamiques Cv (chaleur spécifique) et χ (susceptibilité magnétique) à partir de ces clusters
- Étudier l'effet de la dilution sur ces grandeurs dans différents contextes géométriques
- Comparer les résultats numériques avec des solutions analytiques connues

## Structure des fichiers
| Fichier | Description |
|---------|-------------|
| `square_cluster_opti.py` | Génération optimisée de clusters pour réseaux carrés |
| `triangle_cluster_opti.py` | Génération optimisée de clusters pour réseaux triangulaires |
| `chaine_opti.py` | Génération optimisée de clusters pour chaînes linéaires |
| `square.py` | Calcul des grandeurs thermodynamiques pour réseaux carrés |
| `triangle.py` | Calcul des grandeurs thermodynamiques pour réseaux triangulaires |
| `chaine.py` | Calcul des grandeurs thermodynamiques pour chaînes linéaires |
| `courbes_cluster.py` | Comparaison des résultats entre différents clusters |
| `dvt.py` | Analyse des déviations par rapport aux solutions théoriques |
| `cluster.py` | Module commun de manipulation des clusters |

## Prérequis
- Python ≥ 3.8
- Bibliothèques requises :
  - numpy
  - scipy
  - pandas
  - matplotlib
  - itertools
  - math

Installation :
```bash
pip install numpy scipy pandas matplotlib

## Utilisation

### Étape 1 : Placer les clusters

Les fichiers `.csv` décrivant les clusters doivent être placés dans `CSV_cluster/`.  
Ils doivent suivre le format :

```
clusters_carre_N{N}.csv
clusters_triangle_N{N}.csv
```

Chaque fichier doit contenir :

- `liaisons` : liste des couples de spins liés [(i, j), ...]
- `n_config` : nombre de configurations équivalentes par symétrie
- `n_voisin` : nombre de voisins autour du cluster

### Étape 2 : Lancer le calcul thermodynamique

Pour un réseau carré :

```bash
python3 square_thermodynamics.py
```

Pour un réseau triangulaire :

```bash
python3 triangle_thermodynamics.py
```

Les résultats sont sauvegardés automatiquement dans le dossier `CSV_Thermodynamics/` sous la forme :

```
thermodynamique_reseau_carre_pXX.csv
thermodynamique_reseau_triangle_pXX.csv
```

où `pXX` correspond à la probabilité de dilution `p`, sans virgule (ex : `p01` pour `p = 0.1`).

Les figures affichent Cv/N et chi/N en fonction de la température, avec accumulation des contributions jusqu’à une taille maximale de cluster définie par l’utilisateur.

## Détails techniques

- Le Hamiltonien correspond au modèle de Heisenberg avec champ magnétique longitudinal.
- Les opérateurs de spin sont codés sous forme de matrices creuses (sparse).
- Les grandeurs thermodynamiques Cv et chi sont calculées à partir de moyennes thermiques exactes sur les spectres.
- Chaque cluster est pondéré par une loi de probabilité tenant compte du taux de dilution `p` et du nombre de voisins.

## Auteur

Brendon Geoffrin-Tanner  
L3 Physique Fondamentale – Sorbonne Université  
Stage de recherche au Laboratoire de Physique Théorique de la Matière Condensée (LPTMC) – Sorbonne Université

## Licence

Code distribué sous licence MIT.
