# Étude Thermodynamique de Réseaux de Spins Dilués

Ce projet regroupe plusieurs scripts Python permettant de modéliser et d’analyser les propriétés thermodynamiques de réseaux de spins S = 1/2 avec dilution, à température finie. Les géométries considérées sont les réseaux carrés, triangulaires et chaînes linéaires. Ce travail a été réalisé dans le cadre d’un stage de troisième année de licence de physique à Sorbonne Université, sous la supervision de Laura Messio chercheuse au Laboratoire de Physique Théorique de la Matière Condensée (LPTMC).

## Objectifs

- Générer tous les clusters connexes (à translation, rotation et symétrie près) sur des réseaux discrets.
- Calculer les grandeurs thermodynamiques Cv (chaleur spécifique) et chi (susceptibilité magnétique) à partir de ces clusters.
- Étudier l’effet de la dilution sur ces grandeurs dans différents contextes géométriques.

## Structure des fichiers

| Fichier | Description |
|--------|-------------|
| `programme1.py` | Description courte du programme1 |
| `programme2.py` | Description courte du programme2 |
| `programme3.py` | Description courte du programme3 |
...
## Prérequis

- Python ≥ 3.8
- Bibliothèques requises :
  - numpy
  - scipy
  - pandas
  - matplotlib
  - ...

Installation :
```bash
pip install numpy scipy pandas matplotlib ...
```

## Utilisation

### Étape 1 : Générer les clusters

Pour les réseaux carrés :
```bash
python3 square_cluster_opti.py
```

Pour les réseaux triangulaires :
```bash
python3 triangle_cluster_opti.py
```

Cela produit des fichiers `.csv` de la forme :
```
clusters_carre_N{N}.csv
clusters_triangle_N{N}.csv
```

### Étape 2 : Calcul thermodynamique

Pour un réseau carré :
```bash
python3 cv_chi_square.py
```

Pour un réseau triangulaire :
```bash
python3 cv_chi_triangle.py
```

Pour une chaîne linéaire :
```bash
python3 Calcul_Cv_chi_chaine.py
```

Ces scripts génèrent les courbes Cv/N et chi/N en fonction de la température T, en tenant compte de la dilution.

### Études analytiques et comparatives

Comparer avec les résultats analytiques :
```bash
python3 courbes_cluster.py
python3 dvt.py
```

## Détails techniques

- **Opérateurs de spin** : Les scripts utilisent une représentation matricielle (sparse/dense) des opérateurs Sx, Sy, Sz.
- **Hamiltonien** : Pour un système à N spins et une liste de liaisons, le Hamiltonien est construit comme :

  H = J * somme sur (i,j) de (S_i ⋅ S_j) - h * somme sur i de S^z_i

- **Dilution** : Les clusters sont pondérés par la loi p^N * (1 - p)^n_voisins, représentant la probabilité de leur apparition dans un réseau aléatoirement dilué.

## Auteur

Brendon Geoffrin-Tanner  
L3 Physique Fondamentale – Sorbonne Université  
Stage de recherche au Laboratoire de Physique Théorique de la Matière Condensée (LPTMC) – Sorbonne Université

## Licence

Code distribué sous licence MIT.
