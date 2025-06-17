import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------------
# PARAMÈTRES DE BASE
# ------------------------

N = 4                       # Taille d’un cluster (nombre de points connectés)
L = 2*N +1                  # Taille du réseau (suffisamment grand pour permettre tous les clusters possibles)
origin = (N, N)             # Point de départ (au centre du réseau)

# ------------------------
# FONCTIONS UTILITAIRES
# ------------------------

def draw_square_grid(ax, L):
    """Dessine la grille carrée (segments entre voisins directs)"""
    for y in range(L):
        for x in range(L):
            pt = (x, y)
            for nx, ny in [(x+1, y), (x, y+1)]:
                if 0 <= nx < L and 0 <= ny < L:
                    ax.plot([x, nx], [y, ny], color='lightgray', linewidth=0.5)


def neighbors(p):
    """Retourne les voisins de Von Neumann du point p (haut, bas, gauche, droite)"""
    x, y = p
    return [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]

def normalize(cluster):
    """
    Ramène les coordonnées du cluster de manière à ce que
    le point le plus en haut à gauche soit (0, 0).
    Permet de comparer les clusters indépendamment de leur position absolue.
    """
    xs, ys = zip(*cluster)
    min_x, min_y = min(xs), min(ys)
    return sorted((x - min_x, y - min_y) for x, y in cluster)

def generate_rotations(cluster):
    """
    Génère toutes les versions symétriques du cluster :
    - 4 rotations (0°, 90°, 180°, 270°)
    - Pour chaque rotation, on ajoute aussi son miroir horizontal
    Le but est d'identifier les formes équivalentes (isomorphes).
    """
    def rotate_90(pts): return [(-y, x) for x, y in pts]
    def mirror_x(pts): return [(x, -y) for x, y in pts]

    forms = set()
    current = normalize(cluster)
    for _ in range(4):
        current = normalize(rotate_90(current))  # Applique rotation
        forms.add(tuple(current))                # Ajoute la rotation
        forms.add(tuple(normalize(mirror_x(current))))  # Ajoute son miroir
    return forms

def canonical_form(cluster):
    """
    Retourne la forme canonique d’un cluster,
    c’est-à-dire la plus petite (ordre lexicographique)
    parmi toutes ses versions symétriques.
    """
    return min(generate_rotations(cluster))

# ------------------------
# GÉNÉRATION DES CLUSTERS
# ------------------------

def generate_clusters(start, N, L):
    """
    Génère tous les clusters connexes de taille N, à partir du point 'start',
    en parcourant le réseau par force brute (DFS).
    Les clusters isomorphes (égaux par rotation/symétrie) sont éliminés.
    """
    seen = dict()    # Dictionnaire des formes déjà vues (clé = forme canonique)
    results = []     # Liste des clusters uniques à retourner

    def dfs(cluster):   # Depth-First Search
        """Fonction récursive de parcours (DFS) pour construire les clusters"""
        if len(cluster) == N:
            canon = canonical_form(cluster)
            if canon not in seen:
                seen[canon] = len(generate_rotations(cluster))  # Nombre d’équivalents symétriques
                results.append(cluster)
            return
        for p in cluster:
            for n in neighbors(p):
                if n not in cluster and 0 <= n[0] < L and 0 <= n[1] < L:
                    dfs(cluster | {n})  # Ajoute le voisin et continue

    dfs(frozenset([start]))  # Lancement depuis le point central
    return results, seen     # On retourne la liste des clusters et leur dégénérescence géométrique

# ------------------------
# GÉNÉRATION ET AFFICHAGE
# ------------------------

clusters, config_count = generate_clusters(origin, N, L)
print(f"Nombre de clusters distincts de taille {N} : {len(clusters)}")

# Paramètres d'affichage
cols = 5
clusters_per_figure = 20
figures_needed = int(np.ceil(len(clusters) / clusters_per_figure))

for fig_index in range(figures_needed):
    start_idx = fig_index * clusters_per_figure
    end_idx = min(start_idx + clusters_per_figure, len(clusters))
    n_clusters = end_idx - start_idx
    rows = int(np.ceil(n_clusters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(rows * cols):
        ax = axes[i]
        if start_idx + i < end_idx:
            cluster = clusters[start_idx + i]
            x, y = zip(*cluster)

            ax.scatter(x, y, color='red')
            ax.scatter(*origin, color='red')

            external_neighbors = set()
            for pt in cluster:
                for nb in neighbors(pt):
                    if nb not in cluster and 0 <= nb[0] < L and 0 <= nb[1] < L:
                        external_neighbors.add(nb)
            if external_neighbors:
                x_ext, y_ext = zip(*external_neighbors)
                ax.scatter(x_ext, y_ext, color='green')

            ax.set_xlim(-1, L)
            ax.set_ylim(-1, L)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            draw_square_grid(ax, L)

            canon = canonical_form(cluster)
            count = config_count[canon]
            ax.set_title(f"#{start_idx + i + 1} ({count} configs)", fontsize=8)

            print(f"\nCluster #{start_idx + i + 1} – voisins extérieurs : {len(sorted(external_neighbors))}")
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.suptitle(f"Clusters {start_idx + 1} à {end_idx}", fontsize=14)
    plt.subplots_adjust(top=0.9)
    plt.show()
'''
# Sauvegarde dans un fichier texte
filename = f"CSV_cluster/reseau_carre_cluster_{N}_informations.txt"
with open(filename, 'w') as f:
    f.write(f"Nombre de clusters distincts de taille {N} : {len(clusters)}\n\n")
    for idx, cluster in enumerate(clusters, start=1):
        external_neighbors = set()
        for pt in cluster:
            for nb in neighbors(pt):
                if nb not in cluster and 0 <= nb[0] < L and 0 <= nb[1] < L:
                    external_neighbors.add(nb)
        n_voisins = len(external_neighbors)
        canon = canonical_form(cluster)
        n_configs = config_count[canon]
        f.write(f"Cluster #{idx} – voisins extérieurs : {n_voisins} – nombre de configuration : {n_configs}\n")






'''
