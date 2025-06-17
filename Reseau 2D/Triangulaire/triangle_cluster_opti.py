import numpy as np
import pandas as pd
from itertools import product
from collections import deque
from multiprocessing import Pool, cpu_count

# ------------------------
# PARAMÈTRE MAXIMAL
# ------------------------

N_max = 1  # Taille maximale des clusters à générer (de 1 à N_max)

# ------------------------
# UTILITAIRES
# ------------------------

def neighbors(p):
    x, y = p
    if y % 2 == 0:
        return [(x+1, y), (x-1, y),
                (x, y+1), (x, y-1),
                (x-1, y+1), (x-1, y-1)]
    else:
        return [(x+1, y), (x-1, y),
                (x, y+1), (x, y-1),
                (x+1, y+1), (x+1, y-1)]

def normalize(points):
    xs, ys = zip(*points)
    min_x = min(xs)
    min_y = min(ys)
    translated = [((x - min_x), (y - min_y)) for x, y in points]
    rounded = [tuple(np.round(p, 6)) for p in translated]
    return tuple(sorted(rounded))

def transform_rotate(points, angle_deg):
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return [tuple(np.dot(R, p)) for p in points]

def transform_mirror(points):
    return [(-x, y) for x, y in points]

def to_physical(p):
    x, y = p
    return (x + 0.5 * (y % 2), y * np.sqrt(3) / 2)

def generate_symmetries(cluster):
    phys = [to_physical(p) for p in cluster]
    forms = []

    for angle in range(0, 360, 60):
        rotated = transform_rotate(phys, angle)
        forms.append(normalize(rotated))
        mirrored = transform_mirror(rotated)
        forms.append(normalize(mirrored))

    return set(forms)


def canonical_form(cluster):
    return min(generate_symmetries(cluster))

def get_liaisons(cluster):
    # Calcule les liaisons internes entre les points du cluster
    cluster = list(cluster)
    index_map = {pt: i for i, pt in enumerate(cluster)}  # Associe un index à chaque point
    liaisons = set()
    for i, pt in enumerate(cluster):
        for nb in neighbors(pt):
            if nb in index_map:
                j = index_map[nb]
                if i < j:
                    liaisons.add((i, j))  # Évite les doublons en imposant i < j
    return sorted(liaisons)

def external_neighbors(cluster, L):
    # Compte les voisins extérieurs du cluster (non inclus dans le cluster mais voisins immédiats)
    ext = set()
    for pt in cluster:
        for nb in neighbors(pt):
            if nb not in cluster and 0 <= nb[0] < L and 0 <= nb[1] < L:
                ext.add(nb)
    return len(ext)

# ------------------------
# GÉNÉRATION DE CLUSTERS
# ------------------------

def generate_clusters_iterative(N, L):
    """
    Génère tous les clusters connexes de taille N, à partir du point 'start',
    en parcourant le réseau par force brute (DFS).
    Les clusters isomorphes (égaux par rotation/symétrie) sont éliminés.
    """
    origin = (N, N)  # Point de départ central dans la grille
    queue = deque()
    queue.append(frozenset([origin]))  # Initialisation avec le cluster de taille 1
    seen_forms = dict()  # Mémorise les formes canoniques déjà rencontrées
    results = []

    while queue:
        cluster = queue.popleft()
        if len(cluster) == N:
            canon = canonical_form(cluster)
            if canon not in seen_forms:
                seen_forms[canon] = len(generate_symmetries(cluster))  # Nombre de représentations équivalentes
                results.append(cluster)
            continue
        # Ajoute un point voisin non encore présent dans le cluster
        for pt in cluster:
            for nb in neighbors(pt):
                if nb not in cluster and 0 <= nb[0] < L and 0 <= nb[1] < L:
                    new_cluster = cluster | {nb}
                    if len(new_cluster) <= N:
                        queue.append(new_cluster)

    return results, seen_forms

# ------------------------
# CALCUL POUR UNE VALEUR DE N
# ------------------------

def process_N(N):
    """
    Pour une taille N, génère tous les clusters,
    calcule leurs propriétés, et les enregistre dans un fichier CSV.
    """
    L = 2 * N + 1  # Taille de la grille, suffisante pour permettre la croissance des clusters
    clusters, config_count = generate_clusters_iterative(N, L)
    cluster_data = []

    for cluster in clusters:
        canon = canonical_form(cluster)
        n_configs = config_count[canon]  # Nombre de configurations équivalentes par symétrie
        n_voisins = external_neighbors(cluster, L)  # Nombre de voisins extérieurs
        liaison_list = get_liaisons(cluster)  # Liste des liaisons internes
        cluster_data.append([n_configs, n_voisins, str(liaison_list)])

    df = pd.DataFrame(cluster_data, columns=["n_config", "n_voisin", "liaisons"])
    csv_name = f"CSV_cluster/clusters_triangle_N{N}.csv"
    df.to_csv(csv_name, index=False)
    print(f"{len(clusters)} clusters enregistrés dans {csv_name}")

# ------------------------
# PARALLÉLISATION
# ------------------------

if __name__ == "__main__":
    # Utilise tous les cœurs disponibles pour paralléliser le traitement des différentes tailles N
    with Pool(processes=min(cpu_count(), N_max)) as pool:
        pool.map(process_N, range(1, N_max + 1))  # Lance le calcul pour chaque taille de cluster
