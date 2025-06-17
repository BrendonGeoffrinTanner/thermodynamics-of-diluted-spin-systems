import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Paramètres
# ----------------------------
N = 4
L = 2 * N + 1
origin = (N, N)

# ----------------------------
# Réseau triangulaire (offset odd-q)
# ----------------------------

def neighbors_triangular(p):
    x, y = p
    if y % 2 == 0:
        return [(x+1, y), (x-1, y),
                (x, y+1), (x, y-1),
                (x-1, y+1), (x-1, y-1)]
    else:
        return [(x+1, y), (x-1, y),
                (x, y+1), (x, y-1),
                (x+1, y+1), (x+1, y-1)]

def to_physical(p):
    x, y = p
    return (x + 0.5 * (y % 2), y * np.sqrt(3) / 2)

def draw_triangular_grid(ax, L):
    for y in range(L):
        for x in range(L):
            p1 = (x, y)
            for p2 in neighbors_triangular(p1):
                if 0 <= p2[0] < L and 0 <= p2[1] < L:
                    x1, y1 = to_physical(p1)
                    x2, y2 = to_physical(p2)
                    ax.plot([x1, x2], [y1, y2], color='lightgray', linewidth=0.5)

def transform_rotate(points, angle_deg):
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return [tuple(np.dot(R, p)) for p in points]

def transform_mirror(points):
    return [(-x, y) for x, y in points]

def normalize_physical(points):
    xs, ys = zip(*points)
    min_x = min(xs)
    min_y = min(ys)
    translated = [((x - min_x), (y - min_y)) for x, y in points]
    rounded = [tuple(np.round(p, 6)) for p in translated]
    return tuple(sorted(rounded))

def all_symmetries(cluster):
    phys = [to_physical(p) for p in cluster]
    forms = []

    for angle in range(0, 360, 60):
        rotated = transform_rotate(phys, angle)
        forms.append(normalize_physical(rotated))
        mirrored = transform_mirror(rotated)
        forms.append(normalize_physical(mirrored))

    return set(forms)

def canonical_form(cluster):
    return min(all_symmetries(cluster))

# ----------------------------
# Génération des clusters (DFS)
# ----------------------------

def generate_clusters(start, N, L):
    seen = dict()
    results = []

    def dfs(cluster):
        if len(cluster) == N:
            canon = canonical_form(cluster)
            if canon not in seen:
                seen[canon] = len(all_symmetries(cluster))
                results.append(cluster)
            return
        for p in cluster:
            for n in neighbors_triangular(p):
                if n not in cluster and 0 <= n[0] < L and 0 <= n[1] < L:
                    dfs(cluster | {n})

    dfs(frozenset([start]))
    return results, seen

# ----------------------------
# Affichage
# ----------------------------

clusters, config_count = generate_clusters(origin, N, L)
print(f"Nombre de clusters triangulaires distincts de taille {N} : {len(clusters)}")

cols = 5
rows = int(np.ceil(len(clusters) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < len(clusters):
        cluster = clusters[i]

        pxy = [to_physical(p) for p in cluster]
        px, py = zip(*pxy)
        ax.scatter(px, py, color='red')

        ox, oy = to_physical(origin)
        ax.scatter(ox, oy, color='red')

        external_neighbors = set()
        for pt in cluster:
            for nb in neighbors_triangular(pt):
                if nb not in cluster and 0 <= nb[0] < L and 0 <= nb[1] < L:
                    external_neighbors.add(nb)
        if external_neighbors:
            ext_xy = [to_physical(p) for p in external_neighbors]
            ex, ey = zip(*ext_xy)
            ax.scatter(ex, ey, color='green')

        ax.set_xlim(-1, L + 1)
        ax.set_ylim(-1, (L + 1) * np.sqrt(3) / 2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        draw_triangular_grid(ax, L)

        canon = canonical_form(cluster)
        count = config_count[canon]
        ax.set_title(f"#{i+1} ({count} configs)", fontsize=8)

        print(f"\nCluster #{i+1} – voisins extérieurs : {len(external_neighbors)} – nombre de configurations : {count}")
    else:
        ax.axis('off')

plt.tight_layout()
plt.suptitle(f"Clusters triangulaires distincts de taille {N}", fontsize=14)
plt.subplots_adjust(top=0.9)
plt.show()
