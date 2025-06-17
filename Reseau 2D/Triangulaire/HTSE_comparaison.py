import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chargement des données théoriques cumulatives
csv_file = "CSV_Thermodynamics/thermodynamique_reseau_triangle_p01.csv"
df = pd.read_csv(csv_file)

# Initialisation du plot
plt.figure(figsize=(10, 6))

# Tracé des colonnes cumulées Cv
for col in df.columns:
    if col.startswith("Cv_cumul_N="):
        plt.plot(df["T"], df[col], label=f"Théorie {col.replace('_', ' ')}")

# Chargement et tracé des données expérimentales depuis le fichier .npy
npy_file = "NPY/triangle_p01.npy"
try:
    with open(npy_file, "rb") as file:
        labels = np.load(file)
        for i in range(len(labels)):
            label = labels[i]
            x = np.load(file)
            y = np.load(file)
            plt.plot(x, y, '+', label=f'Données: {label}')
except Exception as e:
    print("Erreur lors du chargement des données expérimentales :", e)

# Mise en forme
plt.xlabel("Température $T$")
plt.ylabel("Capacité thermique $C_v/N$")
plt.ylim(0,0.05)
plt.xlim(0,4)
plt.title("Comparaison théorie/expérience – $C_v/N$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
