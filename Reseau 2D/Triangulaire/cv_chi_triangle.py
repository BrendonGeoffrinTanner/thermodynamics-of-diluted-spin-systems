import numpy as np
import pandas as pd
import ast
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
import matplotlib.pyplot as plt

kB = 1.0

def Sz_op(site, N):
    row = np.arange(2**N, dtype=int)
    z = .5 - ((row >> site) % 2)
    return csr_matrix((z, (row, row)), shape=(2**N, 2**N))

def Sx_op(site, N):
    row = np.arange(2**N, dtype=int)
    col = row ^ (1 << site)
    val = np.ones(2**N) / 2
    return csr_matrix((val, (row, col)), shape=(2**N, 2**N))

def Sy_op(site, N):
    row = np.arange(2**N, dtype=int)
    col = row ^ (1 << site)
    sign = 1 - 2 * ((row >> site) & 1)
    val = 1j * sign / 2
    return csr_matrix((val, (row, col)), shape=(2**N, 2**N))

def total_Sz_operator(N):
    return sum(Sz_op(site, N) for site in range(N)).toarray()

def hamiltonian_dense_from_bonds(J, h, N, bonds):
    H = csr_matrix((2**N, 2**N), dtype=complex)
    for (i, j) in bonds:
        H += Sx_op(i, N) @ Sx_op(j, N)
        H += Sy_op(i, N) @ Sy_op(j, N)
        H += Sz_op(i, N) @ Sz_op(j, N)
    H *= J
    for i in range(N):
        H += -h * Sz_op(i, N)
    return H.toarray()

# Paramètres utilisateur
J = 1.0
h = 0.0
N_max = 7
p = 0.2
T_vals = np.linspace(0.1, 10, 300)

results = pd.DataFrame({"T": T_vals})

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title(r"$C_v/N$ cumulatif")
plt.xlabel("Température $T$")
plt.ylabel(r"$C_v/N$")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title(r"$\chi/N$ cumulatif")
plt.xlabel("Température $T$")
plt.ylabel(r"$\chi/N$")
plt.grid(True)

Cv_cumul = np.zeros_like(T_vals)
chi_cumul = np.zeros_like(T_vals, dtype=np.complex128)

for N in range(1, N_max + 1):
    df = pd.read_csv(f"CSV_cluster/clusters_triangle_N{N}.csv")
    df["liaisons"] = df["liaisons"].apply(ast.literal_eval)

    for _, row in df.iterrows():
        n_config = row['n_config']
        n_voisin = row['n_voisin']
        bonds = row['liaisons']
        poids = n_config * (p**N) * ((1 - p)**n_voisin)

        H = hamiltonian_dense_from_bonds(J, h, N, bonds)
        Sz_tot = total_Sz_operator(N)
        beta_vals = 1 / (kB * T_vals)
        eigvals, eigvecs = eigh(H)

        Z_vals = np.array([np.sum(np.exp(-beta * eigvals)) for beta in beta_vals])
        E_mean = np.array([
            np.sum(eigvals * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])
        E2_mean = np.array([
            np.sum((eigvals**2) * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])
        Cv = beta_vals**2 * (E2_mean - E_mean**2)

        Sz_diag = np.array([v.conj().T @ Sz_tot @ v for v in eigvecs.T])
        Sz_mean = np.array([
            np.sum(Sz_diag * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])
        Sz2_mean = np.array([
            np.sum((Sz_diag**2) * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])
        chi = beta_vals * (Sz2_mean - Sz_mean**2)

        Cv_cumul += poids * Cv
        chi_cumul += poids * chi

    results[f"Cv_cumul_N={N}"] = Cv_cumul.copy()
    results[f"chi_cumul_N={N}"] = chi_cumul.real.copy()

    plt.subplot(1, 2, 1)
    plt.plot(T_vals, Cv_cumul, label=f"N≤{N}")
    plt.subplot(1, 2, 2)
    plt.plot(T_vals, chi_cumul.real, label=f"N≤{N}")

# Sauvegarde
results.to_csv(f"CSV_Thermodynamics/thermodynamique_reseau_triangle_p{str(p).replace('.', '')}.csv", index=False)

# Affichage final
plt.subplot(1, 2, 1)
plt.legend()
plt.subplot(1, 2, 2)
plt.legend()
plt.suptitle(f"Somme cumulée des contributions jusqu'à N (p = {p})", fontsize=14)
plt.tight_layout()
plt.show()
