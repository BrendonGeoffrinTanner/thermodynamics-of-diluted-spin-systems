import numpy as np
import math
import pandas as pd
import ast
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy.polynomial.polynomial as poly

kB = 1.0
M = 10  # Ordre maximal du développement en série

# Opérateurs de spin (identique à triangle.py)
def Sz_op(site, N):
    row = np.arange(2**N, dtype=int)
    z = .5 - ((row >> site) % 2)
    return np.diag(z)

def Sx_op(site, N):
    dim = 2**N
    row = np.arange(dim)
    col = row ^ (1 << site)
    val = 0.5 * np.ones(dim)

    mat = np.zeros((dim, dim), dtype=complex)
    mat[row, col] = val
    return mat


def Sy_op(site, N):
    dim = 2**N
    row = np.arange(dim)
    col = row ^ (1 << site)
    sign = 1 - 2 * ((row >> site) & 1)
    val = 1j * sign / 2

    mat = np.zeros((dim, dim), dtype=complex)
    mat[row, col] = val
    return mat


def total_Sz_operator(N):
    return sum(Sz_op(site, N) for site in range(N))

def hamiltonian_dense_from_bonds(J, h, N, bonds):
    H = np.zeros((2**N, 2**N), dtype=complex)
    for (i, j) in bonds:
        H += Sx_op(i, N) @ Sx_op(j, N)
        H += Sy_op(i, N) @ Sy_op(j, N)
        H += Sz_op(i, N) @ Sz_op(j, N)
    H *= J
    for i in range(N):
        H += -h * Sz_op(i, N)
    return H

# Paramètres
J = 1.0
h = 0.0
p = 0.1
N_max = 7

# Initialisation des polynômes cumulatifs
Cv_poly_cumul = np.zeros(M + 1)
chi_poly_cumul = np.zeros(M + 1)

# Boucle sur les tailles de cluster
for N in range(1, N_max + 1):
    df = pd.read_csv(f"CSV_cluster/clusters_triangle_N{N}.csv")
    df["liaisons"] = df["liaisons"].apply(ast.literal_eval)
    
    for _, row in df.iterrows():
        n_config = row['n_config']
        n_voisin = row['n_voisin']
        bonds = row['liaisons']
        poids = n_config * (p**N) * ((1 - p)**n_voisin)
        
        # Construction des opérateurs
        H = hamiltonian_dense_from_bonds(J, h, N, bonds)
        Sz_tot = total_Sz_operator(N)
        
        # Diagonalisation
        eigvals, eigvecs = eigh(H)
        
        # Calcul des éléments diagonaux nécessaires
        diag_Sz = np.array([eigvecs[:, i].conj().T @ Sz_tot @ eigvecs[:, i] for i in range(len(eigvals))]).real
        diag_Sz2 = np.array([np.linalg.norm(Sz_tot @ eigvecs[:, i])**2 for i in range(len(eigvals))]).real
        
        # Coefficients des séries pour ce cluster
        a_coeffs = np.zeros(M + 1)  # Pour Z
        b_coeffs = np.zeros(M + 1)  # Pour Z_U
        e_coeffs = np.zeros(M + 1)  # Pour Z_{U2}
        c_coeffs = np.zeros(M + 1)  # Pour Z_{Sz}
        d_coeffs = np.zeros(M + 1)  # Pour Z_{Sz2}
        
        # Calcul des coefficients
        for k in range(M + 1):
            fact_k = math.factorial(k)
            sign = (-1)**k
            
            # Tr(H^k)
            tr_Hk = np.sum(eigvals**k)
            a_coeffs[k] = sign * tr_Hk / fact_k
            
            # Tr(H^{k+1})
            if k <= M - 1:
                tr_Hk1 = np.sum(eigvals**(k + 1))
                b_coeffs[k] = sign * tr_Hk1 / fact_k
            
            # Tr(H^{k+2})
            if k <= M - 2:
                tr_Hk2 = np.sum(eigvals**(k + 2))
                e_coeffs[k] = sign * tr_Hk2 / fact_k
            
            # Tr(Sz_tot H^k)
            tr_SzHk = np.sum(diag_Sz * eigvals**k)
            c_coeffs[k] = sign * tr_SzHk / fact_k
            
            # Tr(Sz_tot^2 H^k)
            tr_Sz2Hk = np.sum(diag_Sz2 * eigvals**k)
            d_coeffs[k] = sign * tr_Sz2Hk / fact_k
        
        # Calcul de 1/Z par division de séries
        inv_a = np.zeros(M + 1)
        inv_a[0] = 1 / a_coeffs[0]
        for n in range(1, M + 1):
            inv_a[n] = -np.sum([a_coeffs[j] * inv_a[n - j] for j in range(1, n + 1)]) * inv_a[0]
        
        # Calcul de U = <H> = (1/Z) * Tr(H exp(-βH))
        U_series = np.zeros(M + 1)
        for n in range(M + 1):
            U_series[n] = np.sum([b_coeffs[k] * inv_a[n - k] for k in range(n + 1)])
        
        # Calcul de <H^2>
        U2_series = np.zeros(M + 1)
        for n in range(M + 1):
            U2_series[n] = np.sum([e_coeffs[k] * inv_a[n - k] for k in range(n + 1)])
        
        # Calcul de Cv = β^2 (<H^2> - <H>^2)
        U_sq_series = poly.polymul(U_series, U_series)[:M + 1]
        varE_series = U2_series - U_sq_series
        Cv_series = np.zeros(M + 1)
        for n in range(2, M + 1):
            Cv_series[n] = varE_series[n - 2]  # β^2 * coefficient de β^{n-2}
        
        # Calcul de <Sz_tot> et <Sz_tot^2>
        Sz_series = np.zeros(M + 1)
        for n in range(M + 1):
            Sz_series[n] = np.sum([c_coeffs[k] * inv_a[n - k] for k in range(n + 1)])
        
        Sz2_series = np.zeros(M + 1)
        for n in range(M + 1):
            Sz2_series[n] = np.sum([d_coeffs[k] * inv_a[n - k] for k in range(n + 1)])
        
        # Calcul de χ = β (<Sz_tot^2> - <Sz_tot>^2)
        Sz_sq_series = poly.polymul(Sz_series, Sz_series)[:M + 1]
        varSz_series = Sz2_series - Sz_sq_series
        chi_series = np.zeros(M + 1)
        for n in range(1, M + 1):
            chi_series[n] = varSz_series[n - 1]  # β * coefficient de β^{n-1}
        
        # Ajout pondéré au cumul
        Cv_poly_cumul += poids * Cv_series
        chi_poly_cumul += poids * chi_series

# Affichage des coefficients
print("Développement en série de Cv(T) (coefficients de (1/T)^n):")
for n, coeff in enumerate(Cv_poly_cumul):
    print(f"n = {n}: {coeff:.8f}")

print("\nDéveloppement en série de χ(T) (coefficients de (1/T)^n):")
for n, coeff in enumerate(chi_poly_cumul):
    print(f"n = {n}: {coeff:.8f}")

# Tracé des courbes
T_vals = np.linspace(0.5, 10, 100)
beta_vals = 1 / (kB * T_vals)

# Évaluation des polynômes
Cv_vals = np.polyval(Cv_poly_cumul[::-1], beta_vals)
chi_vals = np.polyval(chi_poly_cumul[::-1], beta_vals)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(T_vals, Cv_vals, 'b-')
plt.title(r"Développement en série de $C_v/N$")
plt.xlabel("Température $T$")
plt.xlim(0)
plt.ylim(0)
plt.ylabel(r"$C_v/N$")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(T_vals, chi_vals, 'r-')
plt.title(r"Développement en série de $\chi/N$")
plt.xlabel("Température $T$")
plt.ylabel(r"$\chi/N$")
plt.xlim(0)
plt.ylim(0)
plt.grid(True)

plt.suptitle(f"Approximation série (ordre M={M})", fontsize=14)
plt.tight_layout()
plt.show()