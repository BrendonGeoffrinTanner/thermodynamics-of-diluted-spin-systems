import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.linalg import eigh

#####################################
# Définition des opérateurs de spin
#####################################

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

#####################################
# Développement en série de Cv(β)
#####################################

def compute_Cv_series(J, h, Nmax, bonds, max_order, p):
    Cv_series = np.zeros(max_order + 1)
    for i in range(1, Nmax + 1):
        bonds_i = [(a, b) for (a, b) in bonds if a < i and b < i]
        H_i = hamiltonian_dense_from_bonds(J, h, i, bonds_i)
        eigvals = eigh(H_i, eigvals_only=True)

        a = np.zeros(max_order + 1)
        b = np.zeros(max_order + 1)
        e = np.zeros(max_order + 1)

        for k in range(max_order + 1):
            f = math.factorial(k)
            s = (-1)**k
            a[k] = s * np.sum(eigvals**k) / f
            if k <= max_order - 1:
                b[k] = s * np.sum(eigvals**(k + 1)) / f
            if k <= max_order - 2:
                e[k] = s * np.sum(eigvals**(k + 2)) / f

        inv_a = np.zeros(max_order + 1)
        inv_a[0] = 1 / a[0]
        for n in range(1, max_order + 1):
            inv_a[n] = -np.sum([a[j] * inv_a[n - j] for j in range(1, n + 1)]) * inv_a[0]

        U = np.zeros(max_order + 1)
        for n in range(max_order + 1):
            U[n] = np.sum([b[k] * inv_a[n - k] for k in range(n + 1)])

        U2 = np.zeros(max_order + 1)
        for n in range(max_order + 1):
            U2[n] = np.sum([e[k] * inv_a[n - k] for k in range(n + 1)])

        U_sq = np.convolve(U, U)[:max_order + 1]
        varE = U2 - U_sq

        Cv_i = np.zeros(max_order + 1)
        for n in range(2, max_order + 1):
            Cv_i[n] = varE[n - 2]

        poids = i * (1 - p)**2 * p**i
        Cv_series += poids * Cv_i / i

    return Cv_series

#####################################
# Paramètres utilisateur
#####################################

J = 1.0
h = 0.0
Nmax = 7
p = 0.1
max_order = 7
bonds = [(i, i + 1) for i in range(Nmax)]

#####################################
# Exécution et affichage des résultats
#####################################

Cv_coeffs = compute_Cv_series(J, h, Nmax, bonds, max_order, p)

print(f"Développement en série de Cv(β, p={p}) jusqu'à l'ordre {max_order} :")
for n, c in enumerate(Cv_coeffs):
    print(f" - Coefficient β^{n} : {c:.8f}")
