import numpy as np
import matplotlib.pyplot as plt
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

def Sa_op(alpha, site, N):
    return [Sx_op, Sy_op, Sz_op][alpha](site, N)

#####################################
# Construction du Hamiltonien
#####################################

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

def total_Sz_operator(N):
    return sum(Sz_op(site, N) for site in range(N)).toarray()

#####################################
# Calculs thermodynamiques avec dilution
#####################################

def compute_thermodynamics(J, h, Nmax, bonds, T_vals, p):
    Cv_total = np.zeros_like(T_vals)
    chi_total = np.zeros_like(T_vals, dtype=np.complex128)

    for i in range(1, Nmax + 1):
        bonds_i = [(a, b) for (a, b) in bonds if a < i and b < i]
        H_i = hamiltonian_dense_from_bonds(J, h, i, bonds_i)
        Sz_tot_i = total_Sz_operator(i)
        beta_vals = 1 / (kB*T_vals)
        eigvals, eigvecs = eigh(H_i)
        Z_vals = np.array([np.sum(np.exp(-beta * eigvals)) for beta in beta_vals])
        lnZ = np.log(Z_vals)

        E_mean = np.array([
            np.sum(eigvals * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])
        E2_mean = np.array([
            np.sum((eigvals**2) * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])
        Cv_i = beta_vals**2 * (E2_mean - E_mean**2)

        Sz_diag = np.array([v.conj().T @ Sz_tot_i @ v for v in eigvecs.T])
        Sz_mean = np.array([
            np.sum(Sz_diag * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])
        Sz2_mean = np.array([
            np.sum((Sz_diag**2) * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])
        chi_i = beta_vals * (Sz2_mean - Sz_mean**2)
        poids = (1 - p)**2 * p**i
        Cv_total += poids * Cv_i
        chi_total += poids * chi_i

    return Cv_total, chi_total

#####################################
# Paramètres
#####################################

kB = 1
J = 1
p = 0.1
h = 0
T_vals = np.linspace(0.1, 10, 300)

'''
N = 2
liaisons = [(i, i+1) for i in range(3)]

# Simulation Cv
Cv_dil, _ = compute_thermodynamics(J, h, N, liaisons, T_vals, p)
'''



#####################################
# Lecture des données expérimentales/fichiers
#####################################
'''
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

file1 = 'data'
with open(file1 + ".npy", chi_dilmode="rb") as file:
    labels = np.load(file)
    for i in range(len(labels)):
        label = labels[i]
        x = np.load(file)
        y = np.load(file)/100
        ax.plot(x, y, label=f'Données: {label}')

# Ajout de la simulation
ax.plot(T_vals, Cv_dil, '--', linewidth=2, label=f'Simulation Cv/N (N={N}, p={p})')

#####################################
# Mise en forme
#####################################

ax.set_xlabel("Température $T$")
ax.set_ylabel(r"$C_v/N$")
ax.set_title("Comparaison données et simulation")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()


'''

#####################################
# Tracé des courbes pour N de 1 à Ni
#####################################

plt.figure(figsize=(8, 5))
plt.title("Chaleur spécifique $C_v/N$")
plt.xlabel("Température $T$")
plt.ylabel(r"$C_v/N$")
plt.grid(True)

for Ni in range(1, 6): 
    liaisons_i = [(i, i+1) for i in range(Ni)]
    #print(liaisons_i)
    Cv_dil,chi_dil = compute_thermodynamics(J, h, Ni, liaisons_i, T_vals, p)
    #print(Cv_dil[-1])
    plt.plot(T_vals, Cv_dil, label=f'N={Ni}')

# Données expérimentales
try:
    with open("Cv/cvp01.npy", "rb") as file:
        labels = np.load(file)
        for i in range(len(labels)):
            label = labels[i]
            x = np.load(file)
            y = np.load(file)
            #print(y[-1])
            plt.plot(x, y, '+', label=f'Données: {label}')
except Exception as e:
    print("Erreur lors du chargement des données:", e)

plt.legend(fontsize='small')
plt.tight_layout()
plt.show()