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
    """
    Construit le Hamiltonien dense d’un système de spins S=1/2 avec couplage pair-à-pair.

    Entrées :
    - J : constante de couplage (float), multipliant les termes d’interaction entre spins.
    - h : champ magnétique externe (float), appliqué selon z.
    - N : nombre total de sites de spin dans la chaîne (int).
    - bonds : liste de tuples (i, j) indiquant les paires de spins interagissant.

    Sortie :
    - H_dense : matrice de Hamiltonien sous forme dense (numpy.ndarray de taille 2^N × 2^N).

    Le Hamiltonien construit est :
    H = J * ∑_{(i,j) ∈ bonds} (Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j) - h * ∑_i Sz_i
    """
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
    """
    Calcule l’opérateur S^z total pour une chaîne de N spins S=1/2.

    Entrée :
    - N : nombre total de spins (int)

    Sortie :
    - Sz_total : matrice dense (numpy.ndarray de taille 2^N × 2^N) représentant S^z total.

    Cet opérateur est la somme des S^z_i sur tous les sites i :
    S^z_total = ∑_i Sz_i
    """
    return sum(Sz_op(site, N) for site in range(N)).toarray()


#####################################
# Calculs thermodynamiques avec dilution
#####################################

def compute_thermodynamics(J, h, Nmax, bonds, T_vals, p):
    """
    Calcule la chaleur spécifique Cv/N et la susceptibilité magnétique χ/N
    moyennées sur un ensemble dilué de clusters de spins jusqu’à une taille Nmax.

    Entrées :
    - J : constante de couplage (float), interaction entre spins.
    - h : champ magnétique externe selon z (float).
    - Nmax : taille maximale des clusters à considérer (int).
    - bonds : liste de paires (i, j) représentant les liens de couplage dans le réseau.
    - T_vals : tableau de températures (numpy.ndarray).
    - p : probabilité de présence d’un spin à un site donné (float entre 0 et 1).

    Sorties :
    - Cv_total / Nmax : chaleur spécifique moyenne par spin (numpy.ndarray).
    - chi_total / Nmax : susceptibilité magnétique moyenne par spin (numpy.ndarray, complexe).
    """

    # Initialisation des observables thermodynamiques
    Cv_total = np.zeros_like(T_vals)
    chi_total = np.zeros_like(T_vals, dtype=np.complex128)

    for i in range(1, Nmax + 1):
        # Sélection des liaisons internes au cluster de taille i
        bonds_i = [(a, b) for (a, b) in bonds if a < i and b < i]

        # Construction du Hamiltonien pour le cluster de taille i
        H_i = hamiltonian_dense_from_bonds(J, h, i, bonds_i)

        # Opérateur total Sz pour ce cluster
        Sz_tot_i = total_Sz_operator(i)

        # Valeurs de β = 1/T
        beta_vals = 1 / (kB*T_vals)

        # Diagonalisation du Hamiltonien
        eigvals, eigvecs = eigh(H_i)

        # Calcul de la partition pour chaque température
        Z_vals = np.array([np.sum(np.exp(-beta * eigvals)) for beta in beta_vals])
        lnZ = np.log(Z_vals)

        # Énergie moyenne et carré moyen de l'énergie
        E_mean = np.array([
            np.sum(eigvals * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])
        E2_mean = np.array([
            np.sum((eigvals**2) * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])

        # Chaleur spécifique Cv_i = β²(⟨E²⟩ - ⟨E⟩²)
        Cv_i = beta_vals**2 * (E2_mean - E_mean**2)

        # Valeurs propres de Sz total dans la base propre de H
        Sz_diag = np.array([v.conj().T @ Sz_tot_i @ v for v in eigvecs.T])
        Sz_mean = np.array([
            np.sum(Sz_diag * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])
        Sz2_mean = np.array([
            np.sum((Sz_diag**2) * np.exp(-beta * eigvals)) / Z
            for beta, Z in zip(beta_vals, Z_vals)
        ])

        # Susceptibilité magnétique χ_i = β(⟨Sz²⟩ - ⟨Sz⟩²)
        chi_i = beta_vals * (Sz2_mean - Sz_mean**2)

        # Poids statistique de ce cluster dans le modèle dilué
        poids = i * (1 - p)**2 * p**i

        #print(i)
        #print(poids)

        # Ajout pondéré aux grandeurs thermodynamiques totales
        Cv_total += poids * Cv_i/i
        chi_total += poids * chi_i/i

    # Normalisation par le nombre total de sites simulés
    return Cv_total , chi_total


#####################################
# Paramètres utilisateur
#####################################

kB = 1                   # Constante de Boltzmann en unités naturelles
J = 1                   # Constante de couplage
N = 2                    # Nombre de sites
p = 0.1                # Probabilité d’occupation
h = 0                    # Champ magnétique
T_vals = np.linspace(0.1, 1, 300)
liaisons = [(i, i+1) for i in range(N)]
      # Réseau 1D linéaire

#####################################
# Calcul et affichage
#####################################

Cv_dil, chi_dil = compute_thermodynamics(J, h, N, liaisons, T_vals, p)

#####################################
# Tracé des courbes
#####################################

plt.figure(figsize=(10, 4))
plt.suptitle("Thermodynamique d’un système de spins dilué (N = {}, p = {})".format(N, p), fontsize=14)

plt.subplot(1, 2, 1)
plt.plot(T_vals, Cv_dil, label=fr'$C_v/N$, dilution $p={p}$')
plt.xlabel("Température $T$")
plt.ylabel(r'$C_v/N$')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(T_vals, chi_dil.real, label=fr'$\chi/N$, dilution $p={p}$', color='darkorange')
plt.xlabel("Température $T$")
plt.ylabel(r'$\chi/N$')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#####################################
# Tracé des courbes pour Nmax de 1 à 20
#####################################

'''
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Chaleur spécifique $C_v/N$")
plt.xlabel("Température $T$")
plt.ylabel(r"$C_v/N$")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Susceptibilité magnétique $\chi/N$")
plt.xlabel("Température $T$")Nmax
plt.ylabel(r"$\chi/N$")
plt.grid(True)

for Nmax in range(1, 11):
    Cv_dil, chi_dil = compute_thermodynamics(J, h, Nmax, liaisons, T_vals, p)
    
    plt.subplot(1, 2, 1)
    plt.plot(T_vals, Cv_dil, label=f'N={Nmax}')
    
    plt.subplot(1, 2, 2)
    plt.plot(T_vals, chi_dil.real, label=f'N={Nmax}')

for i in [1, 2]:
    plt.subplot(1, 2, i)
    plt.legend(fontsize='small', ncol=2)

plt.tight_layout()
plt.suptitle("Thermodynamique diluée ($p=0.2$, $N=1$ à $10$)", fontsize=14)
plt.subplots_adjust(top=0.88)
plt.show()
'''