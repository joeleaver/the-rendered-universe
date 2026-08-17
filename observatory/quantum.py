"""The quantum observatory: distance from entanglement.

The engine here is quantum: free fermions hopping on a graph, in their
ground state. Everything about the state is encoded in the two-point
correlation matrix C_ij = <c_i† c_j> (it is Gaussian), which makes
entanglement entropies of regions exactly computable at real size
(Peschel's method): the entropy of region A comes from the eigenvalues
of C restricted to A.

The renderer defines distance from mutual information:

    d(A, B) := -log I(A : B),   I(A:B) = S_A + S_B - S_AB

High mutual information = close. This is the toy version of the
it-from-qubit program's actual working hypothesis (Ryu-Takayanagi,
Van Raamsdonk): geometry is the shape of the entanglement pattern.
A small staggered mass gaps the system so I decays exponentially —
which makes -log I LINEAR in true distance: an honest ruler.
"""
import numpy as np


def grid_hamiltonian(n, stagger=0.25, extra_bonds=(), cut_bonds=()):
    """Hopping Hamiltonian on an n x n open-boundary grid.
    extra_bonds: (i, j, strength); cut_bonds: iterable of {i, j} sets."""
    N = n * n
    H = np.zeros((N, N))
    cuts = {frozenset(b) for b in cut_bonds}

    def idx(y, x):
        return y * n + x

    for y in range(n):
        for x in range(n):
            i = idx(y, x)
            for (yy, xx) in ((y + 1, x), (y, x + 1)):
                if yy < n and xx < n:
                    j = idx(yy, xx)
                    if frozenset((i, j)) not in cuts:
                        H[i, j] = H[j, i] = -1.0
            H[i, i] = stagger * (1 if (y + x) % 2 else -1)
    for (i, j, g) in extra_bonds:
        H[i, j] = H[j, i] = -g
    return H


def ground_correlations(H):
    """C_ij = <c_i† c_j> in the ground state (all E<0 modes filled)."""
    w, v = np.linalg.eigh(H)
    occ = v[:, w < 0]
    return occ @ occ.T


def region_entropy(C, sites):
    sub = C[np.ix_(sites, sites)]
    nu = np.clip(np.linalg.eigvalsh(sub), 1e-14, 1 - 1e-14)
    return float(-(nu * np.log(nu) + (1 - nu) * np.log(1 - nu)).sum())


def mutual_information(C, A, B):
    return max(region_entropy(C, A) + region_entropy(C, B)
               - region_entropy(C, list(A) + list(B)), 1e-12)


# --- massive scalar field (coupled harmonic oscillators) ----------------
# The canonical 'entanglement builds space' matter: correlations decay
# smoothly and monotonically (no Fermi-surface oscillations), so the
# MI-metric is a clean ruler. Ground state is Gaussian: X = <xx> and
# P = <pp> follow from K = m^2 + graph Laplacian; region entropies come
# from symplectic eigenvalues of X_A P_A.

def scalar_couplings(n, mass=0.2, extra_bonds=(), cut_bonds=()):
    """`mass` may be a scalar or a per-site array of shape (n, n) —
    an inhomogeneous mass profile is how matter density enters."""
    N = n * n
    A = np.zeros((N, N))
    cuts = {frozenset(b) for b in cut_bonds}
    for y in range(n):
        for x in range(n):
            i = y * n + x
            for (yy, xx) in ((y + 1, x), (y, x + 1)):
                if yy < n and xx < n:
                    j = yy * n + xx
                    if frozenset((i, j)) not in cuts:
                        A[i, j] = A[j, i] = 1.0
    for (i, j, g) in extra_bonds:
        A[i, j] += g
        A[j, i] += g
    L = np.diag(A.sum(axis=1)) - A
    m2 = (np.asarray(mass, dtype=float) ** 2).reshape(-1)
    if m2.size == 1:
        m2 = np.full(N, m2[0])
    return np.diag(m2) + L


def scalar_ground_state(K):
    """Returns (X, P): position and momentum correlation matrices."""
    w, v = np.linalg.eigh(K)
    w = np.maximum(w, 1e-12)
    K_half = (v * np.sqrt(w)) @ v.T
    K_ihalf = (v / np.sqrt(w)) @ v.T
    return K_ihalf / 2.0, K_half / 2.0


def scalar_region_entropy(X, P, sites):
    M = X[np.ix_(sites, sites)] @ P[np.ix_(sites, sites)]
    nu = np.sqrt(np.clip(np.real(np.linalg.eigvals(M)), 0.25, None))
    a, b = nu + 0.5, np.maximum(nu - 0.5, 1e-15)
    return float((a * np.log(a) - b * np.log(b)).sum())


def scalar_mutual_information(X, P, A, B):
    return max(scalar_region_entropy(X, P, A)
               + scalar_region_entropy(X, P, B)
               - scalar_region_entropy(X, P, list(A) + list(B)), 1e-12)


def patch_sites(n, corner):
    """2x2 patch of site indices with top-left at grid `corner`."""
    y, x = corner
    return [yy * n + xx for yy in (y, y + 1) for xx in (x, x + 1)]


def mi_distance_matrix(C, n, corners):
    """d = -log I between every pair of landmark patches, floored so
    the minimum distance is 0."""
    patches = [patch_sites(n, c) for c in corners]
    m = len(patches)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = D[j, i] = -np.log(
                mutual_information(C, patches[i], patches[j]))
    if m > 1:
        off = D[~np.eye(m, dtype=bool)]
        D[~np.eye(m, dtype=bool)] = off - off.min()
    return D
