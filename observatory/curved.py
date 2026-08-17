"""Curved space from an inhomogeneous vacuum.

The metric is built LOCALLY: each pair of adjacent landmarks gets an
edge whose length is the mutual-information ruler -log I across that
gap. Long distances are shortest paths through the edge graph.

Calibration: the instrument is zeroed on the empty vacuum. An edge's
length is its true lattice length PLUS whatever extra -log I the
matter profile adds relative to the flat case:

    w_edge = L_lattice + [ (-log I)_matter - (-log I)_vacuum ]

so a universe with no matter is EXACTLY flat by construction, and all
curvature seen afterwards is the measured response to matter. This is
a differential measurement, the same trick as any interferometer.
"""
import numpy as np

from observatory.quantum import (scalar_couplings, scalar_ground_state,
                                 scalar_mutual_information, patch_sites)


def landmark_lattice(n, margin=4, spacing=3):
    coords = list(range(margin, n - margin - 1, spacing))
    corners = [(y, x) for y in coords for x in coords]
    side = len(coords)
    return corners, side


def lattice_edges(side, spacing=3):
    """(i, j, lattice_length) for axis, diagonal, and knight-move
    neighbor pairs. Knight moves matter: with only 8 neighbors the
    shortest-path metric has ~5% octagonal anisotropy, which an MDS
    embedding renders as fake waviness; 16 neighbors cut it to ~2%."""
    edges = []
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1),
            (1, 2), (2, 1), (2, -1), (1, -2))
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for (dr, dc) in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    L = spacing * (dr * dr + dc * dc) ** 0.5
                    edges.append((i, rr * side + cc, L))
    return edges


def edge_rulers(n, corners, edges, mass):
    """-log I across every edge, for the given mass profile."""
    K = scalar_couplings(n, mass=mass)
    X, P = scalar_ground_state(K)
    patches = [patch_sites(n, c) for c in corners]
    w = np.empty(len(edges))
    for e, (i, j, _) in enumerate(edges):
        w[e] = -np.log(scalar_mutual_information(X, P, patches[i],
                                                 patches[j]))
    return w


def calibrated_weights(edges, w_matter, w_vacuum):
    lengths = np.array([L for _, _, L in edges])
    return np.maximum(lengths + (w_matter - w_vacuum), 0.2 * lengths)


def all_pairs(side, edges, weights):
    """Floyd-Warshall with path reconstruction."""
    m = side * side
    D = np.full((m, m), np.inf)
    nxt = np.tile(np.arange(m), (m, 1))
    np.fill_diagonal(D, 0)
    for (i, j, _), w in zip(edges, weights):
        if w < D[i, j]:
            D[i, j] = D[j, i] = w
            nxt[i, j], nxt[j, i] = j, i
    for k in range(m):
        via = D[:, k, None] + D[None, k, :]
        better = via < D
        D = np.where(better, via, D)
        nxt = np.where(better, nxt[:, k, None], nxt)
    return D, nxt


def recover_path(nxt, i, j):
    path = [i]
    while path[-1] != j:
        path.append(int(nxt[path[-1], j]))
        if len(path) > len(nxt):
            return None
    return path


def embed(D, dims=3):
    """Classical MDS into `dims` dimensions."""
    m = len(D)
    J = np.eye(m) - 1.0 / m
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:dims]
    return v[:, order] * np.sqrt(np.maximum(w[order], 0))
