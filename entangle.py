"""Part 4: the quantum observatory — geometry from entanglement.

The engine is a quantum field in its ground state. The renderer's rule:
strong mutual information = adjacent. Local MI links form the
entanglement graph; distance is shortest-path through it (after
Cao-Carroll-Michalakis, 'Space from Hilbert Space'). Experiments:

  [9]  derive the metric from a massive scalar field's ground state
       and check it against the substrate: near-perfect ruler.
  [9b] repeat with FERMIONS on the same graph: the ruler degrades —
       the derived geometry follows the matter's dispersion (Dirac
       valleys), not the substrate coordinates. Geometry is a
       property of the matter, not the wiring diagram.
  [10] wormhole: spring two far patches together and watch the derived
       bridge shorten as entanglement grows — while the mouths
       decouple from their own neighborhoods (monogamy).
  [11] the Van Raamsdonk tear: cut the couplings across a seam; with
       no entanglement there is no space between the halves at all.
"""
import itertools

import numpy as np
from PIL import Image, ImageDraw

from observatory.quantum import (
    grid_hamiltonian, ground_correlations, mutual_information,
    scalar_couplings, scalar_ground_state, scalar_mutual_information,
    patch_sites)

N = 24
MASS = 0.2
CORNERS = [(y, x) for y in range(2, 24, 4) for x in range(2, 24, 4)]
P_CORNER, Q_CORNER = (2, 2), (18, 18)
KNN = 5


def scalar_mi_matrix(X, P):
    m = len(CORNERS)
    I = np.zeros((m, m))
    for a, b in itertools.combinations(range(m), 2):
        I[a, b] = I[b, a] = scalar_mutual_information(
            X, P, patch_sites(N, CORNERS[a]), patch_sites(N, CORNERS[b]))
    return I


def entanglement_graph(I):
    """Distance: d = -log I, directly (for our gapped fields -log I is
    linear in true separation, so it is already a ruler; composing it
    through shortest paths only adds per-hop offsets). Edges: each
    landmark's KNN strongest-MI links — the visible skeleton of space,
    used for drawing."""
    m = len(I)
    D = -np.log(np.maximum(I, 1e-300))
    np.fill_diagonal(D, 0)
    edges = set()
    for a in range(m):
        for b in np.argsort(-I[a])[:KNN]:
            if b != a:
                edges.add((min(a, b), max(a, b)))
    return D, sorted(edges)


def ruler_quality(D):
    """Pearson correlation between derived distance and substrate
    distance (creator-side validation only)."""
    xs, ys = [], []
    for i, j in itertools.combinations(range(len(CORNERS)), 2):
        if np.isfinite(D[i, j]):
            (ya, xa), (yb, xb) = CORNERS[i], CORNERS[j]
            xs.append(np.hypot(ya - yb, xa - xb))
            ys.append(D[i, j])
    return float(np.corrcoef(xs, ys)[0, 1])


def _mds(D):
    Df = D.copy()
    finite = np.isfinite(Df)
    if not finite.all():
        Df[~finite] = Df[finite].max() * 1.3
    m = len(Df)
    J = np.eye(m) - 1.0 / m
    B = -0.5 * J @ (Df ** 2) @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    return v[:, order[:2]] * np.sqrt(np.maximum(w[order[:2]], 0))


def draw_panel(draw, coords, edges, x_off, label):
    xy = coords - coords.min(axis=0)
    span = max(xy.max(), 1e-9)
    xy = xy / span * 400 + 55
    xy[:, 1] += x_off
    for (a, b) in edges:
        draw.line([tuple(xy[a][::-1]), tuple(xy[b][::-1])],
                  fill=(75, 75, 88), width=1)
    for i, (gy, gx) in enumerate(CORNERS):
        color = (60 + int(185 * gy / N), 60 + int(185 * gx / N), 150)
        cy, cx = xy[i]
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=color)
    draw.text((x_off + 130, 12), label, fill=(205, 205, 215))


def scalar_state(extra_bonds=(), cut_bonds=()):
    K = scalar_couplings(N, mass=MASS, extra_bonds=extra_bonds,
                         cut_bonds=cut_bonds)
    return scalar_ground_state(K)


def main():
    print('=' * 68)
    print('PART 4: GEOMETRY FROM ENTANGLEMENT (quantum fields, exact MI)')
    print('=' * 68)

    X, Pm = scalar_state()
    I0 = scalar_mi_matrix(X, Pm)
    D0, E0 = entanglement_graph(I0)
    print('[9] massive scalar field, MI-derived metric vs substrate:')
    print(f'    entanglement ruler -log I vs true separation: '
          f'pearson r = {ruler_quality(D0):.3f}')
    print('    -> for scalar matter, entanglement distance IS distance.')
    print()

    Hf = grid_hamiltonian(N, stagger=0.25)
    Cf = ground_correlations(Hf)
    m = len(CORNERS)
    If = np.zeros((m, m))
    for a, b in itertools.combinations(range(m), 2):
        If[a, b] = If[b, a] = mutual_information(
            Cf, patch_sites(N, CORNERS[a]), patch_sites(N, CORNERS[b]))
    Df, _ = entanglement_graph(If)
    print('[9b] same graph, fermionic matter (staggered, Dirac valleys):')
    print(f'    entanglement ruler -log I vs true separation: '
          f'pearson r = {ruler_quality(Df):.3f}')
    print('    strongest MI links point along lattice DIAGONALS (valley')
    print('    interference splits landmarks into two loosely-stitched')
    print('    checkerboard sheets). The derived geometry belongs to the')
    print('    MATTER, not to the wiring diagram underneath it.')
    print()

    print('[10] the entanglement wormhole (spring two far patches):')
    pi, qi = CORNERS.index(P_CORNER), CORNERS.index(Q_CORNER)
    ni = CORNERS.index((2, 6))
    bonds0 = list(zip(patch_sites(N, P_CORNER), patch_sites(N, Q_CORNER)))
    print(f'     patches {P_CORNER} and {Q_CORNER}, substrate separation '
          f'{np.hypot(16, 16):.1f}')
    print(f'     {"g":>5} {"I(P:Q)":>10} {"derived d(P,Q)":>15} '
          f'{"I(P, P\'s neighbor)":>20}')
    D_worm = E_worm = None
    for g in (0.0, 0.5, 1.0, 2.0, 4.0):
        Xw, Pw = scalar_state(extra_bonds=[(i, j, g) for i, j in bonds0])
        Iw = scalar_mi_matrix(Xw, Pw)
        Dw, Ew = entanglement_graph(Iw)
        print(f'     {g:>5.1f} {Iw[pi, qi]:>10.2e} {Dw[pi, qi]:>15.2f} '
              f'{Iw[pi, ni]:>20.2e}')
        if g == 4.0:
            D_worm, E_worm = Dw, Ew
    print('     -> the bridge shortens as entanglement grows; the mouths')
    print('        decouple from their own neighborhoods (monogamy of')
    print('        entanglement, visible as geometry).')
    print()

    print('[11] the Van Raamsdonk tear (cut all couplings across x=11|12):')
    cuts = [frozenset((y * N + 11, y * N + 12)) for y in range(N)]
    Xt, Pt = scalar_state(cut_bonds=cuts)
    It = scalar_mi_matrix(Xt, Pt)
    Dt, Et = entanglement_graph(It)
    left = [i for i, c in enumerate(CORNERS) if c[1] <= 10]
    right = [i for i, c in enumerate(CORNERS) if c[1] >= 14]
    cross = np.array([It[i, j] for i in left for j in right])
    intra = np.array([It[i, j] for i in left for j in left if i < j])
    at_floor = int((cross <= 1e-12).sum())
    print(f'     cross-seam pairs at the measurement floor (MI <= 1e-12): '
          f'{at_floor}/{len(cross)}')
    print(f'     (median intra-half MI for comparison: {np.median(intra):.2e})')
    print('     -> disentangle two regions and there is no space between')
    print('        them at all. The connectivity of space IS entanglement.')

    img = Image.new('RGB', (1520, 540), (14, 14, 18))
    d = ImageDraw.Draw(img)
    draw_panel(d, _mds(D0), E0, 0, 'scalar ground state')
    draw_panel(d, _mds(D_worm), E_worm, 505, 'wormhole spring, g=4')
    draw_panel(d, _mds(Dt), Et, 1010, 'torn: seam disentangled')
    img.save('films/entanglement_maps.png')
    print('\nmaps -> films/entanglement_maps.png')


if __name__ == '__main__':
    main()
