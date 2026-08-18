"""Part 30: chirality, the hidden dimension, and the mirror.

The knob count named the standing wall: interacting chiral matter on
a lattice. The obstruction is the fermion-doubling theorem (measured
in part 10): a lattice band lives on a circle, so every left-handed
crossing comes with a right-handed twin — and the weak interaction
couples to one hand only. This part builds the two known halves of
the escape and measures the exact point where the open problem
begins.

  [113] the one-handed edge: a two-dimensional engine in a Chern
        phase hosts, on its one-dimensional boundary, a fermion with
        a single sign of velocity — a chiral mode the doubling
        theorem forbids in any standalone 1D lattice. The theorem is
        not violated; it is outmaneuvered: the mandatory twin exists,
        but lives on the OPPOSITE edge, separated in space rather
        than momentum. A trivial-phase control has no such mode.
        Chirality is an edge effect of a dimension the edge does not
        see.
  [114] the ledger between the edges: threading flux through the
        cylinder pumps exactly one electron from one edge to the
        other, through the bulk — measured as spectral flow, one
        edge-localized level crossing zero per flux quantum, with
        the adiabatically tracked edge charges changing by +1 and
        -1. A lone chiral edge does not conserve charge; the bulk is
        the ledger that balances its books. This measured inflow IS
        the anomaly, and it is exactly why the gauge field cannot be
        attached to one edge alone — the forty-year-old obstruction,
        exhibited as a bookkeeping fact.
  [115] erasing the mirror: the known escape route (symmetric mass
        generation) requires interactions to gap the mirror without
        any symmetry-breaking mass. Measured, in the minimal exact
        setting (Fidkowski-Kitaev): for n Majorana mirror modes,
        symmetry forbids every quadratic mass at every n; symmetric
        quartic interactions leave the multiplet degenerate at
        n = 2, 4, 6 — and open a full gap at n = 8. The mirror can
        be erased if and only if the count is right. In 3+1
        dimensions the corresponding magic count is SIXTEEN fermions
        per generation — exactly the SO(10) register of part 11,
        including the right-handed neutrino this program already
        flagged as its dark-matter hook.

The residue, stated plainly: making the erased-mirror construction
dynamical — a lattice chiral GAUGE theory — remains the field's open
problem; the conjecture that it works precisely for the Standard
Model's anomaly-free sixteen (Wang-Wen) is adopted here as exactly
that, a conjecture. What this part adds is the measured mechanism:
the twin banished by a hidden dimension, the anomaly as inflow, and
the mirror erased at the magic count.
"""
import math
import time
from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw

# validated dark-mode categorical palette (dataviz slots 1-3)
C_BLUE, C_ORANGE, C_GREEN = (57, 135, 229), (217, 89, 38), (25, 158, 112)
INK, MUTED, GRIDC = (195, 194, 183), (122, 122, 130), (38, 38, 44)
BG = (14, 14, 18)

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)

NY = 30          # strip width
LX = 24          # cylinder circumference


def strip_h(kx, u):
    """Chern insulator (QWZ) on a strip: periodic x (momentum kx),
    open y. H(k) = sin kx sx + sin ky sy + (u + cos kx + cos ky) sz."""
    ons = np.sin(kx) * SX + (u + np.cos(kx)) * SZ
    hop = 0.5 * SZ - 0.5j * SY
    H = np.zeros((2 * NY, 2 * NY), complex)
    for y in range(NY):
        H[2 * y:2 * y + 2, 2 * y:2 * y + 2] = ons
        if y + 1 < NY:
            H[2 * y + 2:2 * y + 4, 2 * y:2 * y + 2] += hop
            H[2 * y:2 * y + 2, 2 * y + 2:2 * y + 4] += hop.conj().T
    return H


def cylinder_h(u, phi):
    """Real-space cylinder with flux phi through the hole."""
    N = LX * NY * 2
    H = np.zeros((N, N), complex)
    tx = 0.5 * SZ - 0.5j * SX
    ty = 0.5 * SZ - 0.5j * SY

    def idx(x, y):
        return 2 * (y * LX + x)

    ph = np.exp(1j * phi / LX)
    for y in range(NY):
        for x in range(LX):
            i = idx(x, y)
            H[i:i + 2, i:i + 2] += u * SZ
            j = idx((x + 1) % LX, y)
            H[j:j + 2, i:i + 2] += tx * ph
            H[i:i + 2, j:j + 2] += (tx * ph).conj().T
            if y + 1 < NY:
                j = idx(x, y + 1)
                H[j:j + 2, i:i + 2] += ty
                H[i:i + 2, j:j + 2] += ty.conj().T
    return H


# ---- Fidkowski-Kitaev machinery ---------------------------------------


def majoranas(n):
    """n Majorana operators on n/2 qubits (Jordan-Wigner)."""
    nq = n // 2
    I2 = np.eye(2, dtype=complex)

    def kron(ops):
        out = np.array([[1.0 + 0j]])
        for o in ops:
            out = np.kron(out, o)
        return out

    gs = []
    for k in range(nq):
        pre = [SZ] * k
        post = [I2] * (nq - k - 1)
        gs.append(kron(pre + [SX] + post))
        gs.append(kron(pre + [SY] + post))
    return gs


def t_conjugation(gs):
    """T = U K with U the product of the imaginary Majoranas, so that
    T gamma_i T^{-1} = +gamma_i for every i."""
    U = np.eye(gs[0].shape[0], dtype=complex)
    for i in range(1, len(gs), 2):
        U = U @ gs[i]
    return U


def symmetric_terms(gs, U, order):
    """T-symmetric Hermitian products of `order` Majoranas."""
    out = []
    for c in combinations(range(len(gs)), order):
        M = np.eye(gs[0].shape[0], dtype=complex)
        for i in c:
            M = M @ gs[i]
        if np.abs(M - M.conj().T).max() > 1e-9:
            M = 1j * M
        if np.abs(U @ M.conj() @ U.conj().T - M).max() < 1e-9:
            out.append(M)
    return out


# ---- main --------------------------------------------------------------


def main():
    t00 = time.time()
    print('=' * 68)
    print('PART 30: CHIRALITY, THE HIDDEN DIMENSION, AND THE MIRROR')
    print('=' * 68)
    print()
    print('The doubling theorem (part 10) says a lattice fermion always')
    print('comes with its mirror twin. The weak force couples to one')
    print('hand only. This part measures the two known halves of the')
    print('escape: banish the twin to a hidden dimension\'s far edge,')
    print('and erase it with interactions at the right count.')
    print()

    # ---- [113] the one-handed edge ------------------------------------
    print('[113] the one-handed edge (Chern strip, width %d):' % NY)
    ks = np.linspace(-np.pi, np.pi, 161)
    bands = {}
    for u, tag in ((-1.0, 'topological'), (-3.0, 'trivial')):
        pts = []          # (k, E, top-weight) for all states
        for k in ks:
            w, v = np.linalg.eigh(strip_h(k, u))
            wt_top = (np.abs(v[-8:, :]) ** 2).sum(axis=0)
            for e, wt in zip(w, wt_top):
                pts.append((k, float(e), float(wt)))
        bands[tag] = pts
        mid = [(k, e, wt) for (k, e, wt) in pts
               if abs(e) < 0.3 and (wt > 0.7 or wt < 0.05)]
        top_mid = [(k, e) for (k, e, wt) in mid if wt > 0.7]
        if top_mid:
            top_mid.sort()
            karr = np.array([k for k, _ in top_mid])
            earr = np.array([e for _, e in top_mid])
            vel = np.polyfit(karr, earr, 1)[0]
            print(f'     {tag}: {len(top_mid)} top-edge states in the '
                  f'gap, velocity {vel:+.2f} — single-signed')
        else:
            print(f'     {tag}: no edge states in the gap')
    print('     One chiral branch per edge in the topological phase; '
          'none in the')
    print('     control. The mandatory twin exists — on the other '
          'edge: separated in')
    print('     space, not momentum. The doubling theorem is '
          'outmaneuvered, not broken.')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    # ---- [114] the ledger between the edges ----------------------------
    print('[114] the ledger between the edges (flux pump, cylinder '
          f'{LX}x{NY}):')
    ycoord = np.zeros(LX * NY * 2)
    for y in range(NY):
        for x in range(LX):
            i = 2 * (y * LX + x)
            ycoord[i] = ycoord[i + 1] = y
    top = ycoord >= NY - 5
    bot = ycoord < 5
    phis = np.linspace(0, 2 * np.pi, 25)
    Hs = cylinder_h(-1.0, phis[0])
    w0, v0 = np.linalg.eigh(Hs)
    nfill = int((w0 < 0).sum())
    P = v0[:, :nfill]                    # adiabatically tracked filled set
    flow_rows = []                       # (phi, E, side) for midgap states
    q_top0 = float((np.abs(P) ** 2).sum(axis=1)[top].sum())
    q_bot0 = float((np.abs(P) ** 2).sum(axis=1)[bot].sum())
    for phi in phis:
        w, v = np.linalg.eigh(cylinder_h(-1.0, phi))
        # record midgap levels and their edge side
        for j, e in enumerate(w):
            if abs(e) < 0.35:
                dens = np.abs(v[:, j]) ** 2
                side = float(dens[top].sum() - dens[bot].sum())
                flow_rows.append((phi, float(e), side))
        # adiabatic tracking: occupy the nfill states with maximum
        # overlap with the previously tracked set
        ov = (np.abs(v.conj().T @ P) ** 2).sum(axis=1)
        keep = np.argsort(-ov)[:nfill]
        P = v[:, np.sort(keep)]
    dens = (np.abs(P) ** 2).sum(axis=1)
    dq_top = float(dens[top].sum()) - q_top0
    dq_bot = float(dens[bot].sum()) - q_bot0
    print(f'     adiabatic transport over one flux quantum: '
          f'dQ_top = {dq_top:+.3f}, dQ_bot = {dq_bot:+.3f}')
    print('     One electron crosses the bulk per flux quantum. A '
          'lone edge does not')
    print('     conserve charge; the bulk balances its books. This '
          'measured inflow is')
    print('     the anomaly — and the reason a gauge field cannot '
          'couple to one edge')
    print('     alone. An anomaly-free set is one whose inflows '
          'cancel: that is what')
    print('     part 11 measured the Standard Model\'s charges to be.')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    # ---- [115] erasing the mirror --------------------------------------
    print('[115] erasing the mirror (Fidkowski-Kitaev, exact):')
    print('       n   symmetric masses   symmetric quartics   best '
          'symmetric gap')
    rng = np.random.default_rng(7)
    fk_rows = []
    for n in (2, 4, 6, 8):
        gs = majoranas(n)
        err = max(np.abs(gs[i] @ gs[j] + gs[j] @ gs[i]
                         - (2 if i == j else 0)
                         * np.eye(gs[0].shape[0])).max()
                  for i in range(n) for j in range(n))
        assert err < 1e-12
        U = t_conjugation(gs)
        masses = symmetric_terms(gs, U, 2)
        quartics = symmetric_terms(gs, U, 4) if n >= 4 else []
        best = 0.0
        for _ in range(400):
            if not quartics:
                break
            cs = rng.normal(size=len(quartics))
            H = sum(c * M for c, M in zip(cs, quartics))
            w = np.linalg.eigvalsh(H)
            best = max(best, (w[1] - w[0]) / (np.abs(w).max() + 1e-12))
        fk_rows.append((n, len(masses), len(quartics), best))
        print(f'       {n}        {len(masses)}                 '
              f'{len(quartics):3d}                {best:.4f}')
    print('     Symmetry forbids every mass at every n. Interactions '
          'leave the mirror')
    print('     multiplet degenerate at n = 2, 4, 6 — and erase it '
          'completely at n = 8.')
    print('     The mirror can be removed if and only if the count '
          'is right.')
    print('     In 3+1 dimensions the magic count is SIXTEEN per '
          'generation (the')
    print('     Wang-Wen route to a lattice Standard Model, adopted '
          'here as the')
    print('     conjecture it is) — exactly the SO(10) register of '
          'part 11, INCLUDING')
    print('     the right-handed neutrino this program already '
          'flagged as gauge-blind')
    print('     matter. One count touching three riddles: '
          'latticeability, the')
    print('     generation\'s content, and dark matter.')
    print()
    print('     residue: making the erased-mirror construction '
          'dynamical — a lattice')
    print('     chiral gauge theory — remains the field\'s open '
          'problem, and this')
    print('     part does not solve it; it measures the mechanism '
          'and the count.')

    figure(bands, flow_rows, dq_top, dq_bot, fk_rows,
           'films/chirality.png')
    print()
    print(f'     films/chirality.png  ({time.time() - t00:.0f}s)')


def figure(bands, flow_rows, dq_top, dq_bot, fk_rows, path):
    W, Ht = 1560, 660
    img = Image.new('RGB', (W, Ht), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 12), 'PART 30 - CHIRALITY, THE HIDDEN DIMENSION, AND '
           'THE MIRROR', fill=INK)

    # (a) strip bands, edge-colored
    ax0, ay0, ax1, ay1 = 60, 90, 520, 560
    d.text((ax0, ay0 - 34), '[113] strip bands E(k), topological '
           'phase.', fill=INK)
    d.text((ax0, ay0 - 18), 'orange: top edge; green: bottom edge; '
           'grey: bulk.', fill=MUTED)

    def axy(k, e):
        return (ax0 + (ax1 - ax0) * (k + np.pi) / (2 * np.pi),
                (ay0 + ay1) / 2 - (ay1 - ay0) * e / 7.0)
    d.line([axy(-np.pi, 0), axy(np.pi, 0)], fill=GRIDC)
    for (k, e, wt) in bands['topological']:
        if abs(e) > 3.2:
            continue
        if wt > 0.7:
            col = C_ORANGE
        elif wt < 0.05 and abs(e) < 1.0:
            col = C_GREEN if abs(e) < 0.8 else (70, 70, 76)
        else:
            col = (70, 70, 76)
        px, py = axy(k, e)
        d.ellipse([px - 1, py - 1, px + 1, py + 1], fill=col)
    d.text((ax0, ay1 + 8), 'one branch crosses the gap per edge - a '
           'one-handed fermion. The trivial', fill=MUTED)
    d.text((ax0, ay1 + 24), 'control (u = -3) has an empty gap.',
           fill=MUTED)

    # (b) spectral flow
    bx0, by0, bx1, by1 = 600, 90, 1020, 560
    d.text((bx0, by0 - 34), '[114] spectral flow: midgap levels vs '
           'flux.', fill=INK)
    d.text((bx0, by0 - 18), f'per flux quantum: dQ_top = {dq_top:+.2f},'
           f' dQ_bot = {dq_bot:+.2f}.', fill=MUTED)

    def bxy(phi, e):
        return (bx0 + (bx1 - bx0) * phi / (2 * np.pi),
                (by0 + by1) / 2 - (by1 - by0) * e / 0.9)
    d.line([bxy(0, 0), bxy(2 * np.pi, 0)], fill=GRIDC)
    for (phi, e, side) in flow_rows:
        col = C_ORANGE if side > 0.4 else \
            (C_GREEN if side < -0.4 else (70, 70, 76))
        px, py = bxy(phi, e)
        d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=col)
    d.text((bx0, by1 + 8), 'a top-edge level flows down as a '
           'bottom-edge level flows up: one', fill=MUTED)
    d.text((bx0, by1 + 24), 'electron crosses the bulk. The lone '
           'edge does not conserve charge.', fill=MUTED)

    # (c) FK table
    cx0, cy0 = 1100, 90
    d.text((cx0, cy0 - 34), '[115] erasing the mirror '
           '(Fidkowski-Kitaev):', fill=INK)
    d.text((cx0, cy0 - 18), 'best symmetric gap vs Majorana count n.',
           fill=MUTED)
    bw = 70
    for i, (n, nm, nq, gap) in enumerate(fk_rows):
        x0 = cx0 + i * (bw + 30)
        h = 280 * gap / 0.5
        col = C_GREEN if gap > 0.05 else (70, 70, 76)
        d.rectangle([x0, cy0 + 300 - h, x0 + bw, cy0 + 300], fill=col)
        d.text((x0 + bw // 2 - 8, cy0 + 306), f'n={n}', fill=MUTED)
        d.text((x0 + bw // 2 - 16, cy0 + 300 - h - 18),
               f'{gap:.2f}', fill=INK)
    d.text((cx0, cy0 + 340), 'masses forbidden at every n (measured: '
           'zero', fill=MUTED)
    d.text((cx0, cy0 + 356), 'symmetric quadratics). the mirror '
           'erases at n = 8', fill=MUTED)
    d.text((cx0, cy0 + 372), 'and only n = 8. in 3+1d the count is '
           '16 per', fill=MUTED)
    d.text((cx0, cy0 + 388), 'generation = the SO(10) register of '
           'part 11,', fill=MUTED)
    d.text((cx0, cy0 + 404), 'including the right-handed neutrino.',
           fill=MUTED)
    d.text((cx0, cy0 + 436), 'residue: gauging the erased-mirror '
           'edge -', fill=C_ORANGE)
    d.text((cx0, cy0 + 452), 'the lattice chiral gauge theory - '
           'remains', fill=C_ORANGE)
    d.text((cx0, cy0 + 468), 'the field\'s open problem.', fill=C_ORANGE)
    img.save(path)


if __name__ == '__main__':
    main()
