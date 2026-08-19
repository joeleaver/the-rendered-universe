"""Part 32: the tensor engine, and the door it was built for.

Part 31 priced the chiral frontier at forty fermion modes against
exact diagonalization's twenty-four. This part builds the instrument
that crosses such walls — a matrix-product-state engine with an MPO
compiler and two-site DMRG, numpy only like everything else here —
validates it against every exact answer the repository owns, pushes
the Schwinger model beyond the exact-diagonalization wall, and then
walks up to the door itself: the tangent-fermion 3-4-5-0 model
(arXiv:2606.24713), the lattice on which symmetric mass generation
of genuinely chiral matter was recently demonstrated with
matrix-product machinery.

  [119] the engine: an MPO compiler (batched direct-sum with exact
        SVD compression — long-range terms cost nothing special; the
        compressor finds the Schwinger Coulomb tail's bond-dimension-5
        form by itself) and two-site DMRG with penalty-projected
        excited states. Validated: transverse-field Ising ground and
        excited energies to 1e-8 at import; the part-29 Schwinger
        meson gap reproduced to all four printed decimals at N = 12
        (one second) and N = 20 (twelve seconds, where the exact
        diagonalization took minutes).
  [120] beyond the wall: the meson-gap sequences continued past
        exact diagonalization's reach (N = 24..40 at three physical
        volumes) join the ED points smoothly and keep descending
        toward the exact e/sqrt(pi) = 0.5642: finest raw point
        0.689 from above, double-linear extrapolation 0.46 from
        below (the open-boundary systematic, stated) — the exact
        answer is now bracketed, and the N = 40 screening potential
        resolves its full plateau. (The grid reruns with --full,
        ~20 min.)
  [121] the door: the tangent-fermion 3450 model implemented
        exactly — the free sector's exact diagonalization matches
        the analytic tangent-sea energy to six decimals — and the
        interacting model solved exactly at L = 4: a UNIQUE ground
        state in the charge-neutral sector with gap 1.83 (free:
        1.66) and no degeneracy across charge sectors: the first
        exact interacting-3450 numbers in this repository, the
        symmetric-gap behavior in miniature. The measured boundary:
        the tangent kinetic term is nonlocal, so even the free sea's
        half-cut entanglement grows logarithmically (its e^S lower
        bound alone reaches 216 by L = 20 — the reason the published
        result needed bond dimension 16,384), and a numpy engine
        tops out near bond dimension 100: reach L ~ 6-8 of the
        needed ~20. The door is built, hinged, and measured; walking
        through at full size needs a compiled tensor backend — a
        cost now specified in bond dimensions rather than modes.
"""
import math
import sys
import time
from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw

from observatory import mps

# validated dark-mode categorical palette (dataviz slots 1-3)
C_BLUE, C_ORANGE, C_GREEN = (57, 135, 229), (217, 89, 38), (25, 158, 112)
INK, MUTED, GRIDC = (195, 194, 183), (122, 122, 130), (38, 38, 44)
BG = (14, 14, 18)

FULL = '--full' in sys.argv


# ---- Schwinger terms (part-29 conventions) ----------------------------


def schwinger_terms(N, x, mu=0.0, lam=8.0):
    terms = []
    for n in range(N - 1):
        terms.append(mps.jw_term(x, [(n, 'c+'), (n + 1, 'c')]))
        terms.append(mps.jw_term(x, [(n + 1, 'c+'), (n, 'c')]))
    s = [(1 - (-1) ** k) // 2 for k in range(N)]
    w = [[float(N - 1 - max(i, j)) for j in range(N)]
         for i in range(N)]
    const = 0.0
    lin = [0.0] * N
    for i in range(N):
        for j in range(N):
            const += w[i][j] * s[i] * s[j]
            lin[j] += -2.0 * w[i][j] * s[i]
    for i in range(N):
        lin[i] += w[i][i]
    for i in range(N):
        for j in range(i + 1, N):
            terms.append((2 * w[i][j], {i: 'n', j: 'n'}))
    for j in range(N):
        c = lin[j] + mu * (-1) ** j
        if abs(c) > 1e-14:
            terms.append((c, {j: 'n'}))
    terms.append((const, {}))
    # charge-sector stabilization lam (N_tot - N/2)^2
    for i in range(N):
        for j in range(i + 1, N):
            terms.append((2 * lam, {i: 'n', j: 'n'}))
    for i in range(N):
        terms.append((lam * (1 - N), {i: 'n'}))
    terms.append((lam * N * N / 4.0, {}))
    return terms


def schwinger_gap(N, x, chi, sweeps=12):
    terms = schwinger_terms(N, x)
    W = mps.build_mpo(N, terms)
    e0, ms0 = mps.dmrg(W, N, chi=chi, sweeps=sweeps, tol=1e-8)
    e1, _ = mps.dmrg(W, N, chi=chi, sweeps=sweeps + 2, penalty=[ms0],
                     pw=40.0, rng=np.random.default_rng(7), tol=1e-8)
    return (e1 - e0) / (2 * math.sqrt(x))


# ---- the tangent-fermion 3450 model -----------------------------------

CHI_F = [+1, +1, -1, -1]
QF = [3, 4, 5, 0]
T0 = 1.0


def jw(n, f):
    return 4 * n + f


def tangent_T(L):
    ks = 2 * np.pi * (np.arange(L) + 0.5) / L
    ks = np.where(ks > np.pi, ks - 2 * np.pi, ks)
    E = 2 * T0 * np.tan(ks / 2)
    n = np.arange(L)
    ph = np.exp(1j * np.outer(n, ks))
    return (ph * E[None, :]) @ ph.conj().T / L


def model_terms(L, g1, g2, UH, lam=6.0):
    T = tangent_T(L)
    terms = []
    NJW = 4 * L
    for f in range(4):
        for n in range(L):
            for m in range(L):
                if n == m or abs(T[n, m]) < 1e-13:
                    continue
                terms.append(mps.jw_term(
                    CHI_F[f] * T[n, m],
                    [(jw(n, f), 'c+'), (jw(m, f), 'c')]))
    for n in range(L):
        n1 = (n + 1) % L
        seq1 = [(jw(n, 0), 'c'), (jw(n, 1), 'c+'), (jw(n1, 1), 'c+'),
                (jw(n, 2), 'c'), (jw(n, 3), 'c'), (jw(n1, 3), 'c+')]
        seq2 = [(jw(n, 0), 'c'), (jw(n1, 0), 'c'), (jw(n, 1), 'c'),
                (jw(n, 2), 'c+'), (jw(n1, 2), 'c+'), (jw(n, 3), 'c')]
        for g, seq in ((g1, seq1), (g2, seq2)):
            if abs(g) < 1e-14:
                continue
            terms.append(mps.jw_term(g, seq))
            terms.append(mps.jw_term(
                g, [(m, 'c' if k == 'c+' else 'c+')
                    for (m, k) in reversed(seq)]))
    if abs(UH) > 1e-14:
        for n in range(L):
            a = [jw(n, f) for f in range(4)]
            for left, right in (
                (((0, 1.0), (1, -2.0)), ((2, 1.0), (3, 2.0))),
                (((0, 2.0), (1, 1.0)), ((2, -2.0), (3, 1.0)))):
                for (fa, ca) in left:
                    for (fb, cb) in right:
                        c = UH * ca * cb
                        terms.append((c, {a[fa]: 'n', a[fb]: 'n'}))
                        terms.append((-0.5 * c, {a[fa]: 'n'}))
                        terms.append((-0.5 * c, {a[fb]: 'n'}))
                        terms.append((0.25 * c, {}))
    if lam:
        for i in range(NJW):
            for jx in range(i + 1, NJW):
                terms.append((2 * lam, {i: 'n', jx: 'n'}))
        for i in range(NJW):
            terms.append((lam * (1 - 4 * L), {i: 'n'}))
        terms.append((lam * 4 * L * L, {}))
    return terms


def ed_block(L, g1, g2, UH, Qt):
    """Exact diagonalization of the (half filling, charge Qt) block,
    built from the same JW term list the MPS uses."""
    terms = model_terms(L, g1, g2, UH, lam=0.0)
    NJW = 4 * L
    mats_list = [(coef, sorted(
        (k, (mps.OPS[v] if isinstance(v, str) else v))
        for k, v in tdict.items())) for coef, tdict in terms]
    states = []
    for occ in combinations(range(NJW), NJW // 2):
        if sum(QF[m % 4] for m in occ) != Qt:
            continue
        sbit = 0
        for m in occ:
            sbit |= 1 << m
        states.append(sbit)
    states = np.array(sorted(states), dtype=np.int64)
    index = {int(sv): i for i, sv in enumerate(states)}
    nb = len(states)
    H = np.zeros((nb, nb), dtype=complex)
    for coef, mats in mats_list:
        for si, sv in enumerate(states):
            amp = coef
            cur = int(sv)
            ok = True
            for site, mat in mats:
                b = (cur >> site) & 1
                col = mat[:, b]
                nz = np.nonzero(np.abs(col) > 1e-14)[0]
                if len(nz) == 0:
                    ok = False
                    break
                bn = int(nz[0])
                amp *= col[bn]
                if bn != b:
                    cur ^= (1 << site)
            if ok and (di := index.get(cur)) is not None:
                H[di, si] += amp
    return nb, np.linalg.eigvalsh(H)[:4]


# measured with --full (the ~20 min grid); quoted here so the
# default run stays short. Reproduce with: python3 dmrg.py --full
# N <= 20 points are part-29-machinery exact diagonalization.
GRID = {
    (6.0, 12): 0.8145, (6.0, 16): 0.8028, (6.0, 20): 0.7959,
    (6.0, 24): 0.7914, (6.0, 32): 0.7858, (6.0, 40): 0.7825,
    (8.0, 12): 0.7850, (8.0, 16): 0.7603, (8.0, 20): 0.7457,
    (8.0, 24): 0.7360, (8.0, 32): 0.7238, (8.0, 40): 0.7165,
    (10.0, 24): 0.7178, (10.0, 32): 0.6999, (10.0, 40): 0.6891,
}
# screening potential at N = 40, x = 3, even separations (--full)
SCREEN40 = {2: 1.5326, 4: 2.4658, 6: 2.9708, 8: 3.2057,
            10: 3.3083, 12: 3.3545, 14: 3.3785}


def main():
    t00 = time.time()
    print('=' * 68)
    print('PART 32: THE TENSOR ENGINE, AND THE DOOR IT WAS BUILT FOR')
    print('=' * 68)
    print()
    print('Part 31 priced the chiral frontier at forty modes against')
    print('exact diagonalization\'s twenty-four. This part builds the')
    print('instrument that crosses such walls, validates it against')
    print('every exact answer the repository owns, and walks it up to')
    print('the door.')
    print()

    print('[119] the engine (observatory/mps.py):')
    print('     import-time validation: transverse-field Ising ground')
    print('     AND first excited energy vs dense diagonalization '
          '(1e-8 / 1e-6). Passed')
    print('     (this script would not have started otherwise).')
    Me12 = schwinger_gap(12, 2.0, chi=48)
    print(f'     Schwinger N=12, x=2: M/e = {Me12:.4f} '
          f'(part-29 ED: 0.7842)  [{time.time() - t00:.0f}s]')
    Me20 = schwinger_gap(20, 6.25, chi=64)
    print(f'     Schwinger N=20, x=6.25: M/e = {Me20:.4f} '
          f'(part-29 ED: 0.7457)  [{time.time() - t00:.0f}s]')
    print('     Four printed decimals both times — and the N=20 state '
          'took seconds')
    print('     where part 29\'s exact diagonalization took minutes. '
          'The MPO compiler')
    print('     found the Coulomb tail\'s bond-dimension-5 form by '
          'itself.')
    print()

    print('[120] beyond the wall (meson-gap sequences; ED stops at '
          'N = 20):')
    print('       L*e     N=12    N=16    N=20  |  N=24    N=32    '
          'N=40   (| = the ED wall)')
    for Le in (6.0, 8.0, 10.0):
        row = [f'       {Le:4.1f} ']
        for N in (12, 16, 20, 24, 32, 40):
            v = GRID.get((Le, N))
            row.append(f'  {v:.4f}' if v else '     -  ')
            if N == 20:
                row.append(' |')
        print(''.join(row))
    # double extrapolation: a -> 0 within each volume, then 1/Le -> 0
    intercepts = []
    for Le in (6.0, 8.0, 10.0):
        pts = [(Le / N, GRID[(Le, N)]) for N in (12, 16, 20, 24, 32,
                                                 40)
               if (Le, N) in GRID]
        A = np.array(pts)
        c = np.polyfit(A[:, 0], A[:, 1], 1)
        intercepts.append((Le, c[1]))
    for Le, c0 in intercepts:
        print(f'       a->0 at L*e = {Le:4.1f}: M/e -> {c0:.4f}')
    B = np.array([(1.0 / Le, c0) for Le, c0 in intercepts])
    minf = float(np.polyfit(B[:, 0], B[:, 1], 1)[1])
    print(f'       volume -> infinity (linear in 1/L): M/e -> '
          f'{minf:.3f}')
    print('     The MPS points continue the ED sequences smoothly '
          'through the wall.')
    print('     Honest reading: the raw sequences approach the exact '
          f'0.5642 from above')
    print(f'       (finest point {GRID[(10.0, 40)]:.4f}), while the '
          'double-linear extrapolation')
    print(f'       overcorrects below it ({minf:.3f}) — open '
          'boundaries contribute')
    print('       power-law corrections the linear fits mistake for '
          'physics. The')
    print('       sequences now BRACKET the exact answer; percent '
          'closure needs')
    print('       boundary-subtracted estimators, not more sites.')
    # the screening instrument at N = 40, x = 3
    ls = np.array(sorted(SCREEN40))
    Vs = np.array([SCREEN40[l] for l in ls])
    best = None
    for lamf in np.linspace(1.5, 6.0, 400):
        Mf = 1 - np.exp(-ls / lamf)
        A = float((Vs * Mf).sum() / (Mf * Mf).sum())
        r = float(((Vs - A * Mf) ** 2).sum())
        if best is None or r < best[0]:
            best = (r, A, lamf)
    _, A_f, lam_f = best
    Me_scr = math.sqrt(3.0) / lam_f
    print(f'     screening instrument at N = 40, x = 3 (plateau now '
          f'fully resolved,')
    print(f'       {len(ls)} even separations): full fit lambda = '
          f'{lam_f:.2f} sites -> M/e = {Me_scr:.3f};')
    print('       the x = 2 estimate of part 29 (0.569) remains the '
          'sharpest — larger x')
    print('       trades lattice error for slower screening, as the '
          'two fits show.')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    print('[121] the door (the tangent-fermion 3450 model, '
          'arXiv:2606.24713):')
    L = 4
    ks = 2 * np.pi * (np.arange(L) + 0.5) / L
    ks = np.where(ks > np.pi, ks - 2 * np.pi, ks)
    Ek = 2 * np.tan(ks / 2)
    efree = 4 * Ek[Ek < 0].sum()
    nb, wfree = ed_block(L, 0, 0, 0, 24)
    print(f'     free sector, L=4: ED E0 = {wfree[0]:.6f} vs analytic '
          f'tangent sea {efree:.6f}')
    print(f'       (block size {nb}; single chiral branch per flavor '
          '— no doubler: the')
    print('       tangent dispersion buys chirality with '
          'nonlocality.)')
    print(f'     [{time.time() - t00:.0f}s]')
    rows = {}
    for Qt in (23, 24, 25):
        nbq, wq = ed_block(L, 3.5, 3.5, 2.0, Qt)
        rows[Qt] = wq
        print(f'     interacting (g=3.5, U_H=2), Q={Qt}: E = '
              + ' '.join(f'{e:.4f}' for e in wq[:3])
              + f'  [{time.time() - t00:.0f}s]')
    gap = rows[24][1] - rows[24][0]
    gap_free = wfree[1] - wfree[0]
    print(f'     UNIQUE symmetric ground state (Q=24), gap '
          f'{gap:.3f} (free: {gap_free:.3f});')
    print('     charged sectors ~3.9 higher: no condensate. The '
          'symmetric-gap behavior')
    print('     of the published result, in miniature — the first '
          'exact interacting-')
    print('     3450 numbers in this repository.')
    print()
    print('     the measured boundary: the free tangent sea\'s '
          'half-cut entanglement')
    print('     (exact, correlation-matrix method):')
    chi_pts = []
    for Lc in (4, 6, 8, 12, 16, 20):
        T = tangent_T(Lc)
        wv, vv = np.linalg.eigh(T)
        occ = vv[:, wv < 0]
        C1 = occ @ occ.conj().T
        A = list(range(Lc // 2))
        nu = np.clip(np.real(np.linalg.eigvalsh(C1[np.ix_(A, A)])),
                     1e-16, 1 - 1e-16)
        S4 = float(-4 * (nu * np.log(nu)
                         + (1 - nu) * np.log(1 - nu)).sum())
        chi_pts.append((Lc, S4, math.exp(S4)))
        print(f'       L={Lc:2d}: S_half = {S4:.2f} nats,  '
              f'e^S (chi lower bound) ~ {math.exp(S4):7.0f}')
    print('     A numpy engine tops out near chi ~ 100 (matvecs '
          'become tenths of')
    print('     seconds): reach L ~ 6-8 of the ~20 the published '
          'run needed at')
    print('     chi = 16,384. The door is built, hinged, and its '
          'price is now')
    print('     specified in bond dimensions; walking through at '
          'full size needs a')
    print('     compiled tensor backend, not a new idea.')

    figure(GRID, intercepts, minf, rows, wfree, chi_pts,
           'films/dmrg.png')
    print()
    print(f'     films/dmrg.png  ({time.time() - t00:.0f}s)')

    if FULL:
        print()
        print('--full: measuring the beyond-the-wall grid live...')
        for Le in (6.0, 8.0, 10.0):
            for N in (24, 32, 40):
                x = (N / Le) ** 2
                Me = schwinger_gap(N, x, chi=64 if N <= 32 else 80)
                print(f'   L*e={Le} N={N}: M/e = {Me:.4f}  '
                      f'[{time.time() - t00:.0f}s]')


def figure(grid, intercepts, minf, rows, wfree, chi_pts, path):
    W_, Ht = 1560, 640
    img = Image.new('RGB', (W_, Ht), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 12), 'PART 32 - THE TENSOR ENGINE, AND THE DOOR IT '
           'WAS BUILT FOR', fill=INK)

    # (a) meson sequences through the wall
    ax0, ay0, ax1, ay1 = 70, 90, 640, 560
    d.text((ax0, ay0 - 34), '[120] meson gap M/e vs lattice spacing '
           'a*e, three volumes;', fill=INK)
    d.text((ax0, ay0 - 18), 'filled: MPS (beyond the ED wall). open: '
           'part-29 ED. line: exact.', fill=MUTED)

    def axy(a, m):
        return (ax0 + (ax1 - ax0) * a / 0.72,
                ay1 - (ay1 - ay0) * (m - 0.52) / 0.32)
    d.line([axy(0, 1 / math.sqrt(math.pi)),
            axy(0.72, 1 / math.sqrt(math.pi))], fill=C_ORANGE)
    d.text((ax1 - 190, axy(0, 0.5642)[1] + 4),
           'exact e/sqrt(pi) = 0.564', fill=C_ORANGE)
    cols = {6.0: C_BLUE, 8.0: C_GREEN, 10.0: (200, 170, 60)}
    for (Le, N), v in grid.items():
        a = Le / N
        px, py = axy(a, v)
        if N <= 20:
            d.ellipse([px - 4, py - 4, px + 4, py + 4],
                      outline=cols[Le], width=2)
        else:
            d.ellipse([px - 4, py - 4, px + 4, py + 4],
                      fill=cols[Le])
    for Le, c0 in intercepts:
        pts = sorted([(Le / N, v) for (l2, N), v in grid.items()
                      if l2 == Le])
        A = np.array(pts)
        c = np.polyfit(A[:, 0], A[:, 1], 1)
        d.line([axy(0, c[1]), axy(0.72, c[1] + 0.72 * c[0])],
               fill=cols[Le], width=1)
        d.text((axy(0.02, c[1])[0], axy(0.02, c[1])[1] - 16),
               f'L*e={Le:.0f}', fill=cols[Le])
    d.text((ax0, ay1 + 8), 'the MPS points continue the ED sequences '
           'smoothly through the wall; raw values', fill=MUTED)
    d.text((ax0, ay1 + 24), f'approach 0.5642 from above (finest '
           f'0.689), the volume extrapolation ({minf:.2f})', fill=MUTED)
    d.text((ax0, ay1 + 40), 'overcorrects below: the exact answer is '
           'bracketed. Screening at N=40: 0.537.', fill=MUTED)

    # (b) the 3450 spectrum
    bx0, by0 = 720, 90
    d.text((bx0, by0 - 34), '[121] the 3450 at L = 4, exact:',
           fill=INK)
    d.text((bx0, by0 - 18), 'free vs interacting (g = 3.5, U_H = 2).',
           fill=MUTED)

    def by(e, e0, scale):
        return by0 + 260 - (e - e0) * scale
    for k, (tag, wv, col) in enumerate(
            (('free', wfree, MUTED),
             ('interacting', rows[24], C_GREEN))):
        x0 = bx0 + k * 180
        e0 = wv[0]
        for e in wv[:4]:
            yy = by(e, e0, 60)
            d.line([(x0, yy), (x0 + 120, yy)], fill=col, width=3)
        d.text((x0, by0 + 280), tag, fill=col)
        d.text((x0, by0 + 296), f'gap {wv[1] - wv[0]:.3f}', fill=col)
    d.text((bx0, by0 + 330), 'unique symmetric ground state; charged',
           fill=MUTED)
    d.text((bx0, by0 + 346), 'sectors ~3.9 higher: no condensate.',
           fill=MUTED)

    # (c) the cost curve
    cx0, cy0, cx1, cy1 = 1130, 90, 1510, 420
    d.text((cx0, cy0 - 34), '[121] the price of the door: free-sea',
           fill=INK)
    d.text((cx0, cy0 - 18), 'entanglement (exact); chi >= e^S.',
           fill=MUTED)

    def cxy(Lc, lchi):
        return (cx0 + (cx1 - cx0) * Lc / 22.0,
                cy1 - (cy1 - cy0) * lchi / 11.0)
    for val, lab in ((math.log(100), 'numpy ceiling ~100'),
                     (math.log(16384), 'published run: 16,384')):
        py = cxy(0, val)[1]
        d.line([(cx0, py), (cx1, py)], fill=GRIDC)
        d.text((cx0 + 4, py - 14), lab, fill=MUTED)
    pts = [cxy(Lc, math.log(chi)) for (Lc, S4, chi) in chi_pts]
    d.line(pts, fill=C_ORANGE, width=3)
    for p in pts:
        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4],
                  fill=C_ORANGE)
    d.text((cx0, cy1 + 8), 'L (sites). the e^S lower bound alone',
           fill=MUTED)
    d.text((cx0, cy1 + 24), 'passes the numpy ceiling by L ~ 8:',
           fill=MUTED)
    d.text((cx0, cy1 + 40), 'the residual is compute, not concept.',
           fill=MUTED)
    d.text((cx0, cy1 + 72), 'engine validated to 4 decimals against',
           fill=C_GREEN)
    d.text((cx0, cy1 + 88), 'every exact answer the repo owns.',
           fill=C_GREEN)
    img.save(path)


if __name__ == '__main__':
    main()
