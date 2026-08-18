"""Part 29: the first interacting engine.

Every engine in this repository so far has been free: exactly
solvable matter whose particles never exert forces on one another.
This part crosses the wall. The Schwinger model — quantum
electrodynamics in one space dimension — is the hydrogen atom of
interacting field theory: charges confine, the vacuum polarizes,
a composite particle (the "meson", Schwinger's boson) appears with
an exactly known continuum mass M = e/sqrt(pi) = 0.5642 e, and in
the massless model every external charge is screened by pair
creation (Coleman). On the lattice the gauge field can be
integrated out exactly (Gauss's law leaves no radiative photon in
one dimension — the gauge field is pure ledger, the same anatomy
part 25 met in three-dimensional gravity), leaving fermions with a
long-range Coulomb interaction, solved here by exact
diagonalization with a hand-rolled Lanczos (numpy only).

  [110] the instrument, and the wall: Lanczos against dense
        diagonalization to 5e-14; the strong-coupling limit exact.
        And the cost is the point: part 15 measured that
        entanglement is what makes engines expensive, and part 29
        feels it — twenty interacting sites (184,756 amplitudes)
        cost more than six thousand free ones. Precision degrades
        accordingly, and is reported accordingly.
  [111] the meson: in the massless model the screening potential is
        exactly V(l) = Q^2 e sqrt(pi)/2 (1 - e^{-Ml}). Fitting the
        measured V(l) at even separations: the screening length
        gives M/e = 0.57 (exact: 0.5642); the plateau gives it 20%
        high (the coarse lattice's systematic); the direct gap
        sequences approach from above through ~0.77 at the largest
        affordable sizes. An interacting composite particle, its
        mass read off the vacuum's screening cloud.
  [112] confinement, and the string that breaks: with massive
        fermions a half-integer external charge is confined by a
        linear potential forever (measured tension 0.236 vs the
        classical 0.25 — vacuum polarization pays the 6%); an
        integer charge pulls a linear string that BREAKS at l ~ 8
        sites, where the measured slope collapses from 0.93 to
        ~0.05 — the vacuum manufactures a particle-antiparticle
        pair and each half dresses one external charge. The flux
        tube, its absence after breaking, and the created pair are
        all measured in place.

What this begins and does not finish: interacting matter in the
render program. Chiral interacting matter in 3+1 dimensions — the
knob count's standing wall — remains open.
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

MEXACT = 1 / math.sqrt(math.pi)


def basis(N):
    states = []
    for occ in combinations(range(N), N // 2):
        s = 0
        for o in occ:
            s |= 1 << o
        states.append(s)
    return np.array(sorted(states), dtype=np.int64)


def occupations(N, states):
    return ((states[:, None] >> np.arange(N)[None, :]) & 1).astype(
        np.int64)


def diagonal(N, states, F, mu):
    """Electric energy sum (L_n + F_n)^2 with L_n the cumulative
    staggered charge (Gauss's law, open chain), plus the mass term.
    Units: e^2 a / 2; x = 1/(e a)^2; mu = 2 m / (e^2 a)."""
    occ = occupations(N, states)
    stag = np.array([(1 - (-1) ** k) // 2 for k in range(N)])
    q = occ - stag[None, :]
    Lcum = np.cumsum(q, axis=1)[:, :-1]
    E = ((Lcum + F[None, :]) ** 2).sum(axis=1).astype(float)
    E += mu * (occ * np.array([(-1) ** k for k in range(N)])).sum(axis=1)
    return E


def hop_tables(N, states):
    tb = []
    for n in range(N - 1):
        b0, b1 = 1 << n, 1 << (n + 1)
        has = (states & b0 > 0) & (states & b1 == 0)
        src = np.where(has)[0]
        tb.append((src, np.searchsorted(states, states[src] ^ b0 ^ b1)))
    return tb


def lanczos(dg, tb, x, nb, k=1, iters=150, want_vec=False):
    """Lowest k eigenvalues (and the ground vector) by Lanczos with
    full reorthogonalization."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=nb)
    v /= np.linalg.norm(v)
    V = [v]
    al, be = [], []
    for _ in range(iters):
        wv = dg * V[-1]
        for (src, dst) in tb:
            wv[dst] += x * V[-1][src]
            wv[src] += x * V[-1][dst]
        a = float(V[-1] @ wv)
        al.append(a)
        wv = wv - a * V[-1] - (be[-1] * V[-2] if be else 0)
        for u in V:
            wv -= (u @ wv) * u
        b = float(np.linalg.norm(wv))
        if b < 1e-12:
            break
        be.append(b)
        V.append(wv / b)
    T = np.diag(al) + np.diag(be[:len(al) - 1], 1) \
        + np.diag(be[:len(al) - 1], -1)
    ev, evec = np.linalg.eigh(T)
    if not want_vec:
        return ev[:k], None
    g = np.zeros(nb)
    for i, u in enumerate(V[:len(al)]):
        g += evec[i, 0] * u
    return ev[:k], g / np.linalg.norm(g)


def external_field(N, l, Q):
    """Background field of external charges +Q, -Q at separation l,
    centered."""
    F = np.zeros(N - 1)
    a_ = (N - l) // 2
    F[a_:a_ + l] = Q
    return F


def main():
    t00 = time.time()
    print('=' * 68)
    print('PART 29: THE FIRST INTERACTING ENGINE')
    print('=' * 68)
    print()
    print('The Schwinger model: QED in one space dimension. The gauge')
    print('field is integrated out exactly (in 1D Gauss\'s law leaves no')
    print('radiative photon — the gauge sector is pure bookkeeping),')
    print('leaving fermions with a Coulomb interaction. Exact')
    print('diagonalization, hand-rolled Lanczos, charge-zero sector,')
    print('units e^2 a/2 with x = 1/(ea)^2.')
    print()

    # ---- [110] instrument ---------------------------------------------
    print('[110] the instrument, and the wall:')
    N = 12
    st = basis(N)
    tb = hop_tables(N, st)
    dg = diagonal(N, st, np.zeros(N - 1), 0.0)
    Hd = np.diag(dg)
    for (src, dst) in tb:
        for s_, d_ in zip(src, dst):
            Hd[d_, s_] += 2.0
            Hd[s_, d_] += 2.0
    evd = np.linalg.eigvalsh(Hd)[:2]
    evl, _ = lanczos(dg, tb, 2.0, len(st), k=2)
    print(f'     Lanczos vs dense (N=12, x=2): max deviation '
          f'{np.abs(evd - evl[:2]).max():.1e}')
    ev0, _ = lanczos(dg, tb, 0.0, len(st), k=2)
    print(f'     strong-coupling limit x=0: gap = {ev0[1]-ev0[0]:.4f} '
          '(exact: 1 — one link of flux)')
    print('     the wall, felt: 20 interacting sites = 184,756 '
          'amplitudes and minutes')
    print('     of Lanczos, where the free engines of parts 26-28 did '
          '6,400 sites in')
    print('     seconds. Part 15 measured why; this part pays it.')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    # ---- [111] the meson ----------------------------------------------
    print('[111] the meson (massless model, external integer charges):')
    N = 20
    st = basis(N)
    tb = hop_tables(N, st)
    nb = len(st)
    x = 2.0
    dg0 = diagonal(N, st, np.zeros(N - 1), 0.0)
    e_base, _ = lanczos(dg0, tb, x, nb)
    ls = np.array([2, 4, 6, 8, 10], dtype=float)
    V_scr = []
    for l in (2, 4, 6, 8, 10):
        dg = diagonal(N, st, external_field(N, l, 1.0), 0.0)
        ev, _ = lanczos(dg, tb, x, nb)
        V_scr.append(float(ev[0] - e_base[0]))
        print(f'     V(l={l:2d}) = {V_scr[-1]:.3f}   '
              f'[{time.time() - t00:.0f}s]')
    V_scr = np.array(V_scr)
    print(f'     the exact continuum law: V(l) = Q^2 e sqrt(pi)/2 '
          f'(1 - e^(-Ml)).')
    # screening length from increment ratios (plateau-independent):
    # (V(l+2)-V(l)) / (V(l)-V(l-2)) = e^(-2/lambda). The last
    # increment carries a visible boundary creep and is excluded.
    inc = np.diff(V_scr)
    ratios = inc[1:] / inc[:-1]
    lam_fit = -2.0 / math.log(float(np.mean(ratios[:2])))
    Me_lam = math.sqrt(x) / lam_fit
    A_fit = float(V_scr[-1])          # effective plateau at l = 10
    A_exact = math.sqrt(math.pi * x)
    print(f'     increment ratios: {ratios[0]:.3f}, {ratios[1]:.3f} '
          f'(consistent), {ratios[2]:.3f} (boundary')
    print(f'       creep; excluded) -> screening length '
          f'{lam_fit:.2f} sites -> M/e = {Me_lam:.3f}')
    print(f'     effective plateau {A_fit:.2f} vs exact continuum '
          f'{A_exact:.2f}: the +20% is the')
    print('       O(ea) coarse-lattice systematic, which the '
          'length-ratio estimator avoids.')
    print(f'     exact: 1/sqrt(pi) = {MEXACT:.4f}. The meson\'s '
          'mass, read off the vacuum\'s')
    print(f'     screening cloud: {Me_lam:.3f} — within '
          f'{100 * abs(Me_lam - MEXACT) / MEXACT:.0f}% of the exact '
          'answer.')
    # gap sequences at fixed physical volume
    print('     the direct gap, fixed physical volume L*e = 6 '
          '(secondary instrument):')
    gaps = []
    for Ng in (12, 16, 20):
        xg = (Ng / 6.0) ** 2
        stg = basis(Ng)
        tbg = hop_tables(Ng, stg)
        dgg = diagonal(Ng, stg, np.zeros(Ng - 1), 0.0)
        evg, _ = lanczos(dgg, tbg, xg, len(stg), k=2, iters=170)
        Me = (evg[1] - evg[0]) / (2 * math.sqrt(xg))
        gaps.append((Ng, Me))
        print(f'       N={Ng:2d}: M/e = {Me:.4f}   '
              f'[{time.time() - t00:.0f}s]')
    print('       approaching the exact value from above; '
          'percent-level closure needs')
    print('       matrix-product machinery beyond one numpy file — '
          'the honest cost of')
    print('       interaction at exact-diagonalization sizes.')
    print()

    # ---- [112] confinement and the string ------------------------------
    print('[112] confinement, and the string that breaks (mu = 2):')
    MU = 2.0
    dgm = diagonal(N, st, np.zeros(N - 1), MU)
    em_base, _ = lanczos(dgm, tb, x, nb)
    V_half, V_int = [], []
    for l in (2, 4, 6, 8, 10):
        dg = diagonal(N, st, external_field(N, l, 0.5), MU)
        ev, _ = lanczos(dg, tb, x, nb)
        V_half.append(float(ev[0] - em_base[0]))
    for l in (2, 4, 6, 8, 10):
        dg = diagonal(N, st, external_field(N, l, 1.0), MU)
        ev, _ = lanczos(dg, tb, x, nb)
        V_int.append(float(ev[0] - em_base[0]))
    V_half, V_int = np.array(V_half), np.array(V_int)
    sig_half = float(np.polyfit(ls, V_half, 1)[0]) / 1.0
    sl_early = (V_int[2] - V_int[0]) / 4.0
    sl_late = (V_int[4] - V_int[3]) / 2.0
    print(f'     half-integer charge: V(l) linear at slope '
          f'{sig_half:.3f} per site')
    print(f'       (classical field energy: Q^2 = 0.25; vacuum '
          'polarization pays the')
    print(f'       {100 * (0.25 - sig_half) / 0.25:.0f}%). Confined. '
          'The string never breaks.')
    print(f'     integer charge: slope {sl_early:.2f} per site '
          f'(classical 1.0) through l = 6,')
    print(f'       then {sl_late:.2f} beyond l = 8: THE STRING '
          'BREAKS — the vacuum creates')
    print('       a fermion pair and each half dresses one external '
          'charge.')
    print(f'     [{time.time() - t00:.0f}s]')

    # flux and charge profiles at l = 8, baseline-subtracted (the
    # zero-charge ground state carries the same staggered structure;
    # the difference is the flux and charge INDUCED by the pair)
    occ_all = occupations(N, st).astype(float)
    stag = np.array([(1 - (-1) ** k) // 2 for k in range(N)])

    def profiles(Q, mu):
        F = external_field(N, 8, Q) if Q else np.zeros(N - 1)
        dg = diagonal(N, st, F, mu)
        ev, g = lanczos(dg, tb, x, nb, want_vec=True)
        w = g * g
        qmean = (w[:, None] * (occ_all - stag[None, :])).sum(axis=0)
        return np.cumsum(qmean)[:-1] + F, qmean

    base = {mu: profiles(0.0, mu) for mu in (MU, 0.0)}
    profs = {}
    for tag, Q, mu in (('confined (Q=1/2, massive)', 0.5, MU),
                       ('broken (Q=1, massive)', 1.0, MU),
                       ('screened (Q=1, massless)', 1.0, 0.0)):
        Lm, qm = profiles(Q, mu)
        profs[tag] = (Lm - base[mu][0], qm - base[mu][1])
    print('     induced flux on the middle link (baseline-'
          'subtracted), l = 8 apart:')
    mid = (N - 2) // 2
    for tag, (Lm, qm) in profs.items():
        print(f'       {tag:<26} <L_mid> = {Lm[mid]:+.3f}')
    print('     the confined string carries its flux end to end; the '
          'broken and')
    print('     screened strings are empty in the middle — the pair '
          'has been made.')
    print()
    print('     What this begins: interacting matter inside the '
          'program. What stays')
    print('     open: chiral interacting matter in 3+1 dimensions — '
          'the knob count\'s')
    print('     standing wall.')

    figure(ls, V_scr, A_fit, lam_fit, V_half, V_int, profs, gaps, N,
           'films/schwinger.png')
    print()
    print(f'     films/schwinger.png  ({time.time() - t00:.0f}s)')


def figure(ls, V_scr, A_fit, lam_fit, V_half, V_int, profs, gaps, N,
           path):
    W, Ht = 1560, 660
    img = Image.new('RGB', (W, Ht), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 12), 'PART 29 - THE FIRST INTERACTING ENGINE '
           '(THE SCHWINGER MODEL)', fill=INK)

    # (a) V(l) curves
    ax0, ay0, ax1, ay1 = 70, 90, 560, 540
    d.text((ax0, ay0 - 34), 'V(l) between external charges '
           '(units e^2 a/2):', fill=INK)
    d.text((ax0, ay0 - 18), 'orange: massive Q=1 (string breaks); '
           'green: massive Q=1/2 (confined);', fill=MUTED)
    d.text((ax0, ay0 - 2), 'blue: massless Q=1 (screened) with the '
           'exact-law fit.', fill=MUTED)
    vmax = max(V_int.max(), V_scr.max(), V_half.max()) * 1.1

    def axy(l, v):
        return (ax0 + (ax1 - ax0) * l / 11.0,
                ay1 - (ay1 - ay0 - 30) * v / vmax)
    for gv in np.arange(0, vmax, 2):
        d.line([axy(0, gv), axy(11, gv)], fill=GRIDC)
    for vals, col in ((V_scr, C_BLUE), (V_half, C_GREEN),
                      (V_int, C_ORANGE)):
        pts = [axy(l, v) for l, v in zip(ls, vals)]
        d.line(pts, fill=col, width=3)
        for p in pts:
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4],
                      fill=col)
    lf = np.linspace(0.5, 10.5, 60)
    d.line([axy(l, A_fit * (1 - np.exp(-l / lam_fit))) for l in lf],
           fill=(150, 180, 220), width=1)
    d.text((axy(6.4, V_int[3])[0], axy(6.4, V_int[3])[1] - 20),
           'the break', fill=C_ORANGE)
    d.text((ax0, ay1 + 10), 'separation l (sites). The screening fit '
           f'gives M/e = {math.sqrt(2.0)/lam_fit:.3f} (exact 0.564).',
           fill=MUTED)

    # (b) flux profiles
    bx0, by0, bx1, by1 = 640, 90, 1080, 300
    d.text((bx0, by0 - 34), 'induced electric flux <L_n> (baseline '
           'subtracted), charges 8 apart:', fill=INK)
    cols = {('confined (Q=1/2, massive)'): C_GREEN,
            ('broken (Q=1, massive)'): C_ORANGE,
            ('screened (Q=1, massless)'): C_BLUE}

    def bxy(n, v):
        return (bx0 + (bx1 - bx0) * n / (N - 2),
                by1 - (by1 - by0 - 16) * (v + 0.15) / 1.3)
    d.line([bxy(0, 0), bxy(N - 2, 0)], fill=GRIDC)
    for tag, (Lm, qm) in profs.items():
        d.line([bxy(n, Lm[n]) for n in range(N - 1)], fill=cols[tag],
               width=3)
    d.text((bx0, by1 + 6), 'green: the intact flux tube. orange/blue: '
           'the middle is empty - the', fill=MUTED)
    d.text((bx0, by1 + 22), 'string broke (or screened); the vacuum '
           'made a pair.', fill=MUTED)

    # (c) charge profiles
    cx0, cy0, cx1, cy1 = 640, 380, 1080, 560
    d.text((cx0, cy0 - 18), 'induced charge per two-site cell '
           '(broken string): the created pair.', fill=INK)
    qb = profs['broken (Q=1, massive)'][1]
    qc = qb.reshape(N // 2, 2).sum(axis=1)
    qmax = np.abs(qc).max() * 1.2

    def cxy(n, v):
        return (cx0 + (cx1 - cx0) * n / (N // 2 - 1),
                (cy0 + cy1) / 2 - (cy1 - cy0 - 10) * v / (2 * qmax))
    d.line([cxy(0, 0), cxy(N // 2 - 1, 0)], fill=GRIDC)
    for n in range(N // 2):
        p0 = cxy(n, 0)
        p1 = cxy(n, float(qc[n]))
        d.rectangle([min(p0[0] - 7, p1[0] - 7), min(p0[1], p1[1]),
                     p0[0] + 7, max(p0[1], p1[1])], fill=C_ORANGE)
    d.text((cx0, cy1 + 6), 'negative cloud at one external charge, '
           'positive at the other.', fill=MUTED)

    # (d) gap sequence + summary
    sx = 1140
    d.text((sx, 90), 'the meson, three instruments:', fill=INK)
    rows = [
        (f'screening length: M/e = {math.sqrt(2.0)/lam_fit:.3f}',
         C_GREEN),
        (f'plateau height:   M/e = '
         f'{math.sqrt(math.pi)*2.0/(A_fit*math.sqrt(2.0)):.3f}',
         C_BLUE),
        ('direct gap (L*e = 6, a -> 0):', INK),
    ] + [(f'   N={n}: {m:.3f}', MUTED) for n, m in gaps] + [
        ('exact continuum: 1/sqrt(pi) = 0.564', C_ORANGE),
        ('', INK),
        ('the wall, felt: 20 interacting sites =', MUTED),
        ('184,756 amplitudes; free parts did 6,400', MUTED),
        ('sites. Interaction is what costs (part 15).', MUTED),
        ('', INK),
        ('still open: chiral interacting matter in', MUTED),
        ('3+1d - the knob count\'s standing wall.', MUTED),
    ]
    for i, (txt, col) in enumerate(rows):
        d.text((sx, 116 + i * 20), txt, fill=col)
    img.save(path)


if __name__ == '__main__':
    main()
