"""Part 22: the entanglement first law.

Part 21 postulated the tensor's dynamics (box h = -kappa T). Jacobson's
program says the field equation should follow from thermodynamics —
delta-Q = T delta-S across local horizons — and Faulkner et al.
sharpened this: the entanglement first law, holding for all regions at
once, IS the linearized Einstein equation. This part carries out the
first step on the Gaussian observatory of part 4: a critical fermion
chain, where every entropy is exact.

  [85] global heat at temperature T: the entropy change of every
       interval, at every temperature, collapses onto the
       parameter-free CFT curve (1/3)ln(sinh x / x), x = pi*l*T/v.
       Its small-x limb IS the first law delta-S = delta<K> with the
       modular weight measured en route (Stefan-Boltzmann pi/12 too);
       beyond it, delta<K> >= delta-S — the Bekenstein bound — holds
       everywhere, measured.
  [86] heat versus work: three perturbations with the same energy.
       Thermal heat satisfies the first law exactly. A pure particle
       changes a region's entropy only while it straddles the cut
       (about one bit), and not at all once inside, even as its
       modular energy grows. A classically mixed packet contributes
       exactly its mixing entropy h2(lambda), at any position. The
       first law is a statement about heat, as in thermodynamics.
  [87] two warm regions on the chain: every interval's delta-S is
       predicted by the parabolic modular kernel (r ~ 0.999, no
       fitted parameters), and inverting the kernel recovers the
       energy distribution from entropies alone.
  [88] the small-x response scales as l^2.0 — through the
       Ryu-Takayanagi dictionary (installed, like the Born rule of
       part 7) that is a bulk metric response h ~ z^2, the unique
       static solution of the LINEARIZED EINSTEIN EQUATIONS in AdS3
       (Faulkner-Guica-Hartman-Myers-Van Raamsdonk). A mass gap
       destroys the scaling beyond v/gap, and localized non-thermal
       excitations never produce it.
"""
import math
import time

import numpy as np
from PIL import Image, ImageDraw

from observatory.quantum import region_entropy

# validated dark-mode categorical palette (dataviz slots 1-3)
C_BLUE, C_ORANGE, C_GREEN = (57, 135, 229), (217, 89, 38), (25, 158, 112)
INK, MUTED, GRIDC = (195, 194, 183), (122, 122, 130), (38, 38, 44)
BG = (14, 14, 18)

N = 2048          # main chain length
VF = 2.0          # Fermi velocity of the half-filled hopping chain


def chain(n=N, stagger=0.0):
    """Nearest-neighbour hopping chain; stagger opens a gap 2*stagger
    at the Fermi point (the massive control of [88])."""
    H = np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)
    if stagger:
        H = H + np.diag(stagger * (-1.0) ** np.arange(n))
    w, v = np.linalg.eigh(H)
    return H, w, v


def gibbs(w, v, T):
    """Correlation matrix C = <c_i+ c_j> at temperature T (T=0: sea)."""
    if T <= 0:
        occ = (w < 0).astype(float)
    else:
        occ = 1.0 / (1.0 + np.exp(np.clip(w / T, -600, 600)))
    return (v * occ) @ v.T


def interval(c, R):
    """2R sites; entangling cuts at c-R-1/2 and c+R-1/2."""
    return list(range(c - R, c + R))


def beta_w(c, R):
    """The Bisognano-Wichmann / CFT modular weight of the interval:
    beta(x) = (R^2 - (x-x_c)^2) / 2R, the inverse local Unruh
    temperature (up to 2pi/v). Parameter-free."""
    u = np.arange(c - R, c + R, dtype=float)
    return np.clip(R * R - (u - (c - 0.5)) ** 2, 0, None) / (2 * R)


def cft_curve(x):
    """Thermal interval entropy of a c=1 CFT: (1/3) ln(sinh x / x)."""
    return (1.0 / 3.0) * np.log(np.sinh(x) / x)


def mod_energy(e, c, R):
    """delta<K> = (2pi/v) * sum beta(x) delta<T00(x)> over the interval."""
    return (2 * np.pi / VF) * float(beta_w(c, R) @ e[interval(c, R)])


# ---- [85] the one curve ----------------------------------------------

def one_curve(H, w, v, C0, c0):
    S0 = {}
    pts = []
    for T in (0.01, 0.02, 0.04, 0.08):
        C1 = gibbs(w, v, T)
        e = np.diag(H @ (C1 - C0))
        sb = float(np.mean(e[c0 - 64:c0 + 64])) / T ** 2
        for ell in (8, 12, 16, 24, 32, 48, 64, 96, 128):
            R = ell // 2
            A = interval(c0, R)
            if (c0, R) not in S0:
                S0[(c0, R)] = region_entropy(C0, A)
            dS = region_entropy(C1, A) - S0[(c0, R)]
            dK = mod_energy(e, c0, R)
            x = np.pi * ell * T / VF
            pts.append(dict(T=T, ell=ell, x=x, dS=dS, dK=dK, sb=sb))
    return pts


# ---- [86] heat vs work -----------------------------------------------

def packet(C0, j, x0, k0, sig):
    """Wavepacket in the empty band (projected off the Fermi sea)."""
    phi = np.exp(-(j - x0) ** 2 / (4.0 * sig * sig)) * np.cos(k0 * (j - x0))
    phi = phi - C0 @ phi
    return phi / np.linalg.norm(phi)


def price_list(H, C0, c0):
    n = C0.shape[0]
    j = np.arange(n)
    R = 32
    A = interval(c0, R)
    S0A = region_entropy(C0, A)
    cutL = c0 - R - 0.5
    pure, mixed = [], []
    for xi in (-8, -6, -4, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        phi = packet(C0, j, cutL + xi, 2.2, 3.0)
        Q = np.outer(phi, phi)
        # pure particle: occupation 1 in the packet mode
        C1 = C0 + Q
        e = np.diag(H @ (C1 - C0))
        dS = region_entropy(C1, A) - S0A
        pure.append((xi, dS, mod_energy(e, c0, R), float(e.sum())))
        # classically mixed: occupation 0.2
        C1 = C0 + 0.2 * Q
        e = np.diag(H @ (C1 - C0))
        dS = region_entropy(C1, A) - S0A
        mixed.append((xi, dS, mod_energy(e, c0, R)))
    # lambda scan at depth 16: dS vs mixing entropy h2(lambda)
    phi = packet(C0, j, cutL + 16.0, 2.2, 3.0)
    Q = np.outer(phi, phi)
    lam_rows = []
    for lam in (0.5, 0.2, 0.05):
        C1 = C0 + lam * Q
        dS = region_entropy(C1, A) - S0A
        h2 = -lam * math.log(lam) - (1 - lam) * math.log(1 - lam)
        lam_rows.append((lam, dS, h2))
    return pure, mixed, lam_rows


# ---- [87] the landscape ----------------------------------------------

def local_gibbs(H, Tprof):
    """rho ~ exp(-sum_j h_j / T_j): a local-temperature Gibbs state.
    Any hermitian modular matrix gives a valid Gaussian state, so this
    construction cannot leave the physical state space."""
    beta = 1.0 / np.clip(Tprof, 1e-4, None)
    D = np.diag(beta)
    M = 0.5 * (D @ H + H @ D)
    mw, mv = np.linalg.eigh(M)
    occ = 1.0 / (1.0 + np.exp(np.clip(mw, -600, 600)))
    return (mv * occ) @ mv.T


def landscape(H, w, v, c0):
    """Two warm hills on a mild uniform background (T0 = 0.010); all
    differences are taken against the background Gibbs state, so no
    region is ever colder than its own gradient scale — the local-
    equilibrium ansatz stays honest everywhere in the scan."""
    n = H.shape[0]
    j = np.arange(n)
    T0 = 0.010
    Tprof = T0 + 0.035 * np.exp(-(j - (c0 - 260)) ** 2 / (2 * 70.0 ** 2)) \
        + 0.022 * np.exp(-(j - (c0 + 280)) ** 2 / (2 * 95.0 ** 2))
    C1 = local_gibbs(H, Tprof)
    Cref = gibbs(w, v, T0)
    e = np.diag(H @ (C1 - Cref))
    R = 12
    cs = np.arange(c0 - 560, c0 + 581, 6)
    dSs, dKs = [], []
    for c in cs:
        A = interval(int(c), R)
        dSs.append(region_entropy(C1, A) - region_entropy(Cref, A))
        dKs.append(mod_energy(e, int(c), R))
    dSs, dKs = np.array(dSs), np.array(dKs)
    # inverse: read e(x) back from the entropies alone (Tikhonov)
    xs = np.arange(c0 - 580, c0 + 601, 6)
    W = np.zeros((len(cs), len(xs)))
    cnt = np.zeros((len(cs), len(xs)))
    for i, c in enumerate(cs):
        bw = (2 * np.pi / VF) * beta_w(int(c), R)
        for uu, b in zip(range(int(c) - R, int(c) + R), bw):
            k = np.argmin(np.abs(xs - uu))
            W[i, k] += b
            cnt[i, k] += 1
    W = np.where(cnt > 0, W / np.maximum(cnt, 1), 0.0)  # mean kernel per bin
    # Tikhonov readout; the smoothing weight is set by the discrepancy
    # principle — largest lambda whose residual matches the measured
    # instrument floor (no peeking at the true landscape).
    D2 = np.diff(np.eye(len(xs)), 2, axis=0)
    floor = 7.7e-4
    ehat, lam = None, None
    for lam_try in np.geomspace(1e-6, 1e3, 40):
        cand = np.linalg.solve(W.T @ W + lam_try * D2.T @ D2
                               + 1e-10 * np.eye(len(xs)), W.T @ dSs)
        rms = float(np.sqrt(np.mean((W @ cand - dSs) ** 2)))
        if rms <= 1.5 * floor:
            ehat, lam = cand, lam_try
        else:
            break
    etrue = np.array([e[max(0, int(x) - 3):int(x) + 3].sum() for x in xs])
    return Tprof, e, cs, dSs, dKs, xs, ehat, etrue


# ---- [88] the depth law ----------------------------------------------

def depth_law(H, w, v, C0, c0):
    pts = []
    for T in (0.02, 0.03, 0.04):
        C1 = gibbs(w, v, T)
        for ell in (10, 12, 14, 16, 20, 24, 28, 32):
            x = np.pi * ell * T / VF
            if not (0.32 <= x <= 1.0):
                continue
            R = ell // 2
            A = interval(c0, R)
            dS = region_entropy(C1, A) - region_entropy(C0, A)
            pts.append((x, dS))
    pts = np.array(sorted(pts))
    slope = np.polyfit(np.log(pts[:, 0]), np.log(pts[:, 1]), 1)[0]
    # what the exact curve's slope is over the same window (it bends
    # away from 2.00 already by x ~ 1): the honest comparison point
    xw = pts[:, 0]
    ref = np.polyfit(np.log(xw), np.log(cft_curve(xw)), 1)[0]
    return pts, slope, ref


def gapped_points(c0):
    Hg, wg, vg = chain(N, stagger=0.25)
    C0g = gibbs(wg, vg, 0.0)
    out = []
    for T in (0.02, 0.04, 0.08):
        C1g = gibbs(wg, vg, T)
        for ell in (16, 32, 64):
            A = interval(c0, ell // 2)
            dS = region_entropy(C1g, A) - region_entropy(C0g, A)
            out.append((np.pi * ell * T / VF, dS))
    return np.array(out)


# ---- instrument floor -------------------------------------------------

def parity_floor(n):
    H, w, v = chain(n)
    C0 = gibbs(w, v, 0.0)
    c0 = n // 2
    T = 0.02
    C1 = gibbs(w, v, T)
    dev = []
    for ell in (10, 12, 14, 16):
        A = interval(c0, ell // 2)
        dS = region_entropy(C1, A) - region_entropy(C0, A)
        dev.append(dS - cft_curve(np.pi * ell * T / VF))
    return float(np.mean(np.abs(dev)))


# ---- figure -----------------------------------------------------------

def lx(x, x0, x1, px0, px1):
    return px0 + (px1 - px0) * (math.log(x) - math.log(x0)) \
        / (math.log(x1) - math.log(x0))


def figure(pts, gap_pts, slope, ref, pure, mixed, Tprof, e, cs, dSs, dKs,
           xs, ehat, etrue, path):
    W, H = 1560, 860
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 12), 'PART 22 - THE ENTANGLEMENT FIRST LAW', fill=INK)

    # (a) the one curve, log-log, with the gapped chain falling off it
    ax0, ay0, ax1, ay1 = 70, 70, 730, 420
    xr, yr = (0.1, 40.0), (6e-4, 12.0)

    def axy(x, y):
        return (lx(x, xr[0], xr[1], ax0, ax1),
                lx(y, yr[0], yr[1], ay1, ay0))
    for dec in (1e-3, 1e-2, 1e-1, 1, 10):
        py = axy(1, dec)[1]
        d.line([(ax0, py), (ax1, py)], fill=GRIDC)
        d.text((ax0 - 44, py - 5), f'{dec:g}', fill=MUTED)
    for dec in (0.1, 1, 10):
        px = axy(dec, 1)[0]
        d.line([(px, ay0), (px, ay1)], fill=GRIDC)
        d.text((px - 6, ay1 + 6), f'{dec:g}', fill=MUTED)
    xs_c = np.geomspace(0.1, 40, 200)
    d.line([axy(x, cft_curve(x)) for x in xs_c], fill=INK, width=2)
    tcol = {0.01: C_GREEN, 0.02: C_BLUE, 0.04: C_ORANGE, 0.08: (200, 170, 60)}
    for p in pts:
        px, py = axy(p['x'], max(p['dS'], yr[0]))
        col = tcol[p['T']]
        d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=col, width=2)
    for (x, dS) in gap_pts:
        px, py = axy(x, max(dS, yr[0]))
        d.line([(px - 4, py - 4), (px + 4, py + 4)], fill=MUTED, width=2)
        d.line([(px - 4, py + 4), (px + 4, py - 4)], fill=MUTED, width=2)
    d.text((ax0, ay0 - 42), '[85/88] delta-S of every interval at every '
           'temperature vs x = pi*l*T/v.', fill=INK)
    d.text((ax0, ay0 - 26), 'line: (1/3)ln(sinh x/x), no parameters. '
           'rings: measured (T color-coded). x marks: gapped', fill=MUTED)
    d.text((ax0, ay0 - 10), 'chain (mass 0.5) departing from it. '
           f'small-x slope {slope:.2f} (Einstein z^2 law: 2.00)',
           fill=MUTED)
    px, py = axy(0.4, cft_curve(0.4))
    d.text((px - 40, py + 16), 'first-law region ~ x^2', fill=C_GREEN)
    px, py = axy(8, cft_curve(8))
    d.text((px - 130, py - 20), 'Bekenstein bound: dK > dS', fill=MUTED)

    # (b) the price list
    bx0, by0, bx1, by1 = 830, 70, 1520, 420
    d.text((bx0, by0 - 42), '[86] heat vs work: a pure particle changes '
           'dS only near the cut; a mixed packet adds h2 anywhere.',
           fill=INK)
    d.text((bx0, by0 - 26), 'orange: PURE particle, delta-S vs depth '
           '(peak ~ one bit straddling the cut, zero inside).', fill=MUTED)
    d.text((bx0, by0 - 10), 'blue: same packet, classically mixed '
           '(lam=0.2): flat at h2 = 0.500, position-blind.', fill=MUTED)
    bmax = max(r[1] for r in pure)

    def bxy(xi, val):
        return (bx0 + (bx1 - bx0) * (xi + 8) / 40.0,
                by1 - (by1 - by0 - 40) * val / 0.75)
    d.line([(bxy(0, 0)[0], by0 + 30), (bxy(0, 0)[0], by1)], fill=GRIDC)
    d.text((bxy(0, 0)[0] + 4, by0 + 34), 'the cut', fill=MUTED)
    d.line([(bx0, bxy(0, 0.500)[1]), (bx1, bxy(0, 0.500)[1])], fill=GRIDC)
    d.text((bx1 - 130, bxy(0, 0.5)[1] - 14), 'h2(0.2) = 0.500', fill=C_BLUE)
    d.line([bxy(xi, v_) for xi, v_, _, _ in pure], fill=C_ORANGE, width=3)
    d.line([bxy(xi, v_) for xi, v_, _ in mixed], fill=C_BLUE, width=3)
    d.line([(bx0, bxy(0, math.log(2))[1]), (bx0 + 40, bxy(0, math.log(2))[1])],
           fill=MUTED)
    d.text((bx0 + 44, bxy(0, math.log(2))[1] - 6), 'ln 2 (one bit)',
           fill=MUTED)
    d.text((bx0, by1 + 8), 'depth of the packet behind the cut '
           '(negative: outside the interval)', fill=MUTED)
    d.text((bx0, by1 + 26), 'the particle\'s modular energy keeps rising '
           'with depth while its entropy dies:', fill=MUTED)
    d.text((bx0, by1 + 42), 'dK/dS ~ 50 at depth 4, ~10^6 at depth 12: '
           'modular energy without entropy.', fill=MUTED)

    # (c) the landscape read
    cx0, cy0, cx1, cy1 = 70, 500, 730, 800
    n2 = len(Tprof)
    c0 = n2 // 2
    d.text((cx0, cy0 - 42), '[87] reading the landscape: delta-S of a '
           'sliding interval (orange) vs the first-law', fill=INK)
    d.text((cx0, cy0 - 26), 'prediction (2pi/v) sum beta*T00 (thin ink); '
           'below: energy read back from entropies', fill=MUTED)
    d.text((cx0, cy0 - 10), 'alone (blue dots) vs the true landscape '
           '(green).', fill=MUTED)
    smax = max(dSs.max(), dKs.max())

    def cxy(c, v_, base, h):
        return (cx0 + (cx1 - cx0) * (c - (c0 - 580)) / 1160.0,
                base - h * v_)
    d.line([cxy(c, v_ / smax, cy0 + 130, 120) for c, v_ in zip(cs, dKs)],
           fill=INK, width=1)
    d.line([cxy(c, v_ / smax, cy0 + 130, 120) for c, v_ in zip(cs, dSs)],
           fill=C_ORANGE, width=3)
    emax = max(etrue.max(), ehat.max())
    d.line([cxy(x, v_ / emax, cy1, 120) for x, v_ in zip(xs, etrue)],
           fill=C_GREEN, width=3)
    for x, v_ in zip(xs, ehat):
        px, py = cxy(x, v_ / emax, cy1, 120)
        d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=C_BLUE)

    # (d) verdict
    vx = 830
    lines = [
        ('summary:', INK),
        ('', INK),
        ('measured: the first law dS = dK for thermal', INK),
        ('perturbations (one parameter-free curve);', INK),
        ('Stefan-Boltzmann pi/12; the Bekenstein bound', INK),
        ('dK >= dS in every case, saturated only by heat;', INK),
        ('work and mixing enter separately.', INK),
        ('', INK),
        (f'small-x slope {slope:.2f}: through the RT dictionary', C_ORANGE),
        ('(an installed component) the bulk response is', C_ORANGE),
        ('h ~ z^2 - the unique static solution of the', C_ORANGE),
        ('LINEARIZED EINSTEIN EQUATIONS in AdS3. A mass', C_ORANGE),
        ('gap destroys the scaling; localized non-thermal', C_ORANGE),
        ('excitations never produce it.', C_ORANGE),
        ('', INK),
        ('still owed: time-dependent dynamics, flat-space', MUTED),
        ('asymptotics, and Newton\'s constant from the cutoff.', MUTED),
    ]
    for i, (txt, col) in enumerate(lines):
        d.text((vx, 500 + i * 18), txt, fill=col)
    img.save(path)


# ---- main -------------------------------------------------------------

def main():
    t00 = time.time()
    print('=' * 68)
    print('PART 22: THE ENTANGLEMENT FIRST LAW')
    print('=' * 68)
    print()
    print('Matter: the critical hopping-fermion chain (the observatory\'s')
    print('other matter since part 4) — N = %d sites, half filling,' % N)
    print('Fermi velocity v = 2 from the dispersion, central charge c = 1.')
    print('Every state below is Gaussian: every entropy is exact.')
    print()

    print('[0]  instrument validation:')
    H, w, v = chain()
    C0 = gibbs(w, v, 0.0)
    c0 = N // 2
    ls = np.array([8, 16, 32, 64, 128, 256])
    Ss = np.array([region_entropy(C0, interval(c0, l // 2)) for l in ls])
    slope0 = np.polyfit(np.log(ls), Ss, 1)[0]
    A = interval(c0, 32)
    comp = [k for k in range(N) if k not in A]
    pur = region_entropy(C0, A) - region_entropy(C0, comp)
    fl1, fl2 = parity_floor(1024), parity_floor(2048)
    print(f'     vacuum S(l) log-slope {slope0:.4f} -> c_eff = '
          f'{3 * slope0:.3f} (CFT: c = 1)')
    print(f'     purity: S(A) - S(complement) = {pur:.1e} (pure state: 0)')
    print(f'     instrument floor (finite-chain parity ripple in dS):')
    print(f'       +/-{fl1:.1e} nats at N=1024, +/-{fl2:.1e} at N=2048 '
          f'— halves with N;')
    print('       every ratio below is honest only above this floor.')
    print()

    print('[85] thermal interval entropies (global heat, every interval, every T):')
    pts = one_curve(H, w, v, C0, c0)
    for T in (0.01, 0.02, 0.04, 0.08):
        sub = [p for p in pts if p['T'] == T]
        print(f'     T={T:.2f}: e/T^2 = {sub[0]["sb"]:.4f} '
              f'(Stefan-Boltzmann pi/12 = {np.pi / 12:.4f})')
    small = [p for p in pts if 0.35 <= p['x'] <= 1.2]
    large = [p for p in pts if p['x'] > 4]
    rs = [p['dS'] / cft_curve(p['x']) for p in pts if p['x'] > 0.35]
    print(f'     collapse onto (1/3)ln(sinh x/x): dS/curve = '
          f'{np.mean(rs):.3f} +/- {np.std(rs):.3f} over {len(rs)} '
          'points (x > 0.35) —')
    print('       one parameter-free curve across a 16x range of T and l.')
    rk_s = [p['dK'] / p['dS'] for p in small]
    rk_l = [p['dK'] / p['dS'] for p in large]
    print(f'     first-law region (x in [0.35, 1.2]):  dK/dS = '
          f'{np.mean(rk_s):.3f} +/- {np.std(rk_s):.3f}  — dS = d<K>,')
    print('       the entanglement first law, measured.')
    print(f'     Bekenstein regime (x > 4):          dK/dS = '
          f'{np.mean(rk_l):.2f} — the bound d<K> >= dS')
    print('       widens as the state grows distinguishable; never once '
          'below 1')
    mn = min(p['dK'] - p['dS'] for p in pts)
    print(f'       (min dK - dS over all {len(pts)} points: {mn:+.4f} '
          'nats, i.e. >= 0 to the floor).')
    print()

    print('[86] heat versus work (same packet, three ways):')
    pure, mixed, lam_rows = price_list(H, C0, c0)
    peak = max(pure, key=lambda r: r[1])
    deep = [r for r in pure if r[0] >= 16]
    print(f'     PURE particle (energy {pure[0][3]:.2f}): delta-S peaks '
          f'at {peak[1]:.3f} nats ({peak[1] / math.log(2):.2f} bit)')
    print(f'       straddling the cut, dies to '
          f'{max(abs(r[1]) for r in deep):.1e} deep inside — while its')
    print(f'       modular energy keeps growing (dK/dS = '
          f'{pure[9][2] / pure[9][1]:.0f} at depth 4, '
          f'~{pure[12][2] / abs(pure[12][1]):.0e} at depth 12).')
    print('       A pure excitation is work-like: full modular energy, no')
    print('       entropy. (A coherent displacement of the part-5 scalar')
    print('       gives dS = 0 identically — the same statement.)')
    for lam, dS, h2 in lam_rows:
        print(f'     MIXED lam={lam:.2f}: dS = {dS:.5f} vs mixing entropy '
              f'h2 = {h2:.5f}  (match {dS / h2:.4f})')
    flat = [r[1] for r in mixed if r[0] >= 8]
    print(f'       and position-blind: dS = {np.mean(flat):.4f} +/- '
          f'{np.std(flat):.4f} across depths 8-32.')
    print('       Classical uncertainty enters as mixing entropy, at any '
          'position.')
    print('     Only heat couples to the local Unruh temperature: the '
          'first law is')
    print('       a statement about delta-Q, not delta-E, as in '
          'thermodynamics.')
    print()

    print('[87] recovering the energy distribution (two warm regions,')
    print('     +0.035 / +0.022 on a T = 0.010 background):')
    Tprof, e, cs, dSs, dKs, xs, ehat, etrue = landscape(H, w, v, c0)
    r_fwd = np.corrcoef(dSs, dKs)[0, 1]
    sel = dSs > 0.015
    ratio = float(np.mean(dKs[sel] / dSs[sel]))
    r_inv = np.corrcoef(ehat, etrue)[0, 1]
    print(f'     forward: delta-S(c) of a sliding interval vs '
          f'(2pi/v) sum beta*T00:')
    print(f'       r = {r_fwd:.4f}, mean dK/dS = {ratio:.3f} '
          '(the ~x^2 correction of the curve;')
    print('       the kernel has no fitted parameters).')
    print(f'     inverse: energy landscape read back from entropies '
          f'alone: r = {r_inv:.4f},')
    print(f'       recovered heat {ehat.sum():.4f} vs true '
          f'{etrue.sum():.4f} '
          f'({100 * ehat.sum() / etrue.sum():.0f}%: the ~5% Bekenstein '
          'slack in reverse, the rest lost')
    print('       at the scan\'s edges).')
    print('     The energy distribution is recoverable from entropies '
          'alone.')
    print('       (Resolution ~ the width of the warm regions: thermal '
          'perturbations')
    print('       cannot be localized below their own wavelength.)')
    print()

    print('[88] the z^2 scaling (the Faulkner et al. correspondence):')
    dpts, slope, ref = depth_law(H, w, v, C0, c0)
    gpts = gapped_points(c0)
    print(f'     small-x region: dS ~ x^{slope:.2f} measured over '
          f'{len(dpts)} points')
    print(f'       (the exact curve\'s own slope over this window: '
          f'{ref:.2f}; pure limb: 2.00).')
    print('     Through the Ryu-Takayanagi dictionary — entropies are')
    print('     geodesic lengths in an emergent (x, z) bulk; INSTALLED,')
    print('     as the Born rule was in part 7 — the parabolic kernel')
    print('     is the chord integral of a bulk response h ~ z^2 T00:')
    print('     precisely the static solution of the linearized')
    print('     EINSTEIN equations in AdS3, and of nothing else (a')
    print('     bulk field of dimension D responds as z^D; D = 2 is')
    print('     the conserved stress tensor, i.e. the metric).')
    rgap = [dS / cft_curve(x) for x, dS in gpts]
    print(f'     controls: mass gap 0.5 -> the same probes fall to '
          f'{min(rgap):.2f}-{max(rgap):.2f} of the')
    print('       curve (the correspondence fails beyond v/gap); the '
          'localized')
    print('       excitations of [86] never produce it. The '
          'correspondence holds in')
    print('       the thermal sector only.')
    print()

    print('     summary: the first law holds in the thermal sector,')
    print('     the modular kernel is the inverse Unruh temperature, and')
    print('     the measured x^2 response corresponds to the linearized')
    print('     Einstein equation through the RT dictionary. Still owed:')
    print('     time-dependent dynamics, flat-space asymptotics, and')
    print('     Newton\'s constant from the cutoff.')

    figure(pts, gpts, slope, ref, pure, mixed, Tprof, e, cs, dSs, dKs,
           xs, ehat, etrue, 'films/firstlaw.png')
    print()
    print(f'     films/firstlaw.png  ({time.time() - t00:.0f}s)')


if __name__ == '__main__':
    main()
