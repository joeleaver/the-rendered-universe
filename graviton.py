"""Part 26: the first law with a propagating metric.

Part 25 ran the time-dependent first law in 1+1 dimensions and noted
why the result was almost too clean: through the Ryu-Takayanagi
dictionary the boundary lives at the edge of a three-dimensional bulk,
and three-dimensional gravity has no propagating degrees of freedom —
there was no graviton to miss. This part moves the whole program up
one dimension. The matter is a two-dimensional critical fermion
lattice (a pi-flux square lattice: two Dirac cones, velocity 2, the
2+1-dimensional free CFT), whose dual bulk is four-dimensional —
where the metric does propagate.

The handoff hypothesis was that slice-by-slice first-law tracking
should now fail. Measured: it does not. The first law is a constraint
and holds on every slice here exactly as it did in 1+1d. What fails —
and this is the finding — is closure: in 1+1d the energy and momentum
densities on one slice determine the whole future (tracelessness
forces T_xx = T00), while in 2+1d they do not. The missing slice
datum is the anisotropic stress, the spin-2 quantity that the modular
kernel (which reads only T00) is blind to. Two states are built with
the same energy map, identically zero momentum, and oppositely
oriented shear: every disk's entropy agrees at t = 0, and the futures
diverge by an order one amount. Through the dictionary, that
slice-invisible, propagating, spin-2 datum is the boundary shadow of
the bulk graviton.

  [100] the disk first law: the Casini-Huerta-Myers modular weight is
        the same parabola beta(r) = (R^2 - r^2)/2R in any dimension.
        On disks it predicts delta-S with no fitted parameters:
        forward landscape correlation r = 0.9999; global heat gives
        dK/dS near 1 across the accessible window, with the residual
        set by the staircase discretization of a lattice circle. The
        energy density matches the exact band integral and scales as
        T^3.1 (2+1d Stefan-Boltzmann: 3).
  [101] the quench: a released warm bump spreads as a ring at the
        measured speed 2.0 — the one-particle band velocity, not the
        sound speed v/sqrt(2) = 1.41 that any closed, isotropic
        (e, p)-hydrodynamics would give. The first law nevertheless
        holds on every slice through the run: constraints are
        satisfied instant by instant; prediction is what needs more
        data than the slice holds.
  [102] the twins: state B is state A turned by ninety degrees — an
        exact lattice symmetry, found as a gauge-sign pattern. The
        beams' transverse envelope is shaped (a self-calibration: its
        own longitudinal profile is the target) until A's energy map
        is round to about a percent, so A and B agree on the initial
        slice in energy (1.2% of peak), momentum (exactly zero), and
        every disk entropy — but carry opposite shear. The futures
        diverge to order one while a control state (same shear
        orientation, 40% initial energy mismatch) converges. The
        slice-blind datum that decides the future is the shear, and
        it propagates.

What this does not deliver: a bulk wave equation for h_ij derived
from the entanglement data, the flat-space version, or Newton's
constant. Those stay on the owed list; section 6 of the paper says so.
"""
import math
import time
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

# validated dark-mode categorical palette (dataviz slots 1-3)
C_BLUE, C_ORANGE, C_GREEN = (57, 135, 229), (217, 89, 38), (25, 158, 112)
INK, MUTED, GRIDC = (195, 194, 183), (122, 122, 130), (38, 38, 44)
BG = (14, 14, 18)

L = 80             # lattice side
N = L * L
VF = 2.0           # cone velocity of the pi-flux lattice at half filling
CX = CY = (L - 1) / 2

# ---- the matter --------------------------------------------------------


def hamiltonian():
    """Pi-flux square lattice: hopping -1 along x, alternating sign
    along y, so every plaquette carries flux pi. At half filling the
    spectrum is two Dirac cones with velocity 2."""
    H = np.zeros((N, N))
    for y in range(L):
        for x in range(L):
            i = y * L + x
            if x + 1 < L:
                H[i, i + 1] = H[i + 1, i] = -1.0
            if y + 1 < L:
                s = -1.0 if x % 2 == 0 else 1.0
                H[i, i + L] = H[i + L, i] = s
    return H


def rotation_op(H):
    """The 90-degree rotation about the lattice center, as a site
    permutation plus a gauge-sign pattern solved bond by bond so that
    R H R^T = H exactly (flux pi is rotation invariant, so the signs
    always exist on the simply connected open lattice)."""
    perm = np.zeros(N, dtype=int)
    for y in range(L):
        for x in range(L):
            perm[x * L + (L - 1 - y)] = y * L + x
    Hp = H[np.ix_(perm, perm)]
    g = np.zeros(N)
    g[0] = 1.0
    nbrs = [[] for _ in range(N)]
    ii, jj = np.nonzero(H)
    for a, b in zip(ii, jj):
        if a < b:
            nbrs[a].append(b)
            nbrs[b].append(a)
    dq = deque([0])
    while dq:
        a = dq.popleft()
        for b in nbrs[a]:
            if g[b] == 0:
                g[b] = g[a] * (H[a, b] / Hp[a, b])
                dq.append(b)
    err = np.abs((g[:, None] * g[None, :]) * Hp - H).max()
    return perm, g, err


H = hamiltonian()
PERM, GSIGN, ROT_ERR = rotation_op(H)

xg, yg = np.meshgrid(np.arange(L), np.arange(L))
XF, YF = xg.ravel().astype(float), yg.ravel().astype(float)
R2C = (XF - CX) ** 2 + (YF - CY) ** 2

# bond arrays for local energy / shear / current
_xb, _yb = [], []
for y in range(L):
    for x in range(L):
        i = y * L + x
        if x + 1 < L:
            _xb.append((i, i + 1, -1.0))
        if y + 1 < L:
            _yb.append((i, i + L, -1.0 if x % 2 == 0 else 1.0))
XB_I, XB_J, XB_H = (np.array(a) for a in zip(*_xb))
YB_I, YB_J, YB_H = (np.array(a) for a in zip(*_yb))


def site_fields(C):
    """Per-site energy density split into x-bond and y-bond parts
    (each bond's energy shared between its endpoints), plus the
    summed bond-current maps (the momentum density; zero iff C real)."""
    exb = 2 * XB_H * np.real(C[XB_I, XB_J])
    eyb = 2 * YB_H * np.real(C[YB_I, YB_J])
    jxb = 2 * XB_H * np.imag(C[XB_I, XB_J])
    jyb = 2 * YB_H * np.imag(C[YB_I, YB_J])
    ex = np.zeros(N)
    ey = np.zeros(N)
    np.add.at(ex, XB_I, 0.5 * exb)
    np.add.at(ex, XB_J, 0.5 * exb)
    np.add.at(ey, YB_I, 0.5 * eyb)
    np.add.at(ey, YB_J, 0.5 * eyb)
    jmax = max(np.abs(jxb).max(), np.abs(jyb).max())
    return ex + ey, ex, ey, jmax


def disk(cx, cy, R):
    return np.where((XF - cx) ** 2 + (YF - cy) ** 2 < R * R)[0]


def region_entropy(Csub):
    nu = np.clip(np.linalg.eigvalsh(Csub), 1e-14, 1 - 1e-14)
    return float(-(nu * np.log(nu) + (1 - nu) * np.log(1 - nu)).sum())


def mod_energy(e, cx, cy, R):
    """delta<K> = (2pi/v) sum beta(r) delta-e over the disk, with the
    CHM parabola beta(r) = (R^2 - r^2)/2R — the same modular weight
    as the 1D interval of parts 22/25, valid in any dimension."""
    s = disk(cx, cy, R)
    r2 = (XF[s] - cx) ** 2 + (YF[s] - cy) ** 2
    beta = np.clip(R * R - r2, 0, None) / (2 * R)
    return (2 * np.pi / VF) * float(beta @ e[s])


def thermal_e_exact(T):
    """Bulk thermal energy per site from the band integral (two bands
    over the reduced zone = one dispersion sheet over half the full
    zone; the mean over the full zone counts each state twice)."""
    nk = 400
    k = (np.arange(nk) + 0.5) * np.pi / nk
    KX, KY = np.meshgrid(k, k)
    E = 2 * np.sqrt(np.cos(KX) ** 2 + np.cos(KY) ** 2)
    return float(np.mean(E / (1.0 + np.exp(E / T))))


# ---- the twins' machinery ---------------------------------------------

K1 = (np.pi / 2, np.pi / 2)   # one of the two Dirac cones
Q0 = 0.55                     # beam momentum along +x from the cone
SXP = 7.0                     # longitudinal envelope width


def beam_mode(envy_1d, C0):
    """+x-moving beam packet: Gaussian along x, prescribed transverse
    envelope, carrier at K1 + Q0 x, projected into the upper band.
    Its conjugate is the -x partner with the identical energy map."""
    envy = envy_1d[np.round(YF).astype(int)]
    env = np.exp(-(XF - CX) ** 2 / (4 * SXP * SXP)) * envy
    phi = env * np.exp(1j * ((K1[0] + Q0) * XF + K1[1] * YF))
    phi = phi - C0 @ phi
    return phi / np.linalg.norm(phi)


def pair_emap(envy_1d, C0):
    p = beam_mode(envy_1d, C0)
    return 2 * np.real(np.conj(p) * (H @ p))


def shape_transverse(C0, iters=8):
    """Fixed-point shaping of the transverse envelope until the beam
    pair's energy map is round. Self-calibrating: the target profile
    is the pair's own longitudinal profile. Starts from the measured
    factor-two rule (a beam's energy variance transverse to its
    motion is twice its envelope variance, independent of Q0 and
    sigma), i.e. sigma_y = sigma_x / sqrt(2)."""
    yy = np.arange(L) - CY
    envy = np.exp(-yy ** 2 / (4 * (SXP / np.sqrt(2)) ** 2))
    for _ in range(iters):
        m = pair_emap(envy, C0).reshape(L, L)
        prof, profx = m.sum(axis=1), m.sum(axis=0)
        corr = np.sqrt(np.clip(profx / profx.sum() * prof.sum(),
                               1e-12, None) / np.clip(prof, 1e-12, None))
        corr = np.clip(corr, 0.7, 1.4)
        corr = 0.5 * (corr + corr[::-1])
        corr = np.convolve(corr, np.ones(3) / 3, mode='same')
        envy = envy * corr / (envy * corr).max()
    return envy, np.exp(-yy ** 2 / (4 * (SXP / np.sqrt(2)) ** 2))


def rot_odd_fraction(e, k=4, rmax=24.0):
    """Rotation-odd share of the coarse-grained map inside r < rmax."""
    m = e.reshape(L, L).reshape(L // k, k, L // k, k).sum(axis=(1, 3))
    wc = (R2C < rmax * rmax).astype(float).reshape(L, L).reshape(
        L // k, k, L // k, k).sum(axis=(1, 3)) > k * k / 2
    return float(np.abs(m - np.rot90(m))[wc].max() / m[wc].max())


LAM = 0.5   # beam-mode occupation: classically mixed, part-22 style


def twin_fields(modes, e_vac, ex_vac, ey_vac):
    """Energy and shear maps of C0 + LAM sum |phi><phi| without
    building the full matrix; the current audit uses the summed C
    restricted to bonds (conjugate pairs cancel exactly)."""
    dC_x = np.zeros(len(XB_I), dtype=complex)
    dC_y = np.zeros(len(YB_I), dtype=complex)
    for p in modes:
        dC_x += LAM * p[XB_I] * np.conj(p[XB_J])
        dC_y += LAM * p[YB_I] * np.conj(p[YB_J])
    exb = 2 * XB_H * np.real(dC_x)
    eyb = 2 * YB_H * np.real(dC_y)
    jmax = max(np.abs(2 * XB_H * np.imag(dC_x)).max(),
               np.abs(2 * YB_H * np.imag(dC_y)).max())
    ex = np.zeros(N)
    ey = np.zeros(N)
    np.add.at(ex, XB_I, 0.5 * exb)
    np.add.at(ex, XB_J, 0.5 * exb)
    np.add.at(ey, YB_I, 0.5 * eyb)
    np.add.at(ey, YB_J, 0.5 * eyb)
    return ex + ey, ex - ey, jmax


def twin_dS_map(modes, C0, grid, disk_sites, S0_d):
    out = []
    for gpt in grid:
        s = disk_sites[gpt]
        Csub = C0[np.ix_(s, s)].astype(complex)
        for p in modes:
            ps = p[s]
            Csub += LAM * np.outer(ps, ps.conj())
        out.append(region_entropy(Csub) - S0_d[gpt])
    return np.array(out)


# ---- main --------------------------------------------------------------


def main():
    t00 = time.time()
    print('=' * 68)
    print('PART 26: THE FIRST LAW WITH A PROPAGATING METRIC')
    print('=' * 68)
    print()
    print('Matter: a pi-flux square lattice of free fermions, %d x %d,'
          % (L, L))
    print('half filling — two Dirac cones, velocity v = 2: the free CFT')
    print('in 2+1 dimensions. Its dual bulk is four-dimensional, where')
    print('the metric propagates. Every state below is Gaussian; every')
    print('entropy is exact.')
    print()

    w, V = np.linalg.eigh(H)
    occ0 = V[:, w < 0]
    C0 = occ0 @ occ0.T
    e_vac, ex_vac, ey_vac, _ = site_fields(C0)

    print('[0]  instrument validation:')
    print(f'     spectrum: min|E| = {np.abs(w).min():.3f} (no zero '
          f'modes), particle-hole symmetry to '
          f'{np.abs(w + w[::-1]).max():.1e}')
    Rs = np.array([3, 4, 5, 6, 8, 10, 12])
    Ss = np.array([region_entropy(C0[np.ix_(s, s)])
                   for s in (disk(CX, CY, R) for R in Rs)])
    a1, b1 = np.polyfit(2 * np.pi * Rs, Ss, 1)
    resid = float(np.sqrt(np.mean((Ss - (a1 * 2 * np.pi * Rs + b1)) ** 2)))
    print(f'     vacuum disk entropies: S = {a1:.3f} * perimeter '
          f'{b1:+.2f}, rms residual {resid:.2f} nats — the area law')
    A8 = disk(CX, CY, 8)
    comp = np.setdiff1d(np.arange(N), A8)
    pur = region_entropy(C0[np.ix_(A8, A8)]) \
        - region_entropy(C0[np.ix_(comp, comp)])
    print(f'     purity: S(disk) - S(complement) = {pur:.1e}')
    print(f'     90-degree rotation operator (site permutation + '
          f'gauge signs): |R H R^T - H| = {ROT_ERR:.1e}')
    T_val = 0.2
    occT = 1.0 / (1.0 + np.exp(w / T_val))
    CT = (V * occT) @ V.T
    eT = site_fields(CT)[0] - e_vac
    sel = R2C < 100
    print(f'     thermal energy density at T = {T_val}: measured/'
          f'band-integral = '
          f'{float(eT[sel].mean()) / thermal_e_exact(T_val):.3f}')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    # ---- [100] the disk first law -------------------------------------
    print('[100] the disk first law (CHM parabola, no fitted '
          'parameters):')
    Ts = np.array([0.08, 0.11, 0.16, 0.22, 0.30])
    es = []
    for T in Ts:
        occT = 1.0 / (1.0 + np.exp(w / T))
        es.append(float((site_fields((V * occT) @ V.T)[0]
                         - e_vac)[sel].mean()))
    pT = np.polyfit(np.log(Ts), np.log(es), 1)[0]
    print(f'     energy density scales as T^{pT:.2f} '
          '(2+1d Stefan-Boltzmann: 3)')
    S0_R = {R: region_entropy(C0[np.ix_(s, s)])
            for R, s in ((R, disk(CX, CY, R)) for R in (4, 6, 8, 10, 12))}
    heat_pts = []
    for T in (0.03, 0.05, 0.08, 0.12, 0.20):
        occT = 1.0 / (1.0 + np.exp(w / T))
        CT = (V * occT) @ V.T
        eT = site_fields(CT)[0] - e_vac
        for R in (4, 6, 8, 10, 12):
            s = disk(CX, CY, R)
            dS = region_entropy(CT[np.ix_(s, s)]) - S0_R[R]
            dK = mod_energy(eT, CX, CY, R)
            heat_pts.append((2 * np.pi * R * T / VF, T, R, dS, dK))
    window = [(x, dK / dS) for (x, T, R, dS, dK) in heat_pts
              if 0.6 <= x <= 1.6 and dS > 3e-3]
    print(f'     global heat, window x = 2piRT/v in [0.6, 1.6]: '
          f'dK/dS = {np.mean([r for _, r in window]):.3f} +/- '
          f'{np.std([r for _, r in window]):.3f} over {len(window)} '
          '(R, T) pairs')
    print('       below the window the finite lattice\'s discrete '
          'spectrum shows')
    print('       through; above it the thermal x^2 correction of '
          'part 22 grows. The')
    print('       residual is the staircase: a lattice disk\'s cut is '
          'jagged at the')
    print('       lattice scale, and the continuum kernel misweights '
          'it by ~1/R.')
    print(f'     [{time.time() - t00:.0f}s]')

    # landscape: warm bump, sliding disks
    T0B, THOT, SIGH = 0.02, 0.06, 9.0
    Tprof = T0B + THOT * np.exp(-R2C / (2 * SIGH ** 2))
    Dm = np.diag(1.0 / np.clip(Tprof, 1e-4, None))
    mw, mv = np.linalg.eigh(0.5 * (Dm @ H + H @ Dm))
    Cq = (mv * (1.0 / (1.0 + np.exp(np.clip(mw, -600, 600))))) @ mv.T
    occB = 1.0 / (1.0 + np.exp(w / T0B))
    Cref = (V * occB) @ V.T
    e_ref = site_fields(Cref)[0]
    eq0 = site_fields(Cq)[0] - e_ref
    R_MAIN = 6
    cs = np.arange(8, 72, 4).astype(float)
    Sref_d = {c: region_entropy(Cref[np.ix_(s, s)])
              for c, s in ((c, disk(c, CY, R_MAIN)) for c in cs)}
    dSs, dKs = [], []
    for c in cs:
        s = disk(c, CY, R_MAIN)
        dSs.append(region_entropy(Cq[np.ix_(s, s)]) - Sref_d[c])
        dKs.append(mod_energy(eq0, c, CY, R_MAIN))
    dSs, dKs = np.array(dSs), np.array(dKs)
    r_fwd = float(np.corrcoef(dSs, dKs)[0, 1])
    strong = dSs > 0.02
    floor = float(np.abs(dSs[np.abs(cs - CX) > 26]).max())
    print(f'     landscape (warm bump +{THOT} on T = {T0B}, sliding '
          f'disks R = {R_MAIN}):')
    print(f'       delta-S vs kernel prediction: r = {r_fwd:.4f}, '
          f'mean dK/dS = '
          f'{float(np.mean(dKs[strong] / dSs[strong])):.3f}')
    print(f'       instrument floor (far disks): |dS| <= {floor:.4f} '
          'nats — the local-Gibbs')
    print('       tail artifact of part 22, present here too; slices '
          'below ~2x this are')
    print('       excluded everywhere in this part.')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    # ---- [101] the quench ---------------------------------------------
    print('[101] the quench (the warm bump released):')
    DT = 1.5
    NT = 9
    Ud = (V * np.exp(-1j * w * DT)) @ V.T
    Ct = Cq.astype(complex)
    ts, rhalfs, e_rows, dS_rows, dK_rows, etots = [], [], [], [], [], []
    rr = np.sqrt(R2C)
    for it in range(NT):
        t = it * DT
        ts.append(t)
        e_now = np.real(site_fields(Ct)[0]) - e_ref
        etots.append(float(e_now.sum()))
        de = np.clip(e_now, 0, None)
        prof = np.array([float(de[(rr >= r) & (rr < r + 1)].mean())
                         for r in range(36)])
        shell = prof * (2 * np.arange(36) + 1)
        csum = np.cumsum(shell)
        rhalfs.append(float(np.interp(0.5 * csum[-1], csum,
                                      np.arange(36) + 0.5)))
        e_rows.append(list(prof))
        dS_row, dK_row = [], []
        for c in cs:
            s = disk(c, CY, R_MAIN)
            dS_row.append(region_entropy(np.real(Ct)[np.ix_(s, s)])
                          - Sref_d[c])
            dK_row.append(mod_energy(e_now, c, CY, R_MAIN))
        dS_rows.append(dS_row)
        dK_rows.append(dK_row)
        Ct = Ud @ Ct @ Ud.conj().T
    ts = np.array(ts)
    rhalfs = np.array(rhalfs)
    t_end = ts[-1]
    v_end = float(np.sqrt(rhalfs[-1] ** 2 - rhalfs[0] ** 2) / t_end)
    i45, i75 = int(4.5 / DT), int(7.5 / DT)
    v_mid = float(np.sqrt((rhalfs[i75] ** 2 - rhalfs[i45] ** 2)
                          / (7.5 ** 2 - 4.5 ** 2)))
    cons = (max(etots) - min(etots)) / etots[0]
    print(f'     energy conserved to {100 * cons:.2f}% through the run')
    print(f'     ring growth: the half-energy radius follows '
          f'r(t)^2 = r(0)^2 + (vt)^2 with')
    print(f'       v = {v_end:.2f} over the whole run '
          f'({v_mid:.2f} through its clean middle). The band')
    print(f'       velocity is 2.0; the sound speed of any closed '
          f'isotropic (e, p)')
    print(f'       hydrodynamics is v/sqrt(2) = '
          f'{VF / np.sqrt(2):.2f}, which would leave the ring')
    pred_s = math.sqrt(rhalfs[0] ** 2 + (VF / np.sqrt(2) * t_end) ** 2)
    print(f'       at r = {pred_s:.1f} instead of the measured '
          f'{rhalfs[-1]:.1f} by the end. The energy')
    print('       free-streams: the slice\'s (e, p) data plus '
          'isotropy predicts the')
    print('       wrong cone. (The shell is thick and the box small; '
          'the mid-run slope')
    print('       avoids the early transient and the reflective '
          'last ticks.)')
    FLOOR = 8e-3
    ratios = []
    print('        t     mean dK/dS   disks above the floor')
    for i, t in enumerate(ts):
        dS_row = np.array(dS_rows[i])
        dK_row = np.array(dK_rows[i])
        st = dS_row > FLOOR
        if st.sum() < 3:
            continue
        r = float(np.mean(dK_row[st] / dS_row[st]))
        ratios.append(r)
        if i % 2 == 0:
            print(f'      {t:5.1f}     {r:.3f}        {st.sum()}')
    print(f'     across all usable slices: dK/dS = '
          f'{np.mean(ratios):.3f} +/- {np.std(ratios):.3f}')
    print('     The handoff expected slice-by-slice tracking to fail '
          'in 2+1d. It does')
    print('     not: the first law is a constraint, satisfied on '
          'every slice at the')
    print('     static accuracy. What the slice cannot do is predict '
          'the next slice.')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    # ---- [102] the twins ----------------------------------------------
    print('[102] the twins (same slice, opposite shear):')
    envy, envy_ctl = shape_transverse(C0)
    ep_shaped = pair_emap(envy, C0)
    ro4 = rot_odd_fraction(ep_shaped)
    print(f'     beam shaping: a beam pair\'s energy map carries '
          f'twice its envelope')
    print(f'       variance transverse to the motion (measured '
          f'factor 2.0, independent')
    print(f'       of momentum and width). Starting from sigma_y = '
          f'sigma_x/sqrt(2), the')
    print(f'       transverse envelope is reshaped against the '
          f'pair\'s own longitudinal')
    print(f'       profile: rotation-odd residue '
          f'{ro4 * 100:.1f}% of peak after 8 iterations.')
    phi_A = beam_mode(envy, C0)
    modes_A = [phi_A, np.conj(phi_A)]
    modes_B = [GSIGN * p[PERM] for p in modes_A]
    phi_C = beam_mode(envy_ctl, C0)
    modes_C = [phi_C, np.conj(phi_C)]
    eA, shA, jA = twin_fields(modes_A, e_vac, ex_vac, ey_vac)
    eB, shB, jB = twin_fields(modes_B, e_vac, ex_vac, ey_vac)
    eC, shC, jC = twin_fields(modes_C, e_vac, ex_vac, ey_vac)
    print(f'     twin A: two counter-moving beams along x (a '
          f'conjugate pair: its')
    print(f'       correlation matrix is real, so momentum density '
          f'is exactly zero —')
    print(f'       measured max bond current {jA:.1e}). Twin B = R A: '
          f'the same state')
    print('       turned ninety degrees by the exact symmetry.')
    print(f'     the initial slice:')
    print(f'       energy:  max|eA - eB| = '
          f'{np.abs(eA - eB).max() / eA.max() * 100:.1f}% of peak')
    print(f'       shear:   max|T_xx - T_yy| = '
          f'{np.abs(shA).max() / eA.max():.2f} of the energy peak, '
          'and it flips')
    print(f'                sign between the twins: '
          f'max|shA - shB| = '
          f'{np.abs(shA - shB).max() / eA.max():.2f} of peak')
    print(f'       control: a third beam state with the unshaped '
          f'envelope — same shear')
    print(f'                orientation as A, energy mismatch '
          f'{np.abs(eA - eC).max() / eA.max() * 100:.0f}% of peak')

    grid = [(gx, gy) for gx in np.arange(15.5, 64.5, 4)
            for gy in np.arange(15.5, 64.5, 4)]
    disk_sites = {gpt: disk(gpt[0], gpt[1], R_MAIN) for gpt in grid}
    S0_d = {gpt: region_entropy(C0[np.ix_(s, s)])
            for gpt, s in disk_sites.items()}
    mA = [p.copy() for p in modes_A]
    mB = [p.copy() for p in modes_B]
    mC = [p.copy() for p in modes_C]
    div_rows = []
    gif_frames = []
    print('        t    |eA-eB|/pk  |eA-eC|/pk   max|dSA-dSB|  '
          'max|dSA-dSC|  max dS')
    NT2 = 9
    for it in range(NT2):
        t = it * DT
        if it:
            mA = [Ud @ p for p in mA]
            mB = [Ud @ p for p in mB]
            mC = [Ud @ p for p in mC]
        eAt = twin_fields(mA, e_vac, ex_vac, ey_vac)[0]
        eBt = twin_fields(mB, e_vac, ex_vac, ey_vac)[0]
        eCt = twin_fields(mC, e_vac, ex_vac, ey_vac)[0]
        dSA = twin_dS_map(mA, C0, grid, disk_sites, S0_d)
        dSB = twin_dS_map(mB, C0, grid, disk_sites, S0_d)
        dSC = twin_dS_map(mC, C0, grid, disk_sites, S0_d)
        row = (t,
               float(np.abs(eAt - eBt).max() / eAt.max()),
               float(np.abs(eAt - eCt).max() / eAt.max()),
               float(np.abs(dSA - dSB).max()),
               float(np.abs(dSA - dSC).max()),
               float(dSA.max()))
        div_rows.append(row)
        gif_frames.append((eAt.copy(), eBt.copy()))
        if it % 2 == 0:
            print(f'      {t:5.1f}   {row[1] * 100:8.1f}%  '
                  f'{row[2] * 100:8.1f}%      {row[3]:7.4f}      '
                  f'{row[4]:7.4f}  {row[5]:7.4f}')
    d0, dend = div_rows[0], div_rows[-1]
    print(f'     the twins\' energy maps agree to '
          f'{d0[1] * 100:.1f}% at t = 0 and differ by '
          f'{dend[1] * 100:.0f}%')
    print(f'       at t = {dend[0]:.0f}; the control starts '
          f'{d0[2] / d0[1]:.0f}x further away and CONVERGES '
          f'({d0[2] * 100:.0f}% ->')
    print(f'       {dend[2] * 100:.0f}%). Every disk entropy agrees '
          f'at t = 0 (max gap {d0[3]:.4f} nats')
    print(f'       vs signal {d0[5]:.2f}) — the first law reads only '
          f'T00, and the T00 maps')
    print(f'       match; by t = {dend[0]:.0f} the gap is '
          f'{dend[3]:.2f} nats, as large as the signal.')
    print('     In 1+1d this experiment cannot be built: '
          'tracelessness forces')
    print('     T_xx = T00, so matched (e, p) slices have matched '
          'futures — which is')
    print('     why part 25\'s slice-tracking worked. In 2+1d the '
          'shear is free slice')
    print('     data, invisible to every modular kernel, and it '
          'propagates. Through')
    print('     the installed RT dictionary that is the boundary '
          'shadow of the bulk')
    print('     graviton: the metric\'s radiative sector lives '
          'exactly in what the')
    print('     slice\'s energy cannot see.')
    print(f'     [{time.time() - t00:.0f}s]')
    print()
    print('     still owed: a bulk wave equation for h_ij from the '
          'entanglement data,')
    print('     the flat-space first law, and Newton\'s constant '
          'from the cutoff.')

    figure(heat_pts, ts, e_rows, dS_rows, dK_rows, v_end,
           np.mean(ratios), eA, eB, gif_frames, div_rows,
           'films/graviton.png')
    twins_gif(gif_frames, 'films/graviton_twins.gif')
    print()
    print(f'     films/graviton.png, films/graviton_twins.gif  '
          f'({time.time() - t00:.0f}s)')


# ---- figures -----------------------------------------------------------


def _cg2(mat):
    """2x2 coarse-grain for display: removes lattice-scale striping."""
    return mat.reshape(L // 2, 2, L // 2, 2).mean(axis=(1, 3))


def _heatmap(img, mat, x0, y0, wid, hgt, signed=False):
    m = np.array(mat, dtype=float)
    sc = np.percentile(np.abs(m), 99.5) + 1e-15
    z = np.clip(m / sc, -1, 1)
    pos = np.clip(z, 0, 1) ** 0.6
    neg = np.clip(-z, 0, 1) ** 0.6 if signed else np.zeros_like(z)
    rgb = np.zeros(m.shape + (3,), np.uint8)
    rgb[..., 0] = (25 + 210 * pos).astype(np.uint8)
    rgb[..., 1] = (25 + 90 * pos + 90 * neg).astype(np.uint8)
    rgb[..., 2] = (35 + 210 * neg).astype(np.uint8)
    img.paste(Image.fromarray(rgb).resize((wid, hgt), Image.BILINEAR),
              (x0, y0))


def figure(heat_pts, ts, e_rows, dS_rows, dK_rows, v_ring, r_mean,
           eA, eB, gif_frames, div_rows, path):
    W, Ht = 1560, 840
    img = Image.new('RGB', (W, Ht), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 12), 'PART 26 - THE FIRST LAW WITH A PROPAGATING '
           'METRIC', fill=INK)

    # (a) static: dK/dS vs x, log x
    ax0, ay0, ax1, ay1 = 60, 90, 480, 380
    d.text((ax0, ay0 - 34), '[100] global heat on disks: dK/dS vs '
           'x = 2piRT/v.', fill=INK)
    d.text((ax0, ay0 - 18), 'kernel: the CHM parabola, no parameters. '
           'grey band: 10%.', fill=MUTED)
    xlo, xhi = 0.3, 8.0
    ylo, yhi = 0.4, 2.0

    def axy(x, yv):
        px = ax0 + (ax1 - ax0) * (math.log(x / xlo) / math.log(xhi / xlo))
        py = ay1 - (ay1 - ay0) * (yv - ylo) / (yhi - ylo)
        return px, py
    for yv in (0.5, 1.0, 1.5, 2.0):
        p0, p1 = axy(xlo, yv), axy(xhi, yv)
        d.line([p0, p1], fill=GRIDC)
        d.text((ax0 - 34, p0[1] - 6), f'{yv:.1f}', fill=MUTED)
    for xv in (0.5, 1, 2, 4, 8):
        p0 = axy(xv, ylo)
        d.line([p0, axy(xv, yhi)], fill=GRIDC)
        d.text((p0[0] - 6, ay1 + 6), f'{xv:g}', fill=MUTED)
    band = [axy(x, 0.9) for x in (xlo, xhi)] \
        + [axy(x, 1.1) for x in (xhi, xlo)]
    d.polygon(band, fill=(24, 26, 30))
    d.line([axy(xlo, 1.0), axy(xhi, 1.0)], fill=INK)
    tcol = {0.03: MUTED, 0.05: C_GREEN, 0.08: C_BLUE, 0.12: C_ORANGE,
            0.2: (200, 170, 60)}
    for (x, T, R, dS, dK) in heat_pts:
        if dS <= 3e-3:
            continue
        px, py = axy(min(max(x, xlo), xhi),
                     min(max(dK / dS, ylo), yhi))
        d.ellipse([px - 4, py - 4, px + 4, py + 4],
                  outline=tcol[T], width=2)
    d.text((ax0, ay1 + 24), 'rings by T (0.03 grey .. 0.20 gold). '
           'left of the window the discrete', fill=MUTED)
    d.text((ax0, ay1 + 40), 'spectrum shows through; rightward the '
           'x^2 thermal correction grows.', fill=MUTED)

    # (b) quench radial cone
    bx0, by0, bw, bh = 560, 90, 300, 290
    d.text((bx0, by0 - 34), '[101] the quench: radial energy '
           '(r across, t down).', fill=INK)
    d.text((bx0, by0 - 18), f'half-energy radius grows at {v_ring:.2f} '
           f'(band velocity 2.0, sound 1.41);', fill=MUTED)
    _heatmap(img, e_rows, bx0, by0, bw, bh)
    px_per_r = bw / 36.0
    py_per_t = bh / float(ts[-1])
    for v, col in ((2.0, (120, 220, 160)), (2.0 / np.sqrt(2),
                                            (150, 150, 158))):
        tmax = min(ts[-1], 35.0 / v)
        d.line([(bx0, by0), (bx0 + v * tmax * px_per_r,
                             by0 + tmax * py_per_t)], fill=col, width=1)
    d.text((bx0, by0 + bh + 6), 'green: v = 2; grey: v/sqrt2, the '
           'closed-hydro cone the energy ignores.', fill=MUTED)

    # (c) quench ratio map
    cx0 = 940
    d.text((cx0, by0 - 34), '[101] dK/dS - 1 on sliding disks '
           '(x across, t down),', fill=INK)
    d.text((cx0, by0 - 18), f'where delta-S clears the floor; mean '
           f'dK/dS = {r_mean:.3f}.', fill=MUTED)
    ratio = np.array(dK_rows) / np.clip(np.array(dS_rows), 1e-3, None) - 1
    ratio[np.array(dS_rows) < 8e-3] = 0
    ratio = np.clip(ratio / 0.3, -1, 1) * 0.3
    _heatmap(img, ratio, cx0, by0, 300, 290, signed=True)
    d.text((cx0, by0 + bh + 6), 'fixed +/-30% scale (orange high, '
           'blue low): the constraint holds', fill=MUTED)
    d.text((cx0, by0 + bh + 22), 'on every slice while the ring '
           'free-streams away.', fill=MUTED)

    # (d) twins maps
    ty0 = 470
    d.text((60, ty0 - 6), '[102] the twins: B is A turned ninety '
           'degrees. same energy slice at t = 0 (left pair), futures '
           'apart by t = 12 (right pair).', fill=INK)
    sz = 210
    labels = ('A, t=0', 'B, t=0', 'A, t=12', 'B, t=12')
    eA12, eB12 = gif_frames[-1]
    mats = (eA, eB, eA12, eB12)
    for k, (lab, mat) in enumerate(zip(labels, mats)):
        x0 = 60 + k * (sz + 24)
        _heatmap(img, _cg2(mat.reshape(L, L)), x0, ty0 + 16, sz, sz)
        d.text((x0, ty0 + 16 + sz + 4), lab, fill=MUTED)

    # (e) divergence curves
    ex0, ey0_, ex1, ey1 = 1040, ty0 + 16, 1500, ty0 + 16 + sz
    d.text((ex0, ey0_ - 22), 'divergence of the energy maps '
           '(% of peak):', fill=INK)
    for yv in (0, 50, 100):
        py = ey1 - (ey1 - ey0_) * yv / 110.0
        d.line([(ex0, py), (ex1, py)], fill=GRIDC)
        d.text((ex0 - 34, py - 6), f'{yv}', fill=MUTED)

    def exy(t, val):
        return (ex0 + (ex1 - ex0) * t / div_rows[-1][0],
                ey1 - (ey1 - ey0_) * min(val, 110.0) / 110.0)
    d.line([exy(r[0], r[1] * 100) for r in div_rows],
           fill=C_ORANGE, width=3)
    d.line([exy(r[0], r[2] * 100) for r in div_rows],
           fill=C_BLUE, width=3)
    d.text((ex0 + 8, ey0_ + 8), 'orange: twin B (starts 1%, opposite '
           'shear)', fill=C_ORANGE)
    d.text((ex0 + 8, ey0_ + 26), 'blue: control (starts 40%, same '
           'shear)', fill=C_BLUE)
    d.text((ex0, ey1 + 8), 'the future follows the shear, not the '
           'residual shape mismatch.', fill=MUTED)

    # (f) summary
    sy = ty0 + 16 + sz + 40
    lines = [
        'summary: the first law holds on every slice in 2+1d '
        '(constraint), but slices no longer determine the future: '
        'the shear T_xx - T_yy is free',
        'initial data - invisible to the modular kernel, order one '
        'in the beams, and it propagates. In 1+1d tracelessness '
        'forces T_xx = T00 (why',
        'part 25 tracked); in 2+1d the spin-2 sector opens. Through '
        'the RT dictionary: the bulk metric\'s radiative modes are '
        'exactly the slice-',
        'blind data. Still owed: the h_ij wave equation from '
        'entanglement, flat space, Newton\'s constant.',
    ]
    for i, txt in enumerate(lines):
        d.text((60, sy + i * 18), txt, fill=INK if i == 0 else MUTED)
    img.save(path)


def twins_gif(gif_frames, path):
    frames = []
    sz = 220
    for i, (eAt, eBt) in enumerate(gif_frames):
        im = Image.new('RGB', (2 * sz + 60, sz + 56), BG)
        dd = ImageDraw.Draw(im)
        sc = max(eAt.max(), eBt.max())
        for k, mat in enumerate((eAt, eBt)):
            x0 = 20 + k * (sz + 20)
            _heatmap(im, _cg2(mat.reshape(L, L)) / sc, x0, 36, sz, sz)
        dd.text((20, 8), f'twin A (beams along x)      t = '
                f'{i * 1.5:4.1f}', fill=INK)
        dd.text((40 + sz, 8), 'twin B = A turned 90 deg', fill=INK)
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=280, loop=0)


if __name__ == '__main__':
    main()
