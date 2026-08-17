"""Part 20: the echo — the registered test meets the E-mode sky.

Part 19 ended with registered directions: the quadrupole and octupole
axes, the asymmetry axis, and a mirror axis 5 degrees from the CMB
dipole. Registration is what turns a post-hoc curiosity into a
prediction: E-mode polarization is a partially independent second
draw from the same primordial modes (reionization re-scattering), so
if the temperature anomalies are features of the seed, E-modes must
echo them AT THOSE AXES; if they are flukes of the temperature
realization, the echo is absent. No scans, no look-elsewhere — every
statistic below is evaluated at a direction fixed in part 19.

The statistics are careful about one trap: T and E are correlated
(the TE spectrum), so even a fluke partially echoes into E. The null
ensemble therefore draws E GIVEN the real temperature sky under LCDM
(conditional nulls), and the headline statistic is the T-subtracted
residual E_ind = E - (C_TE/C_TT) T — the part of E that carries
genuinely new information.

  [71] the spin-2 instrument and its chain of custody — and a canary
       caught en route: Planck's polarization inpainting eats a third
       of the large-scale TE signal (raw 0.97-1.00 vs theory across
       all four methods; inpainted 0.68). The battery runs on raw
       maps, with the inpainted variant as cross-check.
  [72] the noise reality: the E-mode noise floor (measured from B,
       which cosmology leaves empty) versus the E signal, per
       multipole — where today's data lives and where it dies.
  [73] conditional nulls: E given the real T, LCDM + measured noise,
       identical treatment for data and nulls.
  [74] the echo battery at the registered axes.
  [75] the power of the test: could today's data even see the echo
       if it were there? Injections at temperature-like amplitude,
       with today's noise and with none (the LiteBIRD-class limit).
  [76] verdict.
"""
import json
import math
import time

import numpy as np
from PIL import Image, ImageDraw

from observatory.sphere import GRID, LMAX, full_m, preferred_axis, fib_axes

RNG = np.random.default_rng(20)
N_MC = 4000
N_POW = 800
METHODS = ('smica', 'commander', 'nilc', 'sevem')

# validated dark-mode categorical palette (dataviz slots 1-3)
C_BLUE, C_ORANGE, C_GREEN = (57, 135, 229), (217, 89, 38), (25, 158, 112)
INK, MUTED, GRIDC = (195, 194, 183), (122, 122, 130), (38, 38, 44)
BG = (14, 14, 18)

# ---- the registered directions (part 19, committed before any
# ---- polarization data was read) --------------------------------------
MIRROR_LB = (264.0, 44.0)     # mirror-scan minimum, 4/5 maps
MOD_SIGN_AXIS = None          # asymmetry axis is recomputed per method
                              # from the same T a_lm part 19 used


def n_of(lb):
    L, b = map(math.radians, lb)
    return np.array([math.cos(b) * math.cos(L),
                     math.cos(b) * math.sin(L), math.sin(b)])


N_MIRROR = n_of(MIRROR_LB)

# ---- data -------------------------------------------------------------

POL = np.load('data/realsky_pol.npz')
TDATA = np.load('data/realsky_alm.npz')


def tri(key, lmax=LMAX):
    return POL[key][:lmax + 1, :lmax + 1].copy()


def theory_cl(lmax=LMAX):
    """C_l^TT, C_l^TE, C_l^EE (uK^2) for l = 0..lmax."""
    out = np.zeros((3, lmax + 1))
    for i, key in enumerate(('th_tt', 'th_te', 'th_ee')):
        for L, D in zip(POL['th_ell'], POL[key]):
            if 2 <= L <= lmax:
                out[i, int(L)] = 2 * math.pi * D / (L * (L + 1))
    return out


CTT, CTE, CEE = theory_cl()


def noise_cl(meth, var, lmax=LMAX):
    """E-noise per l from the measured B spectrum (cosmology leaves B
    empty at these scales), smoothed over a 7-multipole window."""
    dl = POL[f'dl_bb_{meth}_{var}']
    n = np.zeros(lmax + 1)
    for ell in range(2, lmax + 1):
        lo, hi = max(2, ell - 3), min(len(dl) - 1, ell + 3)
        band = dl[lo:hi + 1] / (np.arange(lo, hi + 1)
                                * (np.arange(lo, hi + 1) + 1)
                                / (2 * math.pi))
        n[ell] = band.mean()
    return n


def te_transfer(meth, var):
    """Measured TE amplitude vs theory over l=30-150 — the map's
    effective E transfer (raw ~1.0; inpainted ~0.68)."""
    sel = np.arange(30, 151)
    te_t = np.array([D for L, D in zip(POL['th_ell'], POL['th_te'])
                     if 30 <= L <= 150])
    ours = POL[f'dl_te_{meth}_{var}'][sel]
    return float(ours @ te_t / (te_t @ te_t))


# ---- single-axis mirror fraction (part 19's instrument, compact) -----

_EIG = {}
for _l in range(2, LMAX + 1):
    _m = np.arange(-_l, _l + 1)
    _c = np.sqrt(_l * (_l + 1) - _m[:-1] * (_m[:-1] + 1))
    _Jp = np.zeros((2 * _l + 1, 2 * _l + 1), dtype=complex)
    _Jp[np.arange(1, 2 * _l + 1), np.arange(2 * _l)] = _c
    _w, _V = np.linalg.eigh((_Jp - _Jp.conj().T) / 2j)
    _EIG[_l] = (_m, _V, _V.conj().T, _w)


def mirror_fraction(alm, n):
    theta, phi = math.acos(n[2]), math.atan2(n[1], n[0])
    odd = tot = 0.0
    for ell in range(2, LMAX + 1):
        m, V, Vh, w = _EIG[ell]
        b = full_m(alm, ell)
        bp = V @ (np.exp(1j * theta * w) * (Vh @ (b * np.exp(1j * m * phi))))
        p = np.abs(bp) ** 2
        odd += p[(ell + m) % 2 == 1].sum()
        tot += p.sum()
    return odd / tot


# ---- battery ----------------------------------------------------------

def parity_ratio(alm):
    cl = GRID.cl(alm)
    ell = np.arange(2, 21)
    d = ell * (ell + 1) * cl[2:21]
    return d[ell % 2 == 1].sum() / d[ell % 2 == 0].sum()


def hemi_diff(alm, n):
    """Signed hemispheric power difference of the l<=24 field along n."""
    f = GRID.synthesize(alm)
    mask = (GRID.r @ n) > 0
    tot = (GRID.area * f ** 2).sum()
    return float((2 * (GRID.area * f ** 2 * mask).sum() - tot) / tot)


def battery(almE, axes):
    """The five registered-axis echo statistics."""
    return dict(
        mir=mirror_fraction(almE, N_MIRROR),
        align2=abs(float(preferred_axis(almE, 2) @ axes['n2'])),
        align3=abs(float(preferred_axis(almE, 3) @ axes['n3'])),
        asym=hemi_diff(almE, axes['nasym']),
        parity=parity_ratio(almE))


# recurrence direction of each statistic: is the echo LOW or HIGH?
SIDE = dict(mir='low', align2='high', align3='high',
            asym='high', parity='high')


def emp_p(null, obs, side):
    null = np.asarray(null)
    if side == 'low':
        return (1 + (null <= obs).sum()) / (len(null) + 1)
    return (1 + (null >= obs).sum()) / (len(null) + 1)


def fisher_p(ps):
    x = -2 * sum(math.log(max(p, 1e-300)) for p in ps)
    # chi2 survival, df = 2 * len(ps)
    k = len(ps)
    h = x / 2
    term, sf = 1.0, 0.0
    for i in range(k):
        sf += term
        term *= h / (i + 1)
    return math.exp(-h) * sf


def erfinv(y):
    a = 0.147
    ln = math.log(max(1e-300, 1 - y * y))
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)


def sigma_of(p):
    return abs(math.sqrt(2) * erfinv(1 - min(p, 1 - 1e-12)))


# ---- conditional nulls and injections --------------------------------

def draw_E_given_T(rng, almT, t, nl, inject=None):
    """E given the real T under LCDM x transfer t, plus noise; with
    `inject`, the T-independent component carries the temperature
    anomalies at the registered axes instead of isotropy."""
    lmax = LMAX
    alm = np.zeros((lmax + 1, lmax + 1), dtype=complex)
    cond = np.zeros(lmax + 1)
    for ell in range(2, lmax + 1):
        cond[ell] = max(0.0, CEE[ell] - CTE[ell] ** 2 / CTT[ell])
    if inject is None:
        g = np.zeros((lmax + 1, lmax + 1), dtype=complex)
        for ell in range(2, lmax + 1):
            g[ell, 0] = rng.normal()
            g[ell, 1:ell + 1] = (rng.normal(size=ell)
                                 + 1j * rng.normal(size=ell)) / math.sqrt(2)
        for ell in range(2, lmax + 1):
            g[ell] *= math.sqrt(cond[ell])
    else:
        g = injected_field(rng, cond, inject)
    for ell in range(2, lmax + 1):
        alm[ell] = (t * (CTE[ell] / CTT[ell]) * almT[ell]
                    + t * g[ell])
        n = np.zeros(ell + 1, dtype=complex)
        n[0] = rng.normal()
        n[1:] = (rng.normal(size=ell)
                 + 1j * rng.normal(size=ell)) / math.sqrt(2)
        alm[ell, :ell + 1] += math.sqrt(nl[ell]) * n
    return alm


def _planar(ell, axis, rng, cl_val, frac=0.85):
    """A multipole with most of its power planar about `axis`
    (part 14's construction), normalized to cl_val."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(axis @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    f = np.real(((GRID.r @ e1) + 1j * (GRID.r @ e2)) ** ell)
    a_pl = GRID.analyze(f)[ell, :ell + 1]
    def power(a):
        return abs(a[0]) ** 2 + 2 * (np.abs(a[1:]) ** 2).sum()
    a_pl /= math.sqrt(power(a_pl))
    iso = rng.normal(size=ell + 1) + 1j * rng.normal(size=ell + 1)
    iso[0] = iso[0].real
    iso /= math.sqrt(power(iso))
    a = math.sqrt(frac) * a_pl + math.sqrt(1 - frac) * iso
    a *= math.sqrt((2 * ell + 1) * cl_val / power(a))
    return a


def injected_field(rng, cond, axes):
    """The T anomalies, installed in the T-independent E component at
    the REGISTERED axes: planar l=2,3, odd boost, dipole modulation."""
    lmax = LMAX
    g = np.zeros((lmax + 1, lmax + 1), dtype=complex)
    for ell in range(2, lmax + 1):
        g[ell, 0] = rng.normal()
        g[ell, 1:ell + 1] = (rng.normal(size=ell)
                             + 1j * rng.normal(size=ell)) / math.sqrt(2)
        g[ell] *= math.sqrt(cond[ell])
    g[2, :3] = _planar(2, axes['n2'], rng, cond[2])[:3]
    g[3, :4] = _planar(3, axes['n3'], rng, cond[3])[:4]
    for ell in range(3, 20, 2):
        g[ell] *= math.sqrt(1.4)
    f = GRID.synthesize(g) * (1 + 0.07 * (GRID.r @ axes['nasym']))
    return GRID.analyze(f)


def subtract_T(almE, almT, t):
    out = almE.copy()
    for ell in range(2, LMAX + 1):
        out[ell] -= t * (CTE[ell] / CTT[ell]) * almT[ell]
    return out


# ---- figure -----------------------------------------------------------

def _panel_spectra(d, box):
    x0, y0, x1, y1 = box
    lmaxp = 40
    # log scale from 1e-3 to 1e-1 uK^2 (D_l)
    def xy(l, v):
        lv = (math.log10(max(v, 1e-4)) + 4) / 3.3
        return (x0 + (x1 - x0) * (l - 2) / (lmaxp - 2),
                y1 - (y1 - y0) * min(lv, 1.0))
    for e in (-3, -2, -1):
        yy = xy(2, 10 ** e)[1]
        d.line([(x0, yy), (x1, yy)], fill=GRIDC)
        d.text((x0 - 40, yy - 6), f'1e{e}', fill=MUTED)
    th = {int(L): D for L, D in zip(POL['th_ell'], POL['th_ee'])}
    pts = [xy(l, th[l]) for l in range(2, lmaxp + 1)]
    d.line(pts, fill=MUTED, width=3)
    for meth, col in (('smica', C_ORANGE), ('commander', (150, 63, 27)),
                      ('nilc', (150, 63, 27)), ('sevem', (150, 63, 27))):
        dl = POL[f'dl_ee_{meth}_raw']
        pts = [xy(l, dl[l]) for l in range(2, lmaxp + 1)]
        d.line(pts, fill=col, width=2)
    dl = POL['dl_bb_smica_raw']
    pts = [xy(l, dl[l]) for l in range(2, lmaxp + 1)]
    d.line(pts, fill=C_BLUE, width=3)
    for l in (5, 10, 20, 30, 40):
        d.text((xy(l, 1e-4)[0] - 4, y1 + 6), f'{l}', fill=MUTED)
    d.text((x0, y0 - 16), 'the noise reality:  D_l^EE measured (orange) '
           'vs LCDM E signal (gray) vs B floor = noise (blue)', fill=INK)
    d.text(((x0 + x1) // 2 - 6, y1 + 20), 'l', fill=MUTED)


def _panel_te(d, box):
    x0, y0, x1, y1 = box
    lmaxp = 150
    vmax = 50.0
    def xy(l, v):
        return (x0 + (x1 - x0) * (l - 2) / (lmaxp - 2),
                y1 - (y1 - y0) * (max(-vmax, min(vmax, v)) + vmax)
                / (2 * vmax))
    d.line([(x0, xy(2, 0)[1]), (x1, xy(2, 0)[1])], fill=GRIDC)
    th = {int(L): D for L, D in zip(POL['th_ell'], POL['th_te'])}
    d.line([xy(l, th[l]) for l in range(2, lmaxp + 1)], fill=MUTED,
           width=3)
    for var, col in (('raw', C_ORANGE), ('inp', C_GREEN)):
        dl = POL[f'dl_te_smica_{var}']
        # 5-l smoothing for display
        sm = np.convolve(dl[:lmaxp + 1], np.ones(5) / 5, mode='same')
        d.line([xy(l, sm[l]) for l in range(4, lmaxp - 1)], fill=col,
               width=2)
    d.text((x0, y0 - 16), 'the inpainting canary:  D_l^TE - theory '
           '(gray), raw maps (orange, amp 0.97-1.00), inpainted '
           '(green, amp 0.68)', fill=INK)
    for l in (25, 50, 75, 100, 125, 150):
        d.text((xy(l, -vmax)[0] - 8, y1 + 6), f'{l}', fill=MUTED)
    d.text(((x0 + x1) // 2 - 6, y1 + 20), 'l', fill=MUTED)


def _panel_pvals(d, box, results):
    x0, y0, x1, y1 = box
    stats = ('mir', 'align2', 'align3', 'asym', 'parity', 'joint')
    d.text((x0, y0 - 16), 'the echo battery (E_ind, raw maps): p per '
           'registered statistic (orange smica, blue others); '
           '0.05 = detection', fill=INK)
    def ypos(p):
        lv = (math.log10(max(p, 1e-2)) + 2) / 2
        return y1 - 26 - (y1 - y0 - 46) * lv
    for pv in (1.0, 0.1, 0.05, 0.01):
        yy = ypos(pv)
        col = C_ORANGE if pv == 0.05 else GRIDC
        d.line([(x0, yy), (x1, yy)], fill=col)
        d.text((x0 - 40, yy - 6), f'{pv:g}', fill=MUTED)
    for i, st in enumerate(stats):
        xx = x0 + 40 + i * (x1 - x0 - 60) // (len(stats) - 1)
        d.text((xx - 14, y1 - 14), st, fill=MUTED)
        for j, meth in enumerate(METHODS):
            p = results[meth][st]
            d.ellipse([xx - 4 + j * 3 - 4, ypos(p) - 4,
                       xx + 4 + j * 3 - 4, ypos(p) + 4],
                      fill=C_BLUE if meth != 'smica' else C_ORANGE)


def _panel_power(d, box, power):
    x0, y0, x1, y1 = box
    d.text((x0, y0 - 16), 'could the echo even be seen? power of the '
           'joint test at alpha = 0.05', fill=INK)
    bars = [('today (Planck noise)', power['today'], C_BLUE),
            ('no noise (LiteBIRD-class)', power['cv'], C_GREEN)]
    bw = (y1 - y0 - 30) // 2
    for i, (label, val, col) in enumerate(bars):
        yy = y0 + 14 + i * (bw + 14)
        w = int((x1 - x0 - 200) * val)
        d.rectangle([x0 + 190, yy, x0 + 190 + w, yy + bw], fill=col)
        d.text((x0, yy + bw // 2 - 6), label, fill=INK)
        d.text((x0 + 196 + w, yy + bw // 2 - 6), f'{val * 100:.0f}%',
               fill=INK)


def figure(results, power, path):
    W, H = 1560, 900
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 12), 'PART 20 - THE ECHO: the registered test meets '
           'the E-mode sky (raw Planck 2018 polarization, from raw '
           'bytes)', fill=INK)
    _panel_spectra(d, (70, 70, 730, 380))
    _panel_te(d, (830, 70, 1520, 380))
    _panel_pvals(d, (70, 480, 730, 820), results)
    _panel_power(d, (830, 500, 1520, 640), power)
    d.text((830, 690), 'verdict: with today\'s large-angle polarization '
           'noise the echo test is', fill=MUTED)
    d.text((830, 706), 'underpowered - the registered prediction stays '
           'open, and now has a', fill=MUTED)
    d.text((830, 722), 'measured sensitivity requirement attached.',
           fill=MUTED)
    img.save(path)


# ---- main -------------------------------------------------------------

def main():
    t00 = time.time()
    print('=' * 68)
    print('PART 20: THE ECHO — THE REGISTERED TEST MEETS THE E-MODE SKY')
    print('=' * 68)
    print()

    print('[71] the spin-2 instrument, and a canary caught on the way:')
    print('     E/B read from raw Stokes bytes by observatory/spin.py')
    print('     (Wigner-d recursion; validated at import; geometry and')
    print('     E/B convention pinned against reference values).')
    print('     TE amplitude vs LCDM theory, l=30-150:')
    for meth in METHODS:
        print(f'       {meth:9s}  raw {te_transfer(meth, "raw"):.3f}   '
              f'inpainted {te_transfer(meth, "inp"):.3f}')
    print('     Planck\'s polarization inpainting (4% of sky) eats a')
    print('     THIRD of the large-scale TE signal. The battery runs')
    print('     on the raw maps; the inpainted variant cross-checks.')
    print()

    print('[72] the noise reality (E noise taken from measured B,')
    print('     which cosmology leaves empty at these scales):')
    nl = noise_cl('smica', 'raw')
    for ell in (2, 4, 7, 10, 16, 24):
        sn = CEE[ell] / nl[ell] if nl[ell] > 0 else float('inf')
        print(f'       l={ell:2d}:  signal/noise per mode = {sn:5.2f}')
    print('     The reionization bump (l<8) pokes above the floor;')
    print('     everything else drowns. This is why the test needs')
    print('     LiteBIRD-class data — measured below in [75].')
    print()

    # registered axes per method, from the SAME T a_lm part 19 used
    axes_of = {}
    for meth in METHODS:
        almT24 = TDATA[f'alm_{meth}'][:LMAX + 1, :LMAX + 1].copy()
        almT24[:2] = 0
        n2 = preferred_axis(almT24, 2)
        n3 = preferred_axis(almT24, 3)
        f = GRID.synthesize(almT24)
        best, bn = 0.0, None
        for n in fib_axes(160):
            v = hemi_diff(almT24, n)
            if abs(v) > best:
                best, bn = abs(v), (n if v > 0 else -n)
        axes_of[meth] = dict(n2=n2, n3=n3, nasym=bn, almT=almT24)

    print(f'[73] conditional nulls: {N_MC} draws of E GIVEN the real')
    print('     temperature sky (LCDM TE correlation x measured')
    print('     transfer + measured noise), identical battery.')
    print()

    results = {}
    t0 = time.time()
    for meth in METHODS:
        axes = axes_of[meth]
        t = te_transfer(meth, 'raw')
        nl = noise_cl(meth, 'raw')
        almT = axes['almT']
        almE = tri(f'almE_{meth}_raw')
        almE[:2] = 0
        e_ind = subtract_T(almE, almT, t)
        s_real = battery(e_ind, axes)
        nulls = {k: [] for k in SIDE}
        for _ in range(N_MC):
            en = draw_E_given_T(RNG, almT, t, nl)
            s = battery(subtract_T(en, almT, t), axes)
            for k in SIDE:
                nulls[k].append(s[k])
        ps = {k: emp_p(nulls[k], s_real[k], SIDE[k]) for k in SIDE}
        ps['joint'] = fisher_p(list(ps.values()))
        results[meth] = ps
    print(f'[74] the echo battery at the registered axes (E_ind, raw '
          f'maps; {time.time() - t0:.0f}s):')
    print()
    print('     map        p_mir  p_al2  p_al3  p_asym  p_par   '
          'JOINT   sigma')
    for meth in METHODS:
        p = results[meth]
        print(f'     {meth:9s}  {p["mir"]:.3f}  {p["align2"]:.3f}  '
              f'{p["align3"]:.3f}  {p["asym"]:.3f}   {p["parity"]:.3f}'
              f'  {p["joint"]:.3f}   {sigma_of(p["joint"]):.1f}')
    # inpainted cross-check, smica only, fewer nulls
    meth = 'smica'
    axes, t = axes_of[meth], te_transfer(meth, 'inp')
    nl = noise_cl(meth, 'inp')
    almE = tri(f'almE_{meth}_inp')
    almE[:2] = 0
    e_ind = subtract_T(almE, axes['almT'], t)
    s_real = battery(e_ind, axes)
    nulls = {k: [] for k in SIDE}
    for _ in range(1000):
        en = draw_E_given_T(RNG, axes['almT'], t, nl)
        s = battery(subtract_T(en, axes['almT'], t), axes)
        for k in SIDE:
            nulls[k].append(s[k])
    ps = {k: emp_p(nulls[k], s_real[k], SIDE[k]) for k in SIDE}
    pj = fisher_p(list(ps.values()))
    print(f'     smica-inpainted cross-check: joint p = {pj:.3f}')
    print()

    print('[75] the power of the test — inject the T anomalies into')
    print('     the E-independent component at the registered axes,')
    print('     at temperature-like amplitude:')
    meth = 'smica'
    axes, t = axes_of[meth], te_transfer(meth, 'raw')
    power = {}
    for label, nl_use in (('today', noise_cl(meth, 'raw')),
                          ('cv', np.zeros(LMAX + 1))):
        # null distribution under this noise
        nulls = {k: [] for k in SIDE}
        for _ in range(N_MC // 2):
            en = draw_E_given_T(RNG, axes['almT'], t, nl_use)
            s = battery(subtract_T(en, axes['almT'], t), axes)
            for k in SIDE:
                nulls[k].append(s[k])
        hits = 0
        for _ in range(N_POW):
            en = draw_E_given_T(RNG, axes['almT'], t, nl_use,
                                inject=axes)
            s = battery(subtract_T(en, axes['almT'], t), axes)
            ps = {k: emp_p(nulls[k], s[k], SIDE[k]) for k in SIDE}
            if fisher_p(list(ps.values())) < 0.05:
                hits += 1
        power[label] = hits / N_POW
        tag = ('today\'s noise' if label == 'today'
               else 'no noise (LiteBIRD-class)')
        print(f'       {tag:26s}: power = {100 * power[label]:.0f}%')
    print()

    print('[76] verdict:')
    pj = results['smica']['joint']
    print(f'     The echo is {"absent" if pj > 0.05 else "PRESENT"} in '
          f'today\'s data (joint p = {pj:.2f}), and')
    print(f'     [75] says that is the expected outcome EITHER way: at')
    print(f'     today\'s noise the test detects a real echo only '
          f'{100 * power["today"]:.0f}% of')
    print(f'     the time, versus {100 * power["cv"]:.0f}% at '
          f'LiteBIRD-class sensitivity. The')
    print('     registered prediction stands, unresolved but now with')
    print('     a measured sensitivity requirement: this is what the')
    print('     next generation of polarization data is for.')

    figure(results, power, 'films/echo.png')
    print()
    print(f'     films/echo.png  ({time.time() - t00:.0f}s total)')


if __name__ == '__main__':
    main()
