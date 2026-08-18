"""Part 27: the knob count.

The program's standing reply to "why these constants?" has been that
the rule and the seed are contingent data: cornered, tuned, never
derived. That answer is only respectable if the tuning freedom is
smaller than the physics it must reproduce — otherwise "tune every
knob to observation" fits anything and predicts nothing. This part
measures both sides of that comparison in the same unit: bits.

One side is the measured information content of known fundamental
physics — how many bits observation has actually supplied to pin
down the Standard Model's and LCDM's parameters, computed from the
published values and uncertainties. The other side is the tuning
freedom of the program's engine class — the exact number of rules the
constraint ledger (reversibility, isotropy, conservation, stable
vacuum) leaves alive, by alphabet size and dimension, from the orbit
combinatorics of parts 4 and 10. Log2 of a discrete class is its
knob content.

  [103] the price of physics: the Standard Model's parameters carry
        234 bits of measured information (the electron mass alone is
        32 bits; the theta_QCD bound is 36); LCDM adds 44 more for a
        core total of 278 bits, 307 with the neutrino sector.
  [104] the freedom of the class: exact survivor counts across
        alphabets k = 2..8 (2D) and k = 2..4 (3D). The ledger taxes
        at a fixed rate per dimension — it keeps 6.5-7.3% of
        log-rule-space in 2D and 1.5-1.7% in 3D at every alphabet —
        so freedom still grows fast with k: 4 bits (2D binary) to
        15,318 bits (3D 4-state).
  [105] the crossover: the class's freedom passes the data's 278
        bits between k = 4 and k = 5 in 2D, and between k = 2 and
        k = 3 in 3D. A 3D binary engine holds 28 bits of rule
        freedom against 278 bits of measured constraints: if such an
        engine fit observation at all it would be overconstrained
        tenfold — 250 bits of pure prediction. The program's original
        aesthetic bet (a minimal alphabet) and its predictivity
        requirement are the same bet.

What this does not show: that any engine in the class actually
reproduces the 278 bits — that is the standing construction problem
(interacting chiral matter on a lattice). The count says only what
tuning could and could not excuse: below the crossover, a fit would
be forced; above it, a fit would be bought.
"""
import math
import time
from collections import defaultdict
from itertools import permutations, product
from math import lgamma, log, log2

import numpy as np
from PIL import Image, ImageDraw

from rulespace import counting as c2

# validated dark-mode categorical palette (dataviz slots 1-3)
C_BLUE, C_ORANGE, C_GREEN = (57, 135, 229), (217, 89, 38), (25, 158, 112)
INK, MUTED, GRIDC = (195, 194, 183), (122, 122, 130), (38, 38, 44)
BG = (14, 14, 18)


# ---- [103] the measured information content of known physics -----------
# Values and one-sigma uncertainties: PDG 2024 (leptons, quarks in
# MSbar, couplings, Higgs, Wolfenstein CKM), Planck 2018 (LCDM),
# NuFIT-class global fits (neutrino sector). Convention: a positive
# dimensionful quantity measured to relative precision sigma/v has
# supplied log2(v/sigma) bits (units are conventions, ratios are not);
# angles and phases are counted against their full range; the
# theta_QCD bound counts as specification (a measured zero is
# information too). Uncertainties are two-significant-figure honest.

SM_PARAMS = [
    ('m_e',        0.51099895069, 1.6e-10,  'MeV'),
    ('m_mu',       105.6583755,   2.3e-6,   'MeV'),
    ('m_tau',      1776.86,       0.12,     'MeV'),
    ('m_u',        2.16,          0.38,     'MeV'),
    ('m_d',        4.67,          0.32,     'MeV'),
    ('m_s',        93.4,          6.0,      'MeV'),
    ('m_c',        1270.0,        20.0,     'MeV'),
    ('m_b',        4180.0,        30.0,     'MeV'),
    ('m_t',        172570.0,      290.0,    'MeV'),
    ('alpha',      7.2973525643e-3, 1.1e-12, ''),
    ('G_F',        1.1663787e-5,  6.0e-12,  'GeV^-2'),
    ('alpha_s',    0.1180,        0.0009,   '(M_Z)'),
    ('m_H',        125200.0,      110.0,    'MeV'),
    ('lambda_CKM', 0.22501,       0.00068,  ''),
    ('A_CKM',      0.826,         0.015,    ''),
    ('rhobar_CKM', 0.1591,        0.0094,   ''),
    ('etabar_CKM', 0.3523,        0.0073,   ''),
]
THETA_QCD_BOUND = 1e-10          # |theta| < 1e-10 from the neutron EDM
LCDM_PARAMS = [
    ('Omega_b h^2',   0.02237, 0.00015),
    ('Omega_c h^2',   0.1200,  0.0012),
    ('100 theta_MC',  1.04092, 0.00031),
    ('tau_reio',      0.0544,  0.0073),
    ('ln(1e10 A_s)',  3.044,   0.014),
    ('n_s',           0.9649,  0.0042),
]
NU_PARAMS = [
    ('dm2_21',   7.42e-5, 2.1e-6),
    ('dm2_31',   2.51e-3, 2.7e-5),
    ('s2_12',    0.304,   0.012),
    ('s2_23',    0.450,   0.020),
    ('s2_13',    0.02246, 0.00062),
    ('delta_CP', 6.28,    0.7),    # phase: range over sigma
]


def param_bits(rows):
    return [(name, log2(abs(v) / s)) for (name, v, s, *_) in rows]


# ---- [104] exact class freedom ----------------------------------------


def bits_2d(k):
    """Exact log2 of the 2D ledger-survivor count (orbit combinatorics
    of part 4; anchor: k=2 gives 16, matching brute force)."""
    groups = defaultdict(list)
    for digs, orb, stab in c2.orbit_data(k):
        groups[(sum(digs), c2._conj_class(stab))].append(stab)
    b = 0.0
    for stabs in groups.values():
        m, w = len(stabs), c2._weyl(stabs[0])
        b += lgamma(m + 1) / log(2) + m * log2(w)
    return b, lgamma(k ** 4 + 1) / log(2)


VERTS = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
VIDX = {v: i for i, v in enumerate(VERTS)}


def cube_group():
    els = set()
    for p in permutations(range(3)):
        for f in product((0, 1), repeat=3):
            els.add(tuple(VIDX[tuple(v[p[i]] ^ f[i] for i in range(3))]
                          for v in VERTS))
    return sorted(els)


G3 = cube_group()
assert len(G3) == 48


def _compose3(g, h):
    return tuple(g[h[i]] for i in range(8))


def _inv3(g):
    out = [0] * 8
    for i, v in enumerate(g):
        out[v] = i
    return tuple(out)


def _act3(g, digs):
    out = [0] * 8
    for j in range(8):
        out[g[j]] = digs[j]
    return tuple(out)


def _conj3(H):
    return min(tuple(sorted(_compose3(_compose3(g, h), _inv3(g))
                            for h in H)) for g in G3)


def _weyl3(H):
    Hs = frozenset(H)
    NH = [g for g in G3
          if frozenset(_compose3(_compose3(g, h), _inv3(g))
                       for h in H) == Hs]
    return len(NH) // len(H)


def bits_3d(k):
    """Exact log2 of the 3D ledger-survivor count (part 10's 2x2x2
    cube-group combinatorics, generalized from binary to k states;
    anchor: k=2 reproduces part 10's exact 2^28)."""
    seen = set()
    groups = defaultdict(list)
    for s in range(k ** 8):
        if s in seen:
            continue
        d, digs = s, []
        for _ in range(8):
            digs.append(d % k)
            d //= k
        digs = tuple(digs)
        for g in G3:
            a = _act3(g, digs)
            v = 0
            for j in range(8):
                v += a[j] * k ** j
            seen.add(v)
        H = tuple(g for g in G3 if _act3(g, digs) == digs)
        groups[(sum(digs), _conj3(H))].append(H)
    b = 0.0
    for stabs in groups.values():
        m, w = len(stabs), _weyl3(stabs[0])
        b += lgamma(m + 1) / log(2) + m * log2(w)
    return b, lgamma(k ** 8 + 1) / log(2)


# ---- main --------------------------------------------------------------


def main():
    t00 = time.time()
    print('=' * 68)
    print('PART 27: THE KNOB COUNT')
    print('=' * 68)
    print()
    print('Both sides of "tune every knob to observation," measured in')
    print('the same unit. Data side: the bits observation has supplied')
    print('to fix the constants of known physics. Theory side: the bits')
    print('of rule freedom the constraint ledger leaves, by alphabet')
    print('and dimension — exact counts, no enumeration, no sampling.')
    print()

    print('[103] the price of physics '
          '(bits = log2(value / uncertainty)):')
    sm_bits = param_bits(SM_PARAMS)
    b_theta = log2(2 * math.pi / THETA_QCD_BOUND)
    lcdm_bits = [(n, log2(abs(v) / s)) for n, v, s in LCDM_PARAMS]
    nu_bits = [(n, log2(abs(v) / s)) for n, v, s in NU_PARAMS]
    for name, b in sm_bits:
        print(f'       {name:<12} {b:6.1f}')
    print(f'       {"theta_QCD":<12} {b_theta:6.1f}   '
          '(the bound |theta| < 1e-10: a measured zero)')
    print(f'       {"v_Higgs":<12} {0.0:6.1f}   '
          '(fixed by G_F; information counted once)')
    sm_tot = sum(b for _, b in sm_bits) + b_theta
    lcdm_tot = sum(b for _, b in lcdm_bits)
    nu_tot = sum(b for _, b in nu_bits)
    print(f'     Standard Model total:        {sm_tot:6.0f} bits')
    print(f'     LCDM (6 parameters) adds:    {lcdm_tot:6.0f} bits')
    print(f'     CORE TOTAL:                  {sm_tot + lcdm_tot:6.0f} '
          'bits')
    print(f'     neutrino sector (extension): {nu_tot:+6.0f} -> '
          f'{sm_tot + lcdm_tot + nu_tot:.0f} bits')
    print('     Note the shape of this number: it grows with every '
          'decimal of')
    print('     precision. A continuum theory\'s description length is '
          'open-ended;')
    print('     a discrete class\'s is fixed. Every future digit is a '
          'new tunable')
    print('     for the one and a new prediction opportunity for the '
          'other.')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    DATA = sm_tot + lcdm_tot

    print('[104] the freedom of the class (exact orbit combinatorics):')
    print('       dim  k    survivors     full space    kept')
    rows2, rows3 = [], []
    for k in (2, 3, 4, 5, 6, 7, 8):
        b, full = bits_2d(k)
        rows2.append((k, b, full))
        print(f'       2D   {k}   2^{b:8.1f}    2^{full:9.1f}   '
              f'{100 * b / full:5.2f}%')
    for k in (2, 3, 4):
        b, full = bits_3d(k)
        rows3.append((k, b, full))
        print(f'       3D   {k}   2^{b:8.1f}    2^{full:9.1f}   '
              f'{100 * b / full:5.2f}%')
    print('     anchors: 2D k=2 gives 2^4.0 = 16 (brute-force match, '
          'part 4);')
    print('     3D k=2 gives 2^28.0, part 10\'s exact count.')
    print('     The tax is a per-dimension constant: the ledger keeps '
          '6.5-7.3% of')
    print('     log-rule-space in 2D and 1.5-1.7% in 3D at every '
          'alphabet —')
    print('     isotropy sets the rate, and the rate does not save '
          'rich alphabets:')
    print('     freedom still grows from 4 bits (2D binary) to '
          f'{rows3[-1][1]:,.0f} bits')
    print('     (3D 4-state).')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    print('[105] the crossover:')
    print(f'     measured physics to reproduce: {DATA:.0f} bits '
          f'({DATA + nu_tot:.0f} with neutrinos)')
    k2lo = max(k for k, b, _ in rows2 if b < DATA)
    k2hi = min(k for k, b, _ in rows2 if b > DATA)
    k3lo = max(k for k, b, _ in rows3 if b < DATA)
    k3hi = min(k for k, b, _ in rows3 if b > DATA)
    b2lo = [b for k, b, _ in rows2 if k == k2lo][0]
    b3lo = [b for k, b, _ in rows3 if k == k3lo][0]
    print(f'     2D: freedom crosses the data between k = {k2lo} '
          f'(2^{b2lo:.0f}) and k = {k2hi};')
    print(f'     3D: between k = {k3lo} (2^{b3lo:.0f}) and '
          f'k = {k3hi}.')
    print(f'     A 3D binary engine holds {b3lo:.0f} bits of rule '
          f'freedom against {DATA:.0f} bits')
    print(f'     of measured constraints: if such an engine fit '
          f'observation at all, it')
    print(f'     would be overconstrained {DATA / b3lo:.0f}-fold — '
          f'{DATA - b3lo:.0f} bits of pure prediction.')
    print('     Above the crossover the same fit would be bought, '
          'not forced.')
    print('     The seed column: part 13 measured the detectability '
          'ceiling for seed')
    print('     generators at ~40 bytes = 320 bits — initial-condition '
          'freedom of the')
    print('     same order as the whole data budget, which is why the '
          'program\'s')
    print('     falsifiable seed content is the SYMMETRY class '
          '(survives dynamics),')
    print('     not the seed bytes. And two of LCDM\'s six parameters '
          '(A_s, n_s) are')
    print('     seed-statistics, not law: 16 of the 278 bits already '
          'live in the')
    print('     seed column.')
    print()
    print('     What this does not show: that any engine in the class '
          'reproduces')
    print('     the 278 bits. That is the standing construction '
          'problem (chiral')
    print('     interacting matter on a lattice, part 10). The count '
          'fixes what')
    print('     tuning could excuse: below the crossover, nothing.')

    figure(rows2, rows3, DATA, nu_tot, sm_bits, b_theta, lcdm_bits,
           'films/knobs.png')
    print()
    print(f'     films/knobs.png  ({time.time() - t00:.0f}s)')


# ---- figure ------------------------------------------------------------


def figure(rows2, rows3, DATA, nu_tot, sm_bits, b_theta, lcdm_bits,
           path):
    W, Ht = 1560, 700
    img = Image.new('RGB', (W, Ht), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 12), 'PART 27 - THE KNOB COUNT', fill=INK)

    # (a) class freedom vs alphabet, log2-bits on log axis
    ax0, ay0, ax1, ay1 = 80, 90, 700, 560
    d.text((ax0, ay0 - 34), '[104/105] rule-class freedom (bits, log '
           'scale) vs alphabet size k.', fill=INK)
    d.text((ax0, ay0 - 18), 'line at 278 bits: the measured information '
           'content of SM + LCDM.', fill=MUTED)
    blo, bhi = 2.0, 20000.0

    def axy(k, bits):
        px = ax0 + (ax1 - ax0) * (k - 2) / 6.0
        py = ay1 - (ay1 - ay0) * (math.log(bits / blo)
                                  / math.log(bhi / blo))
        return px, py
    for dec in (10, 100, 1000, 10000):
        py = axy(2, dec)[1]
        d.line([(ax0, py), (ax1, py)], fill=GRIDC)
        d.text((ax0 - 52, py - 6), f'{dec:>5}', fill=MUTED)
    for k in range(2, 9):
        px = axy(k, blo)[0]
        d.line([(px, ay0), (px, ay1)], fill=GRIDC)
        d.text((px - 3, ay1 + 8), str(k), fill=MUTED)
    pd = axy(2, DATA)[1]
    d.line([(ax0, pd), (ax1, pd)], fill=INK, width=2)
    d.text((ax1 - 210, pd - 18), f'measured physics: {DATA:.0f} bits',
           fill=INK)
    pdn = axy(2, DATA + nu_tot)[1]
    d.line([(ax0, pdn), (ax1, pdn)], fill=(90, 90, 96), width=1)
    for rows, col, lab in ((rows2, C_BLUE, '2D (D4)'),
                           (rows3, C_ORANGE, '3D (cube group)')):
        pts = [axy(k, b) for k, b, _ in rows]
        d.line(pts, fill=col, width=3)
        for p in pts:
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4],
                      fill=col)
        d.text((pts[-1][0] - 70, pts[-1][1] - 22), lab, fill=col)
    d.text((ax0, ay1 + 28), 'every point exact (orbit combinatorics). '
           'crossings: 2D between k = 4 and 5; 3D between k = 2 and 3.',
           fill=MUTED)
    d.text((ax0, ay1 + 44), 'a 3D binary engine: 28 bits of freedom '
           'vs 278 of data - tenfold overconstrained if it fits.',
           fill=MUTED)

    # (b) the data side: per-parameter bits
    bx0, by0, bx1, by1 = 800, 90, 1520, 560
    d.text((bx0, by0 - 34), '[103] where the 278 bits live: measured '
           'information per parameter.', fill=INK)
    d.text((bx0, by0 - 18), 'blue: SM; grey: theta_QCD bound; green: '
           'LCDM.', fill=MUTED)
    bars = ([(n, b, C_BLUE) for n, b in sm_bits]
            + [('theta_QCD', b_theta, (130, 130, 138))]
            + [(n, b, C_GREEN) for n, b in lcdm_bits])
    bars.sort(key=lambda r: -r[1])
    bw = (bx1 - bx0) / len(bars)
    bmax = max(b for _, b, _ in bars)
    for i, (name, b, col) in enumerate(bars):
        x0 = bx0 + i * bw + 2
        h = (by1 - by0 - 30) * b / bmax
        d.rectangle([x0, by1 - h, x0 + bw - 4, by1], fill=col)
        d.text((x0 + bw / 2 - 8, by1 - h - 16), f'{b:.0f}', fill=MUTED)
    d.text((bx0, by1 + 8), 'left to right: ' + ', '.join(
        n for n, _, _ in bars[:6]) + ', ...', fill=MUTED)
    d.text((bx0, by1 + 28), 'precision is information: the electron '
           'mass and alpha carry more bits than every quark mass '
           'combined.', fill=MUTED)
    d.text((bx0, by1 + 44), 'this number grows with every measured '
           'digit; a discrete class\'s freedom does not.', fill=MUTED)
    img.save(path)


if __name__ == '__main__':
    main()
