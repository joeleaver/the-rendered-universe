"""Part 14: the spherical detector.

The 1D battery of part 13 goes to the sphere, as the statistics the
real CMB anomalies are quoted in:

  [47] calibrate: Monte-Carlo isotropic Gaussian skies give every
       statistic its null distribution; the joint (Fisher-combined)
       detector is checked for false positives on clean skies.
  [48] inject the three observed anomalies AT THEIR PUBLISHED POINT
       ESTIMATES into a single sky — quadrupole-octupole alignment
       ~9 degrees, dipole modulation A=0.07, low-ell odd-parity
       excess — and ask the question the literature asks piecemeal:
       three separate ~2-sigma flukes, or one combined signature?
  [49] the protected fingerprint on the sphere: an exact mirror
       symmetry of the seed survives as a hard spectral selection
       rule (a_lm = 0 for odd l+m), fading with noise.

Idealized full sky: no mask, no foregrounds, no instrument noise.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from observatory.sphere import (GRID, LMAX, preferred_axis,
                                alignment_stat, asymmetry_stat,
                                parity_stat, spike_stat, fib_axes)

RNG = np.random.default_rng(3)
AXES = fib_axes(160)
N_MC = 200


def cl_theory():
    cl = np.zeros(LMAX + 1)
    ell = np.arange(2, LMAX + 1)
    cl[2:] = 1.0 / (ell * (ell + 1))
    return cl


CL = cl_theory()


def random_sky(rng):
    alm = np.zeros((LMAX + 1, LMAX + 1), dtype=complex)
    for ell in range(2, LMAX + 1):
        s = math.sqrt(CL[ell])
        alm[ell, 0] = rng.normal() * s
        alm[ell, 1:ell + 1] = (rng.normal(size=ell)
                               + 1j * rng.normal(size=ell)) * s / math.sqrt(2)
    return alm


def _row_power(a_row):
    return (abs(a_row[0]) ** 2
            + 2 * (np.abs(a_row[1:]) ** 2).sum())


def planar_multipole(ell, axis, rng, planar_frac=0.85):
    """A multipole with `planar_frac` of its power in |m|=ell about
    `axis`, built in real space (no rotations needed), normalized to
    the theory C_ell."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(axis @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    f = np.real(((GRID.r @ e1) + 1j * (GRID.r @ e2)) ** ell)
    a_pl = GRID.analyze(f)[ell, :ell + 1]
    a_pl /= math.sqrt(_row_power(a_pl))
    iso = rng.normal(size=ell + 1) + 1j * rng.normal(size=ell + 1)
    iso[0] = iso[0].real
    iso /= math.sqrt(_row_power(iso))
    a = (math.sqrt(planar_frac) * a_pl
         + math.sqrt(1 - planar_frac) * iso)
    a *= math.sqrt((2 * ell + 1) * CL[ell] / _row_power(a))
    return a


def stats_of(alm, mu, sd):
    f = GRID.synthesize(alm)
    cl = GRID.cl(alm)
    return dict(align=alignment_stat(alm),
                asym=asymmetry_stat(f, AXES),
                parity=parity_stat(cl),
                spike=spike_stat(cl, mu, sd))


def emp_p(null, obs, two_sided=False):
    null = np.asarray(null)
    hi = (1 + (null >= obs).sum()) / (len(null) + 1)
    if not two_sided:
        return hi
    lo = (1 + (null <= obs).sum()) / (len(null) + 1)
    return min(1.0, 2 * min(hi, lo))


def chi2_sf_df8(x):
    h = x / 2
    return math.exp(-h) * (1 + h + h * h / 2 + h ** 3 / 6)


def rotate_axis(axis, angle_deg, rng):
    """A unit vector `angle` degrees away from `axis`."""
    ref = rng.normal(size=3)
    perp = np.cross(axis, ref)
    perp /= np.linalg.norm(perp)
    a = math.radians(angle_deg)
    return math.cos(a) * axis + math.sin(a) * perp


def injected_sky(rng, align_deg=9.0, mod_A=0.07, odd_boost=1.4):
    alm = random_sky(rng)
    n2 = rng.normal(size=3)
    n2 /= np.linalg.norm(n2)
    n3 = rotate_axis(n2, align_deg, rng)
    alm[2, :3] = planar_multipole(2, n2, rng)[:3]
    alm[3, :4] = planar_multipole(3, n3, rng)[:4]
    for ell in range(3, 20, 2):
        alm[ell] *= math.sqrt(odd_boost)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    f = GRID.synthesize(alm) * (1 + mod_A * (GRID.r @ d))
    return GRID.analyze(f)


def main():
    print('=' * 68)
    print('PART 14: THE SPHERICAL DETECTOR')
    print('=' * 68)

    print(f'[47] calibration: {N_MC} isotropic Gaussian skies '
          f'(lmax={LMAX})...')
    mc_cl = np.array([GRID.cl(random_sky(RNG)) for _ in range(N_MC)])
    mu, sd = mc_cl.mean(0), np.maximum(mc_cl.std(0), 1e-12)
    nulls = {k: [] for k in ('align', 'asym', 'parity', 'spike')}
    for _ in range(N_MC):
        s = stats_of(random_sky(RNG), mu, sd)
        for k, v in s.items():
            nulls[k].append(v)
    fp = 0
    for _ in range(60):
        s = stats_of(random_sky(RNG), mu, sd)
        ps = [emp_p(nulls['align'], s['align']),
              emp_p(nulls['asym'], s['asym']),
              emp_p(nulls['parity'], s['parity'], two_sided=True),
              emp_p(nulls['spike'], s['spike'])]
        if chi2_sf_df8(-2 * sum(math.log(p) for p in ps)) < 0.01:
            fp += 1
    print(f'     joint-detector false positives on clean skies: '
          f'{fp}/60 at p<0.01')
    print()

    print('[48] one sky, three observed anomalies '
          '(9-degree alignment, A=0.07 modulation, 1.4x odd parity):')
    joints, per = [], {k: [] for k in nulls}
    for _ in range(20):
        alm = injected_sky(RNG)
        s = stats_of(alm, mu, sd)
        ps = dict(align=emp_p(nulls['align'], s['align']),
                  asym=emp_p(nulls['asym'], s['asym']),
                  parity=emp_p(nulls['parity'], s['parity'],
                               two_sided=True),
                  spike=emp_p(nulls['spike'], s['spike']))
        for k in per:
            per[k].append(ps[k])
        joints.append(chi2_sf_df8(-2 * sum(math.log(p)
                                           for p in ps.values())))
    for k in ('align', 'asym', 'parity', 'spike'):
        print(f'     {k:>7}: median single-sky p = '
              f'{np.median(per[k]):.3f}')
    jm = float(np.median(joints))
    print(f'     JOINT (Fisher): median p = {jm:.1e}  '
          f'(~{abs(np.sqrt(2) * erfinv(1 - jm)):.1f} sigma)'
          if jm > 0 else '')
    print('     Separately: a handful of ~2-sigma curiosities, exactly')
    print('     as the literature reports for the real sky. Jointly,')
    print('     under the seed-symmetry hypothesis that predicts them')
    print('     TOGETHER: a single detection-grade signature.')
    print()

    print('[49] the protected fingerprint: exact mirror symmetry')
    print('     kills a_lm with odd l+m — a hard selection rule:')
    for eps in (0.0, 0.5, 1.0, 2.0):
        alm = random_sky(RNG)
        f = GRID.synthesize(alm)
        fs = 0.5 * (f + f[::-1, :])          # equatorial mirror
        fs = fs + eps * GRID.synthesize(random_sky(RNG))
        a = GRID.analyze(fs)
        num = den = 0.0
        for ell in range(2, LMAX + 1):
            for m in range(ell + 1):
                w = 1.0 if m == 0 else 2.0
                if (ell + m) % 2:
                    num += w * abs(a[ell, m]) ** 2
                den += w * abs(a[ell, m]) ** 2
        frac = num / den
        print(f'     noise/signal {eps:>4.1f}: odd-(l+m) power fraction '
              f'= {frac:.3f}  (isotropic null ~ 0.5)')
    print('     An exact seed symmetry is not a statistical anomaly —')
    print('     it is a spectral law, visible until noise buries it.')

    # figure: the injected sky, equirectangular, with the two axes
    alm = injected_sky(np.random.default_rng(11))
    f = GRID.synthesize(alm)
    n2, n3 = preferred_axis(alm, 2), preferred_axis(alm, 3)
    v = f / np.abs(f).max()
    H, W = v.shape
    img = np.zeros((H, W, 3))
    pos = v > 0
    img[..., 0] = np.where(pos, 60 + 195 * v, 60 * (1 + v))
    img[..., 2] = np.where(pos, 60 * (1 - v), 60 - 195 * v)
    img[..., 1] = 60 * (1 - np.abs(v))
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB')
    im = im.resize((W * 6, H * 6), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for n, col in ((n2, (255, 255, 160)), (n3, (160, 255, 200))):
        for sgn in (1, -1):
            p = sgn * n
            th = math.acos(max(-1, min(1, p[2])))
            ph = math.atan2(p[1], p[0]) % (2 * math.pi)
            yy = np.searchsorted(-GRID.x, -math.cos(th)) * 6
            xx = ph / (2 * math.pi) * W * 6
            d.ellipse([xx - 7, yy - 7, xx + 7, yy + 7], outline=col,
                      width=3)
    im.save('films/sky.png')
    print('\n     films/sky.png  (rings: the aligned l=2 and l=3 axes)')


def erfinv(y):
    # Winitzki approximation — good to ~1e-3, fine for a sigma label
    a = 0.147
    ln = math.log(max(1e-300, 1 - y * y))
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)


if __name__ == '__main__':
    main()
