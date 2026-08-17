"""The sky's polarization, from the same raw bytes: Q/U -> E/B a_lm.

Reads the inpainted Stokes Q/U columns of the four Planck 2018
component-separated maps already fetched by tools/fetch_sky.py,
analyzes them with the spin-2 machinery of observatory/spin.py
(lmax 200), recomputes temperature a_lm to the same lmax, and writes
data/realsky_pol.npz: E/B/T a_lm per method, our measured EE/BB/TE
spectra, the published Planck TE/EE spectra and best-fit theory for
validation, and SHA-256 provenance.

WMAP is deliberately temperature-only in this program: its large-angle
polarization is a likelihood-level measurement requiring the mission's
pixel-space noise covariance, not a map one can honestly battery-test.

    python3 tools/fetch_pol.py        # needs data/raw/ from fetch_sky
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observatory.healpix import read_column, nest2ring, ring_info  # noqa: E402
from observatory.spin import d_stack, _ring_cm                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')
OUT = os.path.join(ROOT, 'data', 'realsky_pol.npz')
LMAX = 200
NSIDE = 2048
METHODS = ('smica', 'commander', 'nilc', 'sevem')

COSMO = ('https://irsa.ipac.caltech.edu/data/Planck/release_3/'
         'ancillary-data/cosmoparams/')
SPECTRA = {'tt': 'COM_PowerSpect_CMB-TT-full_R3.01.txt',
           'te': 'COM_PowerSpect_CMB-TE-full_R3.01.txt',
           'ee': 'COM_PowerSpect_CMB-EE-full_R3.01.txt'}
THEORY = ('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-'
          'minimum-theory_R3.01.txt')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def cross_cl(a, b):
    lmax = a.shape[0] - 1
    return np.array([(np.real(a[l, 0] * np.conj(b[l, 0]))
                      + 2 * np.real(a[l, 1:l + 1]
                                    * np.conj(b[l, 1:l + 1])).sum())
                     / (2 * l + 1) for l in range(lmax + 1)])


def main():
    import urllib.request
    for name in list(SPECTRA.values()) + [THEORY]:
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print(f'   downloading {name} ...')
            urllib.request.urlretrieve(COSMO + name, path)

    print('   nest->ring permutation ...')
    perm = nest2ring(NSIDE)
    z = ring_info(NSIDE)[0]
    dA = 4 * np.pi / (12 * NSIDE ** 2)

    # stage 1: per-ring azimuthal coefficient tables for every map,
    # both polarization variants: inpainted ('inp') and untouched
    # ('raw') — inpainting turns out to eat large-scale TE (the
    # validation below measures it), so the battery needs both
    Cp, Cm, Ct, prov = {}, {}, {}, {}
    for meth in METHODS:
        path = os.path.join(
            RAW, f'COM_CMB_IQU-{meth}_2048_R3.00_full.fits')
        t0 = time.time()
        f_ring = np.empty(12 * NSIDE ** 2)
        cols = {}
        for col in ('I_STOKES_INP', 'Q_STOKES_INP', 'U_STOKES_INP',
                    'Q_STOKES', 'U_STOKES'):
            v, cards = read_column(path, col)
            assert cards['ORDERING'].startswith('NEST')
            f_ring[perm] = v * 1e6                     # K -> uK
            cols[col] = f_ring.copy()
        Ct[meth] = _ring_cm(cols['I_STOKES_INP'], NSIDE, LMAX)
        for var, suf in (('inp', '_INP'), ('raw', '')):
            P = cols['Q_STOKES' + suf] + 1j * cols['U_STOKES' + suf]
            Cp[meth, var] = _ring_cm(P, NSIDE, LMAX)
            Cm[meth, var] = _ring_cm(np.conj(P), NSIDE, LMAX)
        prov[meth] = dict(file=os.path.basename(path),
                          sha256=sha256(path))
        print(f'   {meth:9s} ring FFT tables  {time.time() - t0:.0f}s')

    # stage 2: one Wigner-d table per m, applied to all maps at once
    print(f'   spin transforms to lmax {LMAX} ...')
    t0 = time.time()
    norm = np.sqrt((2 * np.arange(LMAX + 1) + 1) / (4 * np.pi))
    VARS = [(meth, var) for meth in METHODS for var in ('inp', 'raw')]
    E = {k: np.zeros((LMAX + 1, LMAX + 1), dtype=complex) for k in VARS}
    B = {k: np.zeros((LMAX + 1, LMAX + 1), dtype=complex) for k in VARS}
    T = {m: np.zeros((LMAX + 1, LMAX + 1), dtype=complex)
         for m in METHODS}
    for m in range(LMAX + 1):
        lam0 = d_stack(LMAX, m, 0, z) * norm[:, None]
        lam_p = d_stack(LMAX, m, -2, z) * norm[:, None]
        lam_m = d_stack(LMAX, m, +2, z) * norm[:, None]
        for meth in METHODS:
            T[meth][:, m] = (lam0 * Ct[meth][m + LMAX]).sum(axis=1) * dA
        for k in VARS:
            a_p2 = (lam_p * Cp[k][m + LMAX]).sum(axis=1) * dA
            a_m2 = (lam_m * Cm[k][m + LMAX]).sum(axis=1) * dA
            E[k][:, m] = -(a_p2 + a_m2) / 2
            B[k][:, m] = 1j * (a_p2 - a_m2) / 2
    print(f'   done in {time.time() - t0:.0f}s')

    # spectra + validation against the published Planck points
    ell = np.arange(LMAX + 1)
    fac = ell * (ell + 1) / (2 * np.pi)
    dls = {}
    for meth, var in VARS:
        k = (meth, var)
        dls[f'ee_{meth}_{var}'] = fac * cross_cl(E[k], E[k])
        dls[f'bb_{meth}_{var}'] = fac * cross_cl(B[k], B[k])
        dls[f'te_{meth}_{var}'] = fac * cross_cl(T[meth], E[k])
    pub = {}
    for key, name in SPECTRA.items():
        tab = np.loadtxt(os.path.join(RAW, name))
        keep = tab[:, 0] <= LMAX
        pub[key] = (tab[keep, 0].astype(int), tab[keep, 1],
                    tab[keep, 2:4].T)
    tab = np.loadtxt(os.path.join(RAW, THEORY))
    keep = tab[:, 0] <= LMAX
    th_ell = tab[keep, 0].astype(int)
    theory = dict(tt=tab[keep, 1], te=tab[keep, 2], ee=tab[keep, 3])

    lo, hi = 30, 150
    lsel = np.arange(lo, hi + 1)
    te_th = np.array([t for L, t in zip(th_ell, theory['te'])
                      if lo <= L <= hi])
    print('   TE amplitude vs LCDM theory (l=30-150) — the inpainting'
          ' canary:')
    for meth in METHODS:
        amps = {var: float(dls[f'te_{meth}_{var}'][lsel] @ te_th
                           / (te_th @ te_th)) for var in ('raw', 'inp')}
        print(f'     {meth:9s} raw {amps["raw"]:.3f}   '
              f'inpainted {amps["inp"]:.3f}')
    r = np.corrcoef(dls['te_smica_raw'][lsel], te_th)[0, 1]
    print(f'   TE shape vs theory (smica raw): r = {r:+.3f}')
    assert r > 0.5, ('TE anti-correlates with theory — '
                     'spin-2 sign convention broken')

    provenance = dict(
        note=('Planck 2018 component-separated inpainted Q/U analyzed '
              'to E/B a_lm (uK) by observatory/spin.py (Wigner-d '
              'recursion, per-ring FFT); T recomputed to same lmax; '
              'published spectra + best-fit theory bundled for '
              'validation.'),
        fetched=time.strftime('%Y-%m-%d'), files=prov)
    np.savez_compressed(
        OUT,
        ell=ell, th_ell=th_ell,
        th_tt=theory['tt'], th_te=theory['te'], th_ee=theory['ee'],
        pub_te_ell=pub['te'][0], pub_te_dl=pub['te'][1],
        pub_te_err=pub['te'][2],
        pub_ee_ell=pub['ee'][0], pub_ee_dl=pub['ee'][1],
        pub_ee_err=pub['ee'][2],
        provenance=json.dumps(provenance, indent=1),
        **{f'dl_{k}': v for k, v in dls.items()},
        **{f'almE_{m}_{v}': E[m, v] for m, v in VARS},
        **{f'almB_{m}_{v}': B[m, v] for m, v in VARS},
        **{f'almT_{m}': T[m] for m in METHODS})
    print(f'   wrote {OUT} ({os.path.getsize(OUT) >> 20} MB)')


if __name__ == '__main__':
    main()
