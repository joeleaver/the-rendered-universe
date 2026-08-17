"""Raw spacecraft bytes -> a_lm the battery can eat.

Downloads (if absent) the four Planck 2018 component-separated CMB
temperature maps (inpainted, Nside 2048, NESTED, Galactic, K_CMB),
the WMAP 9-year ILC map (Nside 512, NESTED, mK), and the Planck 2018
best-fit LCDM theory spectrum; reads them with the hand-rolled
FITS/HEALPix code in observatory/healpix.py; reorders NESTED -> RING;
analyzes each map to a_lm (lmax 48) in microkelvin; and writes the
tiny result to data/realsky_alm.npz with full provenance (URL and
SHA-256 of every raw byte stream).

The npz is committed so part 19 (firstlight.py) runs offline; this
script exists so nobody has to trust it.

    python3 tools/fetch_sky.py            # uses data/raw/, ~7 GB
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observatory.healpix import read_column, nest2ring, map2alm  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'raw')
OUT = os.path.join(os.path.dirname(RAW), 'realsky_alm.npz')
LMAX = 48

IRSA = ('https://irsa.ipac.caltech.edu/data/Planck/release_3/'
        'all-sky-maps/maps/component-maps/cmb/')
LAMBDA = 'https://lambda.gsfc.nasa.gov/data/map/dr5/dfp/ilc/'
COSMO = ('https://irsa.ipac.caltech.edu/data/Planck/release_3/'
         'ancillary-data/cosmoparams/')
THEORY = ('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-'
          'minimum-theory_R3.01.txt')

SOURCES = {m: (IRSA, f'COM_CMB_IQU-{m}_2048_R3.00_full.fits')
           for m in ('smica', 'commander', 'nilc', 'sevem')}
SOURCES['wmap9'] = (LAMBDA, 'wmap_ilc_9yr_v5.fits')
SOURCES['theory'] = (COSMO, THEORY)
SOURCES['tt_measured'] = (COSMO, 'COM_PowerSpect_CMB-TT-full_R3.01.txt')


def fetch(base, name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        print(f'   downloading {name} ...')
        urllib.request.urlretrieve(base + name, path)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(RAW, exist_ok=True)
    alms, prov = {}, {}
    for key, (base, name) in SOURCES.items():
        path = fetch(base, name)
        prov[key] = dict(url=base + name, sha256=sha256(path),
                         bytes=os.path.getsize(path))
        if key in ('theory', 'tt_measured'):
            continue
        column = 'I_STOKES_INP' if key != 'wmap9' else 'TEMPERATURE'
        to_uK = 1e6 if key != 'wmap9' else 1e3
        t0 = time.time()
        col, cards = read_column(path, column)
        nside = cards['NSIDE']
        assert cards['ORDERING'].startswith('NEST'), cards['ORDERING']
        assert np.isfinite(col).all() and np.abs(col).max() * to_uK < 1e5, \
            'map values implausible'
        f_ring = np.empty(col.size)
        f_ring[nest2ring(nside)] = col * to_uK
        alms[key] = map2alm(f_ring, nside, LMAX)
        c2 = np.sum(np.abs(alms[key][2, :3]) ** 2
                    * np.array([1, 2, 2])) / 5
        d2 = 6 * c2 / (2 * np.pi)
        print(f'   {key:9s} nside {nside}  ->  a_lm (lmax {LMAX})  '
              f'[D_2 ~ {d2:.0f} uK^2]  {time.time() - t0:.1f}s')
    tab = np.loadtxt(os.path.join(RAW, THEORY))
    ell = tab[:, 0].astype(int)
    keep = ell <= 64
    meas = np.loadtxt(os.path.join(RAW, SOURCES['tt_measured'][1]))
    mell = meas[:, 0].astype(int)
    mkeep = mell <= 64
    provenance = dict(
        note=('Planck 2018 component-separated CMB temperature maps '
              '(I_STOKES_INP: inpainted, Galactic coords) and WMAP '
              '9yr ILC, analyzed to a_lm in uK by '
              'observatory/healpix.py (nested->ring, per-ring FFT, '
              'unit-weight quadrature); theory = Planck 2018 best-fit '
              'LCDM D_l in uK^2.'),
        fetched=time.strftime('%Y-%m-%d'), files=prov)
    np.savez_compressed(
        OUT,
        ell_theory=ell[keep], dl_theory=tab[keep, 1],
        ell_tt=mell[mkeep], dl_tt=meas[mkeep, 1],
        dl_tt_err=meas[mkeep, 2:4].T,
        provenance=json.dumps(provenance, indent=1),
        **{f'alm_{k}': v for k, v in alms.items()})
    print(f'   wrote {OUT} ({os.path.getsize(OUT)} bytes)')


if __name__ == '__main__':
    main()
