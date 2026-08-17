"""Reading the real sky from raw bytes: FITS + HEALPix, hand-rolled.

No astropy, no healpy. A FITS binary table is 2880-byte header blocks
of 80-character cards followed by big-endian rows; a HEALPix sphere is
12 nside^2 equal-area pixels on 4 nside - 1 iso-latitude rings. Both
are simple enough to own — and owning them means the chain from
spacecraft bytes to a_lm has no step this repository cannot inspect.

Everything is validated at import: the nested->ring map must be a
bijection whose neighbors stay neighbors, pixel centers must integrate
low multipoles to zero, and analyze(synthesize(a)) must return a.
The spherical-harmonic conventions match observatory/sphere.py exactly,
so real-sky a_lm feed the part-14 battery unchanged.
"""
import numpy as np

# ---- FITS binary tables ----------------------------------------------

_TFORM_DTYPE = {'E': '>f4', 'D': '>f8', 'J': '>i4', 'K': '>i8',
                'I': '>i2', 'B': 'u1', 'L': 'u1', 'A': 'S1'}


def _read_header(fh):
    """Read one FITS header at the current position. Returns (cards
    dict, data offset just past the padded header)."""
    cards = {}
    while True:
        block = fh.read(2880)
        if len(block) < 2880:
            return None
        done = False
        for i in range(0, 2880, 80):
            card = block[i:i + 80].decode('ascii', 'replace')
            key = card[:8].strip()
            if key == 'END':
                done = True
                break
            if card[8:10] != '= ':
                continue
            val = card[10:].split('/')[0].strip()
            if val.startswith("'"):
                cards[key] = val.strip("'").strip()
            elif val in ('T', 'F'):
                cards[key] = (val == 'T')
            else:
                try:
                    cards[key] = int(val)
                except ValueError:
                    try:
                        cards[key] = float(val)
                    except ValueError:
                        cards[key] = val
        if done:
            return cards


def fits_tables(path):
    """Walk the HDUs. Returns a list of (cards, data_offset) for every
    binary-table extension."""
    out = []
    with open(path, 'rb') as fh:
        while True:
            cards = _read_header(fh)
            if cards is None:
                break
            offset = fh.tell()
            nbytes = 0
            if cards.get('NAXIS', 0) > 0:
                nbytes = cards.get('NAXIS1', 1) * cards.get('NAXIS2', 1)
                nbytes += cards.get('PCOUNT', 0)
            if cards.get('XTENSION', '') == 'BINTABLE':
                out.append((cards, offset))
            fh.seek(offset + ((nbytes + 2879) // 2880) * 2880)
    return out


def read_column(path, ttype):
    """Extract one named column from whichever binary table holds it,
    as a native-endian float64 array (vector columns flattened)."""
    for cards, offset in fits_tables(path):
        names, dtypes = [], []
        found = False
        for i in range(1, cards['TFIELDS'] + 1):
            name = cards[f'TTYPE{i}'].strip()
            tform = cards[f'TFORM{i}'].strip()
            rep = int(tform[:-1]) if tform[:-1] else 1
            code = tform[-1]
            names.append(name)
            dtypes.append((name, _TFORM_DTYPE[code], (rep,)))
            found = found or (name == ttype)
        if not found:
            continue
        rows = np.memmap(path, dtype=np.dtype(dtypes), mode='r',
                         offset=offset, shape=(cards['NAXIS2'],))
        col = np.asarray(rows[ttype], dtype=np.float64).ravel()
        return col, cards
    raise KeyError(f'{ttype} not found in {path}')


# ---- HEALPix geometry ------------------------------------------------

_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])


def ring_info(nside):
    """Per-ring geometry: cos(theta), first-center phi, pixel count,
    start index — rings jr = 1 .. 4 nside - 1, north to south."""
    jr = np.arange(1, 4 * nside)
    nr = np.minimum(np.minimum(jr, 4 * nside - jr), nside)
    cap_n, cap_s = jr < nside, jr > 3 * nside
    z = (2 * nside - jr) * 2.0 / (3 * nside)
    z = np.where(cap_n, 1 - jr ** 2 / (3.0 * nside ** 2), z)
    z = np.where(cap_s, -1 + (4 * nside - jr) ** 2 / (3.0 * nside ** 2), z)
    kshift = np.where(cap_n | cap_s, 0, (jr - nside) & 1)
    phi0 = (1 - kshift / 2.0) * (np.pi / (2 * nr))  # center of pixel j=0
    # start indices
    npix = 12 * nside ** 2
    ncap = 2 * nside * (nside - 1)
    start = ncap + (jr - nside) * 4 * nside
    start = np.where(cap_n, 2 * jr * (jr - 1), start)
    start = np.where(cap_s, npix - 2 * nr * (nr + 1), start)
    return z, phi0, 4 * nr, start


def _compact_bits(v):
    """Keep the even-position bits of v, packed (Morton decode)."""
    v = v & 0x5555555555555555
    v = (v | (v >> 1)) & 0x3333333333333333
    v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FF
    v = (v | (v >> 8)) & 0x0000FFFF0000FFFF
    v = (v | (v >> 16)) & 0x00000000FFFFFFFF
    return v


def nest2ring(nside, p=None):
    """Ring index for every nested index (vectorized). With p omitted,
    the full permutation: ring_map[nested] = ring."""
    if p is None:
        p = np.arange(12 * nside ** 2, dtype=np.int64)
    p = np.asarray(p, dtype=np.int64)
    face = p // (nside * nside)
    within = p - face * nside * nside
    ix = _compact_bits(within)
    iy = _compact_bits(within >> 1)
    jr = _JRLL[face] * nside - ix - iy - 1
    nr = np.minimum(np.minimum(jr, 4 * nside - jr), nside)
    cap = (jr < nside) | (jr > 3 * nside)
    kshift = np.where(cap, 0, (jr - nside) & 1)
    n = _JPLL[face] * nr + ix - iy + 1 + kshift
    jp = (n + 8 * nr) // 2          # 1..4nr after wrap; +8nr keeps n > 0
    jp = 1 + (jp - 1) % (4 * nr)
    npix = 12 * nside ** 2
    ncap = 2 * nside * (nside - 1)
    start = ncap + (jr - nside) * 4 * nside
    start = np.where(jr < nside, 2 * jr * (jr - 1), start)
    start = np.where(jr > 3 * nside, npix - 2 * nr * (nr + 1), start)
    return start + jp - 1


def pix2ang_ring(nside, p):
    """(cos(theta), phi) of ring-ordered pixel centers."""
    p = np.asarray(p, dtype=np.int64)
    z, phi0, count, start = ring_info(nside)
    r = np.searchsorted(start, p, side='right') - 1
    j = p - start[r]
    return z[r], phi0[r] + j * 2 * np.pi / count[r]


# ---- spherical-harmonic analysis on HEALPix rings --------------------

def plm_ring(lmax, x):
    """Normalized associated Legendre P[l, m, i] at x = cos(theta) —
    the same recursion (and sign convention) as observatory/sphere.py."""
    x = np.asarray(x, dtype=np.float64)
    st = np.sqrt(np.maximum(0.0, 1 - x ** 2))
    P = np.zeros((lmax + 1, lmax + 1, x.size))
    P[0, 0] = np.sqrt(1 / (4 * np.pi))
    for m in range(1, lmax + 1):
        P[m, m] = -np.sqrt(1 + 1 / (2 * m)) * st * P[m - 1, m - 1]
    for m in range(lmax):
        P[m + 1, m] = x * np.sqrt(2 * m + 3) * P[m, m]
    for m in range(lmax + 1):
        for ell in range(m + 2, lmax + 1):
            a = np.sqrt((4 * ell ** 2 - 1) / (ell ** 2 - m ** 2))
            b = np.sqrt(((ell - 1) ** 2 - m ** 2)
                        / (4 * (ell - 1) ** 2 - 1))
            P[ell, m] = a * (x * P[ell - 1, m] - b * P[ell - 2, m])
    return P


def map2alm(f_ring, nside, lmax):
    """a_lm of a ring-ordered HEALPix map by per-ring FFT plus
    unit-weight quadrature (exact to ~(l/nside)^2 — at nside 2048 and
    l <= 48, parts in 10^5). Convention matches Grid.analyze."""
    z, phi0, count, start = ring_info(nside)
    dA = 4 * np.pi / (12 * nside ** 2)
    nring = z.size
    C = np.zeros((lmax + 1, nring), dtype=complex)
    for r in range(nring):
        F = np.fft.fft(f_ring[start[r]:start[r] + count[r]])
        m = np.arange(lmax + 1)
        C[:, r] = F[m % count[r]] * np.exp(-1j * m * phi0[r])
    P = plm_ring(lmax, z)
    alm = np.zeros((lmax + 1, lmax + 1), dtype=complex)
    for m in range(lmax + 1):
        alm[:, m] = (P[:, m, :] * C[m]).sum(axis=1) * dA
    return alm


def synth_at(alm, z, phi):
    """Direct (slow) synthesis of a real field at arbitrary points —
    the validation oracle for every fast path."""
    lmax = alm.shape[0] - 1
    P = plm_ring(lmax, np.atleast_1d(z))
    phi = np.atleast_1d(phi)
    f = np.zeros(phi.size)
    for m in range(lmax + 1):
        g = (P[:, m, :] * alm[:, m][:, None]).sum(axis=0)
        w = 1.0 if m == 0 else 2.0
        f += w * np.real(g * np.exp(1j * m * phi))
    return f


# ---- import-time validation ------------------------------------------

def _validate():
    rng = np.random.default_rng(7)
    for nside in (4, 16):
        npix = 12 * nside ** 2
        r_of_n = nest2ring(nside)
        assert np.array_equal(np.sort(r_of_n), np.arange(npix)), \
            'nest->ring is not a bijection'
        # nested neighbors (lowest-bit siblings) must be adjacent on sky
        z, phi = pix2ang_ring(nside, r_of_n)
        v = np.stack([np.sqrt(1 - z ** 2) * np.cos(phi),
                      np.sqrt(1 - z ** 2) * np.sin(phi), z], axis=1)
        sib = np.arange(npix) ^ 1
        gap = np.linalg.norm(v - v[sib], axis=1)
        assert gap.max() < 4.0 / nside, 'nested siblings not adjacent'
    # pixel centers must integrate low multipoles to ~zero: a constant
    # sky has exactly one nonzero coefficient. Unit-weight HEALPix
    # quadrature leaks into even-l m=0 at O(1/nside^2) — check the
    # level AND the convergence rate (so nside 2048 sits at ~1e-7).
    leak = {}
    for nside in (4, 16):
        quad = map2alm(np.ones(12 * nside ** 2), nside, nside)
        assert abs(quad[0, 0] - np.sqrt(4 * np.pi)) < 1e-12, \
            'quadrature broken (monopole)'
        quad[0, 0] = 0.0
        leak[nside] = np.abs(quad).max() / np.sqrt(4 * np.pi)
    assert leak[4] < 0.03 and leak[16] < 3e-3, 'quadrature leakage'
    assert leak[16] < leak[4] / 8, 'quadrature not converging'
    # round trip: random band-limited sky, synthesized on rings,
    # analyzed back — conventions pinned against synth_at, error
    # falling as 1/nside^2 (so the real maps sit at ~4e-7)
    lmax = 8
    alm = np.zeros((lmax + 1, lmax + 1), dtype=complex)
    for ell in range(lmax + 1):
        alm[ell, 0] = rng.normal()
        alm[ell, 1:ell + 1] = (rng.normal(size=ell)
                               + 1j * rng.normal(size=ell)) / np.sqrt(2)
    err = {}
    for nside in (32, 64):
        z, phi = pix2ang_ring(nside, np.arange(12 * nside ** 2))
        back = map2alm(synth_at(alm, z, phi), nside, lmax)
        err[nside] = np.abs(back - alm).max()
    assert err[32] < 5e-3, f'HEALPix round-trip failed ({err[32]:.1e})'
    assert err[64] < err[32] / 3, 'HEALPix round-trip not converging'


_validate()
