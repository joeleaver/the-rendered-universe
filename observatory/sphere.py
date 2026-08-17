"""A spherical-harmonic laboratory, self-contained and self-validated.

Maps live on a Gauss-Legendre x uniform-longitude grid, so the
forward transform is EXACT for band-limited skies (Gauss quadrature in
cos(theta), FFT in phi). Normalized associated Legendre functions come
from the standard stable recursion. Everything is validated at import:
Y orthonormality and analysis/synthesis round-trip must hold to 1e-10
or we refuse to run.
"""
import numpy as np

LMAX = 24
NTHETA = LMAX + 6
NPHI = 2 * LMAX + 8


class Grid:
    def __init__(self, lmax=LMAX, ntheta=NTHETA, nphi=NPHI):
        self.lmax, self.nt, self.np_ = lmax, ntheta, nphi
        x, w = np.polynomial.legendre.leggauss(ntheta)
        self.x, self.w = x, w                      # cos(theta), weights
        self.phi = 2 * np.pi * np.arange(nphi) / nphi
        st = np.sqrt(1 - x ** 2)
        # unit vectors r[i, j, 3]
        self.r = np.stack([st[:, None] * np.cos(self.phi)[None, :],
                           st[:, None] * np.sin(self.phi)[None, :],
                           x[:, None] * np.ones(nphi)[None, :]], axis=2)
        self.area = (self.w[:, None] * (2 * np.pi / nphi)
                     * np.ones(nphi)[None, :])
        # normalized associated Legendre P[l, m, itheta]
        P = np.zeros((lmax + 1, lmax + 1, ntheta))
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
        self.P = P

    def analyze(self, f):
        """Real map -> alm[l, m] for m >= 0 (complex)."""
        g = np.fft.fft(f, axis=1) * (2 * np.pi / self.np_)
        alm = np.zeros((self.lmax + 1, self.lmax + 1), dtype=complex)
        for m in range(self.lmax + 1):
            alm[:, m] = (self.P[:, m, :] * (self.w * g[:, m])).sum(axis=1)
        return alm

    def synthesize(self, alm):
        """alm (m >= 0) -> real map."""
        g = np.zeros((self.nt, self.np_), dtype=complex)
        for m in range(self.lmax + 1):
            col = (self.P[:, m, :] * alm[:, m][:, None]).sum(axis=0)
            g[:, m] += col
            if m > 0:
                g[:, self.np_ - m] += np.conj(col)
        return np.real(np.fft.ifft(g, axis=1)) * self.np_

    def cl(self, alm):
        c = np.zeros(self.lmax + 1)
        for ell in range(self.lmax + 1):
            c[ell] = (np.abs(alm[ell, 0]) ** 2
                      + 2 * (np.abs(alm[ell, 1:ell + 1]) ** 2).sum()) \
                     / (2 * ell + 1)
        return c


GRID = Grid()

# ---- import-time validation ------------------------------------------
_rng = np.random.default_rng(0)
_alm = np.zeros((LMAX + 1, LMAX + 1), dtype=complex)
for _l in range(2, LMAX + 1):
    _alm[_l, 0] = _rng.normal()
    _alm[_l, 1:_l + 1] = (_rng.normal(size=_l)
                          + 1j * _rng.normal(size=_l)) / np.sqrt(2)
_round = GRID.analyze(GRID.synthesize(_alm))
assert np.abs(_round - _alm)[2:].max() < 1e-10, 'SHT round-trip failed'
_e = np.zeros((LMAX + 1, LMAX + 1), dtype=complex)
_e[3, 2] = 1.0
_f = GRID.synthesize(_e)   # f = 2 Re(Y32): integral of f^2 must be 2
assert abs((GRID.area * _f ** 2).sum() - 2.0) < 1e-9, \
    'Y normalization failed'


# ---- statistics -------------------------------------------------------

def full_m(alm, ell):
    """Coefficients for m = -ell..ell from the m>=0 store (real map)."""
    b = np.zeros(2 * ell + 1, dtype=complex)
    for m in range(ell + 1):
        b[ell + m] = alm[ell, m]
        if m > 0:
            b[ell - m] = (-1) ** m * np.conj(alm[ell, m])
    return b


def preferred_axis(alm, ell):
    """de Oliveira-Costa axis: principal eigenvector of the angular
    momentum inertia tensor M_ab = <L_a f, L_b f>, built with ladder
    operators — no rotations, no search."""
    b = full_m(alm, ell)
    ms = np.arange(-ell, ell + 1)
    cp = np.sqrt(ell * (ell + 1) - ms * (ms + 1))   # L+ from m
    cm = np.sqrt(ell * (ell + 1) - ms * (ms - 1))   # L- from m
    up = np.zeros_like(b)
    up[1:] = cp[:-1] * b[:-1]
    um = np.zeros_like(b)
    um[:-1] = cm[1:] * b[1:]
    ux, uy, uz = (up + um) / 2, (up - um) / (2j), ms * b
    u = [ux, uy, uz]
    M = np.zeros((3, 3))
    for a in range(3):
        for c in range(3):
            M[a, c] = np.real(np.vdot(u[a], u[c]))
    vals, vecs = np.linalg.eigh(M)
    return vecs[:, -1]


def alignment_stat(alm):
    n2, n3 = preferred_axis(alm, 2), preferred_axis(alm, 3)
    return abs(float(n2 @ n3))


def asymmetry_stat(f, axes):
    tot = (GRID.area * f ** 2).sum()
    best = 0.0
    for n in axes:
        mask = (GRID.r @ n) > 0
        p = (GRID.area * f ** 2 * mask).sum()
        best = max(best, abs(2 * p - tot) / tot)
    return best


def parity_stat(cl):
    ell = np.arange(2, 21)
    d = ell * (ell + 1) * cl[2:21]
    return d[ell % 2 == 1].sum() / d[ell % 2 == 0].sum()


def spike_stat(cl, mu, sd):
    return float(np.abs((cl[2:] - mu[2:]) / sd[2:]).max())


def fib_axes(n=160):
    k = np.arange(n)
    z = 1 - (k + 0.5) * 2 / n
    r = np.sqrt(1 - z ** 2)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
