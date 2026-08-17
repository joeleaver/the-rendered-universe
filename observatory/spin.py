"""Spin-2 spherical harmonics: polarization read the same way we read
temperature — from first principles, validated at import.

Polarization is a spin-2 field: the Stokes pair (Q, U) transforms
under rotation of the local basis, so scalar harmonics cannot expand
it. The right basis functions are spin-weighted harmonics, and those
are Wigner d-functions in disguise:

    sYlm(theta, phi) = sqrt((2l+1)/4pi) * d^l_{m,-s}(theta) * e^{i m phi}

We build d^l_{m,s} by the standard three-term recursion in l (the
cos-beta recurrence, seeded at l0 = max(|m|,|s|) by the closed-form
single-term Wigner expression). Both the seeds' signs and the
recursion were pinned against the matrix exponential exp(-i beta Jy)
— the same operator part 19's rotation oracle validated against
direct synthesis — so temperature and polarization share one
convention chain. E and B are then the parity-even and parity-odd
combinations of the spin-(+2)/spin-(-2) coefficients (healpy sign
convention: Q + iU = -Sigma (E + iB) 2Ylm).

Import-time validation: d-recursion vs matrix exponential; the s=0
column vs the scalar Legendre recursion of sphere.py/healpix.py;
E/B -> (Q,U) -> E/B round trip on HEALPix rings; B-purity of pure-E
skies; and the parity law (E -> (-1)^l E, B -> (-1)^{l+1} B under
point reflection) — the test that catches E/B mixing.
"""
import math

import numpy as np

from observatory.healpix import ring_info, plm_ring


# ---- Wigner d columns -------------------------------------------------

def d_stack(lmax, m, mp, x):
    """d^l_{m,mp}(theta) for l = 0..lmax at x = cos(theta) (array).
    Rows below l0 = max(|m|,|mp|) are zero."""
    x = np.asarray(x, dtype=np.float64)
    c = np.sqrt((1 + x) / 2)          # cos(theta/2)
    s = np.sqrt((1 - x) / 2)          # sin(theta/2)
    l0 = max(abs(m), abs(mp))
    out = np.zeros((lmax + 1, x.size))
    if l0 > lmax:
        return out
    mu = mp if abs(m) >= abs(mp) else m
    lg = 0.5 * (math.lgamma(2 * l0 + 1) - math.lgamma(l0 + mu + 1)
                - math.lgamma(l0 - mu + 1))
    if abs(m) >= abs(mp):
        sign = (-1) ** ((l0 - mp) % 2) if m >= 0 else 1
    else:
        sign = (-1) ** ((l0 - m) % 2) if mp < 0 else 1
    with np.errstate(under='ignore'):
        out[l0] = (sign * np.exp(lg)
                   * c ** abs(m + mp) * s ** abs(m - mp))
    for ell in range(l0, lmax):
        A = (math.sqrt(((ell + 1) ** 2 - m ** 2)
                       * ((ell + 1) ** 2 - mp ** 2))
             / ((ell + 1) * (2 * ell + 1)))
        B = m * mp / (ell * (ell + 1)) if ell > 0 else 0.0
        C = (math.sqrt((ell ** 2 - m ** 2) * (ell ** 2 - mp ** 2))
             / (ell * (2 * ell + 1))) if ell > l0 else 0.0
        prev = out[ell - 1] if ell - 1 >= l0 else 0.0
        out[ell + 1] = ((x - B) * out[ell] - C * prev) / A
    return out


# ---- spin-2 analysis on HEALPix rings --------------------------------

def _ring_cm(f_ring, nside, lmax):
    """Per-ring azimuthal coefficients of a (possibly complex) map:
    C[j, r] = sum_pix f e^{-i m phi}, for m = -lmax..lmax (j = m+lmax)."""
    z, phi0, count, start = ring_info(nside)
    ms = np.arange(-lmax, lmax + 1)
    C = np.zeros((2 * lmax + 1, z.size), dtype=complex)
    for r in range(z.size):
        F = np.fft.fft(f_ring[start[r]:start[r] + count[r]])
        C[:, r] = F[ms % count[r]] * np.exp(-1j * ms * phi0[r])
    return C


def map2alm_spin2(q_ring, u_ring, nside, lmax):
    """(Q, U) HEALPix ring maps -> (E_lm, B_lm), m >= 0 store, in the
    convention of Grid.analyze / healpix.map2alm for scalars."""
    z = ring_info(nside)[0]
    dA = 4 * np.pi / (12 * nside ** 2)
    Cp = _ring_cm(q_ring + 1j * u_ring, nside, lmax)   # (Q+iU)
    Cm = _ring_cm(q_ring - 1j * u_ring, nside, lmax)   # (Q-iU)
    E = np.zeros((lmax + 1, lmax + 1), dtype=complex)
    B = np.zeros((lmax + 1, lmax + 1), dtype=complex)
    norm = np.sqrt((2 * np.arange(lmax + 1) + 1) / (4 * np.pi))
    for m in range(lmax + 1):
        lam_p = d_stack(lmax, m, -2, z) * norm[:, None]   # 2Ylm theta part
        lam_m = d_stack(lmax, m, +2, z) * norm[:, None]   # -2Ylm theta part
        a_p2 = (lam_p * Cp[m + lmax]).sum(axis=1) * dA
        a_m2 = (lam_m * Cm[m + lmax]).sum(axis=1) * dA
        E[:, m] = -(a_p2 + a_m2) / 2
        B[:, m] = 1j * (a_p2 - a_m2) / 2
    return E, B


def synth_pol_at(E, B, z, phi):
    """(Q, U) of an E/B field at arbitrary points — the slow oracle."""
    lmax = E.shape[0] - 1
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    phi = np.atleast_1d(phi)
    norm = np.sqrt((2 * np.arange(lmax + 1) + 1) / (4 * np.pi))
    P = np.zeros(phi.size, dtype=complex)      # Q + iU
    for m in range(-lmax, lmax + 1):
        am = abs(m)
        lam = d_stack(lmax, m, -2, z) * norm[:, None]
        if m >= 0:
            a = -(E[:, am] + 1j * B[:, am])
        else:
            a = -((-1) ** am) * np.conj(E[:, am] - 1j * B[:, am])
        P += (lam * a[:, None]).sum(axis=0) * np.exp(1j * m * phi)
    return P.real, P.imag


# ---- import-time validation ------------------------------------------

def _validate():
    rng = np.random.default_rng(20)
    # (1) d-recursion against the matrix exponential of Jy
    for beta in (0.4, 1.7):
        x = np.array([math.cos(beta)])
        for ell in (2, 5, 8):
            mm = np.arange(-ell, ell + 1)
            cp = np.sqrt(ell * (ell + 1) - mm[:-1] * (mm[:-1] + 1))
            Jp = np.zeros((2 * ell + 1, 2 * ell + 1), dtype=complex)
            Jp[np.arange(1, 2 * ell + 1), np.arange(2 * ell)] = cp
            w, V = np.linalg.eigh((Jp - Jp.conj().T) / 2j)
            D = (V @ np.diag(np.exp(-1j * beta * w)) @ V.conj().T).real
            for m in (-ell, -1, 0, 2, ell):
                for mp in (-2, 0, 2):
                    got = d_stack(ell, m, mp, x)[ell, 0]
                    assert abs(got - D[m + ell, mp + ell]) < 1e-12, \
                        'wigner-d recursion broken'
    # (2) s = 0 must reproduce the scalar Legendre functions
    xs = np.linspace(-0.95, 0.95, 7)
    P = plm_ring(8, xs)
    for m in range(9):
        lam = d_stack(8, m, 0, xs) * np.sqrt(
            (2 * np.arange(9) + 1) / (4 * np.pi))[:, None]
        assert np.abs(lam[m:] - P[m:, m, :]).max() < 1e-12, \
            'spin-0 limit disagrees with scalar transform'
    # (3) round trip on HEALPix rings, with B-purity of a pure-E sky
    nside, lmax = 32, 8
    z, phi0, count, start = ring_info(nside)
    zs = np.repeat(z, count)
    phis = np.concatenate([p0 + 2 * np.pi * np.arange(c) / c
                           for p0, c in zip(phi0, count)])
    def rand_alm():
        a = np.zeros((lmax + 1, lmax + 1), dtype=complex)
        for ell in range(2, lmax + 1):
            a[ell, 0] = rng.normal()
            a[ell, 1:ell + 1] = (rng.normal(size=ell)
                                 + 1j * rng.normal(size=ell)) / np.sqrt(2)
        return a
    E0, B0 = rand_alm(), rand_alm()
    q, u = synth_pol_at(E0, B0, zs, phis)
    E1, B1 = map2alm_spin2(q, u, nside, lmax)
    err = max(np.abs(E1 - E0).max(), np.abs(B1 - B0).max())
    assert err < 5e-3, f'spin-2 round trip failed ({err:.1e})'
    qe, ue = synth_pol_at(E0, 0 * B0, zs, phis)
    _, Bp = map2alm_spin2(qe, ue, nside, lmax)
    assert np.abs(Bp).max() < 5e-3, 'pure-E sky leaks into B'
    # (4) parity: (Q,U)(n) -> (Q, -U)(-n) sends E -> (-1)^l E and
    # B -> (-1)^{l+1} B — the test that catches E/B mixing
    qp, up = synth_pol_at(E0, B0, -zs, (phis + np.pi) % (2 * np.pi))
    E2, B2 = map2alm_spin2(qp, -up, nside, lmax)
    par = np.array([(-1) ** ell for ell in range(lmax + 1)])
    assert np.abs(E2 - par[:, None] * E1).max() < 5e-3, \
        'E parity law broken'
    assert np.abs(B2 + par[:, None] * B1).max() < 5e-3, \
        'B parity law broken'


_validate()
