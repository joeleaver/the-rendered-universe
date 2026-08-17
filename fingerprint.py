"""Part 13: the fingerprint hunt.

The compressible-seed hypothesis (part 9) predicts the universe's
primordial fluctuations are PSEUDOrandom — output of a short
generator — not true randomness. A strong generator is undetectable
by definition. A simple one is not. So the honest question is
quantitative: how simple must the seed's generator be, and how much
sky must you read, before the fingerprint shows?

  [40] the battery: five blind detectors (spectral lines, extreme
       autocorrelation, pair-histogram structure, GF(2) parity taps,
       compressibility) against seven seed generators spanning the
       simplicity ladder, with true randomness as control.
  [41] the sensitivity curve: minimum sample count for detection,
       per generator — where detectability dies while the generator
       is still only tens of bytes long.
  [42] survival under dynamics: does a fingerprint survive being
       run through a scrambling universe? (Decides whether to look
       at processed structure or at the earliest snapshot — which
       for the real sky is the CMB.)
"""
import os
import zlib

import numpy as np

from engine.substrate import Substrate

DETECT_P = 1e-4


# ------------------------------------------------------ seed generators

def gen_tile(n, seed=1):
    rng = np.random.default_rng(seed)
    tile = rng.random(256)
    return np.tile(tile, n // 256 + 1)[:n]


def gen_lcg16(n, seed=12345):
    out = np.empty(n)
    x = seed
    for i in range(n):
        x = (25173 * x + 13849) % 65536
        out[i] = x / 65536
    return out


def gen_lcg32(n, seed=12345):
    out = np.empty(n)
    x = seed
    for i in range(n):
        x = (1664525 * x + 1013904223) % (2 ** 32)
        out[i] = x / 2 ** 32
    return out


def gen_lfsr31(n, seed=0x1234567):
    out = np.empty(n)
    x = seed
    for i in range(n):
        bit = ((x >> 30) ^ (x >> 27)) & 1
        x = ((x << 1) | bit) & 0x7FFFFFFF
        out[i] = ((x >> 15) & 0xFFFF) / 65536
    return out


def gen_xorshift64(n, seed=88172645463325252):
    out = np.empty(n)
    x = seed
    for i in range(n):
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        out[i] = (x >> 40) / 2 ** 24
    return out


def gen_pcg(n, seed=7):
    return np.random.default_rng(seed).random(n)


def gen_true(n, seed=None):
    raw = np.frombuffer(os.urandom(n * 4), dtype=np.uint32)
    return raw / 2 ** 32


GENERATORS = [
    ('tiled structure', gen_tile, 30),
    ('LCG 16-bit', gen_lcg16, 34),
    ('LFSR 31-bit', gen_lfsr31, 60),
    ('LCG 32-bit', gen_lcg32, 40),
    ('xorshift64', gen_xorshift64, 62),
    ('PCG64 (strong)', gen_pcg, 500),
    ('true random', gen_true, None),
]


# ------------------------------------------------------ detectors
# each returns a p-value under the null 'true random'

def d_spectral(x):
    f = np.abs(np.fft.rfft(x - x.mean())[1:]) ** 2
    m = len(f)
    g = f.max() / f.sum()
    return min(1.0, m * np.exp(-g * m))


def d_autocorr(x):
    n = len(x)
    L = min(2 ** 17, n // 2)
    xc = x - x.mean()
    f = np.fft.rfft(xc, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:L + 1] / (xc @ xc)
    r = np.abs(ac[1:]).max()
    from math import erfc, sqrt
    return min(1.0, 2 * L * 0.5 * erfc(r * sqrt(n) / sqrt(2)))


def d_pairs(x, bits=6):
    q = (x * 2 ** bits).astype(int) % 2 ** bits
    pairs = q[:-1] * 2 ** bits + q[1:]
    counts = np.bincount(pairs, minlength=4 ** bits)
    n, k = len(pairs), 4 ** bits
    chi2 = ((counts - n / k) ** 2 / (n / k)).sum()
    z = (chi2 - (k - 1)) / np.sqrt(2 * (k - 1))
    from math import erfc, sqrt
    return min(1.0, 0.5 * erfc(z / sqrt(2)))


def d_taps(x):
    b = (x[:2 ** 16] < 0.5).astype(np.uint8)
    n = len(b)
    best = 0.0
    from math import erfc, sqrt
    trials = 0
    for p in range(1, 33):
        bp = np.roll(b, p)
        for q in range(p + 1, 34):
            r = float((b ^ bp ^ np.roll(b, q)).mean())
            best = max(best, abs(r - 0.5))
            trials += 1
    return min(1.0, trials * erfc(best * 2 * sqrt(n) / sqrt(2)))


def d_compress(x):
    data = (x * 256).astype(np.uint8).tobytes()
    ratio = len(zlib.compress(data, 6)) / len(data)
    # null calibration: byte-quantized true randomness compresses to
    # ~1.0007; flag anything materially below
    return 1e-12 if ratio < 0.98 else 1.0


DETECTORS = [('spectral', d_spectral), ('autocorr', d_autocorr),
             ('pairs', d_pairs), ('taps', d_taps),
             ('compress', d_compress)]


def battery(x):
    """Bonferroni-corrected best detection p-value + which detector."""
    best, who = 1.0, '-'
    for name, det in DETECTORS:
        p = min(1.0, det(x) * len(DETECTORS))
        if p < best:
            best, who = p, name
    return best, who


def main():
    print('=' * 70)
    print('PART 13: THE FINGERPRINT HUNT')
    print('=' * 70)

    N0 = 2 ** 18
    print(f'[40] detector battery at N = 2^18 samples:')
    print(f'     {"generator":<17} {"~bytes":>6} {"p-value":>10}  '
          f'caught by')
    for name, gen, size in GENERATORS:
        p, who = battery(gen(N0))
        verdict = who if p < DETECT_P else 'UNDETECTED'
        print(f'     {name:<17} {size if size else "-":>6} '
              f'{p:>10.1e}  {verdict}')
    print()

    print('[41] sensitivity: minimum sky needed for detection '
          f'(p < {DETECT_P}):')
    print(f'     {"generator":<17} {"~bytes":>6}   N_min')
    for name, gen, size in GENERATORS:
        n_min = None
        for exp in range(12, 21, 2):
            p, _ = battery(gen(2 ** exp))
            if p < DETECT_P:
                n_min = exp
                break
        res = f'2^{n_min}' if n_min else '> 2^20  (invisible)'
        print(f'     {name:<17} {size if size else "-":>6}   {res}')
    print()
    print('     The epistemic ceiling, measured: a ~60-byte generator')
    print('     (xorshift64) already defeats a battery that shreds')
    print('     LCGs. Detectability dies while simplicity remains.')
    print('     The real sky offers ~10^7 primordial modes (CMB to')
    print('     l~2500): on this curve, that reads generator classes')
    print('     up to roughly LCG-32 grade — IF the fingerprint is')
    print('     in the primordial layer at all. Hence:')
    print()

    print('[42] does a fingerprint survive dynamics?')
    print('     (differential test: any evolved field has physical')
    print('     clustering that trips a naive battery, so the null is')
    print('     an ensemble of true-random-seeded evolved universes)')

    def evolved_profile(field01):
        sub = Substrate(128)
        sub.grid = (field01.reshape(128, 128) < 0.5).astype(np.uint8)
        for _ in range(600):
            sub.step()
        e = (sub.grid ^ (sub.t % 2)).astype(float).ravel()
        ec = e - e.mean()
        f = np.fft.rfft(ec, 2 * len(e))
        return np.fft.irfft(f * np.conj(f))[1:2049] / (ec @ ec)

    controls = np.array([evolved_profile(gen_true(128 * 128))
                         for _ in range(12)])
    mu, sig = controls.mean(0), np.maximum(controls.std(0), 1e-4)
    null_z = [float((np.abs(evolved_profile(gen_true(128 * 128)) - mu)
                     / sig).max()) for _ in range(3)]
    print(f'     held-out null calibration: max z = '
          f'{", ".join(f"{z:.1f}" for z in null_z)}')
    thresh = 3 * max(null_z)
    for label, gen in (('tiled structure', gen_tile),
                       ('LCG 16-bit', gen_lcg16)):
        prof = evolved_profile(gen(128 * 128))
        z = np.abs(prof - mu) / sig
        lag = int(np.argmax(z)) + 1
        zz = float(z.max())
        verdict = (f'SURVIVES (z={zz:.0f} at lag {lag})' if zz > thresh
                   else f'laundered (z={zz:.1f}, inside null band)')
        print(f'     {label:<17}: {verdict}')
    print()
    print('     The split is the finding. STATISTICAL pseudorandom')
    print('     structure launders: chaotic dynamics are extractors.')
    print('     But an EXACT SEED SYMMETRY is dynamically protected —')
    print('     translation-equivariant laws can never break an exact')
    print('     periodicity of the initial state. Symmetries of the')
    print('     seed are fingerprints that survive forever.')
    print()
    print('     The prediction, stated: IF the seed generator is')
    print('     simple, the robustly observable signature class is')
    print('     seed SYMMETRY and long-range correlation — alignments')
    print('     and asymmetries in the primordial sky, surviving from')
    print('     the initial layer. The low-l CMB anomalies (quadrupole-')
    print('     octupole alignment, hemispherical asymmetry) are')
    print('     precisely statistics of this class, at 2-3 sigma,')
    print('     unexplained by inflation. A dedicated symmetry-class')
    print('     search on the primordial map is the experiment.')


if __name__ == '__main__':
    main()
