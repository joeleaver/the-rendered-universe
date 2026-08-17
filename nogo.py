"""Part 10c: the boss fight's arena, mapped.

We cannot derive quantum mechanics from a classical engine (nobody
can). What we CAN do is show exactly where classical engines die:

  [28] the interference deficit. Any engine in which each particle
       takes ONE slit, and slits don't act on each other at a
       distance, predicts P_both(y) = P_1(y) + P_2(y): opening a
       second slit can never DECREASE arrivals anywhere. Run the wave
       engine three times (both slits, slit 1 only, slit 2 only): at
       the interference minima, arrivals with both slits open fall
       far BELOW either single slit alone. Amplitudes cancel;
       probabilities cannot.

  [29] the amplitude wall. If the engine stores the quantum state
       classically, n entangled qubits cost 2^n amplitudes. Measure
       the wall directly: simulate random circuits and time them.
"""
import math
import time

import numpy as np
from PIL import Image, ImageDraw

from engine.field import Field
from duality import (NY, NX, M0, WALL_X, WALL_W, WALL_M, SLIT_CENTERS,
                     SLIT_HALF, SCREEN_X, K_X)

# validated dark-mode categorical palette (dataviz slots 1-2)
C_BOTH, C_CLASS = (57, 135, 229), (217, 89, 38)
INK, MUTED, GRID_C = (195, 194, 183), (122, 122, 130), (38, 38, 44)


def slit_intensity(open_slits, t_max=220.0):
    m = np.full((NY, NX), M0)
    wall = np.ones(NY, dtype=bool)
    for i in open_slits:
        c = SLIT_CENTERS[i]
        wall[c - SLIT_HALF:c + SLIT_HALF + 1] = False
    m[wall, WALL_X:WALL_X + WALL_W] = WALL_M
    f = Field(NY, NX, m)
    f.add_packet(48, 30, K_X, wy=26, wx=6)
    intensity = np.zeros(NY)
    while f.t < t_max:
        f.step(2)
        intensity += f.energy()[:, SCREEN_X]
    return intensity


def chart(i_both, i_sum, path):
    W, H, ml, mr, mt, mb = 1000, 480, 70, 30, 54, 50
    img = Image.new('RGB', (W, H), (14, 14, 18))
    d = ImageDraw.Draw(img)
    top = max(i_sum.max(), i_both.max()) * 1.05

    def xy(y, v):
        return (ml + (W - ml - mr) * y / (NY - 1),
                H - mb - (H - mt - mb) * v / top)

    for fr in (0.25, 0.5, 0.75, 1.0):
        yy = xy(0, top * fr / 1.05)[1]
        d.line([(ml, yy), (W - mr, yy)], fill=GRID_C)
    d.text((10, 12), 'arrivals at the screen (time-integrated)', fill=INK)
    d.text((W // 2 - 60, H - 24), 'screen position', fill=MUTED)
    for series, color in ((i_sum, C_CLASS), (i_both, C_BOTH)):
        d.line([xy(y, v) for y, v in enumerate(series)],
               fill=color, width=2)
    for label, color, yy in (
            ('classical bound: slit1 + slit2', C_CLASS, mt + 4),
            ('measured, both slits open', C_BOTH, mt + 24)):
        d.line([(ml + 12, yy + 6), (ml + 38, yy + 6)], fill=color, width=3)
        d.text((ml + 46, yy), label, fill=INK)
    img.save(path)


def apply_1q(psi, U, q, n):
    psi = psi.reshape(2 ** (n - q - 1), 2, 2 ** q)
    return np.einsum('ij,ajb->aib', U, psi).reshape(-1)


def apply_cnot(psi, c, t, n):
    idx = np.arange(len(psi))
    sel = ((idx >> c) & 1) == 1
    out = psi.copy()
    out[idx[sel]] = psi[idx[sel] ^ (1 << t)]
    return out


def circuit_time(n, depth=6, seed=1):
    rng = np.random.default_rng(seed)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    t0 = time.perf_counter()
    for layer in range(depth):
        for q in range(n):
            m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            U, _ = np.linalg.qr(m)
            psi = apply_1q(psi, U, q, n)
        for q in range(layer % 2, n - 1, 2):
            psi = apply_cnot(psi, q, q + 1, n)
    return time.perf_counter() - t0


def main():
    print('=' * 68)
    print('PART 10c: WHERE CLASSICAL ENGINES DIE')
    print('=' * 68)
    print('[28] the interference deficit (double slit, three runs):')
    i1 = slit_intensity((0,))
    i2 = slit_intensity((1,))
    ib = slit_intensity((0, 1))
    lo, hi = 30, 66
    ratio = ib[lo:hi] / np.maximum(i1[lo:hi] + i2[lo:hi], 1e-12)
    y_min = lo + int(np.argmin(ratio))
    below_single = int(((ib[lo:hi] < i1[lo:hi])
                        & (ib[lo:hi] < i2[lo:hi])).sum())
    print(f'     deepest deficit at screen y={y_min}: opening BOTH slits')
    print(f'     delivers only {ratio.min():.1%} of slit1+slit2 there —')
    print(f'     and at {below_single} of {hi - lo} central positions,')
    print(f'     both-open arrivals fall below EACH single slit alone.')
    print('     Classical exclusive-path engines are bounded below by')
    print('     P1 + P2 everywhere. The render layer needs amplitudes')
    print('     that cancel — negative numbers under the probabilities.')
    chart(ib, i1 + i2, 'films/nogo.png')
    print('     -> films/nogo.png')
    print()

    print('[29] the amplitude wall (classical cost of n entangled qubits):')
    print(f'     {"qubits":>7} {"amplitudes":>12} {"seconds":>9}')
    times = {}
    for n in range(10, 23, 2):
        times[n] = circuit_time(n)
        print(f'     {n:>7} {2 ** n:>12,} {times[n]:>9.3f}')
    doubling = (times[22] / times[14]) ** (1 / 8)
    t100 = times[22] * doubling ** (100 - 22)
    print(f'     cost multiplies by ~{doubling:.2f} per added qubit;')
    print(f'     at that rate, 100 qubits would take ~{t100 / 3.15e7:.1e} '
          f'years.')
    print('     A classical engine rendering our universe pays this for')
    print('     every entangled system it draws. Either the engine is')
    print('     quantum, or quantum computers will hit a ceiling —')
    print('     and the fabs are running that experiment right now.')


if __name__ == '__main__':
    main()
