"""Part 8: measurement — which-path detection and the eraser.

Decoherence, told in render language: fringes live in the engine's
amplitudes. A which-path detector is anything whose state becomes
CORRELATED with the wave's route — and to correlate it must couple,
and coupling kicks the phase it measures. Average the render layer's
clicks over the detector's states and interference is gone. Nothing
collapsed. Nothing was destroyed. The coherence moved into the
correlation between wave and detector — and sorting the clicks BY the
detector's record brings the fringes back, shifted per record.

  [19] complementarity dial: detector coupling g up, visibility down,
       matching the phase-average prediction V = |<a e^{i dphi}>|.
  [20] the eraser: the g=0.3 ensemble is washed out; conditioned on
       the detector record (quartile bins), each bin shows fringes
       again. Information was never lost — this universe is unitary.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from engine.field import Field
from duality import (NY, NX, M0, WALL_X, WALL_W, SLIT_CENTERS, SLIT_HALF,
                     SCREEN_X, K_X, slit_mass, render_clicks)

DET_DEPTH = 16          # the detector cell sits in slit 1's channel
OMEGA = math.sqrt(M0 ** 2 + 4 * math.sin(K_X / 2) ** 2)
SHOTS = 48
COUPLINGS = (0.0, 0.15, 0.3, 0.5)


def run_shot(dm, t_max=210.0):
    m = slit_mass()
    c = SLIT_CENTERS[0]
    m[c - SLIT_HALF:c + SLIT_HALF + 1,
      WALL_X:WALL_X + DET_DEPTH] += dm
    f = Field(NY, NX, m)
    f.add_packet(48, 30, K_X, wy=26, wx=6)
    intensity = np.zeros(NY)
    while f.t < t_max:
        f.step(2)
        intensity += f.energy()[:, SCREEN_X]
    return intensity


def predicted_amp_phase(dm):
    """Plane-wave transmission through the detector cell: phase kick if
    still propagating, exponential suppression if evanescent."""
    m = M0 + dm
    k0 = 2 * math.asin(math.sqrt((OMEGA ** 2 - M0 ** 2) / 4))
    if m < OMEGA:
        k = 2 * math.asin(math.sqrt((OMEGA ** 2 - m ** 2) / 4))
        return 1.0, (k0 - k) * DET_DEPTH
    q = 2 * math.asinh(math.sqrt((m ** 2 - OMEGA ** 2) / 4))
    return math.exp(-q * DET_DEPTH), k0 * DET_DEPTH


def visibility(intensity, lo=34, ln=28):
    win = np.convolve(intensity, np.ones(3) / 3, mode='same')[lo:lo + ln]
    return float((win.max() - win.min()) / (win.max() + win.min()))


def panel(d, y0, intensity, label, rng, n=2600):
    clicks = render_clicks(intensity, rng, n)
    for y in clicks:
        px = 130 + int(rng.integers(0, 330))
        py = y0 + int(y * 2.2) + int(rng.integers(0, 2))
        d.point((px, py), fill=(120, 190, 235))
    curve = intensity / intensity.max()
    d.line([(478 + 120 * curve[y], y0 + y * 2.2) for y in range(NY)],
           fill=(245, 170, 90), width=2)
    d.text((8, y0 + 8), label, fill=(200, 200, 210))


def main():
    print('=' * 68)
    print('PART 8: MEASUREMENT — WHICH-PATH DETECTION AND THE ERASER')
    print('=' * 68)
    rng = np.random.default_rng(11)

    print('[19] complementarity dial (detector in slit 1, %d shots each):'
          % SHOTS)
    print(f'     {"coupling g":>10} {"V measured":>11} {"V predicted":>12}')
    ensembles = {}
    for g in COUPLINGS:
        if g == 0:
            dms = np.zeros(1)
        else:
            dms = np.abs(rng.normal(0, g, SHOTS))
        shots = [run_shot(dm) for dm in dms]
        mean_i = np.mean(shots, axis=0)
        ensembles[g] = (dms, shots, mean_i)
        aps = [predicted_amp_phase(dm) for dm in dms]
        num = abs(sum(a * complex(math.cos(p), math.sin(p))
                      for a, p in aps))
        v_pred = num / max(sum(a for a, _ in aps), 1e-12)
        print(f'     {g:>10.2f} {visibility(mean_i):>11.2f} '
              f'{v_pred:>12.2f}')
    print('     -> the more the detector state records, the harder it')
    print('        kicks the phase, the fainter the fringes. Coupling')
    print('        buys information at the price of coherence.')
    print()

    g = 0.3
    dms, shots, mean_i = ensembles[g]
    phases = np.array([predicted_amp_phase(dm)[1] for dm in dms])
    order = np.argsort(phases)
    nb = 6
    print(f'[20] the eraser (g={g}): sort the SAME clicks by the')
    print('     detector record (its predicted phase kick, 6 bins):')
    print(f'     full ensemble: V = {visibility(mean_i):.2f}')
    bin_is = []
    for b in range(nb):
        idx = order[b * len(order) // nb:(b + 1) * len(order) // nb]
        bi = np.mean([shots[i] for i in idx], axis=0)
        v = visibility(bi)
        ph = float(phases[idx].mean())
        print(f'     bin {b + 1} (mean kick {ph:4.2f} rad): V = {v:.2f}')
        if b in (0, 2, 5):
            bin_is.append((f'record bin {b + 1} '
                           f'({ph:.1f} rad kick)', bi))
    print('     -> conditioned on the record, interference returns,')
    print('        fringes shifting with the recorded kick. The')
    print('        coherence was in the wave-detector correlation all')
    print('        along: unitarity keeps its books.')

    img = Image.new('RGB', (620, 5 * int(NY * 2.2) + 60), (14, 14, 18))
    d = ImageDraw.Draw(img)
    rows = [('no detector', ensembles[0.0][2]),
            (f'detector on (g={g}), all shots', mean_i)] + bin_is
    for i, (label, inten) in enumerate(rows):
        panel(d, 10 + i * int(NY * 2.2 + 10), inten, label, rng)
    img.save('films/eraser.png')
    print('\n     films/eraser.png')


if __name__ == '__main__':
    main()
