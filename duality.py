"""Part 7: wave/particle duality, read as render architecture.

In this program's language the old paradox is a type distinction:

    the WAVE is the engine object   — the amplitude field, the thing
                                      the mechanism actually evolves;
    the PARTICLE is the render event — the discrete pixel the screen
                                      writes when a detector fires.

Asking 'is it REALLY a wave or a particle?' is asking whether the
universe is REALLY the mechanism or the output. It is a wave in the
engine and a particle on the screen, and there is no third fact.

The demo is the double slit:

  [18] the engine propagates ONE field through BOTH slits (that is
       just what wave dynamics does — no mystery, nothing to explain);
       the renderer samples discrete clicks from the arriving
       intensity (Born rule as the render rule, put in by hand as in
       part 2 — the demonstration is architectural). Dot by dot, the
       interference pattern assembles.

Honest note, as always: we do not derive the Born rule here; we show
that 'wave computes, particle renders' reproduces what detectors see.
Real physics agrees particles are the detector-relative layer: in QFT
particle NUMBER is observer-dependent (Unruh effect), while the field
is the invariant object.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from engine.field import Field

NY, NX = 96, 192
M0 = 0.2
WALL_X, WALL_W, WALL_M = 68, 4, 3.0
SLIT_CENTERS, SLIT_HALF = (34, 62), 3
SCREEN_X = 125
K_X = 2 * math.pi / 6
N_CLICKS = 6000


def slit_mass():
    m = np.full((NY, NX), M0)
    wall = np.zeros(NY, dtype=bool)
    wall[:] = True
    for c in SLIT_CENTERS:
        wall[c - SLIT_HALF:c + SLIT_HALF + 1] = False
    m[wall, WALL_X:WALL_X + WALL_W] = WALL_M
    return m


def run_slits(t_max=220.0, snap_t=95.0):
    f = Field(NY, NX, slit_mass())
    f.add_packet(48, 30, K_X, wy=26, wx=6)
    intensity = np.zeros(NY)
    snap = None
    while f.t < t_max:
        f.step(2)
        intensity += f.energy()[:, SCREEN_X]
        if snap is None and f.t >= snap_t:
            snap = f.phi.copy()
    return intensity, snap


def fringe_spacing(intensity):
    """Distance between interference maxima on the screen."""
    peaks = [y for y in range(6, NY - 6)
             if intensity[y] == intensity[y - 4:y + 5].max()
             and intensity[y] > 0.25 * intensity.max()]
    gaps = np.diff(peaks)
    return float(np.median(gaps)) if len(gaps) else float('nan'), peaks


def render_clicks(intensity, rng, n):
    p = intensity / intensity.sum()
    return rng.choice(NY, size=n, p=p)


def film_buildup(clicks, path, per_frame=50):
    """Tonomura-style: the pattern assembles one detection at a time."""
    scale = 5
    w = 220
    rng = np.random.default_rng(1)
    xs = rng.integers(6, w - 6, size=len(clicks))
    screen = np.full((NY * scale, w, 3), 16.0)
    frames = []
    for i, (y, x) in enumerate(zip(clicks, xs)):
        cy = y * scale + int(rng.integers(0, scale))
        screen[max(cy - 1, 0):cy + 2, max(x - 1, 0):x + 2] += (
            95, 165, 205)
        if (i + 1) % per_frame == 0:
            img = Image.fromarray(
                np.clip(screen, 0, 255).astype(np.uint8), 'RGB')
            d = ImageDraw.Draw(img)
            d.text((8, 6), f'{i + 1} detections', fill=(200, 200, 210))
            frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=50, loop=0)


def render_still(snap, intensity, clicks, path):
    scale = 4
    a = np.clip(np.abs(snap) / 0.5, 0, 1)
    pos = snap > 0
    img = np.zeros((NY, NX, 3))
    img[..., 0] = np.where(pos, 235 * a, 80 * a) + 14
    img[..., 1] = np.where(pos, 160 * a, 130 * a) + 14
    img[..., 2] = np.where(pos, 80 * a, 230 * a) + 18
    m = slit_mass()
    img[m > 1] = (150, 150, 160)
    img[:, SCREEN_X] = np.maximum(img[:, SCREEN_X], 50)
    img = np.clip(img, 0, 255).astype(np.uint8)
    big = Image.fromarray(np.kron(img, np.ones((scale, scale, 1),
                                               dtype=np.uint8)), 'RGB')
    canvas = Image.new('RGB', (NX * scale + 360, NY * scale), (14, 14, 18))
    canvas.paste(big, (0, 0))
    d = ImageDraw.Draw(canvas)
    # detection panel: final clicks + the wave's intensity curve
    x0 = NX * scale + 20
    rng = np.random.default_rng(1)
    for y in clicks[:3000]:
        px = x0 + int(rng.integers(0, 250))
        py = y * scale + int(rng.integers(0, scale))
        d.point((px, py), fill=(120, 190, 235))
    curve = intensity / intensity.max()
    pts = [(x0 + 250 + 90 * curve[y], y * scale) for y in range(NY)]
    d.line(pts, fill=(245, 170, 90), width=2)
    d.text((x0, 8), 'clicks (render)', fill=(150, 200, 240))
    d.text((x0 + 240, 8), '|wave|^2 (engine)', fill=(245, 170, 90))
    canvas.save(path)


def main():
    print('=' * 68)
    print('PART 7: WAVE/PARTICLE DUALITY AS RENDER ARCHITECTURE')
    print('=' * 68)
    intensity, snap = run_slits()
    spacing, peaks = fringe_spacing(intensity)
    lam = 2 * math.pi / K_X
    d_slits = SLIT_CENTERS[1] - SLIT_CENTERS[0]
    L = SCREEN_X - WALL_X
    print(f'[18] the double slit (wavelength {lam:.0f}, slit separation '
          f'{d_slits}, screen at L={L}):')
    print(f'     wave optics predicts fringe spacing lam*L/d = '
          f'{lam * L / d_slits:.1f} cells')
    print(f'     engine wave shows {len(peaks)} maxima, median spacing '
          f'{spacing:.1f} cells')
    rng = np.random.default_rng(7)
    clicks = render_clicks(intensity, rng, N_CLICKS)
    print(f'     renderer emitted {N_CLICKS} discrete clicks from the')
    print(f'     arriving intensity; their histogram IS the fringe pattern.')
    print()
    print('     One field went through both slits (engine fact: waves')
    print('     do that). Each detection is one pixel event (render')
    print('     fact: screens do that). The "paradox" dissolves into')
    print('     the type distinction this whole repo is built on.')
    film_buildup(clicks, 'films/duality_buildup.gif')
    render_still(snap, intensity, clicks, 'films/duality.png')
    print('\n     films/duality_buildup.gif, films/duality.png')


if __name__ == '__main__':
    main()
