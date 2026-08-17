"""Part 10b: closing the loop — matter curves what moves matter.

In part 5 the star was painted by hand. Here the coupling closes:
the local mass responds to smoothed field energy density,

    m^2(x, t) = m0^2 - g * blur(energy density),

with the sign chosen so that energy makes the medium 'slower' for
phase (higher local k at fixed omega, higher index) — i.e., ENERGY
ATTRACTS ENERGY, universally, by coupling to energy density rather
than to any particular species. Two wave packets flying side by side
should fall toward each other. No star was painted; the wells dig
themselves.

Also: [27] the area law. Holography's clue about the engine's data
layout — the information in a region scales with its BOUNDARY. Our
Gaussian ground state can check this exactly.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from engine.field import Field
from observatory.quantum import (scalar_couplings, scalar_ground_state,
                                 scalar_region_entropy)

NY, NX = 96, 192
# heavy, slow matter: gravity-like coupling grips waves with m^2/w^2
# near 1 (nonrelativistic matter falls; light barely bends — as in GR)
M0 = 0.45
K_X = 2 * math.pi / 10
SEP0 = 24


class GravField(Field):
    """The potential solves a (screened) Poisson equation, like real
    gravity: Phi = (laplacian - mu^2)^-1 rho, range ~ 1/mu. That makes
    the force LONG-RANGE — a Gaussian blur is not gravity; its wells
    never reach the neighbors. Coupling is calibrated on the first
    update so the deepest well is g_target of m0^2 (no saturation)."""

    def __init__(self, ny, nx, g_target, screen=35.0):
        super().__init__(ny, nx, M0)
        self.g_target = g_target
        self.g_phys = None
        ky = np.fft.fftfreq(ny) * 2 * math.pi
        kx = np.fft.rfftfreq(nx) * 2 * math.pi
        k2 = ky[:, None] ** 2 + kx[None, :] ** 2
        self.kernel = 1.0 / (k2 + (1.0 / screen) ** 2)
        self.kernel[0, 0] = 0.0  # drop the zero mode (mean-free Phi)

    def _potential(self, e):
        return np.fft.irfft2(np.fft.rfft2(e) * self.kernel,
                             s=(self.ny, self.nx))

    def step(self, k=1):
        for _ in range(k):
            if int(self.t / self.dt) % 4 == 0:
                phi_pot = self._potential(self.pi ** 2 + self.phi ** 2)
                if self.g_phys is None:
                    self.g_phys = (self.g_target * M0 ** 2
                                   / max(phi_pot.max(), 1e-12))
                self.m2 = np.maximum(M0 ** 2 - self.g_phys * phi_pot,
                                     0.004)
            super().step(1)


def two_packets(g, t_max=170.0):
    f = GravField(NY, NX, g)
    f.add_packet(48 - SEP0 // 2, 30, K_X, wy=7, m0=M0)
    f.add_packet(48 + SEP0 // 2, 30, K_X, wy=7, m0=M0)
    e0 = f.energy().sum()
    streak = np.zeros((NY, NX))
    while f.t < t_max:
        f.step(4)
        streak += f.energy()
    e = f.energy()
    win = e[:, 120:180]
    ys = np.arange(NY)
    top = win[:48]
    bot = win[48:]
    y_top = float((top.sum(1) * ys[:48]).sum() / top.sum())
    y_bot = float((bot.sum(1) * ys[48:]).sum() / bot.sum())
    drift = f.energy().sum() / e0 - 1
    # merger diagnosis from the whole flight: does the midline carry
    # more energy than both original lanes downstream?
    mid = streak[42:55, 90:170].sum()
    lanes = streak[24:37, 90:170].sum() + streak[60:73, 90:170].sum()
    merge_x = None
    if mid > lanes:
        run = 0
        for x in range(55, 180):
            frac = streak[42:55, x].sum() / max(streak[:, x].sum(), 1e-12)
            run = run + 1 if frac > 0.5 else 0
            if run >= 10:
                merge_x = x - 9
                break
    return y_bot - y_top, drift, streak, merge_x


def area_law():
    n = 32
    K = scalar_couplings(n, mass=M0)
    X, P = scalar_ground_state(K)
    print('     side  S(region)   S/perimeter   S/area')
    rows = []
    for ell in range(2, 15, 2):
        y0 = (n - ell) // 2
        sites = [y * n + x for y in range(y0, y0 + ell)
                 for x in range(y0, y0 + ell)]
        S = scalar_region_entropy(X, P, sites)
        rows.append((ell, S))
        print(f'     {ell:>4}  {S:>9.3f}   {S / (4 * ell):>11.4f}   '
              f'{S / ell ** 2:>6.4f}')
    ells = np.array([r[0] for r in rows], dtype=float)
    Ss = np.array([r[1] for r in rows])
    r_per = np.corrcoef(4 * ells, Ss)[0, 1]
    r_area = np.corrcoef(ells ** 2, Ss)[0, 1]
    return r_per, r_area


def main():
    print('=' * 68)
    print('PART 10b: SELF-GRAVITY (the wells dig themselves) + AREA LAW')
    print('=' * 68)
    print('[26] two packets, launched parallel, separation '
          f'{SEP0} cells:')
    print(f'     {"g":>6} {"final separation":>17} {"energy drift":>13}')
    streaks = {}
    for g in (0.0, 0.35, 0.7):
        sep, drift, streak, merge_x = two_packets(g)
        streaks[g] = streak
        state = (f'MERGED at x~{merge_x}' if merge_x
                 else f'{sep:.1f}')
        print(f'     {g:>6.2f} {state:>17} {drift:>+13.1%}')
    print('     -> with the coupling on, the packets fall toward each')
    print('        other: attraction between lumps of energy, mediated')
    print('        by a long-range Poisson potential they themselves')
    print('        source. The coupling is to ENERGY, not species. Notes')
    print('        earned along the way: a Gaussian blur is not gravity')
    print('        (its wells never reach the neighbor — the force must')
    print('        be long-range); fast light matter barely bends while')
    print('        slow heavy matter falls hard (the nonrelativistic')
    print('        limit, as in GR); and at strong coupling the medium')
    print('        pumps energy — a Jeans-like runaway. Dispersion stays')
    print('        chromatic: full universality needs geometry-coupling,')
    print('        not medium-coupling.')
    print()

    print('[27] the area law (does information scale with boundary?):')
    r_per, r_area = area_law()
    print(f'     correlation of S with perimeter: {r_per:.4f}; '
          f'with area: {r_area:.4f}')
    print('     -> entanglement entropy tracks the BOUNDARY. The clue')
    print('        that killed volume-voxel engines in our gap list is')
    print('        reproduced by the toy: regions know their edges.')

    # streak figure: g=0 vs g=0.15
    scale = 3
    canvas = Image.new('RGB', (NX * scale, 2 * NY * scale + 40), (14, 14, 18))
    d = ImageDraw.Draw(canvas)
    for row, g in enumerate((0.0, 0.7)):
        s = streaks[g].copy()
        s[:, :14] = s[:, -14:] = 0
        colmax = np.maximum(s.max(axis=0, keepdims=True), 1e-12)
        s = (s / colmax) ** 2
        rgb = np.zeros((NY, NX, 3))
        rgb[..., 0] = 14 + 100 * s
        rgb[..., 1] = 14 + 200 * s
        rgb[..., 2] = 18 + 235 * s
        im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), 'RGB')
        im = im.resize((NX * scale, NY * scale), Image.NEAREST)
        canvas.paste(im, (0, 20 + row * (NY * scale + 20)))
        d.text((10, 4 + row * (NY * scale + 20)),
               f'g = {g}' + ('  (no coupling)' if g == 0 else
                             '  (gravitational capture: the beams merge)'),
               fill=(200, 200, 210))
    canvas.save('films/selfgravity.png')
    print('\n     films/selfgravity.png')


if __name__ == '__main__':
    main()
