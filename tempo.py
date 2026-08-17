"""Part 6: time.

Part 5 derived a SPATIAL metric from entanglement statics and found
the measured law  ds = (1 + kappa * dm(x)) |dx|  with kappa = 1.03
(fit across the A-sweep: 1.06, 1.06, 1.03, 0.97). A metric is only
physics if matter actually moves by it. So:

  [15] the Shapiro race — twin wave packets, one lane through empty
       space, one through the star. The statics metric PREDICTS the
       arrival delay before the race is run.
  [16] deflection and chromaticity — a packet passing the star at
       impact parameter b bends. Run it at two wavelengths: if the
       bend depends on the wavelength, this universe violates the
       equivalence principle — every analog-medium 'gravity' does,
       and real gravity does not. That difference is a constraint
       any render-program must eventually answer.
  [17] films: the race, and the bent trajectories.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from engine.field import Field

NY, NX = 96, 192
M0, SIGMA = 0.2, 8.0
CENTER = (48, 96)
KAPPA = 1.03                     # measured in part 5
LAUNCH_X, FINISH_X = 30, 150
LANE_FAR, LANE_STAR = 20, 48
K_HI = 2 * math.pi / 8           # short wavelength (high frequency)
K_LO = 2 * math.pi / 16          # long wavelength (low frequency)


def mass_profile(A):
    yy, xx = np.mgrid[0:NY, 0:NX]
    r2 = (yy - CENTER[0]) ** 2 + (xx - CENTER[1]) ** 2
    return M0 + A * np.exp(-r2 / (2 * SIGMA ** 2))


def group_speed(k_x):
    om = math.sqrt(M0 ** 2 + 4 * math.sin(k_x / 2) ** 2)
    return math.sin(k_x) / om


def strip_centroid(e, lane, half=4):
    """On-axis arrival: a narrow strip, so the packet's wings (which
    miss the star) don't dilute the measured delay."""
    strip = e[lane - half:lane + half]
    tot = strip.sum()
    if tot <= 0:
        return 0.0
    return float((strip.sum(axis=0) * np.arange(NX)).sum() / tot)


def wave_delay(A, k_x):
    """Ground truth from wave optics: the group-delay integral through
    the star's mass profile, with the lattice dispersion relation."""
    om = math.sqrt(M0 ** 2 + 4 * math.sin(k_x / 2) ** 2)
    vg0 = math.sin(k_x) / om
    xs = np.arange(-44.0, 44.0, 0.25)
    m = M0 + A * np.exp(-xs ** 2 / (2 * SIGMA ** 2))
    s2 = np.clip((om ** 2 - m ** 2) / 4, 1e-9, None)
    k_loc = 2 * np.arcsin(np.clip(np.sqrt(s2), 0, 1))
    vg = np.maximum(np.sin(k_loc) / om, 1e-9)
    return float(np.trapezoid(1 / vg - 1 / vg0, xs))


def race(A, k_x=K_HI, t_max=400.0):
    """Two packets, same tick zero. Returns arrival times at FINISH_X."""
    f = Field(NY, NX, mass_profile(A))
    f.add_packet(LANE_FAR, LAUNCH_X, k_x, wy=14)
    f.add_packet(LANE_STAR, LAUNCH_X, k_x, wy=14)
    arrive = {}
    prev = {LANE_FAR: (0.0, LAUNCH_X), LANE_STAR: (0.0, LAUNCH_X)}
    while f.t < t_max and len(arrive) < 2:
        f.step(2)
        e = f.energy()
        for lane in (LANE_FAR, LANE_STAR):
            if lane in arrive:
                continue
            xc = strip_centroid(e, lane)
            t0, x0 = prev[lane]
            if xc >= FINISH_X > x0:
                frac = (FINISH_X - x0) / max(xc - x0, 1e-9)
                arrive[lane] = t0 + frac * (f.t - t0)
            prev[lane] = (f.t, xc)
    return arrive.get(LANE_FAR), arrive.get(LANE_STAR)


def deflection(A, k_x, y0=32, t_max=340.0):
    """One packet passing the star at impact parameter |y0 - 48|.

    Measurement: the TRANSMITTED beam only — the y-centroid within a
    downstream window (x in [120, 178]) at the moment that window's
    energy peaks. The full-domain centroid is meaningless once part of
    the beam reflects or scatters. Also returns the transmitted energy
    fraction and a time-integrated energy map (a long-exposure streak
    photo of the beam).
    """
    f = Field(NY, NX, mass_profile(A))
    f.add_packet(y0, LAUNCH_X, k_x, wy=8)
    e0 = f.energy().sum()
    streak = np.zeros((NY, NX))
    best = (0.0, y0, 0.0)  # window energy, y-centroid, time
    while f.t < t_max:
        f.step(4)
        e = f.energy()
        streak += e
        win = e[:, 120:178]
        we = float(win.sum())
        if we > best[0]:
            yc = float((win.sum(axis=1) * np.arange(NY)).sum() / we)
            best = (we, yc, f.t)
    return {'shift': best[1] - y0, 'transmitted': best[0] / e0,
            'streak': streak}


def film_race(A, k_x, path):
    f = Field(NY, NX, mass_profile(A))
    f.add_packet(LANE_FAR, LAUNCH_X, k_x)
    f.add_packet(LANE_STAR, LAUNCH_X, k_x)
    frames = []
    scale = 3
    yy, xx = np.mgrid[0:NY, 0:NX]
    ring = np.abs(np.hypot(yy - CENTER[0], xx - CENTER[1]) - 2 * SIGMA) < 0.7
    for _ in range(150):
        f.step(6)
        a = np.clip(np.abs(f.phi) / 0.8, 0, 1)
        pos = f.phi > 0
        img = np.zeros((NY, NX, 3))
        img[..., 0] = np.where(pos, 235 * a, 80 * a) + 14
        img[..., 1] = np.where(pos, 160 * a, 130 * a) + 14
        img[..., 2] = np.where(pos, 80 * a, 230 * a) + 18
        img[ring] = (110, 110, 120)
        img[:, FINISH_X] = np.maximum(img[:, FINISH_X], 55)
        img = np.clip(img, 0, 255).astype(np.uint8)
        big = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))
        frames.append(Image.fromarray(big, 'RGB'))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=45, loop=0)


def draw_streaks(runs, path):
    """Long-exposure photo of the three beams: control in blue,
    short wavelength in green, long wavelength in red."""
    scale = 4
    rgb = np.full((NY, NX, 3), 14.0)
    order = ['long wavelength', 'short wavelength', 'A=0 (no star)']
    for ch, label in enumerate(order):
        s = runs[label]['streak'].copy()
        s[:14] = s[-14:] = 0
        s[:, :14] = s[:, -14:] = 0
        colmax = np.maximum(s.max(axis=0, keepdims=True), 1e-12)
        s = (s / colmax) ** 2  # per-column: the beam's ridge line
        rgb[..., ch] += 225 * s
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), 'RGB')
    img = img.resize((NX * scale, NY * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    for r in (SIGMA, 2 * SIGMA):
        d.ellipse([(CENTER[1] - r) * scale, (CENTER[0] - r) * scale,
                   (CENTER[1] + r) * scale, (CENTER[0] + r) * scale],
                  outline=(120, 120, 132), width=2)
    for label, col in (('long wavelength', (245, 90, 90)),
                       ('short wavelength', (90, 235, 120)),
                       ('control (no star)', (110, 150, 250))):
        i = ['long wavelength', 'short wavelength',
             'control (no star)'].index(label)
        d.text((20, 16 + 20 * i), label, fill=col)
    img.save(path)


def main():
    print('=' * 68)
    print('PART 6: TIME (the entanglement metric makes dynamical predictions)')
    print('=' * 68)
    vg = group_speed(K_HI)
    extra_len = KAPPA * SIGMA * math.sqrt(2 * math.pi)  # per unit A
    print(f'[15] the Shapiro race (short-wavelength packets, v_g = {vg:.3f})')
    print(f'     entanglement-metric prediction (part 5, kappa={KAPPA}): '
          f'delay = {extra_len / vg:.1f}*A ticks')
    print(f'     {"A":>5} {"measured":>9} {"wave optics":>12} '
          f'{"MI metric":>10}')
    for A in (0.1, 0.2, 0.3, 0.4, 0.5):
        t_far, t_star = race(A)
        dt = t_star - t_far
        print(f'     {A:>5.1f} {dt:>9.1f} {wave_delay(A, K_HI):>12.1f} '
              f'{extra_len * A / vg:>10.1f}')
    print('     -> the packet through the star arrives late. Measured')
    print('        delays track the wave-optics integral; the MI-metric')
    print('        ruler (linear in m) overshoots at small A and meets')
    print('        wave optics at mid-strength. The static entanglement')
    print('        geometry predicts real time-of-flight physics, up to')
    print('        dispersion corrections.')
    print()

    print('[16] deflection and the equivalence principle (A=2, b=16):')
    runs = {'A=0 (no star)': deflection(0.0, K_HI),
            'short wavelength': deflection(2.0, K_HI),
            'long wavelength': deflection(2.0, K_LO)}
    for label, r in runs.items():
        print(f'     {label:<18}: transmitted beam shifted {r["shift"]:+6.1f} '
              f'cells, transmission {r["transmitted"]:.0%}')
    print('     -> the star deflects the transmitted beam away (paths')
    print('        through it are longer — consistent with part 5) and')
    print('        the effect depends on wavelength, in both bend and')
    print('        opacity. Chromatic gravity: this analog universe')
    print('        VIOLATES the equivalence principle. Real gravity does')
    print('        not — attraction via a universal metric, not via a')
    print('        medium, is exactly what the render program still owes.')
    print()

    print('[17] films...')
    film_race(0.5, K_HI, 'films/shapiro_race.gif')
    draw_streaks(runs, 'films/deflection.png')
    print('     films/shapiro_race.gif, films/deflection.png')


if __name__ == '__main__':
    main()
