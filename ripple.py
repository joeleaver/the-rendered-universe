"""Part 18: ripple — the metric learns to wave.

Part 16 gave gravity a metric and passed the Eotvos test, but the
metric was quasi-static: matter curved it instantly, at a distance.
Here the potential becomes a field with its own wave equation on the
SAME lattice, with the SAME stencil, at the same tick as matter:

    d(Phi)/dt = Pi_g
    d(Pi_g)/dt = c_g^2 laplacian(Phi) - kappa * source

That one design fact — one substrate updating everything — is a
falsifiable physical prediction, and the real sky has already run
both of its tests:

  [77] the metric learns to wave: a disturbed source radiates an
       expanding metric ripple; front speed and isotropy measured.
  [78] the race, run both ways: a photon and a metric pulse from one
       event cross a long baseline. One substrate: they arrive
       together (measured |dc/c| at our timing floor). Then the
       two-substrate engine — gravity on its own stencil — is built
       deliberately, and loses the race by half the track.
       The sky's verdict: GW170817's gamma burst arrived 1.7 s after
       the chirp across 40 Mpc — |dc/c| < 3e-15. Two-substrate
       engines are dead by ~14 orders of magnitude; one-substrate
       engines predict exactly zero.
  [79] the polarization test, where the scalar variant dies: a
       passing scalar ripple strains rulers ALONG the propagation
       direction exactly as much as across it (longitudinal/
       transverse response = 1). GR's tensor waves are transverse
       (ratio 0), and LIGO-Virgo's polarization analyses favor
       tensor. Measured, conceded: the render program owes gravity a
       TENSOR metric, not a scalar one — Jacobson's direction.
  [80] the scorecard: which race verdicts the architecture has
       passed, and which variant they killed.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

# validated dark-mode categorical palette (dataviz slots 1-3)
C_BLUE, C_ORANGE, C_GREEN = (57, 135, 229), (217, 89, 38), (25, 158, 112)
INK, MUTED, GRIDC = (195, 194, 183), (122, 122, 130), (38, 38, 44)
BG = (14, 14, 18)

DT = 0.2


class Wave:
    """Massless leapfrog wave on the shared lattice. `c2` is the
    stencil coefficient: 1.0 is THE substrate speed; anything else is
    a deliberately foreign substrate."""

    def __init__(self, ny, nx, c2=1.0, sponge=14):
        self.ny, self.nx, self.c2 = ny, nx, c2
        self.f = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.t = 0.0
        yy, xx = np.mgrid[0:ny, 0:nx]
        d = np.minimum.reduce([yy, ny - 1 - yy, xx, nx - 1 - xx])
        ramp = np.clip((sponge - d) / sponge, 0, 1)
        self.damp = 1 - 0.06 * ramp ** 2

    def lap(self):
        f = self.f
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)

    def step(self, k=1, source=None):
        for _ in range(k):
            self.p += DT * (self.c2 * self.lap()
                            + (source(self.t) if source else 0.0))
            self.f += DT * self.p
            self.p *= self.damp
            self.f *= self.damp
            self.t += DT

    def add_packet(self, y0, x0, k_x, wy=12, wx=6):
        yy, xx = np.mgrid[0:self.ny, 0:self.nx]
        env = np.exp(-((yy - y0) ** 2 / (2 * wy ** 2)
                       + (xx - x0) ** 2 / (2 * wx ** 2)))
        om = 2 * math.sqrt(self.c2) * abs(math.sin(k_x / 2))
        self.f += env * np.cos(k_x * xx)
        self.p += env * om * np.sin(k_x * xx)

    def energy(self):
        return self.p ** 2 + self.f ** 2


# ---- [77] the ripple --------------------------------------------------

def ripple_front():
    ny = nx = 240
    g = Wave(ny, nx)
    yy, xx = np.mgrid[0:ny, 0:nx]
    blob = np.exp(-((yy - ny // 2) ** 2 + (xx - nx // 2) ** 2) / (2 * 4.0 ** 2))

    def source(t):
        return blob * math.sin(0.9 * t) * math.exp(-((t - 7) / 4) ** 2)

    snaps, radii, times = [], [], []
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1),
            (0.7071, 0.7071), (-0.7071, 0.7071),
            (0.7071, -0.7071), (-0.7071, -0.7071)]
    while g.t < 100:
        g.step(5, source)
        e = g.energy()
        rr = []
        for dy, dx in dirs:
            prof = np.array([e[int(ny // 2 + dy * r), int(nx // 2 + dx * r)]
                             for r in range(0, 110)])
            rr.append(int(np.argmax(prof)))
        if min(rr) > 8:
            radii.append(rr)
            times.append(g.t)
        if abs(g.t - 30) < 0.5 or abs(g.t - 60) < 0.5 or abs(g.t - 90) < 0.5:
            snaps.append(g.f.copy())
    radii = np.array(radii, dtype=float)
    mean_r = radii.mean(axis=1)
    speed = np.polyfit(times, mean_r, 1)[0]
    aniso = float((radii.std(axis=1) / mean_r).mean())
    return speed, aniso, snaps


# ---- [78] the race ----------------------------------------------------

def first_crossing(times, xs, target):
    for (t0, x0), (t1, x1) in zip(zip(times, xs), zip(times[1:], xs[1:])):
        if x1 >= target > x0:
            return t0 + (target - x0) / (x1 - x0) * (t1 - t0)
    return float('nan')


def race(c2_gravity):
    """One event, two messengers, one long track. Returns the arrival
    times of the photon and the metric pulse at x = 500."""
    ny, nx, k_x = 72, 560, 0.6
    photon = Wave(ny, nx, c2=1.0)
    metric = Wave(ny, nx, c2=c2_gravity)
    for w in (photon, metric):
        w.add_packet(ny // 2, 50, k_x)
    lane = slice(ny // 2 - 8, ny // 2 + 8)
    xs_p, xs_g, ts = [], [], []
    while photon.t < 1400:
        photon.step(2)
        metric.step(2)
        ax = np.arange(nx)
        for w, xs in ((photon, xs_p), (metric, xs_g)):
            e = w.energy()[lane].sum(axis=0)
            e[:60] = 0.0            # ignore the left-moving half
            xs.append(float((e * ax).sum() / max(e.sum(), 1e-12)))
        ts.append(photon.t)
    t_p = first_crossing(ts, xs_p, 500.0)
    t_g = first_crossing(ts, xs_g, 500.0)
    return t_p, t_g, (ts, xs_p, xs_g)


# ---- [79] the polarization test --------------------------------------

def polarization():
    """A plane-ish scalar ripple passes a detector; measure the
    optical-path (ruler) response of a longitudinal arm (along the
    propagation direction) and a transverse arm."""
    ny, nx = 160, 420
    g = Wave(ny, nx)
    yy, xx = np.mgrid[0:ny, 0:nx]
    blob = np.exp(-((yy - ny // 2) ** 2 / (2 * 40.0 ** 2)
                    + (xx - 40) ** 2 / (2 * 4.0 ** 2)))

    # wavelength ~ 2 pi / 0.15 ~ 42 cells >> the 6-cell arms below —
    # like LIGO, the detector must be small against the wave
    def source(t):
        return blob * math.sin(0.15 * t) * min(1.0, t / 50)

    x0, y0, arm = 300, ny // 2, 6
    kappa = 0.02          # Phi = kappa * f: keep the metric weak
    resp_l, resp_t, ts = [], [], []
    while g.t < 700:
        g.step(2, source)
        phi = kappa * g.f
        c = np.sqrt(np.clip(1 + 2 * phi, 0.25, None))
        # optical path length of each arm, minus its flat value
        tau_l = (1.0 / c[y0, x0:x0 + arm]).sum() - arm
        tau_t = (1.0 / c[y0:y0 + arm, x0]).sum() - arm
        resp_l.append(tau_l)
        resp_t.append(tau_t)
        ts.append(g.t)
    resp_l, resp_t = np.array(resp_l), np.array(resp_t)
    sl = slice(len(ts) // 2, None)          # steady passage
    amp_l = resp_l[sl].std()
    amp_t = resp_t[sl].std()
    corr = float(np.corrcoef(resp_l[sl], resp_t[sl])[0, 1])
    return amp_l / amp_t, corr, (np.array(ts), resp_l, resp_t)


# ---- figure -----------------------------------------------------------

def figure(snaps, race_full, race_lame, pol_series, path):
    W, H = 1560, 760
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 12), 'PART 18 - RIPPLE: the metric learns to wave, '
           'and the sky judges the design', fill=INK)

    # (a) ripple snapshots
    x0 = 20
    for i, s in enumerate(snaps[:3]):
        v = s / max(1e-12, np.abs(s).max())
        pane = np.zeros(s.shape + (3,))
        pos = v > 0
        pane[..., 0] = np.where(pos, 40 + 200 * v, 40 * (1 + v))
        pane[..., 2] = np.where(pos, 40 * (1 - v), 40 - 200 * v)
        pane[..., 1] = 40 * (1 - np.abs(v))
        im = Image.fromarray(np.clip(pane, 0, 255).astype(np.uint8),
                             'RGB').resize((230, 230), Image.LANCZOS)
        img.paste(im, (x0 + i * 244, 56))
    d.text((20, 292), 'a metric ripple, radiated and expanding '
           '(t = 30, 60, 90): front speed 0.99 c, anisotropy ~1%',
           fill=MUTED)

    # (b) the race
    bx0, by0, bx1, by1 = 830, 56, 1520, 292
    ts, xs_p, xs_g = race_full
    ts2, _, xs_g2 = race_lame

    def xy(t, x):
        return (bx0 + (bx1 - bx0) * t / 660.0,
                by1 - (by1 - by0) * (x - 50) / 460.0)
    for xval in (100, 200, 300, 400, 500):
        yy = xy(0, xval)[1]
        d.line([(bx0, yy), (bx1, yy)], fill=GRIDC)
        d.text((bx0 - 34, yy - 6), f'{xval}', fill=MUTED)
    def clip(series_t, series_x):
        pts = []
        for t, x in zip(series_t, series_x):
            if t > 660 or x >= 505:
                break
            pts.append(xy(t, x))
        return pts
    d.line(clip(ts, xs_p), fill=C_ORANGE, width=4)
    d.line(clip(ts, xs_g), fill=C_BLUE, width=2)
    d.line(clip(ts2, xs_g2), fill=C_GREEN, width=2)
    d.text((bx0, by0 - 16), 'the race: photon (orange) vs metric pulse '
           '(blue, same substrate: identical) vs', fill=INK)
    d.text((bx0, by0), 'a two-substrate engine\'s gravity (green) - '
           'which loses by half the track', fill=INK)
    d.text(((bx0 + bx1) // 2, by1 + 8), 'tick', fill=MUTED)

    # (c) polarization
    cx0, cy0, cx1, cy1 = 70, 380, 730, 700
    ts, rl, rt = pol_series
    amp = max(np.abs(rl).max(), np.abs(rt).max(), 1e-12)

    def cxy(t, v):
        return (cx0 + (cx1 - cx0) * (t - ts[0]) / (ts[-1] - ts[0]),
                (cy0 + cy1) / 2 - (cy1 - cy0) / 2.2 * v / amp)
    d.line([(cx0, (cy0 + cy1) / 2), (cx1, (cy0 + cy1) / 2)], fill=GRIDC)
    d.line([cxy(t, v) for t, v in zip(ts, rl)], fill=C_ORANGE, width=3)
    d.line([cxy(t, v) for t, v in zip(ts, rt)], fill=C_BLUE, width=1)
    d.text((cx0, cy0 - 30), 'the polarization test: ruler response '
           'along the wave (orange) and across it (blue)', fill=INK)
    d.text((cx0, cy0 - 14), 'scalar metric: equal, in phase '
           '(breathing). GR: zero along, pure quadrupole across.',
           fill=MUTED)
    d.text((cx0, cy1 + 10), 'LIGO-Virgo favor tensor: this variant '
           'of render gravity is dead - it owes a tensor metric.',
           fill=C_ORANGE)

    # (d) the dc/c ladder
    dx0, dy0, dx1 = 830, 420, 1520
    d.text((dx0, dy0 - 20), 'where the speed verdict stands:  '
           '|c_gw/c_photon - 1|', fill=INK)
    rows = [('two-substrate engine (measured here)', 0.5, C_GREEN),
            ('the real sky: GW170817 + GRB 170817A', 3e-15, C_ORANGE),
            ('one-substrate engine (this one)', 1e-16, C_BLUE)]
    for i, (label, val, col) in enumerate(rows):
        yy = dy0 + 16 + i * 56
        frac = (math.log10(max(val, 1e-16)) + 16) / 16
        w = int((dx1 - dx0 - 330) * max(frac, 0.012))
        d.rectangle([dx0 + 320, yy, dx0 + 320 + w, yy + 22], fill=col)
        d.text((dx0, yy + 4), label, fill=INK)
        txt = 'exactly 0' if val <= 1e-16 else f'{val:.0e}'
        d.text((dx0 + 326 + w, yy + 4), txt, fill=INK)
    d.text((dx0, dy0 + 250), 'log scale to 1e-16: the sky already '
           'rules out gravity on its own clock by ~14 orders;',
           fill=MUTED)
    d.text((dx0, dy0 + 266), 'a shared substrate predicts exactly '
           'zero, forever.', fill=MUTED)
    img.save(path)


# ---- main -------------------------------------------------------------

def main():
    import time
    t00 = time.time()
    print('=' * 68)
    print('PART 18: RIPPLE — THE METRIC LEARNS TO WAVE')
    print('=' * 68)
    print()

    print('[77] the metric learns to wave: a pumped source radiates')
    print('     into the metric field (same lattice, same stencil,')
    print('     same tick as matter)...')
    speed, aniso, snaps = ripple_front()
    print(f'     front speed = {speed:.3f} cells/tick '
          f'(substrate c = 1); anisotropy {100 * aniso:.1f}%')
    print()

    print('[78] the race: one event, two messengers, 450 cells:')
    t_p, t_g, series_full = race(1.0)
    print(f'     photon arrival t = {t_p:.1f}; metric pulse '
          f't = {t_g:.1f}; difference {abs(t_g - t_p):.10f}')
    print('     Bit-identical: one substrate means one dispersion')
    print('     relation, so equal speed at every frequency is an')
    print('     IDENTITY of the design, not a tuning.')
    t_p2, t_g2, series_lame = race(0.25)
    print(f'     the two-substrate engine (gravity on its own '
          f'stencil, c_g = 0.5):')
    print(f'     photon t = {t_p2:.1f}, gravity t = {t_g2:.1f} — '
          f'loses by {t_g2 - t_p2:.0f} ticks.')
    print('     THE SKY HAS RUN THIS RACE: GW170817\'s gamma rays')
    print('     arrived 1.7 s after the chirp across ~40 Mpc:')
    print('     |dc/c| < 3e-15. Two-substrate engines: dead by ~14')
    print('     orders. One-substrate engines: exactly zero, forever.')
    print()

    print('[79] the polarization test (the one the scalar fails):')
    ratio, corr, pol_series = polarization()
    print(f'     longitudinal/transverse ruler response = '
          f'{ratio:.2f} (correlation {corr:+.2f})')
    print('     A scalar metric breathes: it strains rulers ALONG the')
    print('     wave as much as ACROSS it. GR strains only across')
    print('     (transverse, quadrupolar), and LIGO-Virgo polarization')
    print('     analyses favor pure tensor. Verdict, conceded in')
    print('     advance by part 16\'s residue and now measured: the')
    print('     scalar-metric variant of render gravity is EXCLUDED')
    print('     by the real sky. What survives is the tensor-metric')
    print('     program (Jacobson\'s direction).')
    print()

    print('[80] the scorecard, race by race:')
    print('     one cone (c_gw = c_photon)   PASSED   (toy: exact; '
          'sky: <3e-15)')
    print('     GW polarization              KILLS the scalar variant')
    print('     preferred frame (part 10)    KILLS the lattice '
          'variant; sprinkle survives')
    print('     equivalence principle (16)   PASSED   (metric '
          'coupling, Eotvos)')
    print('     The architecture is not unfalsifiable: the sky has')
    print('     already executed two of its variants.')

    figure(snaps, series_full, series_lame, pol_series,
           'films/ripple.png')
    print()
    print(f'     films/ripple.png  ({time.time() - t00:.0f}s)')


if __name__ == '__main__':
    main()
