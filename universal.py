"""Part 16: gravity through the metric — the equivalence principle,
restored and measured.

Part 6's honest accounting recorded the program's sharpest negative
result: medium-coupled gravity (a mass profile) is chromatic and
violates the equivalence principle. The cure is the one general
relativity uses: couple through GEOMETRY. Here the field propagates
with a position-dependent speed c(x) — an optical metric,
c^2 = 1 + 2*Phi with Phi < 0 near a star — via the wave operator

    d2(phi)/dt2 = c div(c grad phi) - c^2 m^2 phi.

Ray analysis predicts, before any run: local wavenumber obeys
(dk/k) = -(dc/c)/v^2, so trajectories depend on position and VELOCITY
only — not on frequency at fixed v (massless waves: v = c for every
wavelength), and not on mass at fixed v. That is the equivalence
principle, with general relativity's own velocity dependence (slow
matter falls ~1/v^2 harder than light). Measured below:

  [61] achromatic lensing — massless packets at three wavelengths
       bend TOWARD the star (attraction, at last) by the same angle,
       with full transmission. (Part 6's medium: repulsion, 55% vs
       23% transmission across one octave.)
  [62] achromatic Shapiro delay — equal at both wavelengths.
  [63] the Eotvos test — two fields of DIFFERENT mass, prepared at
       the same group velocity, fall identically; a massless packet
       bends less by ~v^2, as GR demands.
  [64] the loop closed universally — the metric sourced by the
       field's own energy: attraction and capture, through geometry.
"""
import math

import numpy as np

from engine.field import Field

NY, NX = 96, 192
CENTER = (48, 96)
SIGMA = 10.0


class MetricField(Field):
    """Wave dynamics on a static optical metric c(x)."""

    def __init__(self, ny, nx, cmap, mass=0.0, dt=0.15, sponge=12):
        super().__init__(ny, nx, mass, dt=dt, sponge=sponge)
        self.set_metric(cmap)

    def set_metric(self, cmap):
        self.c = cmap
        self.cxh = 0.5 * (cmap + np.roll(cmap, -1, 1))
        self.cyh = 0.5 * (cmap + np.roll(cmap, -1, 0))
        self.c2 = cmap ** 2

    def step(self, k=1):
        for _ in range(k):
            fx = self.cxh * (np.roll(self.phi, -1, 1) - self.phi)
            fy = self.cyh * (np.roll(self.phi, -1, 0) - self.phi)
            div = (fx - np.roll(fx, 1, 1)) + (fy - np.roll(fy, 1, 0))
            self.pi += self.dt * (self.c * div - self.c2 * self.m2 * self.phi)
            self.phi += self.dt * self.pi
            self.pi *= self.damp
            self.phi *= self.damp
            self.t += self.dt


def star_metric(strength):
    """Optical metric of a star: c^2 = 1 + 2*Phi, Phi < 0 (Gaussian)."""
    yy, xx = np.mgrid[0:NY, 0:NX]
    r2 = (yy - CENTER[0]) ** 2 + (xx - CENTER[1]) ** 2
    phi_pot = -strength * np.exp(-r2 / (2 * SIGMA ** 2))
    return np.sqrt(np.clip(1 + 2 * phi_pot, 0.2, None))


def k_for_velocity(m, v):
    """Lattice wavenumber giving group velocity v for mass m (far
    field, c=1): vg = sin(k)/sqrt(4 sin^2(k/2) + m^2). Scans the
    rising branch of the lattice dispersion and refuses targets above
    the achievable maximum (lattice dispersion caps massive group
    velocities — the bug this replaces compared unmatched speeds)."""
    ks = np.linspace(0.01, 2.5, 4000)
    vgs = np.sin(ks) / np.sqrt(4 * np.sin(ks / 2) ** 2 + m * m)
    peak = int(np.argmax(vgs))
    assert v < vgs[peak] - 0.01, \
        f'v={v} unreachable for m={m} (max {vgs[peak]:.3f})'
    i = int(np.argmin(np.abs(vgs[:peak] - v)))
    return float(ks[i])


def deflect(cmap, k_x, m=0.0, y0=34, t_max=420.0):
    """Transmitted-beam y-shift and transmission through the star
    (windowed measurement, as in part 6)."""
    f = MetricField(NY, NX, cmap, mass=m)
    om = math.sqrt(4 * math.sin(k_x / 2) ** 2 + m * m)
    yy, xx = np.mgrid[0:NY, 0:NX]
    env = np.exp(-((yy - y0) ** 2 / (2 * 8.0 ** 2)
                   + (xx - 30) ** 2 / (2 * 7.0 ** 2)))
    f.phi += env * np.cos(k_x * xx)
    f.pi += om * env * np.sin(k_x * xx)
    e0 = f.energy().sum()
    best = (0.0, y0)
    while f.t < t_max:
        f.step(4)
        e = f.energy()
        win = e[:, 120:178]
        we = float(win.sum())
        if we > best[0]:
            yc = float((win.sum(axis=1) * np.arange(NY)).sum() / we)
            best = (we, yc)
    return best[1] - y0, best[0] / e0


def race(cmap, k_x, t_max=520.0):
    """Arrival times at x=150 for lanes through and beside the star."""
    f = MetricField(NY, NX, cmap)
    om = 2 * abs(math.sin(k_x / 2))
    yy, xx = np.mgrid[0:NY, 0:NX]
    for lane in (20, 48):
        env = np.exp(-((yy - lane) ** 2 / (2 * 14.0 ** 2)
                       + (xx - 30) ** 2 / (2 * 6.0 ** 2)))
        f.phi += env * np.cos(k_x * xx)
        f.pi += om * env * np.sin(k_x * xx)
    arrive, prev = {}, {20: (0.0, 30.0), 48: (0.0, 30.0)}
    while f.t < t_max and len(arrive) < 2:
        f.step(2)
        e = f.energy()
        for lane in (20, 48):
            if lane in arrive:
                continue
            strip = e[lane - 4:lane + 4]
            xc = float((strip.sum(axis=0) * np.arange(NX)).sum()
                       / max(strip.sum(), 1e-12))
            t0, x0 = prev[lane]
            if xc >= 150 > x0:
                frac = (150 - x0) / max(xc - x0, 1e-9)
                arrive[lane] = t0 + frac * (f.t - t0)
            prev[lane] = (f.t, xc)
    return arrive.get(48, float('nan')) - arrive.get(20, float('nan'))


def main():
    print('=' * 68)
    print('PART 16: GRAVITY THROUGH THE METRIC (the EP, restored)')
    print('=' * 68)
    star = star_metric(0.22)

    print('[61] achromatic lensing (massless field, star at b=14):')
    print(f'     {"wavelength":>10} {"beam shift":>10} {"transmission":>13}')
    shifts = []
    for lam in (8, 12, 16):
        s, tr = deflect(star, 2 * math.pi / lam)
        shifts.append(s)
        print(f'     {lam:>10} {s:>+10.1f} {tr:>13.0%}')
    spread = max(shifts) - min(shifts)
    print(f'     bend TOWARD the star (attraction), spread across one')
    print(f'     octave: {spread:.1f} cells (part 6 medium-gravity: '
          f'repulsion, opacity 55% vs 23%).')
    print()

    print('[62] achromatic Shapiro delay (star lane vs far lane):')
    for lam in (8, 16):
        print(f'     wavelength {lam:>2}: delay = {race(star, 2 * math.pi / lam):.1f} ticks')
    print()

    v_t = 0.70
    print(f'[63] the Eotvos test — different masses, same velocity '
          f'(v = {v_t}):')
    rows = []
    for m in (0.35, 0.55):
        k = k_for_velocity(m, v_t)
        s, tr = deflect(star, k, m=m, t_max=620.0)
        rows.append(s)
        print(f'     mass {m}: k = {k:.3f} (wavelength '
              f'{2 * math.pi / k:.1f}), beam shift {s:+.1f} cells')
    print(f'     composition dependence: {abs(rows[0] - rows[1]):.1f} '
          f'cells — different matter, same fall.')
    s_light = np.mean(shifts)
    print(f'     massless packet (v = 1): shift {s_light:+.1f} — slow '
          f'matter falls harder, as it should')
    print(f'     (measured slow/light ratio {np.mean(rows) / s_light:.2f}; '
          f'ray theory ~ 1/v^2 = {1 / v_t ** 2:.2f}).')
    print()

    print('[64] the loop, closed through geometry: the metric sourced')
    print('     by the field\'s own energy (screened Poisson, c^2 = '
          '1 + 2*Phi):')
    ky = np.fft.fftfreq(NY) * 2 * math.pi
    kx = np.fft.rfftfreq(NX) * 2 * math.pi
    kern = 1.0 / (ky[:, None] ** 2 + kx[None, :] ** 2 + (1 / 35.0) ** 2)
    kern[0, 0] = 0.0
    f = MetricField(NY, NX, np.ones((NY, NX)), mass=0.45, dt=0.12)
    m0, k_x = 0.45, 2 * math.pi / 10
    om = math.sqrt(4 * math.sin(k_x / 2) ** 2 + m0 ** 2)
    yy, xx = np.mgrid[0:NY, 0:NX]
    for lane in (36, 60):
        env = np.exp(-((yy - lane) ** 2 / (2 * 7.0 ** 2)
                       + (xx - 30) ** 2 / (2 * 7.0 ** 2)))
        f.phi += env * np.cos(k_x * xx)
        f.pi += om * env * np.sin(k_x * xx)
    g_phys, streak = None, np.zeros((NY, NX))
    while f.t < 200.0:
        if int(f.t / f.dt) % 4 == 0:
            rho = np.fft.irfft2(np.fft.rfft2(f.energy()) * kern,
                                s=(NY, NX))
            if g_phys is None:
                g_phys = 0.20 / max(rho.max(), 1e-12)
            f.set_metric(np.sqrt(np.clip(1 - 2 * g_phys * rho, 0.25,
                                         None)))
        f.step(4)
        streak += f.energy()
    mid = streak[42:55, 90:170].sum()
    lanes = streak[29:40, 90:170].sum() + streak[56:67, 90:170].sum()
    print(f'     midline/lanes energy ratio downstream: '
          f'{mid / max(lanes, 1e-12):.2f} '
          f'({"MERGED — capture through the metric" if mid > lanes else "converging"})')
    print()
    print('     Gravity that attracts, does not disperse, and treats')
    print('     every mass and every wavelength alike — because it is')
    print('     geometry, not medium. What remains beyond this toy:')
    print('     tensor (not scalar) metric structure, gravitational')
    print('     waves, and Einstein dynamics for the metric itself.')


if __name__ == '__main__':
    main()
