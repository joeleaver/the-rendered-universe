"""Part 10a: hiding the lattice — the random chart.

The gap: our engines have a grid, and grids have preferred directions;
the real universe's Lorentz symmetry is exact to absurd precision.
The known loophole (causal set theory): RANDOM discreteness. A Poisson
sprinkle has no rows, no columns, no rest frame.

  [24] the speed of light should not depend on direction. On the
       square lattice it does (group speed at k=pi/2 differs axis vs
       diagonal by ~20%). On a Poisson random graph, same test:
       direction-independent to within noise.
  [25] the sprinkle has no rest frame. Boost a spacetime LATTICE of
       events and its nearest-neighbor statistics change completely
       (every observer can measure their absolute velocity). Boost a
       Poisson sprinkle: statistically indistinguishable — an
       area-preserving map of a uniform Poisson process is the same
       process. Discreteness without a preferred frame.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

rng = np.random.default_rng(2)


# ------------------------------------------------- [24] wave isotropy

def lattice_speed(angle_deg, k_mag=math.pi / 2, T=120, n=220):
    """Group speed of a wave packet on the square lattice."""
    th = math.radians(angle_deg)
    kx, ky = k_mag * math.cos(th), k_mag * math.sin(th)
    om = math.sqrt(4 * math.sin(kx / 2) ** 2 + 4 * math.sin(ky / 2) ** 2)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    c = n / 2
    env = np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * 9.0 ** 2))
    phase = kx * xx + ky * yy
    phi = env * np.cos(phase)
    pi = om * env * np.sin(phase)
    dt = 0.2

    def lap(f):
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)

    def centroid():
        e = pi ** 2 + phi ** 2
        t = e.sum()
        return (float((e.sum(1) * np.arange(n)).sum() / t),
                float((e.sum(0) * np.arange(n)).sum() / t))

    c0 = centroid()
    for _ in range(int(T / dt)):
        pi += dt * lap(phi)
        phi += dt * pi
    c1 = centroid()
    return math.hypot(c1[0] - c0[0], c1[1] - c0[1]) / T


def build_graph(L=200.0, radius=1.8, seed=4):
    """Poisson sprinkle in a box, edges within `radius`."""
    r2 = rng if seed is None else np.random.default_rng(seed)
    n_pts = r2.poisson(L * L)
    pts = r2.random((n_pts, 2)) * L
    cell = radius
    grid = {}
    for i, p in enumerate(pts):
        grid.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(i)
    ei, ej = [], []
    for (cx, cy), members in grid.items():
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neigh += grid.get((cx + dx, cy + dy), [])
        for i in members:
            for j in neigh:
                if j > i and np.hypot(*(pts[i] - pts[j])) < radius:
                    ei.append(i)
                    ej.append(j)
    return pts, np.array(ei), np.array(ej)


def graph_speed(pts, ei, ej, angle_deg, k_mag=2 * math.pi / 12, T=12,
                om_guess=None):
    om_guess = om_guess or 4.9 * k_mag
    th = math.radians(angle_deg)
    kx, ky = k_mag * math.cos(th), k_mag * math.sin(th)
    c = pts.mean(axis=0)
    d2 = ((pts - c) ** 2).sum(axis=1)
    env = np.exp(-d2 / (2 * 16.0 ** 2))
    phase = kx * (pts[:, 0] - c[0]) + ky * (pts[:, 1] - c[1])
    phi = env * np.cos(phase)
    pi = om_guess * env * np.sin(phase)
    dt = 0.04

    def lap(f):
        out = np.zeros_like(f)
        np.add.at(out, ei, f[ej] - f[ei])
        np.add.at(out, ej, f[ei] - f[ej])
        return out

    def centroid():
        e = pi ** 2 + phi ** 2
        return (e[:, None] * pts).sum(axis=0) / e.sum()

    c0 = centroid()
    for _ in range(int(T / dt)):
        pi += dt * lap(phi)
        phi += dt * pi
    c1 = centroid()
    return float(np.hypot(*(c1 - c0)) / T)


# ------------------------------------------------- [25] boost test

def boost(pts, v):
    g = 1.0 / math.sqrt(1 - v * v)
    t, x = pts[:, 0], pts[:, 1]
    return np.stack([g * (t - v * x), g * (x - v * t)], axis=1)


def nn_stats(pts, window):
    """Mean/std of nearest-neighbor (Euclidean) distance for points in
    a central window (avoids boundary artifacts)."""
    lo, hi = window
    sel = ((pts[:, 0] > lo) & (pts[:, 0] < hi)
           & (pts[:, 1] > lo) & (pts[:, 1] < hi))
    core = pts[sel]
    d = np.full(len(core), np.inf)
    for i, p in enumerate(core):
        dd = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
        dd[dd == 0] = np.inf
        d[i] = dd.min()
    return float(d.mean()), float(d.std())


def main():
    print('=' * 68)
    print('PART 10a: HIDING THE LATTICE (the random chart)')
    print('=' * 68)

    print('[24] does the speed of light depend on direction?')
    angles = (0.0, 22.5, 45.0)
    lat = [lattice_speed(a) for a in angles]
    print(f'     lattice (wavelength = 4 cells), exactly reproducible:')
    print(f'       speeds {["%.3f" % v for v in lat]} at '
          f'{[f"{a}°" for a in angles]}'
          f' -> SYSTEMATIC anisotropy {(max(lat) / min(lat) - 1):.0%},'
          f' variance zero')
    reals = []
    for s in range(4):
        pts, ei, ej = build_graph(L=140.0, radius=2.8, seed=20 + s)
        reals.append([graph_speed(pts, ei, ej, a) for a in angles])
    reals = np.array(reals)
    mean, std = reals.mean(axis=0), reals.std(axis=0)
    print(f'     sprinkle (wavelength = 12 spacings, 4 realizations):')
    for i, a in enumerate(angles):
        print(f'       {a:>5.1f}°: v = {mean[i]:.3f} ± {std[i]:.3f}')
    print(f'       -> systematic spread of means '
          f'{(mean.max() / mean.min() - 1):.1%} (within noise ± '
          f'{(std / mean).mean():.1%}); direction of the per-run scatter')
    print('       is RANDOM, not fixed. The lattice has grain; the')
    print('       sprinkle has noise — and noise averages away with')
    print('       scale, which is why a Planck-scale sprinkle hides')
    print('       completely while a Planck-scale grid never could.')
    print()

    print('[25] does the substrate have a rest frame? (boost v = 0.6c)')
    L, v = 70, 0.6
    tt, xx = np.mgrid[0:L, 0:L].astype(float)
    lattice_ev = np.stack([tt.ravel(), xx.ravel()], axis=1)
    sprink_ev = np.random.default_rng(9).random((L * L, 2)) * L
    win = (L * 0.35, L * 0.65)
    rows = {}
    for name, ev in (('spacetime lattice', lattice_ev),
                     ('causal sprinkle', sprink_ev)):
        m0, s0 = nn_stats(ev, win)
        bv = boost(ev - L / 2, v) + L / 2
        m1, s1 = nn_stats(bv, win)
        rows[name] = (ev, bv)
        print(f'     {name:<17}: NN distance {m0:.3f}±{s0:.3f} -> '
              f'boosted {m1:.3f}±{s1:.3f}  '
              f'({abs(m1 - m0) / m0:.0%} shift)')
    print('     -> every observer in the lattice universe can measure')
    print('        their absolute velocity. In the sprinkled universe,')
    print('        no measurement can find the frame: an area-preserving')
    print('        map of a Poisson process is the same process.')

    # figure: four scatter panels
    S, pad = 340, 30
    img = Image.new('RGB', (4 * S + 5 * pad, S + 2 * pad + 20), (14, 14, 18))
    d = ImageDraw.Draw(img)
    panels = [('lattice', rows['spacetime lattice'][0]),
              ('lattice, boosted', rows['spacetime lattice'][1]),
              ('sprinkle', rows['causal sprinkle'][0]),
              ('sprinkle, boosted', rows['causal sprinkle'][1])]
    for k, (label, ev) in enumerate(panels):
        x0 = pad + k * (S + pad)
        lo, hi = L * 0.35, L * 0.65
        sel = ((ev[:, 0] > lo) & (ev[:, 0] < hi)
               & (ev[:, 1] > lo) & (ev[:, 1] < hi))
        for (t, x) in ev[sel]:
            px = x0 + (x - lo) / (hi - lo) * S
            py = pad + 20 + (t - lo) / (hi - lo) * S
            d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(170, 190, 220))
        d.text((x0 + 10, 8), label, fill=(200, 200, 210))
    img.save('films/lorentz.png')
    print('\n     films/lorentz.png')


if __name__ == '__main__':
    main()
