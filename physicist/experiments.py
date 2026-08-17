"""The experimental program.

Six experiments, in the order a civilization might run them:

  1. measure_lightspeed — how fast can any disturbance propagate?
  2. check_conservation — is anything exactly conserved?
  3. hunt_particles     — do small lumps of matter form stable objects?
  4. capture_from_debris— catch free-flying particles in cosmic debris
  5. test_isotropy      — are the laws the same in every direction?
  6. ballistic_survey   — fire particles across the whole sky; do the
                          laws hold at every point in space?

Everything below sees only frames. `lab` is a zero-argument callable
that creates a fresh universe and returns its Instruments.
"""
import math
from itertools import combinations

import numpy as np


# ---------------------------------------------------------------- helpers

def _tordist(p, q, n):
    """Chebyshev distance on the torus."""
    dy = abs(p[0] - q[0]); dy = min(dy, n - dy)
    dx = abs(p[1] - q[1]); dx = min(dx, n - dx)
    return max(dy, dx)


def _circ_mean(vals, n):
    """Mean of positions on a ring of circumference n."""
    th = np.asarray(vals, dtype=float) * (2 * math.pi / n)
    m = math.atan2(np.sin(th).mean(), np.cos(th).mean())
    return (m * n / (2 * math.pi)) % n


def _clusters(frame, link=3):
    """Group live cells into blobs (toroidal, Chebyshev linkage).
    Returns list of (cy, cx, size)."""
    n = frame.shape[0]
    ys, xs = np.nonzero(frame)
    cells = list(zip(ys.tolist(), xs.tolist()))
    if not cells:
        return []
    parent = list(range(len(cells)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if _tordist(cells[i], cells[j], n) <= link:
                parent[find(i)] = find(j)
    groups = {}
    for i, c in enumerate(cells):
        groups.setdefault(find(i), []).append(c)
    out = []
    for g in groups.values():
        cy = _circ_mean([c[0] for c in g], n)
        cx = _circ_mean([c[1] for c in g], n)
        out.append((cy, cx, len(g)))
    return out


def _verify_glider(lab, seed, parity, max_period=24):
    """Place `seed` in vacuum at the given (y, x) parity and demand that
    the WHOLE universe repeats as an exact translated copy of itself over
    at least three full periods. Shuttling oscillators fail this.

    Returns {'period', 'velocity', 'speed'} or None.
    """
    u = lab()
    n = u.frame().shape[0]
    u.poke(n // 2 + parity[0], n // 2 + parity[1], seed)
    frames = [u.frame().copy()]
    for _ in range(3 * max_period):
        u.tick()
        frames.append(u.frame().copy())
    f0 = frames[0]
    if f0.sum() == 0:
        return None
    ys, xs = np.nonzero(f0)
    com0 = (_circ_mean(ys, n), _circ_mean(xs, n))
    for p in range(1, max_period + 1):
        fp = frames[p]
        if fp.sum() != f0.sum():
            continue
        ys2, xs2 = np.nonzero(fp)
        dy = round(float(_circ_mean(ys2, n) - com0[0]))
        dx = round(float(_circ_mean(xs2, n) - com0[1]))
        if (dy, dx) == (0, 0):
            if np.array_equal(f0, fp):
                return None  # oscillator or still life
            continue
        if all(np.array_equal(np.roll(f0, (k * dy, k * dx), axis=(0, 1)),
                              frames[k * p]) for k in (1, 2, 3)):
            return {'period': p, 'velocity': (dy, dx),
                    'speed': max(abs(dy), abs(dx)) / p}
    return None


# ------------------------------------------------------------ experiments

def measure_lightspeed(lab, trials=6, ticks=48, seed=1):
    """Drop dense blobs into vacuum; track the fastest front.
    Returns the maximum observed propagation speed (cells/tick)."""
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(trials):
        u = lab()
        n = u.frame().shape[0]
        c0 = (n // 2, n // 2)
        blob = (rng.random((6, 6)) < 0.5).astype(np.uint8)
        u.poke(c0[0] - 3, c0[1] - 3, blob)
        r0 = 3
        for t in range(1, ticks + 1):
            u.tick()
            f = u.frame()
            ys, xs = np.nonzero(f)
            if len(ys) == 0:
                continue
            r = max(_tordist((y, x), c0, n) for y, x in zip(ys, xs))
            best = max(best, (r - r0) / t)
    return best


def check_conservation(lab, ticks=400, seed=2):
    """Fill a region with random soup; watch total brightness."""
    rng = np.random.default_rng(seed)
    u = lab()
    n = u.frame().shape[0]
    soup = (rng.random((n // 2, n // 2)) < 0.4).astype(np.uint8)
    u.poke(n // 4, n // 4, soup)
    counts = []
    for _ in range(ticks):
        counts.append(int(u.frame().sum()))
        u.tick()
    return min(counts), max(counts)


def hunt_particles(lab, settle=48):
    """Drop every small seed (2-6 cells in a 3x3) into vacuum and
    classify what it becomes. Returns a taxonomy dict."""
    taxonomy = {'evaporated': 0, 'bound': 0, 'complex': 0, 'travelers': 0}
    for k in range(2, 7):
        for combo in combinations(range(9), k):
            s = np.zeros(9, dtype=np.uint8)
            s[list(combo)] = 1
            seed = s.reshape(3, 3)
            u = lab()
            n = u.frame().shape[0]
            u.poke(n // 2, n // 2, seed)
            u.tick(settle)
            f0 = u.frame().copy()
            if f0.sum() == 0:
                taxonomy['evaporated'] += 1
                continue
            res = _verify_glider(lab, seed, (0, 0), max_period=16)
            if res:
                taxonomy['travelers'] += 1
                continue
            # bound = stays localized near origin; complex = spreads out
            u.tick(64)
            f1 = u.frame()
            ys, xs = np.nonzero(f1)
            r = max((_tordist((y, x), (n // 2, n // 2), n)
                     for y, x in zip(ys, xs)), default=0)
            taxonomy['bound' if r <= 12 else 'complex'] += 1
    return taxonomy


def capture_from_debris(lab, n_soups=8, seed=3):
    """Detonate random blobs, wait for the debris to clear, and chase
    every free-flying cluster. Each steady mover is extracted cell by
    cell and re-created in a clean vacuum; only exact repeaters count.

    Returns list of species: {'seed', 'parity', 'period', 'velocity',
    'speed', 'cells'}.
    """
    rng = np.random.default_rng(seed)
    species = []
    seen = set()
    for _ in range(n_soups):
        u = lab()
        n = u.frame().shape[0]
        u.poke(60, 60, (rng.random((8, 8)) < 0.5).astype(np.uint8))
        u.tick(150)
        frames = [u.frame().copy()]
        for _ in range(40):
            u.tick()
            frames.append(u.frame().copy())
        for (cy, cx, sz) in _clusters(frames[0], link=2):
            pos, ok = (cy, cx), True
            for f in frames[1:]:
                cl = _clusters(f, link=2)
                nearest = min(cl, key=lambda c: _tordist(c[:2], pos, n),
                              default=None)
                if nearest is None or _tordist(nearest[:2], pos, n) > 4:
                    ok = False
                    break
                pos = nearest[:2]
            if not ok:
                continue
            dy = (pos[0] - cy + n / 2) % n - n / 2
            dx = (pos[1] - cx + n / 2) % n - n / 2
            if max(abs(dy), abs(dx)) / 40 < 0.2:
                continue  # drifting debris, not a ballistic mover
            # extract the mover's exact cells and clone it into vacuum
            f0 = frames[0]
            ys, xs = np.nonzero(f0)
            cells = [(y, x) for y, x in zip(ys.tolist(), xs.tolist())
                     if _tordist((y, x), (cy, cx), n) <= 3]
            y0 = min(c[0] for c in cells)
            x0 = min(c[1] for c in cells)
            pat = np.zeros((max(c[0] for c in cells) - y0 + 1,
                            max(c[1] for c in cells) - x0 + 1), dtype=np.uint8)
            for y, x in cells:
                pat[y - y0, x - x0] = 1
            key = (pat.tobytes(), pat.shape, y0 % 2, x0 % 2)
            if key in seen:
                continue
            seen.add(key)
            res = _verify_glider(lab, pat, (y0 % 2, x0 % 2))
            if res:
                species.append({'seed': pat, 'parity': (y0 % 2, x0 % 2),
                                'cells': int(pat.sum()), **res})
    # dedup by velocity+shape signature
    uniq = {}
    for s in species:
        uniq[(s['seed'].tobytes(), s['seed'].shape)] = s
    return list(uniq.values())


def test_isotropy(lab, sp):
    """If space has no preferred direction, a rotated particle should fly
    too. Rotate/flip a captured species and test each transform (trying
    all four lattice parities). Returns list of verified variants."""
    variants = []
    forms = []
    for k in range(4):
        forms.append(np.rot90(sp['seed'], k))
    forms.append(np.fliplr(sp['seed']))
    forms.append(np.flipud(sp['seed']))
    seen = set()
    for pat in forms:
        key = (pat.tobytes(), pat.shape)
        if key in seen:
            continue
        seen.add(key)
        for py in (0, 1):
            for px in (0, 1):
                res = _verify_glider(lab, np.ascontiguousarray(pat), (py, px))
                if res:
                    variants.append({'seed': np.ascontiguousarray(pat),
                                     'parity': (py, px),
                                     'cells': int(pat.sum()), **res})
                    break
            else:
                continue
            break
    uniq = {}
    for v in variants:
        uniq[v['velocity']] = v
    return list(uniq.values())


def track_blob(u, ticks, jump_threshold=16):
    """Follow the nearest blob frame by frame. Returns (positions, events)
    where events are apparent discontinuities: (t, from, to, distance)."""
    n = u.frame().shape[0]
    cl = _clusters(u.frame())
    if not cl:
        return [], []
    pos = max(cl, key=lambda c: c[2])[:2]
    positions = [pos]
    events = []
    for t in range(1, ticks + 1):
        u.tick()
        cl = _clusters(u.frame())
        if not cl:
            positions.append(pos)
            continue
        nearest = min(cl, key=lambda c: _tordist(c[:2], pos, n))
        d = _tordist(nearest[:2], pos, n)
        if d > jump_threshold:
            events.append((t, pos, nearest[:2], d))
        pos = nearest[:2]
        positions.append(pos)
    return positions, events


def ballistic_survey(lab, probes, max_ticks=1200):
    """Fire probe particles from launch sites covering the sky, one
    universe per shot. In a universe with honest geometry every
    trajectory is a straight line at the probe's speed. Report every
    violation."""
    u0 = lab()
    n = u0.frame().shape[0]
    report = []
    for sp in probes:
        ticks = min(max_ticks, int(96 / max(sp['speed'], 1e-9)) + 64)
        vertical = sp['velocity'][0] != 0
        if vertical:
            starts = [(n // 2 + sp['parity'][0], x + sp['parity'][1])
                      for x in range(0, n, 8)]
        else:
            starts = [(y + sp['parity'][0], n // 2 + sp['parity'][1])
                      for y in range(0, n, 8)]
        for (y0, x0) in starts:
            u = lab()
            u.poke(y0, x0, sp['seed'])
            _, events = track_blob(u, ticks)
            report.append({'start': (y0, x0), 'velocity': sp['velocity'],
                           'events': events})
    return report
