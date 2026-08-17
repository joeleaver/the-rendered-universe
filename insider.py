"""Part 10d: the first observer made of pixels.

Every physicist in this repo so far has lived OUTSIDE the universe,
watching frames through a firewall. A real observer is a pattern in
the render, made of the same matter it measures. The minimum viable
observer is a detector: a stable structure that changes to a
DIFFERENT stable structure when something passes — a 1-bit memory
written by a physical event.

The search: settle every small seed into a bound state, fire the
part-1 glider at it across a range of impact parameters, and look for
collisions whose outcome near the target is stable, isolated, and
different from the original — a record. (Part 9 taught what to
expect: records are exactly the thing collisions make and vacuum
preserves.)
"""
import numpy as np
from PIL import Image
from itertools import combinations

from engine.substrate import Substrate

SIZE = 96
CENTER = (48, 48)
GLIDER = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 1, 1, 0]],
                  dtype=np.uint8)  # part-1 species: v=(-2,0)/4, odd parity
SETTLE = 60
T_ANALYZE = 212
R_LOCAL, R_RING = 10, 18


def frame_of(sub):
    return sub.grid ^ (sub.t % 2)


def local_cells(frame, radius=R_LOCAL):
    ys, xs = np.nonzero(frame)
    out = set()
    for y, x in zip(ys.tolist(), xs.tolist()):
        dy = min(abs(y - CENTER[0]), SIZE - abs(y - CENTER[0]))
        dx = min(abs(x - CENTER[1]), SIZE - abs(x - CENTER[1]))
        if max(dy, dx) <= radius:
            out.add((y, x))
    return frozenset(out)


def ring_empty(frame):
    ys, xs = np.nonzero(frame)
    for y, x in zip(ys.tolist(), xs.tolist()):
        dy = min(abs(y - CENTER[0]), SIZE - abs(y - CENTER[0]))
        dx = min(abs(x - CENTER[1]), SIZE - abs(x - CENTER[1]))
        if R_LOCAL < max(dy, dx) <= R_RING:
            return False
    return True


def collide(bound_grid, b):
    sub = Substrate(SIZE)
    sub.grid = bound_grid.copy()
    sub.t = SETTLE
    y0, x0 = 81, 49 + b
    sub.grid[y0:y0 + 3, x0:x0 + 4] = GLIDER
    while sub.t < T_ANALYZE:
        sub.step()
    f = frame_of(sub)
    after = local_cells(f)
    isolated = ring_empty(f)
    stable = False
    after_phases = {after}
    for _ in range(24):
        sub.step()
        fp = local_cells(frame_of(sub))
        if fp == after and ring_empty(frame_of(sub)):
            stable = True
            break
        after_phases.add(fp)
    return after, isolated and stable, after_phases, sub


def main():
    print('=' * 68)
    print('PART 10d: THE FIRST OBSERVER MADE OF PIXELS')
    print('=' * 68)

    # build the bound-state library (settled in place, phases recorded)
    lib = []
    seen = set()
    for k in range(2, 7):
        for combo in combinations(range(9), k):
            s = np.zeros(9, dtype=np.uint8)
            s[list(combo)] = 1
            sub = Substrate(SIZE)
            sub.grid[CENTER[0]:CENTER[0] + 3,
                     CENTER[1]:CENTER[1] + 3] = s.reshape(3, 3)
            for _ in range(SETTLE):
                sub.step()
            f0 = local_cells(frame_of(sub), radius=8)
            if not f0 or not ring_empty(frame_of(sub)):
                continue
            snap = sub.grid.copy()
            phases, stable = {f0}, False
            for _ in range(24):
                sub.step()
                fp = local_cells(frame_of(sub), radius=8)
                if fp == f0:
                    stable = True
                    break
                phases.add(fp)
            if stable and f0 not in seen:
                seen.add(f0)
                lib.append((snap, phases))
    print(f'library: {len(lib)} distinct stable bound states '
          f'(from 456 seeds)')

    outcomes = {'unchanged': 0, 'RECORD': 0, 'annihilated': 0, 'messy': 0}
    detectors = []
    for i, (snap, phases) in enumerate(lib):
        for b in range(-6, 7, 2):
            after, ok, after_phases, sub = collide(snap, b)
            if not ok:
                outcomes['messy'] += 1
            elif not after:
                outcomes['annihilated'] += 1
            elif after in phases:
                outcomes['unchanged'] += 1
            else:
                outcomes['RECORD'] += 1
                detectors.append((i, b, snap, phases, after,
                                  after_phases))
    total = sum(outcomes.values())
    print(f'{total} collision experiments '
          f'({len(lib)} targets x 7 impact parameters):')
    for k, v in outcomes.items():
        print(f'     {k:>12}: {v:>4}  ({v / total:.0%})')

    if detectors:
        # showcase the detector whose matter count changed the most:
        # an unambiguous transformation, not just a nudged twin
        detectors.sort(key=lambda d: -abs(len(d[4])
                                          - len(next(iter(d[3])))))
        i, b, snap, phases, after, after_phases = detectors[0]
        n_before = len(next(iter(phases)))
        print(f'\nshowcase detector: target #{i}, impact parameter {b:+d}:')
        print(f'     before: stable {n_before}-cell bound state')
        print(f'     after:  stable {len(after)}-cell bound state — '
              f'the passage is recorded in matter')
        # persistence check: isolate the record in vacuum first (the
        # departing glider wraps the torus and would re-collide — a
        # small-universe artifact), then wait 400 ticks
        _, ok, _, sub = collide(snap, b)
        f = frame_of(sub)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        dy = np.minimum(np.abs(yy - CENTER[0]), SIZE - np.abs(yy - CENTER[0]))
        dx = np.minimum(np.abs(xx - CENTER[1]), SIZE - np.abs(xx - CENTER[1]))
        far = np.maximum(dy, dx) > R_RING
        sub.grid[far] = sub.t % 2  # vacuum, at the current phase
        for _ in range(400):
            sub.step()
        persist = local_cells(frame_of(sub))
        print(f'     record after 400 more ticks: '
              f'{"INTACT" if persist in after_phases else "changed"}'
              f' ({len(persist)} cells)')
        print('     A pattern inside the universe, obeying only the')
        print('     three-line rule, that irreversibly-in-practice')
        print('     remembers that something passed. Observers are not')
        print('     extra physics. They are weather.')

        # figure: before / during / after
        frames = []
        sub2 = Substrate(SIZE)
        sub2.grid = snap.copy()
        sub2.t = SETTLE
        y0, x0 = 81, 49 + b
        sub2.grid[y0:y0 + 3, x0:x0 + 4] = GLIDER
        snap_ts = {SETTLE + 4: 'before (glider incoming)',
                   126: 'collision', T_ANALYZE: 'after: the record'}
        while sub2.t <= T_ANALYZE:
            if sub2.t in snap_ts:
                frames.append((snap_ts[sub2.t], frame_of(sub2).copy()))
            sub2.step()
        scale = 6
        W = len(frames) * (64 * scale + 20) + 20
        img = Image.new('RGB', (W, 64 * scale + 46), (14, 14, 18))
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        for k, (label, f) in enumerate(frames):
            crop = np.roll(np.roll(f, 32 - CENTER[0], 0),
                           32 - CENTER[1], 1)[:64, :64]
            x_off = 20 + k * (64 * scale + 20)
            for (yy, xx) in zip(*np.nonzero(crop)):
                d.rectangle([x_off + xx * scale, 40 + yy * scale,
                             x_off + xx * scale + scale - 1,
                             40 + yy * scale + scale - 1],
                            fill=(225, 225, 232))
            d.text((x_off, 12), label, fill=(200, 200, 210))
        img.save('films/insider.png')
        print('\n     films/insider.png')


if __name__ == '__main__':
    main()
