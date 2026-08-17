"""The dynamical ledger: checks that require rendering a universe.

For each candidate rule the observatory measures:

  c_b   butterfly speed — 0 means causally frozen: no physics
  dim   emergent spatial dimension from causal ball growth
  prt   free particles found chasing soup debris

Verdict: a rule 'makes a universe' if influence propagates at a
finite nonzero speed, space is 2-dimensional to the causal probe,
and matter supports free-traveling excitations.
"""
import numpy as np

from engine.substrate import Substrate
from observatory.causal import butterfly_and_dimension


def _frame(sub, family):
    g = sub.grid
    if family == 'complement' and sub.t % 2 == 1:
        g = 1 - g
    return (g != 0).astype(np.uint8)


def _tordist(p, q, n):
    dy = abs(p[0] - q[0]); dy = min(dy, n - dy)
    dx = abs(p[1] - q[1]); dx = min(dx, n - dx)
    return max(dy, dx)


def _blob_centroids(frame, link=2):
    n = frame.shape[0]
    ys, xs = np.nonzero(frame)
    cells = list(zip(ys.tolist(), xs.tolist()))
    if not cells or len(cells) > 400:
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
    return [(float(np.mean([c[0] for c in g])),
             float(np.mean([c[1] for c in g])), len(g))
            for g in groups.values()]


def count_free_particles(lut, family, size=96, soups=6, seed=3, states=2):
    """Detonate blobs, wait, chase debris; count clusters that travel
    steadily (speed >= 0.2 cells/tick over 30 ticks)."""
    rng = np.random.default_rng(seed)
    movers = 0
    for _ in range(soups):
        sub = Substrate(size, lut=lut, states=states)
        if states == 2:
            blob = (rng.random((8, 8)) < 0.5).astype(np.uint8)
        else:
            blob = (rng.integers(1, states, (8, 8)) *
                    (rng.random((8, 8)) < 0.5)).astype(np.uint8)
        sub.grid[28:36, 28:36] = blob
        for _ in range(150):
            sub.step()
        frames = [(_frame(sub, family)).copy()]
        for _ in range(30):
            sub.step()
            frames.append((_frame(sub, family)).copy())
        for (cy, cx, sz) in _blob_centroids(frames[0]):
            pos, ok = (cy, cx), True
            for f in frames[1:]:
                cl = _blob_centroids(f)
                near = min(cl, key=lambda c: _tordist(c[:2], pos, size),
                           default=None)
                if near is None or _tordist(near[:2], pos, size) > 4:
                    ok = False
                    break
                pos = near[:2]
            if not ok:
                continue
            dy = (pos[0] - cy + size / 2) % size - size / 2
            dx = (pos[1] - cx + size / 2) % size - size / 2
            if max(abs(dy), abs(dx)) / 30 >= 0.2:
                movers += 1
    return movers


def score_rule(lut, family, size=56):
    speed, dim, conv = butterfly_and_dimension(size, lut)
    particles = count_free_particles(lut, family) if speed > 0 else 0
    if speed <= 0.05:
        verdict = 'dead (frozen)'
    elif not conv:
        verdict = 'unresolved (probe too small)'
    elif not (1.6 <= dim <= 2.5):
        verdict = f'dead (space is {dim:.1f}-dimensional)'
    elif particles == 0:
        verdict = 'space, no matter'
    else:
        verdict = 'UNIVERSE'
    return {'c_b': speed, 'dim': dim, 'converged': conv,
            'particles': particles, 'verdict': verdict}
