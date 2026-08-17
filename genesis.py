"""Part 17: genesis — the three-line program.

The whole program's claim, reduced to three postulates:

    1.  U is a bijection             (nothing is ever lost)
    2.  d(x, y) = -log I(x : y)      (nearness IS dependence)
    3.  K(seed) is small             (the beginning is written)

This file is the existence proof that the three lines RUN: a universe
from (rule, seed) — both lofted variables, the full specification a
few dozen bytes — with everything else derived.

The honesty device: the renderer never sees the engine's wiring.
Sites are opaque shuffled IDs — no coordinates, no adjacency, no
lattice. If space appears, it was output; there was nothing to peek
at.

Two renderers are run, and their disagreement is a finding:

  OBSERVATIONAL (watch only): mutual information between sensor
  histories. It FAILS (validation r ~ 0.07): the gas is full of
  period-locked bound states, and two oscillators sharing a period
  have large mutual information at ANY separation — observational
  correlation fakes wormholes everywhere. In a deterministic world,
  correlation is not causation, and space is causal.

  INTERVENTIONAL (poke, then watch): flip a site, time the arrival
  of the difference at other sites. Postulate 2 in its do-calculus
  form. Phase coincidence cannot fool an intervention. Space emerges:
  dimension ~2, geometry matching the hidden wiring at r ~ 0.9.

Four worlds are lofted from the part-3 ledger: two universes, one
space without matter, one stillborn.
"""
import heapq

import numpy as np
from PIL import Image, ImageDraw

from engine.substrate import Substrate
from rulespace.families import ledger_rules

SIZE = 64
SETTLE = 300
L = 500              # anonymous sensors
T_HIST = 4000        # observational history length
T_LINK = 8           # interventional reach for adjacency
T_CONE = 160         # long probes for the light-cone check
RULES = ledger_rules()

WORLDS = [
    ('universe A (rule 24, Critters)', 24, 'complement'),
    ('universe B (rule 11)', 11, 'strict'),
    ('space, no matter (rule 6)', 6, 'strict'),
    ('stillborn (rule 0)', 0, 'strict'),
]


def run_engine(rule_idx, family, seed=7):
    """Run a world. Returns only ID-keyed data; positions are kept
    creator-side for disclosed validation."""
    lut = RULES[rule_idx][0]
    rng = np.random.default_rng(seed)
    sub = Substrate(SIZE, lut=lut)
    sub.grid[8:56, 8:56] = (rng.random((48, 48)) < 0.5).astype(np.uint8)
    for _ in range(SETTLE):
        sub.step()
    base, t0 = sub.grid.copy(), sub.t
    idx = rng.choice(SIZE * SIZE, L, replace=False)
    ys, xs = idx // SIZE, idx % SIZE

    def frame(s):
        return s.grid ^ (s.t % 2) if family == 'complement' else s.grid

    # observational sensors: 3x3 'any matter' patches
    hist = np.zeros((L, T_HIST), dtype=np.uint8)
    for t in range(T_HIST):
        sub.step()
        f = frame(sub)
        patch = sum(np.roll(np.roll(f, a, 0), b, 1)
                    for a in (-1, 0, 1) for b in (-1, 0, 1))
        hist[:, t] = patch[ys, xs] > 0

    def probe(source, ticks):
        a = Substrate(SIZE, lut=lut)
        b = Substrate(SIZE, lut=lut)
        a.grid = base.copy()
        b.grid = base.copy()
        a.t = b.t = t0
        b.grid[ys[source], xs[source]] ^= 1
        out = np.full(L, np.inf)
        for t in range(1, ticks + 1):
            a.step()
            b.step()
            diff = (a.grid != b.grid)[ys, xs]
            fresh = diff & ~np.isfinite(out)
            out[fresh] = t
        return out

    arr = np.stack([probe(s, T_LINK) for s in range(L)])
    cone_src = rng.choice(L, 12, replace=False)
    cones = np.stack([probe(int(s), T_CONE) for s in cone_src])
    return hist, arr, cone_src, cones, (ys, xs)


# ---------------------------------------------------------------- render

def obs_mi(hist):
    H = hist.astype(np.float64)
    T = H.shape[1]
    n11 = (H @ H.T) / T
    p = H.mean(axis=1)
    p1a, p1b = p[:, None], p[None, :]
    I = np.zeros((L, L))
    for ja, jb in ((1, 1), (1, 0), (0, 1), (0, 0)):
        pab = (n11 if (ja, jb) == (1, 1) else
               p1a - n11 if (ja, jb) == (1, 0) else
               p1b - n11 if (ja, jb) == (0, 1) else
               1 - p1a - p1b + n11)
        qa = p1a if ja else 1 - p1a
        qb = p1b if jb else 1 - p1b
        with np.errstate(divide='ignore', invalid='ignore'):
            I += np.where(pab > 1e-12,
                          np.nan_to_num(pab * np.log(pab / (qa * qb))), 0)
    np.fill_diagonal(I, 0)
    return I


def causal_metric(arr):
    """Weighted graph on interventional reach; distance by Dijkstra."""
    W = np.minimum(arr, arr.T)
    edges = [[] for _ in range(L)]
    n_edges = 0
    for s in range(L):
        for j in np.where(W[s] <= T_LINK)[0]:
            if j != s:
                edges[s].append((int(j), float(W[s, j])))
                n_edges += 1
    if n_edges < L:
        return None
    D = np.full((L, L), np.inf)
    for s in range(L):
        dist = D[s]
        dist[s] = 0
        pq = [(0.0, s)]
        while pq:
            du, u = heapq.heappop(pq)
            if du > dist[u]:
                continue
            for v, w in edges[u]:
                nd = du + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
    return D


def dimension(D):
    iu = np.triu_indices(L, 1)
    off = D[iu][np.isfinite(D[iu])]
    if len(off) < 500 or off.std() < 1e-9:
        return None
    rs = np.quantile(off, np.linspace(0.08, 0.5, 12))
    if rs[-1] <= rs[0] * 1.05:
        return None
    counts = [(off < r).sum() for r in rs]
    return float(np.polyfit(np.log(rs), np.log(counts), 1)[0])


def embed2(D):
    Df = np.nan_to_num(D, posinf=np.nanmax(np.where(np.isfinite(D), D, 0))
                       * 1.2 + 1)
    J = np.eye(L) - 1 / L
    B = -0.5 * J @ (Df ** 2) @ J
    w, v = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    return v[:, o[:2]] * np.sqrt(np.maximum(w[o[:2]], 0))


def main():
    print('=' * 68)
    print('PART 17: GENESIS — THREE POSTULATES, UNIVERSES AS OUTPUT')
    print('=' * 68)
    print('specification per world: rule = 16 bytes (ledger); seed =')
    print('"48x48 coin flips" (~30 bytes); renderer = postulate 2.')
    print(f'renderer input: {L} anonymous sensors. No coordinates')
    print('exist anywhere in the render path.')
    print()

    panels = []
    for label, ridx, family in WORLDS:
        hist, arr, cone_src, cones, (ys, xs) = run_engine(ridx, family)
        dy = np.abs(ys[:, None] - ys[None, :])
        dx = np.abs(xs[:, None] - xs[None, :])
        td = np.maximum(np.minimum(dy, SIZE - dy),
                        np.minimum(dx, SIZE - dx))
        iu = np.triu_indices(L, 1)

        I = obs_mi(hist)
        d_obs = -np.log(np.clip(I, 1.5 / T_HIST, None) /
                        max(I.max(), 1e-9))
        ok_o = np.isfinite(d_obs[iu]) & (d_obs[iu] > 0)
        r_obs = (float(np.corrcoef(d_obs[iu][ok_o], td[iu][ok_o])[0, 1])
                 if ok_o.sum() > 500 and d_obs[iu][ok_o].std() > 1e-9
                 else 0.0)

        D = causal_metric(arr)
        print(f'--- {label}')
        print(f'     observational renderer (watch only): validation '
              f'r = {r_obs:+.2f}')
        if D is None:
            print('     interventional renderer: influence does not '
                  'spread. NO SPACE. Stillborn.')
            panels.append((label, None, (ys, xs), None))
        else:
            dim = dimension(D)
            ok = np.isfinite(D[iu])
            r_int = float(np.corrcoef(D[iu][ok], td[iu][ok])[0, 1])
            cx, cy = [], []
            for pi, s in enumerate(cone_src):
                fin = np.isfinite(cones[pi]) & np.isfinite(D[int(s)])
                fin[int(s)] = False
                cx += list(D[int(s)][fin])
                cy += list(cones[pi][fin])
            r_cone = float(np.corrcoef(cx, cy)[0, 1]) if len(cx) > 50 \
                else float('nan')
            print(f'     interventional renderer (poke, then watch): '
                  f'SPACE EMERGES')
            print(f'       dimension {dim:.2f} · connected '
                  f'{ok.mean():.0%} · light cones linear in the metric '
                  f'r = {r_cone:.2f}')
            print(f'       hidden-wiring recovery (validation, '
                  f'disclosed): r = {r_int:.2f}')
            panels.append((label, embed2(D), (ys, xs), dim))
        print()

    print('The observational renderer fails the same way in every')
    print('living world: period-locked matter gives distant sensors')
    print('shared rhythm, and rhythm masquerades as proximity. Space')
    print('could not be read off by watching; it had to be established')
    print('by acting. Correlation is not causation — and in a')
    print('deterministic world, SPACE is causation, rendered.')
    print()
    print('Three lines, ~50 lofted bytes per world, and geometry,')
    print('dimension, and causal order come out the other side — none')
    print('of which appear anywhere in the input. Not our universe:')
    print('A universe, on demand. The seed of ours is the only')
    print('missing argument.')

    S, pad = 340, 26
    img = Image.new('RGB', (2 * S + 3 * pad, 2 * S + 3 * pad + 30),
                    (14, 14, 18))
    dr = ImageDraw.Draw(img)
    for k, (label, xy, (ys, xs), dim) in enumerate(panels):
        x0 = pad + (k % 2) * (S + pad)
        y0 = pad + 20 + (k // 2) * (S + pad + 8)
        if xy is None:
            dr.text((x0 + 90, y0 + S // 2), 'no space', fill=(120, 120, 130))
        else:
            p = xy - xy.min(axis=0)
            p = p / max(p.max(), 1e-9) * (S - 20) + 10
            for i in range(L):
                col = (60 + int(185 * ys[i] / SIZE),
                       60 + int(185 * xs[i] / SIZE), 150)
                dr.ellipse([x0 + p[i, 1] - 2, y0 + p[i, 0] - 2,
                            x0 + p[i, 1] + 2, y0 + p[i, 0] + 2], fill=col)
        tag = label + (f'  (dim {dim:.1f})' if dim else '')
        dr.text((x0, y0 - 16), tag, fill=(200, 200, 210))
    img.save('films/genesis.png')
    print('\nfilms/genesis.png — maps colored by the HIDDEN wiring:')
    print('smooth gradients mean the renderer reconstructed, by poking')
    print('alone, the space it was never shown.')


if __name__ == '__main__':
    main()
