"""Films for part 17: the birth of space.

  genesis_birth.gif      — the derived map rebuilt as interventional
                           probes accumulate: dust -> islands -> a
                           coherent sheet. Frames Procrustes-aligned
                           so the crystallization is watchable.
  genesis_watch_vs_poke.png — the finding in one image: the
                           observational renderer's hairball beside
                           the interventional renderer's sheet.
"""
import heapq
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, '.')
import genesis as G

SIZE = G.SIZE
L = G.L


def partial_metric(arr, k):
    """All-pairs distances using only the first k probe rows."""
    edges = [[] for _ in range(L)]
    for s in range(k):
        for j in np.where(arr[s] <= G.T_LINK)[0]:
            if j != s:
                w = float(arr[s, j])
                edges[s].append((int(j), w))
                edges[int(j)].append((s, w))
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


def embed_visible(D):
    """2D MDS of the largest connected island; returns (ids, coords)."""
    reach = np.isfinite(D).sum(axis=1)
    hub = int(np.argmax(reach))
    core = np.where(np.isfinite(D[hub]))[0]
    if len(core) < 8:
        return np.array([], dtype=int), np.zeros((0, 2))
    Dc = D[np.ix_(core, core)]
    fin = np.isfinite(Dc)
    if not fin.all():
        Dc = np.where(fin, Dc, np.nanmax(np.where(fin, Dc, np.nan)) * 1.2)
    m = len(core)
    J = np.eye(m) - 1 / m
    B = -0.5 * J @ (Dc ** 2) @ J
    w, v = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    return core, v[:, o[:2]] * np.sqrt(np.maximum(w[o[:2]], 0))


def procrustes(prev_ids, prev_xy, ids, xy):
    """Rotate/reflect/scale xy to best match prev on shared ids."""
    common, ia, ib = np.intersect1d(prev_ids, ids, return_indices=True)
    if len(common) < 8:
        return xy
    A = prev_xy[ia] - prev_xy[ia].mean(0)
    B = xy[ib] - xy[ib].mean(0)
    U, s, Vt = np.linalg.svd(B.T @ A)
    R = U @ Vt
    scale = s.sum() / max((B ** 2).sum(), 1e-12)
    return (xy - xy[ib].mean(0)) @ R * scale + prev_xy[ia].mean(0)


def draw(xy, ids, ys, xs, k, frac):
    S = 420
    img = Image.new('RGB', (S, S + 30), (14, 14, 18))
    d = ImageDraw.Draw(img)
    d.text((12, 8), f'interventions: {k:>3}   space: {frac:.0%} of '
                    f'sensors placed', fill=(200, 200, 210))
    if len(ids):
        p = xy - xy.min(axis=0)
        p = p / max(p.max(), 1e-9) * (S - 30) + 15
        for n, i in enumerate(ids):
            col = (60 + int(185 * ys[i] / SIZE),
                   60 + int(185 * xs[i] / SIZE), 150)
            d.ellipse([p[n, 1] - 2, 30 + p[n, 0] - 2,
                       p[n, 1] + 2, 30 + p[n, 0] + 2], fill=col)
    return img


def main():
    print('running universe A (Critters) with anonymized sensors...')
    hist, arr, cone_src, cones, (ys, xs) = G.run_engine(24, 'complement')

    print('rendering the birth of space...')
    frames = []
    prev_ids, prev_xy = np.array([], dtype=int), np.zeros((0, 2))
    ks = list(range(10, 151, 10)) + list(range(170, 501, 30))
    for k in ks:
        D = partial_metric(arr, k)
        ids, xy = embed_visible(D)
        if len(ids):
            xy = procrustes(prev_ids, prev_xy, ids, xy)
            prev_ids, prev_xy = ids, xy
        frames.append(draw(xy, ids, ys, xs, k, len(ids) / L))
    durations = [200] * (len(frames) - 1) + [2500]
    frames[0].save('films/genesis_birth.gif', save_all=True,
                   append_images=frames[1:], duration=durations, loop=0)
    print('films/genesis_birth.gif')

    # watch vs poke
    I = G.obs_mi(hist)
    d_obs = -np.log(np.clip(I, 1.5 / G.T_HIST, None) / max(I.max(), 1e-9))
    np.fill_diagonal(d_obs, 0)
    core_o = np.arange(L)
    m = L
    J = np.eye(m) - 1 / m
    B = -0.5 * J @ (d_obs ** 2) @ J
    w, v = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    xy_obs = v[:, o[:2]] * np.sqrt(np.maximum(w[o[:2]], 0))
    D_full = G.causal_metric(arr)
    ids_i, xy_int = embed_visible(D_full)

    S, pad = 420, 24
    img = Image.new('RGB', (2 * S + 3 * pad, S + 2 * pad + 26), (14, 14, 18))
    d = ImageDraw.Draw(img)
    for panel, (label, ids, xy) in enumerate(
            (('WATCH: d = -log I on histories (r = 0.09)', core_o, xy_obs),
             ('POKE: d from interventions (r = 0.89)', ids_i, xy_int))):
        x0 = pad + panel * (S + pad)
        p = xy - xy.min(axis=0)
        p = p / max(p.max(), 1e-9) * (S - 24) + 12
        for n, i in enumerate(ids):
            col = (60 + int(185 * ys[i] / SIZE),
                   60 + int(185 * xs[i] / SIZE), 150)
            d.ellipse([x0 + p[n, 1] - 2, pad + 22 + p[n, 0] - 2,
                       x0 + p[n, 1] + 2, pad + 22 + p[n, 0] + 2], fill=col)
        d.text((x0 + 8, 10), label, fill=(200, 200, 210))
    img.save('films/genesis_watch_vs_poke.png')
    print('films/genesis_watch_vs_poke.png')


if __name__ == '__main__':
    main()
