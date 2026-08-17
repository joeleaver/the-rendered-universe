"""Part 5: curved space — the Einstein cartoon, from entanglement.

A Gaussian 'star' of extra mass sits in the middle of the scalar
field's world. Nothing else changes: same springs, same graph, same
renderer rule (distance = -log MI, vacuum-calibrated). Experiments:

  [12] the metric responds to matter: local rulers stretch where the
       matter sits, monotonically in the matter density.
  [13] geodesics bend: the shortest path between two points on
       opposite sides of the star detours around it.
  [14] the well: embed the derived metric in 3D and look at it —
       a flat sheet where there is no matter, a funnel where there is.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from observatory.curved import (landmark_lattice, lattice_edges,
                                edge_rulers, calibrated_weights,
                                all_pairs, recover_path, embed)

N = 45
M0, SIGMA = 0.2, 5.0
CENTER = (22, 22)
SWEEP = (0.0, 0.25, 0.5, 1.0, 2.0)

CORNERS, SIDE = landmark_lattice(N)
EDGES = lattice_edges(SIDE)
CIDX = CORNERS.index(CENTER)


def mass_profile(A):
    yy, xx = np.mgrid[0:N, 0:N]
    r2 = (yy - CENTER[0]) ** 2 + (xx - CENTER[1]) ** 2
    return M0 + A * np.exp(-r2 / (2 * SIGMA ** 2))


def node_edge_ids(node):
    return [e for e, (i, j, _) in enumerate(EDGES) if node in (i, j)]


def lm(r, c):
    return r * SIDE + c


def render_heatmap(draw, x0, dw, path, label):
    cell = 30
    vmax = max(dw.max(), 1e-9)
    for r in range(SIDE):
        for c in range(SIDE):
            t = dw[lm(r, c)] / vmax
            col = (int(25 + 230 * t), int(25 + 130 * t), int(90 - 40 * t))
            x, y = x0 + 55 + c * cell, 70 + r * cell
            draw.rectangle([x, y, x + cell - 2, y + cell - 2], fill=col)

    def px(node):
        r, c = divmod(node, SIDE)
        return (x0 + 55 + c * cell + cell // 2, 70 + r * cell + cell // 2)

    chord = [px(lm(SIDE // 2, c)) for c in (0, SIDE - 1)]
    draw.line(chord, fill=(110, 110, 120), width=2)
    draw.line([px(p) for p in path], fill=(120, 235, 140), width=4)
    draw.text((x0 + 110, 30), label, fill=(205, 205, 215))


def render_surface(draw, x0, coords, label, tilt=1.05):
    xyz = coords - coords.mean(axis=0)
    if xyz[CIDX, 2] > 0:
        xyz[:, 2] *= -1  # draw the well downward
    span = max(np.abs(xyz[:, :2]).max(), 1e-9)
    s = 195 / span
    ct, st = math.cos(tilt), math.sin(tilt)
    pts, depth = [], []
    for (x, y, z) in xyz:
        pts.append((x0 + 250 + s * x, 280 + s * (y * ct - z * st) * 0.9))
        depth.append(y * st + z * ct)
    segs = []
    for r in range(SIDE):
        for c in range(SIDE):
            i = lm(r, c)
            for (rr, cc) in ((r, c + 1), (r + 1, c)):
                if rr < SIDE and cc < SIDE:
                    j = lm(rr, cc)
                    segs.append(((depth[i] + depth[j]) / 2, i, j))
    segs.sort()
    dmin = min(depth); dspan = max(depth) - dmin + 1e-9
    for (d, i, j) in segs:
        b = 70 + int(140 * (d - dmin) / dspan)
        draw.line([pts[i], pts[j]], fill=(b, b, int(b * 1.15)), width=2)
    cx, cy = pts[CIDX]
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(240, 120, 90))
    draw.text((x0 + 160, 30), label, fill=(205, 205, 215))


def main():
    print('=' * 68)
    print('PART 5: CURVED SPACE (a star made of mass, a metric made of MI)')
    print('=' * 68)
    print(f'vacuum calibration run (A=0)...')
    w_vac = edge_rulers(N, CORNERS, EDGES, mass_profile(0.0))

    central = node_edge_ids(CIDX)
    lat_len = np.array([L for _, _, L in EDGES])
    thr_i, thr_j = lm(SIDE // 2, 0), lm(SIDE // 2, SIDE - 1)
    far_i, far_j = lm(0, 0), lm(0, SIDE - 1)

    print()
    print('[12] the metric responds to matter (vacuum-calibrated rulers):')
    print(f'     {"A":>5} {"central stretch":>16} {"through/around":>15} '
          f'{"well depth":>11}')
    results, depth0 = {}, 0.0
    for A in SWEEP:
        w = w_vac if A == 0 else edge_rulers(N, CORNERS, EDGES,
                                             mass_profile(A))
        cw = calibrated_weights(EDGES, w, w_vac)
        D, nxt = all_pairs(SIDE, EDGES, cw)
        stretch = float((cw[central] / lat_len[central]).mean())
        ratio = float(D[thr_i, thr_j] / D[far_i, far_j])
        coords = embed(D, dims=3)
        z = coords[:, 2] - np.median(coords[:, 2])
        depth = float(np.abs(z).max())
        if A == 0:
            depth0 = depth
        results[A] = (w, cw, D, nxt, coords)
        print(f'     {A:>5.2f} {stretch:>15.2f}x {ratio:>15.2f} '
              f'{max(depth - depth0, 0):>11.1f}')
    print('     -> rulers stretch where the matter is, monotonically in')
    print('        the matter density. Matter tells space how to curve.')
    print()

    A = SWEEP[-1]
    w, cw, D, nxt, coords = results[A]
    path = recover_path(nxt, thr_i, thr_j)
    rows = [divmod(p, SIDE)[0] for p in path]
    deflect = max(abs(r - SIDE // 2) for r in rows) * 3
    straight = sum(cw[e] for e, (i, j, _) in enumerate(EDGES)
                   if divmod(i, SIDE)[0] == divmod(j, SIDE)[0] == SIDE // 2)
    print(f'[13] geodesic bending (A={A}):')
    print(f'     shortest path between opposite sides of the star bows')
    print(f'     {deflect} cells around it; going straight through would')
    print(f'     cost {straight:.1f} ruler units vs {D[thr_i, thr_j]:.1f} '
          f'for the detour')
    print()

    print('[14] the well: derived space, embedded and looked at ->')
    print('     films/curved_space.png')
    img = Image.new('RGB', (1520, 560), (14, 14, 18))
    d = ImageDraw.Draw(img)
    dw_map = np.array([np.mean([(w - w_vac)[e] for e in node_edge_ids(i)])
                       for i in range(SIDE * SIDE)])
    render_heatmap(d, 0, dw_map, path,
                   f'metric perturbation + geodesic (A={A})')
    render_surface(d, 505, results[0.0][4], 'derived space, no matter')
    render_surface(d, 1010, coords, f'derived space, star at center (A={A})')
    img.save('films/curved_space.png')


if __name__ == '__main__':
    main()
