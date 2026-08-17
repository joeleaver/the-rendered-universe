"""Corner the rule, then derive the map.

Stage 1 — the static ledger deletes rule-space by pure combinatorics:
          16! candidate reversible block rules -> the survivors.
Stage 2 — the observatory renders every survivor and scores it:
          does influence propagate? what dimension is space? is there
          matter that travels? (This is where 'renderer v2' becomes
          stage 1's measurement instrument.)
Stage 3 — for a rule that makes a universe, derive its geometry from
          causal structure alone — then wire a shortcut into the
          ENGINE's dynamics and watch the derived map grow a wormhole
          nobody drew.
"""
import numpy as np
from PIL import Image, ImageDraw

from engine.substrate import critters_lut
from rulespace.families import ledger_rules, FULL_SPACE
from observatory.causal import arrival_field
from observatory.scorecard import score_rule

SIZE = 64
LANDMARKS = [(y, x) for y in range(8, 56, 6) for x in range(8, 56, 6)]
WIRES = [((12, 12), (44, 44), (6, 6))]
TICKS = 70


def stage1():
    print('=' * 66)
    print('STAGE 1: THE STATIC LEDGER (combinatorics, zero simulation)')
    print('=' * 66)
    rules = ledger_rules()
    print(f'rule space (bijections on 2x2 block states): {FULL_SPACE:.2e}')
    print(f'+ isotropy (D4-equivariance) + exact matter conservation')
    print(f'+ stable vacuum  ->  {len(rules)} surviving rules')
    ck = critters_lut().tobytes()
    idx = next(i for i, (l, _) in enumerate(rules) if l.tobytes() == ck)
    print(f'(our part-1/2 universe, Critters, is survivor #{idx})')
    return rules, idx


def stage2(rules, critters_idx):
    print()
    print('=' * 66)
    print('STAGE 2: THE DYNAMICAL LEDGER (observatory scores each survivor)')
    print('=' * 66)
    print(f'{"rule":>4}  {"family":<10} {"c_b":>5}  {"dim":>5}  '
          f'{"particles":>9}  verdict')
    universes = []
    for i, (lut, fam) in enumerate(rules):
        s = score_rule(lut, fam)
        mark = ' *' if i == critters_idx else ''
        print(f'{i:>4}  {fam:<10} {s["c_b"]:>5.2f}  {s["dim"]:>5.2f}  '
              f'{s["particles"]:>9}  {s["verdict"]}{mark}')
        if s['verdict'] == 'UNIVERSE':
            universes.append(i)
    print(f'\nrules that make a universe: {universes}'
          f'  ({len(universes)} of {len(rules)}; * = Critters)')
    return universes


def _landmark_distances(lut, wires, seed=21):
    rng = np.random.default_rng(seed)
    n = len(LANDMARKS)
    arr = [arrival_field(SIZE, lut, s, TICKS, rng, wires=wires, refs=2)
           for s in LANDMARKS]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = 0.5 * (arr[i][LANDMARKS[j]] + arr[j][LANDMARKS[i]])
    return D


def _mds(D):
    D2 = D ** 2
    n = len(D)
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ D2 @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    coords = v[:, order[:2]] * np.sqrt(np.maximum(w[order[:2]], 0))
    return coords


def _draw_map(draw, coords, origin_x, title_y=14):
    xy = coords - coords.min(axis=0)
    span = max(xy.max(), 1e-9)
    xy = xy / span * 380 + 60
    xy[:, 1] += origin_x
    k = 8  # landmarks per row
    for i, (y, x) in enumerate(LANDMARKS):
        for j in (i + 1, i + k):  # right and down neighbors in engine order
            if j < len(LANDMARKS) and \
               (j != i + 1 or (i % k) != k - 1):
                draw.line([tuple(xy[i][::-1]), tuple(xy[j][::-1])],
                          fill=(70, 70, 80), width=1)
    for i, (gy, gx) in enumerate(LANDMARKS):
        r = 4
        color = (60 + int(180 * gy / SIZE), 60 + int(180 * gx / SIZE), 150)
        cy, cx = xy[i]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def stage3(lut):
    print()
    print('=' * 66)
    print('STAGE 3: GEOMETRY IS AN OUTPUT (causal cartography)')
    print('=' * 66)
    print('distance := how soon a flipped cell can influence another')
    print('(64 landmark probes, thermal medium; no cell index is ever')
    print(' treated as a position)')

    D_honest = _landmark_distances(lut, wires=())
    D_wired = _landmark_distances(lut, wires=WIRES)

    li = LANDMARKS.index((14, 14))
    lj = LANDMARKS.index((44, 44))
    grid_d = 30
    print(f'\nlandmarks at engine cells (14,14) and (44,44), '
          f'lattice distance {grid_d}:')
    print(f'  honest engine: causal distance = {D_honest[li, lj]:.1f} ticks')
    print(f'  wired engine:  causal distance = {D_wired[li, lj]:.1f} ticks')
    med = np.median(D_wired[D_wired > 0])
    print(f'  (wired-engine median landmark distance: {med:.1f} ticks)')
    print('\nThe derived map of the wired engine contains a wormhole.')
    print('Nobody drew it. It is IN the metric, because the metric is')
    print('computed from what can influence what. -> maps.png')

    img = Image.new('RGB', (1000, 520), (14, 14, 18))
    draw = ImageDraw.Draw(img)
    _draw_map(draw, _mds(D_honest), 0)
    _draw_map(draw, _mds(D_wired), 500)
    draw.text((170, 12), 'derived map: honest engine', fill=(200, 200, 210))
    draw.text((650, 12), 'derived map: wired engine', fill=(200, 200, 210))
    img.save('films/maps.png')


if __name__ == '__main__':
    rules, ck = stage1()
    universes = stage2(rules, ck)
    stage3(critters_lut())
