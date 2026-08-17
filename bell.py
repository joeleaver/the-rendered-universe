"""Part 2: entanglement cartography.

Same universe as run.py — same rule, same chart — plus the substrate's
deep-matter excitation. The physicist, having located the doors in
part 1, now tests the chart theory's sharpest prediction: two patches
of sky that are far apart on screen but adjacent in the mechanism.
"""
import itertools
import math

import numpy as np

from engine.substrate import Substrate
from engine.deep import DeepField, COHERENCE_LENGTH
from render.screen import Renderer
from physicist.instruments import Instruments
from physicist import bell_experiments as bx

SIZE = 128
PORTALS = [((16, 16), (96, 32), (12, 12))]

_seed_counter = itertools.count(1000)
_last_deep = []  # engine-side handles, for the creator's log only


def make_lab():
    def new_universe():
        sub = Substrate(SIZE)
        ren = Renderer(sub, swaps=PORTALS)
        deep = DeepField(sub, np.random.default_rng(next(_seed_counter)))
        _last_deep.append(deep)

        def tick(k=1):
            for _ in range(k):
                sub.step()
                deep.tick()

        def emit_pair(sa, sb):
            ea = (int(ren.src_y[sa]), int(ren.src_x[sa]))
            eb = (int(ren.src_y[sb]), int(ren.src_x[sb]))
            deep.emit(ea, eb)

        def set_analyzer(s, theta):
            e = (int(ren.src_y[s]), int(ren.src_x[s]))
            deep.set_setting(e, theta)

        return Instruments(ren.frame, ren.poke, tick, emit_pair, set_analyzer)
    return new_universe


def main():
    lab = make_lab()

    print('=' * 64)
    print("PHYSICIST'S LAB NOTEBOOK, VOLUME II: THE DOORS ARE ADJACENT")
    print('=' * 64)
    print('Chart theory prediction: the strip left of door D1 and the')
    print('interior of door D2 are mechanism-adjacent, though far apart')
    print('on screen. Test: play the CHSH game across that gap, timed so')
    print('no screen-space influence at c could connect the stations.')
    print()

    res = bx.pixel_chsh_game(lab)
    if res is None:
        print('[7] pixel CHSH game: no violating configuration found')
    else:
        print(f'[7] pixel CHSH game (raw matter, patch-parity outcomes, '
              f'readout at t={bx.GAME_T}):')
        print(f'    screen gap between stations: {res["gap_main"]} cells; '
              f'light needs {res["gap_main"]} ticks > {bx.GAME_T} — '
              f'stations are screen-causally isolated')
        print(f'    predicted placement:  |S| = {abs(res["S"])}   '
              f'(locality bound: 2, found after {res["configs_tried"]} '
              f'pattern configs)')
        print(f'    control placement (gap {res["gap_ctrl"]}): '
              f'|S| = {abs(res["S_control"])}')
        table = {xy: ab for xy, ab in sorted(res['outcomes'].items())}
        print(f'    outcome table (A,B per settings x,y): {table}')
        print('    -> outcomes at one station depend on the OTHER station\'s')
        print('       setting: the doors are a signaling channel. Crude.')
    print()

    print(f'[8] deep-matter CHSH (analyzer angles, 12000 pairs/placement):')
    main_r = bx.deep_chsh(lab, bx.DEEP_L, bx.DEEP_R)
    ctrl_r = bx.deep_chsh(lab, bx.DEEP_CTRL_L, bx.DEEP_CTRL_R, seed=18)
    se = 2 / math.sqrt(main_r['n_used'] / 4)
    print(f'    predicted placement:  |S| = {abs(main_r["S"]):.3f} '
          f'± {se:.3f}   (classical bound 2, quantum bound 2.828)')
    print(f'    control placement:    |S| = {abs(ctrl_r["S"]):.3f} '
          f'± {se:.3f}')
    print(f'    no-signaling audit, predicted placement: max marginal shift '
          f'when the far setting changes = {main_r["signal_leak"]:.3f} '
          f'(consistent with zero)')
    print()
    print('    -> Across the hidden adjacency: correlations violate the')
    print('       screen-locality bound at the Tsirelson value, yet carry')
    print('       NO signal — each station sees a fair coin, always.')
    print('       No mechanism local in SCREEN space can produce this.')
    print()

    print('=' * 64)
    print("CREATOR'S LOG (what actually happened in the engine)")
    print('=' * 64)
    joint = next(t for d in _last_deep for t in d.trace if t['mode'] == 'joint')
    cls = next(t for d in _last_deep for t in d.trace if t['mode'] == 'classical')
    n = SIZE

    def sdist(a, b):
        dy = min(abs(a[0] - b[0]), n - abs(a[0] - b[0]))
        dx = min(abs(a[1] - b[1]), n - abs(a[1] - b[1]))
        return max(dy, dx)

    print(f'sample pair, predicted placement: screen separation '
          f'{sdist(bx.DEEP_L, bx.DEEP_R)} cells, engine separation '
          f'{joint["engine_dist"]} cells (coherence length '
          f'{COHERENCE_LENGTH}) -> resolved as ONE object with ONE draw')
    print(f'sample pair, control placement:   screen separation '
          f'{sdist(bx.DEEP_CTRL_L, bx.DEEP_CTRL_R)} cells, engine separation '
          f'{cls["engine_dist"]} cells -> decohered, two independent '
          f'classical fragments')
    print()
    print('The "entangled pair" never had two locations in the mechanism.')
    print('Separation was a property of the rendering. ER = EPR, in 128px.')


if __name__ == '__main__':
    main()
