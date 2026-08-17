"""Create a universe; let a physicist loose on its screen.

Part 1 (creator's log) verifies engine-side properties the physicist
can never check directly. Part 2 is the physicist's lab notebook: every
number in it was derived from rendered frames alone.
"""
import subprocess
import sys

import numpy as np

from engine.substrate import Substrate
from render.screen import Renderer
from physicist.instruments import Instruments
from physicist import experiments

SIZE = 128
# The chart's secret: these two engine rectangles trade screen positions.
PORTALS = [((16, 16), (96, 32), (12, 12))]


def make_lab(swaps=PORTALS):
    def new_universe():
        sub = Substrate(SIZE)
        ren = Renderer(sub, swaps=swaps)

        def tick(k=1):
            for _ in range(k):
                sub.step()

        return Instruments(ren.frame, ren.poke, tick)
    return new_universe


def creators_log():
    print('=' * 64)
    print("CREATOR'S LOG (engine-side checks the physicist can't run)")
    print('=' * 64)

    rng = np.random.default_rng(0)
    sub = Substrate(SIZE)
    sub.grid = (rng.random((SIZE, SIZE)) < 0.3).astype(np.uint8)
    start = sub.grid.copy()
    for _ in range(400):
        sub.step()
    for _ in range(400):
        sub.unstep()
    ok = np.array_equal(sub.grid, start)
    print(f'reversibility: 400 ticks forward, 400 back -> '
          f'{"EXACT recovery" if ok else "FAILED"}')

    r = subprocess.run([sys.executable, 'tests/test_firewall.py'],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode or not ok:
        sys.exit(1)


def physicists_notebook():
    lab = make_lab()
    print()
    print('=' * 64)
    print("PHYSICIST'S LAB NOTEBOOK (frames only — see the firewall)")
    print('=' * 64)

    c = experiments.measure_lightspeed(lab)
    print(f'[1] causal speed limit: no disturbance front observed faster '
          f'than c = {c:.3f} cells/tick')

    lo, hi = experiments.check_conservation(lab)
    print(f'[2] conservation: total brightness over 400 ticks of soup: '
          f'min={lo} max={hi} -> {"EXACTLY conserved" if lo == hi else "varies"}')

    tax = experiments.hunt_particles(lab)
    print(f'[3] small-seed taxonomy (456 seeds): {tax["evaporated"]} evaporated, '
          f'{tax["bound"]} formed bound states, {tax["complex"]} went complex, '
          f'{tax["travelers"]} travelers')

    species = experiments.capture_from_debris(lab)
    print(f'[4] debris capture: {len(species)} verified free particle(s)')
    for s in species:
        print(f'    - {s["cells"]} cells, period {s["period"]}, '
              f'velocity {s["velocity"]}, speed {s["speed"]:.3f} '
              f'({s["speed"] / c:.2f} c), lattice parity {s["parity"]}')
    if not species:
        print('    (no particles captured; survey skipped)')
        return None

    variants = experiments.test_isotropy(lab, species[0])
    vels = sorted(v['velocity'] for v in variants)
    print(f'[5] isotropy: rotated/flipped copies fly in {len(variants)} '
          f'directions {vels} -> laws are '
          f'{"symmetric under the square group" if len(variants) >= 4 else "anisotropic"}')

    # survey with one vertical and one horizontal probe
    vert = next((v for v in variants if v['velocity'][0] != 0), None)
    horiz = next((v for v in variants if v['velocity'][1] != 0), None)
    probes = [p for p in (vert, horiz) if p]
    report = experiments.ballistic_survey(lab, probes)
    n_lines = len(report)
    anomalous = [r for r in report if r['events']]
    print(f'[6] ballistic survey: {n_lines} trajectories fired across the sky')
    print(f'    consistent with straight-line motion: '
          f'{n_lines - len(anomalous)}/{n_lines}')
    print(f'    ANOMALOUS: {len(anomalous)}')
    for r in anomalous:
        for (t, p, q, d) in r['events']:
            print(f'    - launch {r["start"]} v={r["velocity"]}: at t={t} probe '
                  f'vanished near ({p[0]:.0f},{p[1]:.0f}), reappeared at '
                  f'({q[0]:.0f},{q[1]:.0f}) — {d} cells in 1 tick '
                  f'= {d / max(c, 1e-9):.0f}x the speed of light')
    if anomalous:
        print()
        print('    CONCLUSION: locality holds everywhere except two fixed')
        print('    regions of the sky, which behave as a matched pair of')
        print('    doors. Either physics is broken there, or screen distance')
        print('    is not mechanism distance. The pixels are not the machine.')
        print('    The pixels are the rendering of the machine.')
    return probes, anomalous


if __name__ == '__main__':
    creators_log()
    physicists_notebook()
