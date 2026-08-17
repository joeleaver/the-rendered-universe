"""Deep matter: a second excitation of the substrate.

Pixel matter (the block CA field) is not the only thing the engine
evolves. A deep pair is a SINGLE engine object with two ends. Its rule
is as local as the block rule: nothing about a pair is ever computed
using state farther than COHERENCE_LENGTH engine cells from its ends.

When both ends of a pair have received analyzer settings ta, tb:

  - ends within COHERENCE_LENGTH of each other: the pair is still one
    object at one locus. ONE random draw resolves both outcomes:
    A is a fair coin, B agrees with A with probability sin^2((ta-tb)/2)
    (the singlet law, E = -cos(ta - tb)).

  - ends farther apart: the object has decohered into two classical
    fragments. Each resolves ALONE from the polarization angle `lam`
    it has carried since emission: A = sign(cos(ta - lam)),
    B = -sign(cos(tb - lam)). This is the best a shared classical
    memory can do.

Note what is NOT here: no engine rule ever consults screen coordinates.
Whether two ends are "near" is judged in engine space only. If the
chart happens to render near ends far apart, that is the renderer's
business, not ours.

Outcomes are written into the pixel field as 2x2 flashes so observers
can read them off the screen: +1 flashes at the end's locus, -1
flashes 4 cells to its right.
"""
import math

import numpy as np

COHERENCE_LENGTH = 12


def _tordist(p, q, n):
    dy = abs(p[0] - q[0]); dy = min(dy, n - dy)
    dx = abs(p[1] - q[1]); dx = min(dx, n - dx)
    return max(dy, dx)


class DeepField:
    def __init__(self, substrate, rng):
        self.sub = substrate
        self.rng = rng
        self.pairs = []
        self.trace = []  # engine-side log, for the creator's eyes only

    def emit(self, locus_a, locus_b):
        self.pairs.append({
            'ends': [tuple(locus_a), tuple(locus_b)],
            'settings': [None, None],
            'lam': float(self.rng.uniform(0.0, 2 * math.pi)),
            'resolved': False,
        })

    def set_setting(self, engine_locus, theta):
        """Attach an analyzer setting to the pair end at this locus."""
        n = self.sub.n
        for pair in self.pairs:
            for i, end in enumerate(pair['ends']):
                if not pair['resolved'] and pair['settings'][i] is None \
                        and _tordist(end, engine_locus, n) <= 3:
                    pair['settings'][i] = float(theta)
                    return True
        return False

    def _flash(self, locus, outcome):
        """Write the outcome into the pixel field as a 2x2 flash."""
        g, n, t = self.sub.grid, self.sub.n, self.sub.t
        bright = 1 ^ (t % 2)  # phase-corrected 'lit'
        y, x = locus
        if outcome < 0:
            x += 4
        for dy in (0, 1):
            for dx in (0, 1):
                g[(y + dy) % n, (x + dx) % n] = bright

    def tick(self):
        n = self.sub.n
        for pair in self.pairs:
            if pair['resolved'] or None in pair['settings']:
                continue
            (ea, eb), (ta, tb) = pair['ends'], pair['settings']
            d = _tordist(ea, eb, n)
            if d <= COHERENCE_LENGTH:
                # one object, one locus, one draw
                a = 1 if self.rng.random() < 0.5 else -1
                same = self.rng.random() < math.sin((ta - tb) / 2) ** 2
                b = a if same else -a
                mode = 'joint'
            else:
                # two decohered fragments, each on its own
                lam = pair['lam']
                a = 1 if math.cos(ta - lam) >= 0 else -1
                b = -(1 if math.cos(tb - lam) >= 0 else -1)
                mode = 'classical'
            pair['resolved'] = True
            self._flash(ea, a)
            self._flash(eb, b)
            self.trace.append({'ends': (ea, eb), 'engine_dist': d,
                               'mode': mode, 'outcomes': (a, b)})
