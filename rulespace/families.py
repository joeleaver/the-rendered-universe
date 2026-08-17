"""The static ledger: constraints that delete rule-space without
running anything.

  L1  reversibility      — the rule is a bijection on block states
                           (built into the parameterization: free)
  L2  isotropy           — the rule commutes with every rotation and
                           reflection of the block (the square group D4)
  L3  conservation       — a matter count is exactly conserved:
                           'strict'     popcount(rule(s)) == popcount(s)
                           'complement' popcount(rule(s)) == 4 - popcount(s)
                           (complement rules conserve phase-corrected
                           matter, like Critters)
  L4  stable vacuum      — forced by L3: empty maps to empty (strict)
                           or to full (complement, a blinking vacuum
                           that phase-corrects to stillness)

Everything here is exact combinatorics, not simulation.
"""
from itertools import product

import numpy as np

FULL_SPACE = 20922789888000  # 16!


def _state_bits(s):
    return ((s >> 3) & 1, (s >> 2) & 1, (s >> 1) & 1, s & 1)  # a, b, c, d


def _bits_state(b):
    return b[0] * 8 + b[1] * 4 + b[2] * 2 + b[3]


def _d4_elements():
    """The 8 symmetries of the square, as permutations of the 16 states.
    Block layout: a b / c d."""
    def rot(b):  # 90 degrees clockwise: a b / c d -> c a / d b
        a, bb, c, d = b
        return (c, a, d, bb)

    def mirror(b):  # left-right: a b / c d -> b a / d c
        a, bb, c, d = b
        return (bb, a, d, c)

    perms = []
    for k in range(4):
        for m in (False, True):
            table = np.zeros(16, dtype=np.uint8)
            for s in range(16):
                b = _state_bits(s)
                for _ in range(k):
                    b = rot(b)
                if m:
                    b = mirror(b)
                table[s] = _bits_state(b)
            perms.append(table)
    return perms


D4 = _d4_elements()


def popcount(s):
    return bin(s).count('1')


def ledger_rules():
    """Enumerate every rule satisfying L1-L4, by brute force over
    equivariant orbit assignments. Returns list of (lut, family)."""
    # orbits of the D4 action on states
    orbits = []
    seen = set()
    for s in range(16):
        if s in seen:
            continue
        orb = sorted({int(g[s]) for g in D4})
        orbits.append(orb)
        seen.update(orb)

    results = []
    for family in ('strict', 'complement'):
        target = (lambda k: k) if family == 'strict' else (lambda k: 4 - k)
        # candidate images per orbit representative
        options = []
        for orb in orbits:
            rep = orb[0]
            cands = []
            for v in range(16):
                if popcount(v) != target(popcount(rep)):
                    continue
                # equivariant extension must be well-defined:
                # every symmetry fixing rep must fix v
                if all(int(g[v]) == v for g in D4 if int(g[rep]) == rep):
                    cands.append(v)
            options.append((rep, cands))
        for choice in product(*[c for _, c in options]):
            lut = np.full(16, 255, dtype=np.uint8)
            ok = True
            for (rep, _), v in zip(options, choice):
                for g in D4:
                    src, dst = int(g[rep]), int(g[v])
                    if lut[src] not in (255, dst):
                        ok = False
                        break
                    lut[src] = dst
                if not ok:
                    break
            if ok and len(set(lut.tolist())) == 16:
                results.append((lut, family))
    # dedup
    uniq, out = set(), []
    for lut, fam in results:
        key = lut.tobytes()
        if key not in uniq:
            uniq.add(key)
            out.append((lut, fam))
    return out
