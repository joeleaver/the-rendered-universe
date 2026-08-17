"""Part 10f: the ledger in three dimensions.

The 2D census used 2x2 blocks and the 8-element square group. Real
space is 3D: blocks become 2x2x2 cubes (256 states — the same state
count as 2D four-state cells) and isotropy means the full 48-element
symmetry group of the cube. Same exact orbit combinatorics:

    survivors = product over (conserved-sum, stabilizer-class) groups
                of m! * w^m,   w = |N(H)| / |H|

The question: with the same amount of raw state, does the richer
symmetry of 3D corner harder? (Formula validated against brute force
at k=2 in 2D, part 3.)
"""
from collections import defaultdict
from itertools import permutations, product
from math import lgamma, log10, log

# --- the octahedral group as permutations of the 8 cube vertices ------

VERTS = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
VIDX = {v: i for i, v in enumerate(VERTS)}


def group_elements():
    els = set()
    for p in permutations(range(3)):
        for f in product((0, 1), repeat=3):
            perm = []
            for v in VERTS:
                w = tuple(v[p[i]] ^ f[i] for i in range(3))
                perm.append(VIDX[w])
            els.add(tuple(perm))
    return sorted(els)


G = group_elements()
assert len(G) == 48


def compose(g, h):
    return tuple(g[h[i]] for i in range(8))


def inverse(g):
    out = [0] * 8
    for i, v in enumerate(g):
        out[v] = i
    return tuple(out)


def act(g, s):
    out = 0
    for j in range(8):
        if s >> j & 1:
            out |= 1 << g[j]
    return out


def popcount(s):
    return bin(s).count('1')


def conj_class(H):
    variants = []
    for g in G:
        gi = inverse(g)
        variants.append(tuple(sorted(compose(compose(g, h), gi)
                                     for h in H)))
    return min(variants)


def weyl(H):
    Hs = frozenset(H)
    NH = [g for g in G
          if frozenset(compose(compose(g, h), inverse(g))
                       for h in H) == Hs]
    return len(NH) // len(H)


def main():
    print('=' * 68)
    print('PART 10f: THE LEDGER IN 3D (2x2x2 blocks, the cube group)')
    print('=' * 68)

    seen, groups = set(), defaultdict(list)
    n_orbits = 0
    for s in range(256):
        if s in seen:
            continue
        orb = {act(g, s) for g in G}
        seen.update(orb)
        n_orbits += 1
        H = tuple(g for g in G if act(g, s) == s)
        groups[(popcount(s), conj_class(H))].append(H)

    total = 1
    for stabs in groups.values():
        m, w = len(stabs), weyl(stabs[0])
        f = 1
        for i in range(2, m + 1):
            f *= i
        total *= f * (w ** m)

    full = lgamma(257) / log(10)
    print(f'256 block states form {n_orbits} orbits under the '
          f'48-element cube group.')
    print(f'full rule space: 256! = 10^{full:.1f}')
    print(f'ledger survivors (reversible + isotropic + conserving + '
          f'stable vacuum):')
    print(f'    {total:,}   (10^{log10(total):.2f})')
    print()
    print('comparison across the program:')
    print(f'    {"space":<22} {"states":>7} {"|G|":>5} '
          f'{"survivors":>12} {"log-fraction":>13}')
    rows = [('2D, 2-state (part 3)', 16, 8, 32, log10(32) / 13.3),
            ('2D, 3-state', 81, 8, 67108864, 7.83 / 120.8),
            ('2D, 4-state', 256, 8, None, 33.14 / 506.9),
            ('3D, 2-state (here)', 256, 48, total,
             log10(total) / full)]
    for name, st, gg, surv, frac in rows:
        s = f'{surv:,}' if surv and surv < 10 ** 12 else \
            (f'10^{33.14:.1f}' if surv is None else f'10^{log10(surv):.1f}')
        print(f'    {name:<22} {st:>7} {gg:>5} {s:>12} {frac:>13.3f}')
    print()
    print('Same 256 raw states as 2D four-state cells — but where the')
    print('square group left 10^33 survivors, the cube group leaves')
    print('10^8.4: TWENTY-FIVE more orders of magnitude deleted by the')
    print('extra isotropy alone. Symmetry, not alphabet, is where the')
    print('ledger gets its teeth — and 2.7e8 rules is small enough to')
    print('SAMPLE: a 3D observatory census is now a feasible project,')
    print('not a fantasy.')


if __name__ == '__main__':
    main()
