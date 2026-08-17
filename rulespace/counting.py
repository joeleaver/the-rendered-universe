"""How sharply do constraints corner rule space as it grows?

For cells with k states, a block rule is a bijection on the k^4 block
states: the full space has (k^4)! rules. The ledger (reversibility +
D4-equivariance + exact sum conservation + stable vacuum) admits a
closed-form count via orbit combinatorics:

    survivors = product over (conserved-sum class, stabilizer class) of
                m! * w^m

where m is the number of D4-orbits in that class and w = |N(H)|/|H| is
the number of equivariant bijections between two orbits with stabilizer
H. No enumeration, no simulation — the count is exact for any k.

(Sanity anchor: k=2 must give 16, matching the brute-force enumeration
in families.py.)
"""
from collections import defaultdict
from math import lgamma, log10

import numpy as np

# D4 as permutations of the four cell positions [(0,0),(0,1),(1,0),(1,1)];
# g acts by new[i] = old[g[i]]
_R90 = (2, 0, 3, 1)
_MIRROR = (1, 0, 3, 2)


def _compose(g, h):  # (g∘h)(t) = g(h(t))
    return tuple(h[g[i]] for i in range(4))


def d4_group():
    els = {(0, 1, 2, 3)}
    frontier = [(0, 1, 2, 3)]
    while frontier:
        g = frontier.pop()
        for s in (_R90, _MIRROR):
            for new in (_compose(g, s), _compose(s, g)):
                if new not in els:
                    els.add(new)
                    frontier.append(new)
    return sorted(els)


G = d4_group()
assert len(G) == 8


def _inv(g):
    out = [0] * 4
    for i, v in enumerate(g):
        out[v] = i
    return tuple(out)


def act(g, digits):
    return tuple(digits[g[i]] for i in range(4))


def _states(k):
    out = []
    for s in range(k ** 4):
        d, digs = s, []
        for _ in range(4):
            digs.append(d % k)
            d //= k
        out.append(tuple(reversed(digs)))
    return out


def _encode(digits, k):
    v = 0
    for d in digits:
        v = v * k + d
    return v


def orbit_data(k):
    """Orbits of D4 on block states: list of
    (rep_digits, orbit_state_ids, stabilizer)."""
    states = _states(k)
    seen, orbits = set(), []
    for s, digs in enumerate(states):
        if s in seen:
            continue
        orb = {_encode(act(g, digs), k) for g in G}
        seen.update(orb)
        stab = frozenset(g for g in G if act(g, digs) == digs)
        orbits.append((digs, sorted(orb), stab))
    return orbits


def _conj_class(H):
    """Canonical label for the conjugacy class of subgroup H in D4."""
    variants = []
    for g in G:
        gi = _inv(g)
        variants.append(frozenset(_compose(_compose(g, h), gi) for h in H))
    return min(tuple(sorted(v)) for v in variants)


def _weyl(H):
    NH = [g for g in G
          if frozenset(_compose(_compose(g, h), _inv(g)) for h in H) == H]
    return len(NH) // len(H)


def ledger_count(k):
    """Exact number of ledger-surviving rules for k-state cells."""
    groups = defaultdict(list)
    for digs, orb, stab in orbit_data(k):
        groups[(sum(digs), _conj_class(stab))].append(stab)
    total = 1
    for stabs in groups.values():
        m, w = len(stabs), _weyl(stabs[0])
        f = 1
        for i in range(2, m + 1):
            f *= i
        total *= f * (w ** m)
    return total


def full_space_log10(k):
    return lgamma(k ** 4 + 1) / np.log(10)


def sample_rule(k, rng):
    """Draw a uniformly random ledger-surviving rule as a LUT."""
    groups = defaultdict(list)
    for digs, orb, stab in orbit_data(k):
        groups[(sum(digs), _conj_class(stab))].append((digs, orb, stab))
    states = _states(k)
    lut = np.full(k ** 4, -1, dtype=np.int64)
    for members in groups.values():
        targets = list(rng.permutation(len(members)))
        for src_i, tgt_i in enumerate(targets):
            u_digs, _, H_u = members[src_i]
            _, tgt_orb, _ = members[tgt_i]
            cands = [v for v in tgt_orb
                     if frozenset(g for g in G
                                  if act(g, states[v]) == states[v]) == H_u]
            v_digs = states[int(rng.choice(cands))]
            for g in G:
                lut[_encode(act(g, u_digs), k)] = _encode(act(g, v_digs), k)
    assert (lut >= 0).all() and len(set(lut.tolist())) == k ** 4
    for s in range(k ** 4):
        assert sum(states[s]) == sum(states[int(lut[s])])
    return lut.astype(np.uint8 if k ** 4 <= 256 else np.int64)
