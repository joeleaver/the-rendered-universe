"""The frontier: does the ledger keep cornering as rule space grows?

Two measurements:

  1. Exact survivor counts for k = 2, 3, 4 cell states (closed-form
     orbit combinatorics — no simulation). The question: what fraction
     of the rule space's log-volume do the structural constraints
     delete, and does that fraction hold as the space explodes?

  2. A sampled census of the ternary (k=3) survivors: draw random
     ledger rules and run the observatory on each. The question: among
     rules that pass the STRUCTURAL ledger, how common are universes —
     and does that rate survive the jump from binary to ternary?
"""
from math import log10

import numpy as np

from rulespace.counting import ledger_count, full_space_log10, sample_rule
from observatory.causal import butterfly_and_dimension
from observatory.scorecard import count_free_particles

N_SAMPLES = 24
SCALES = [(56, 56), (96, 96)]


def verdict(speed, dim, conv, particles):
    if speed <= 0.05:
        return 'dead (frozen)'
    if not conv:
        return 'unresolved'
    if not (1.6 <= dim <= 2.5):
        return f'dead ({dim:.1f}-dim space)'
    if particles == 0:
        return 'space, no matter'
    return 'UNIVERSE'


def main():
    print('=' * 70)
    print('STATIC LEDGER COMPRESSION vs ALPHABET SIZE (exact, no simulation)')
    print('=' * 70)
    for k in (2, 3, 4):
        n = ledger_count(k)
        fs = full_space_log10(k)
        print(f'  k={k}: full space 10^{fs:>5.1f}  ->  survivors '
              f'10^{log10(n):>5.2f}  ({n:,})'
              f'   log-fraction kept: {log10(n) / fs:.3f}')
    print()
    print('The constraints delete ~93% of the log-volume at every k, but')
    print('the absolute survivor count explodes: structure alone stops')
    print('cornering. The rest of the work belongs to dynamics.')
    print()

    print('=' * 70)
    print(f'SAMPLED TERNARY CENSUS ({N_SAMPLES} random k=3 ledger rules)')
    print('=' * 70)
    rng = np.random.default_rng(42)
    print(f'{"rule":>4} {"c_b":>5} {"dim":>5} {"prt":>4}  verdict')
    tally = {}
    for i in range(N_SAMPLES):
        lut = sample_rule(3, rng)
        speed = dim = 0.0
        conv = True
        for (size, ticks) in SCALES:
            speed, dim, conv = butterfly_and_dimension(
                size, lut, ticks=ticks, states=3)
            if conv:
                break
        particles = count_free_particles(lut, 'strict', states=3) \
            if speed > 0.05 else 0
        v = verdict(speed, dim, conv, particles)
        tally[v] = tally.get(v, 0) + 1
        print(f'{i:>4} {speed:>5.2f} {dim:>5.2f} {particles:>4}  {v}')
    print()
    unis = tally.get('UNIVERSE', 0)
    print(f'ternary universe rate: {unis}/{N_SAMPLES} '
          f'= {unis / N_SAMPLES:.0%}   (binary census: 9/32 = 28%)')
    for v, c in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f'  {v}: {c}')


if __name__ == '__main__':
    main()
