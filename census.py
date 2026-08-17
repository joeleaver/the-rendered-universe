"""The complete census.

Stage 2 left 12 rules 'unresolved (probe too small)'. A bigger probe is
not a different theory — it is a bigger telescope. Escalate the causal
probe per rule until the dimension estimator converges (or admit defeat
at the largest scale), then issue final verdicts for all 32 survivors.
"""
import numpy as np

from rulespace.families import ledger_rules
from observatory.causal import butterfly_and_dimension
from observatory.scorecard import count_free_particles

SCALES = [(56, 56), (96, 96), (128, 128)]


def final_verdict(speed, dim, conv, particles):
    if speed <= 0.05:
        return 'dead (frozen)'
    if not conv:
        return 'UNRESOLVED at max probe'
    if not (1.6 <= dim <= 2.5):
        return f'dead (space is {dim:.1f}-dimensional)'
    if particles == 0:
        return 'space, no matter'
    return 'UNIVERSE'


def main():
    rules = ledger_rules()
    print('=' * 70)
    print('THE COMPLETE CENSUS (adaptive probe scale)')
    print('=' * 70)
    print(f'{"rule":>4} {"family":<10} {"c_b":>5} {"dim":>5} {"scale":>6} '
          f'{"prt":>4}  verdict')
    verdicts = {}
    for i, (lut, fam) in enumerate(rules):
        speed = dim = 0.0
        conv, used = True, SCALES[0][0]
        for (size, ticks) in SCALES:
            speed, dim, conv = butterfly_and_dimension(size, lut, ticks=ticks)
            used = size
            if conv:
                break
        particles = count_free_particles(lut, fam) if speed > 0.05 else 0
        v = final_verdict(speed, dim, conv, particles)
        verdicts[i] = v
        print(f'{i:>4} {fam:<10} {speed:>5.2f} {dim:>5.2f} {used:>6} '
              f'{particles:>4}  {v}')

    unis = [i for i, v in verdicts.items() if v == 'UNIVERSE']
    unresolved = [i for i, v in verdicts.items() if v.startswith('UNRESOLVED')]
    print(f'\nuniverses: {unis}  ({len(unis)} of {len(rules)})')
    print(f'still unresolved at 128-cell probe: {unresolved}')
    # collapse by the i <-> 31-i twin involution
    seen, classes = set(), []
    for i in verdicts:
        j = 31 - i
        if i not in seen:
            seen.update({i, j})
            classes.append((i, j, verdicts[i], verdicts[j]))
    same = sum(1 for a, b, va, vb in classes if va == vb)
    print(f'twin pairs (i, 31-i) with matching verdicts: {same}/{len(classes)}')


if __name__ == '__main__':
    main()
