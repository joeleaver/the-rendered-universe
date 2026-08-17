"""Part 11b: cornering the Standard Model.

Nobody derives the Standard Model. But its strangest numbers — the
fermion hypercharges, those weird 1/6, -2/3, 1/3 fractions — are not
free parameters. Quantum consistency (gauge anomaly cancellation) is
a ledger, and it corners them:

  [38] search every rational hypercharge assignment for one
       generation of fermions. Demand the four anomaly conditions.
       Result: up to normalization and labeling, the chiral solution
       is UNIQUE — the Standard Model's charges. Corollary: the
       proton charge exactly balances the electron's. Atoms are
       neutral because the ledger says so.
  [39] the compression: one generation = the even-parity states of a
       5-bit register (the SO(10) half-spinor). Three color bits, two
       weak bits, one parity rule — and every particle of matter,
       with its exact charges, falls out. The seed compresses.
"""
from fractions import Fraction
from itertools import product

import numpy as np


def anomaly_search(rng_units=12):
    """Hypercharges in units of 1/6, integer grid. Conditions:
    [SU(3)]^2 U(1), [SU(2)]^2 U(1), [grav]^2 U(1), [U(1)]^3."""
    sols = []
    for yq in range(-rng_units, rng_units + 1):
        yl = -3 * yq
        if abs(yl) > 2 * rng_units:
            continue
        for yu in range(-2 * rng_units, 2 * rng_units + 1):
            yd = -2 * yq - yu
            ye = -(6 * yq + 3 * yu + 3 * yd + 2 * yl)
            if abs(yd) > 2 * rng_units or abs(ye) > 2 * rng_units:
                continue
            cubic = (6 * yq ** 3 + 3 * yu ** 3 + 3 * yd ** 3
                     + 2 * yl ** 3 + ye ** 3)
            if cubic == 0:
                sols.append((yq, yu, yd, yl, ye))
    return sols


def main():
    print('=' * 68)
    print('PART 11b: CORNERING THE STANDARD MODEL')
    print('=' * 68)
    sols = anomaly_search()
    chiral = [s for s in sols if s[0] != 0]
    vector = [s for s in sols if s[0] == 0]
    # reduce chiral rays to primitive form
    prim = set()
    for s in chiral:
        g = np.gcd.reduce([abs(v) for v in s if v]) or 1
        sgn = 1 if s[0] > 0 else -1
        prim.add(tuple(sgn * v // g for v in s))
    print(f'[38] anomaly-cancellation search over hypercharge space:')
    print(f'     {len(sols)} integer solutions; {len(vector)} have '
          f'Y_Q = 0 (vector-like, no chiral quarks);')
    print(f'     chiral solutions reduce to {len(prim)} primitive '
          f'ray(s):')
    for p in sorted(prim):
        names = ('Y_Q', 'Y_u', 'Y_d', 'Y_L', 'Y_e')
        vals = [Fraction(v, 6) for v in p]
        print('       ' + ', '.join(f'{n}={v}' for n, v in
                                    zip(names, vals)))
    print('     Up to normalization and the u/d label swap: the')
    print('     Standard Model, and nothing else. Electric charges')
    print('     (Q = T3 + Y):')
    yq, yu, yd, yl, ye = Fraction(1, 6), Fraction(-2, 3), \
        Fraction(1, 3), Fraction(-1, 2), Fraction(1)
    table = [('u', yq + Fraction(1, 2)), ('d', yq - Fraction(1, 2)),
             ('e', yl - Fraction(1, 2)), ('nu', yl + Fraction(1, 2))]
    print('       ' + '   '.join(f'{n}: {q:+}' for n, q in table))
    p_charge = 2 * (yq + Fraction(1, 2)) + (yq - Fraction(1, 2))
    print(f'     proton (uud): {p_charge:+}, electron: -1 -> atoms are')
    print('     exactly neutral. Charge quantization is not an input;')
    print('     it is anomaly bookkeeping.')
    print()

    print('[39] one generation = a 5-bit register (even parity):')
    # bits: 3 color, 2 weak; state included iff # of minus bits is even
    alpha, beta = None, None
    target = sorted([Fraction(1, 6)] * 6 + [Fraction(-2, 3)] * 3
                    + [Fraction(1, 3)] * 3 + [Fraction(-1, 2)] * 2
                    + [Fraction(1), Fraction(0)])
    for a_num in range(-12, 13):
        for b_num in range(-12, 13):
            a, bb = Fraction(a_num, 12), Fraction(b_num, 12)
            ys = []
            for bits in product((1, -1), repeat=5):
                if bits.count(-1) % 2:
                    continue
                ys.append(a * sum(bits[:3]) + bb * sum(bits[3:]))
            if sorted(ys) == target:
                alpha, beta = a, bb
                break
        if alpha is not None:
            break
    print(f'     hypercharge law found by search: '
          f'Y = {alpha}*(color bits) + {beta}*(weak bits)')
    print(f'     {"bits":<8} {"Y":>6} {"T3":>5} {"Q_em":>6}  particle')
    names = {(Fraction(0), 'singlet', Fraction(0)): 'nu_R (sterile)',
             (Fraction(1), 'singlet', Fraction(0)): 'e^c  (positron)',
             (Fraction(-2, 3), '3bar', Fraction(0)): 'u^c',
             (Fraction(1, 3), '3bar', Fraction(0)): 'd^c',
             (Fraction(1, 6), '3', Fraction(1, 2)): 'u_L',
             (Fraction(1, 6), '3', Fraction(-1, 2)): 'd_L',
             (Fraction(-1, 2), 'singlet', Fraction(1, 2)): 'nu_L',
             (Fraction(-1, 2), 'singlet', Fraction(-1, 2)): 'e_L'}
    count = 0
    for bits in product((1, -1), repeat=5):
        if bits.count(-1) % 2:
            continue
        count += 1
        Y = alpha * sum(bits[:3]) + beta * sum(bits[3:])
        c_minus = bits[:3].count(-1)
        rep = 'singlet' if c_minus in (0, 3) else \
            ('3' if c_minus == 1 else '3bar')
        T3 = Fraction(bits[3] - bits[4], 4)
        Q = T3 + Y
        key = (Y, rep, T3)
        nm = names.get(key, '?')
        s = ''.join('+' if b > 0 else '-' for b in bits)
        print(f'     {s:<8} {str(Y):>6} {str(T3):>5} {str(Q):>6}  {nm}')
    print(f'     {count} states — one full generation, from three')
    print('     color bits, two weak bits, and one parity rule. The')
    print('     Standard Model\'s matter content compresses to five')
    print('     bits. Compressible seeds, all the way down.')


if __name__ == '__main__':
    main()
