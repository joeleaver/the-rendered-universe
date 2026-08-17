"""Part 12: the axiom collider.

The method of this whole program, made explicit: physics as a
constraint-satisfaction problem. Axioms that each seem obviously true
are slammed together; each collision either CONVERGES (jointly forces
a unique structure), is COMPATIBLE (proven consistent), is
INCOMPATIBLE (a no-go theorem — the crash teaches the escape), or is
OPEN (the research frontier). Every verdict cites its authority: a
named theorem, or a measurement made in this repo.

The no-go theorems are the fossil record of collisions past. The
black-hole information paradox is a collision in progress.
"""

AXIOMS = {
    'REV':  'reversibility — no information ever destroyed',
    'LOC':  'locality — dynamics couple only neighbors',
    'LOR':  'exact Lorentz invariance',
    'DISC': 'discreteness — finitely many DOF per region',
    'GRID': 'REGULAR discreteness (a lattice)',
    'SPRK': 'RANDOM discreteness (a sprinkle)',
    'AREA': 'area-law information scaling (holography)',
    'CHI':  'net chirality (Standard Model matter)',
    'CONT': 'continuity of reversible change',
    'PUR':  'purification — mixed states come from pure global ones',
    'TOM':  'local tomography — states readable by local statistics',
    'REAL': 'realism — definite values before measurement',
    'IND':  'statistical independence — free measurement settings',
    'EP':   'equivalence principle — gravity couples universally',
    'EMG':  'graviton emergent IN THE SAME spacetime',
    'CONS': 'exact local conservation / anomaly consistency',
    'UNI':  'unitarity of black hole evaporation',
    'EFT':  'effective field theory valid outside horizons',
}

COLLISIONS = [
    dict(ax=('LOC', 'REAL', 'IND'), verdict='INCOMPATIBLE',
         why='measured Bell violations exceed the bound',
         auth='Bell 1964; Aspect 1982; in-repo part 2: S=2.84, '
              'no-signaling',
         escape='drop REAL, or accept locality only in the render '
                '(engine adjacency != screen adjacency — the chart)'),
    dict(ax=('REAL', 'LOC'), verdict='INCOMPATIBLE',
         why='interference: both-slits arrivals fall below each '
             'single slit',
         auth='PBR 2012; in-repo part 10c: 4.8% of classical floor',
         escape='amplitudes in the engine; particles only in the '
                'render (part 7)'),
    dict(ax=('GRID', 'LOR'), verdict='INCOMPATIBLE',
         why='a lattice has systematic anisotropy and a rest frame',
         auth='in-repo part 10a: 20% anisotropy, zero variance; '
              'boosted NN stats shift 29%',
         escape='SPRK — randomness has no grain and no frame'),
    dict(ax=('SPRK', 'LOR'), verdict='COMPATIBLE',
         why='a Poisson sprinkle is statistically boost-invariant',
         auth='Bombelli-Henson-Sorkin 2006; in-repo part 10a: 1% '
              'shift (noise)',
         escape=None),
    dict(ax=('DISC', 'LOC', 'CHI'), verdict='INCOMPATIBLE',
         why='the Brillouin zone is closed: doublers are topological',
         auth='Nielsen-Ninomiya 1981; in-repo part 10e: 2 species in '
              '1D, 4 in 2D, net chirality 0',
         escape='pay with an extra dimension (domain wall) or '
                'non-locality (overlap)'),
    dict(ax=('EMG', 'LOR', 'CONS'), verdict='INCOMPATIBLE',
         why='a composite massless spin-2 particle cannot carry a '
             'Lorentz-covariant conserved current',
         auth='Weinberg-Witten 1980',
         escape='the graviton lives in the EMERGENT space (bulk of '
                'holography) — gravity must be part of the render '
                'layer. The no-go that demands the architecture.'),
    dict(ax=('EP', 'LOC'), verdict='INCOMPATIBLE',
         why='medium-coupled gravity is chromatic: bend and opacity '
             'depend on wavelength',
         auth='in-repo part 6: transmission 55% vs 23% by wavelength; '
              'MICROSCOPE bounds EP violation < 1e-15',
         escape='couple through geometry, not a medium (Jacobson '
                '1995: Einstein eqs from entanglement thermodynamics)'),
    dict(ax=('DISC', 'AREA'), verdict='INCOMPATIBLE',
         why='volume-voxel engines scale information with volume; '
             'regions scale with their boundary',
         auth="Bekenstein 1973; 't Hooft 1993; in-repo part 10b: "
              'S vs perimeter r=1.0000',
         escape='the engine stores ~2D-worth of state; the 3D screen '
                'is over-rendered'),
    dict(ax=('REV', 'CONT', 'TOM'), verdict='CONVERGES',
         why='the only surviving probability theory is complex QM',
         auth='Hardy 2001; Chiribella-D\'Ariano-Perinotti 2011; '
              'in-repo part 11a: classical fails CONT, real fails '
              'TOM (9 vs 8), complex passes (15=15)',
         escape=None),
    dict(ax=('PUR', 'REV'), verdict='CONVERGES',
         why='branch-swap symmetry forces P=|amplitude|^2',
         auth='Zurek 2003 (envariance); in-repo part 11a: verified '
              'numerically',
         escape=None),
    dict(ax=('CONS', 'CHI'), verdict='CONVERGES',
         why='anomaly cancellation leaves ONE chiral hypercharge ray: '
             'the Standard Model\'s; atoms exactly neutral',
         auth='in-repo part 11b: 65 solutions, 1 chiral ray up to '
              'u/d relabel',
         escape=None),
    dict(ax=('REV', 'LOC'), verdict='CONVERGES',
         why='+ isotropy + conservation: rule space collapses '
             '(2.1e13 -> 32 in 2D; 10^507 -> 2^28 in 3D)',
         auth='in-repo parts 3, 10f: exact orbit counts',
         escape=None),
    dict(ax=('REV',), verdict='COMPATIBLE',
         why='with a coarse render: the arrow of time emerges; with '
             'collisions: records and observers emerge',
         auth='in-repo parts 9, 10d: entropy 3831->7160 bits, exact '
              'reversal; 21% of collisions leave stable records',
         escape=None),
    dict(ax=('UNI', 'EFT', 'EP'), verdict='OPEN',
         why='black hole evaporation cannot satisfy all three: '
             'the information paradox, currently on fire',
         auth='Hawking 1976; AMPS 2012; replica wormholes 2019 '
              '(lean: keep UNI, bend EFT locality)',
         escape='unknown — the next fossil is forming here'),
    dict(ax=('REAL', 'DISC', 'LOC'), verdict='OPEN',
         why='can any sub-quantum engine reproduce QM? Bell/PBR '
             'block the obvious routes; superdeterminism (drop IND) '
             'is the surviving loophole',
         auth="'t Hooft 2016 program; in-repo part 10c: the 2^n "
              'amplitude wall, 2.14x per qubit',
         escape='quantum computing scale-up is the running experiment'),
]


def main():
    print('=' * 72)
    print('PART 12: THE AXIOM COLLIDER')
    print('=' * 72)
    print('axioms in play:')
    for k, v in AXIOMS.items():
        print(f'   {k:<5} {v}')
    print()
    order = ['INCOMPATIBLE', 'CONVERGES', 'COMPATIBLE', 'OPEN']
    in_repo = 0
    for verdict in order:
        rows = [c for c in COLLISIONS if c['verdict'] == verdict]
        print(f'--- {verdict} ({len(rows)}) ' + '-' * (50 - len(verdict)))
        for c in rows:
            if 'in-repo' in c['auth']:
                in_repo += 1
            print(f'   {" + ".join(c["ax"]):<22} {c["why"]}')
            print(f'   {"":<22} [{c["auth"]}]')
            if c['escape']:
                print(f'   {"":<22} escape: {c["escape"]}')
            print()
    print(f'{len(COLLISIONS)} collisions; {in_repo} carry in-repo '
          f'demonstrations.')
    print()
    print('The avalanche, measured (deletion power of each slam):')
    print('   2D structural ledger : 2.1e13 rules -> 32')
    print('   3D isotropy          : same 256 states, 25 more orders '
          'of magnitude gone')
    print('   anomaly consistency  : all hypercharges -> 1 ray (the SM)')
    print('   QM reconstruction    : all probability theories -> 1 '
          '(complex QM)')
    print('   Reasonable axioms do not whittle. They avalanche.')

    with open('COLLIDER.md', 'w') as f:
        f.write('# The Axiom Collider\n\nPhysics as constraint '
                'satisfaction. Verdicts: theorem, or in-repo '
                'measurement.\n\n## Axioms\n\n')
        for k, v in AXIOMS.items():
            f.write(f'- **{k}** — {v}\n')
        for verdict in order:
            f.write(f'\n## {verdict}\n\n')
            for c in COLLISIONS:
                if c['verdict'] != verdict:
                    continue
                f.write(f'- **{" + ".join(c["ax"])}** — {c["why"]}.  \n'
                        f'  *{c["auth"]}*\n')
                if c['escape']:
                    f.write(f'  - escape: {c["escape"]}\n')
    print('\nwritten: COLLIDER.md')


if __name__ == '__main__':
    main()
