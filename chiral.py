"""Part 10e: the doubling theorem — what stands between lattices and
the Standard Model, demonstrated.

Nature's weak force couples ONLY to left-handed fermions. Nielsen and
Ninomiya proved that a local lattice theory can't have that: put one
chiral fermion on a lattice and the lattice manufactures a partner of
the opposite chirality. Watch it happen:

  [30] naive lattice fermion, E(k) = sin k: the Brillouin zone is a
       circle, and sin k must cross zero an EVEN number of times —
       one crossing at k=0 (right-handed), an uninvited one at k=pi
       (left-handed). In 2D: four species. Net chirality: exactly 0,
       always. Topology, not accident.
  [31] the standard fix (Wilson term) gaps the doublers — by adding a
       term that BREAKS chiral symmetry explicitly. You may have the
       lattice or chirality, not both, naively. (Modern escapes —
       domain-wall and overlap fermions — buy chirality back with an
       extra dimension or non-locality: a constraint any engine
       hosting our matter must budget for. New ledger line.)
"""
import numpy as np
from PIL import Image, ImageDraw

R_WILSON = 0.5
C_NAIVE, C_WILSON = (57, 135, 229), (217, 89, 38)  # dataviz dark slots 1-2
INK, MUTED, GRID_C = (195, 194, 183), (122, 122, 130), (38, 38, 44)


def main():
    print('=' * 68)
    print('PART 10e: THE DOUBLING THEOREM (chirality vs the lattice)')
    print('=' * 68)

    ks = np.linspace(-np.pi, np.pi, 2001)
    e_naive = np.sin(ks)
    # the Brillouin zone is a CIRCLE: scan crossings periodically
    kc = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    ec = np.sin(kc + 1e-9)
    zeros = [float(((kc[i] + np.pi) % (2 * np.pi)) - np.pi)
             for i in range(len(kc))
             if ec[i - 1] < 0 <= ec[i] or ec[i - 1] > 0 >= ec[i]]
    print('[30] naive 1D lattice fermion, E(k) = sin k:')
    print(f'     gapless points in the Brillouin zone: '
          f'{[f"{z:+.2f}" for z in zeros]}  ({len(zeros)} species)')
    chir = [int(np.sign(np.cos(z))) for z in zeros]
    print(f'     chirality (slope sign) at each: {chir}  ->  net: '
          f'{sum(chir)}')
    print('     2D naive fermion: gapless where sin kx = sin ky = 0:')
    pts = [(kx, ky) for kx in (0.0, np.pi) for ky in (0.0, np.pi)]
    ch2 = [int(np.sign(np.cos(kx) * np.cos(ky))) for kx, ky in pts]
    print(f'     {len(pts)} species at (0,0),(0,pi),(pi,0),(pi,pi); '
          f'chiralities {ch2} -> net {sum(ch2)}')
    print('     One ordered, three manufactured. The count is even by')
    print('     topology: sin k must come back down. No local lattice')
    print('     engine hosts a NET-chiral fermion this way — and the')
    print('     Standard Model is net-chiral. Ledger line acquired.')
    print()

    w = R_WILSON * (1 - np.cos(ks))
    e_wilson = np.sqrt(np.sin(ks) ** 2 + w ** 2)
    gap_pi = 2 * R_WILSON
    breaking = float((2 * np.abs(w)).max())
    print(f'[31] Wilson term (r={R_WILSON}): doubler at k=pi gains mass '
          f'{gap_pi:.1f};')
    print(f'     chiral-symmetry violation ||g5 H g5 + H|| peaks at '
          f'{breaking:.1f} (zero for the naive fermion).')
    print('     The cure gaps the ghost by breaking the very symmetry')
    print('     chirality needs. Pick your poison — or pay for overlap')
    print('     fermions with non-locality.')

    # chart: two bands
    W, H, ml, mr, mt, mb = 1000, 460, 70, 30, 52, 50
    img = Image.new('RGB', (W, H), (14, 14, 18))
    d = ImageDraw.Draw(img)
    top = float(e_wilson.max()) * 1.1

    def xy(k, e):
        return (ml + (W - ml - mr) * (k + np.pi) / (2 * np.pi),
                H - mb - (H - mt - mb) * e / top)

    for fr in (0.0, 0.5, 1.0):
        yy = xy(-np.pi, top * fr / 1.1)[1]
        d.line([(ml, yy), (W - mr, yy)], fill=GRID_C)
    for kk, lab in ((-np.pi, '-pi'), (0, '0'), (np.pi, 'pi')):
        x = xy(kk, 0)[0]
        d.text((x - 8, H - mb + 8), lab, fill=MUTED)
    d.text((10, 12), '|E(k)|  (gapless = a particle species lives here)',
           fill=INK)
    d.line([xy(k, abs(e)) for k, e in zip(ks, e_naive)],
           fill=C_NAIVE, width=2)
    d.line([xy(k, e) for k, e in zip(ks, e_wilson)],
           fill=C_WILSON, width=2)
    for z in zeros:
        x, y = xy(z, 0)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=C_NAIVE, width=2)
    for label, color, yy in ((f'naive: gapless at 0 AND pi (doubled)',
                              C_NAIVE, mt + 4),
                             (f'Wilson r={R_WILSON}: doubler gapped, '
                              f'chiral symmetry broken', C_WILSON, mt + 24)):
        d.line([(ml + 12, yy + 6), (ml + 38, yy + 6)], fill=color, width=3)
        d.text((ml + 46, yy), label, fill=INK)
    img.save('films/doubling.png')
    print('\n     films/doubling.png')


if __name__ == '__main__':
    main()
