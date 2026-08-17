"""Part 9: the arrow of time.

The engine is a bijection — part 1 proved it by running the universe
backwards, exactly. So the arrow of time cannot live in the laws.
Boltzmann's answer, in this repo's language: the arrow lives in the
RENDER layer. Coarse-grain the pixels into blocks and count matter per
block; the Boltzmann entropy of that coarse description,

    S = sum over blocks of log2 C(block_size, n_block),

counts the microstates compatible with what the renderer shows. The
engine never loses a bit; the rendering does. Experiments:

  [21] the rise — matter starts as a dense blob (a low-entropy past);
       S climbs to equilibrium and stays there.
  [22] Loschmidt — at t=T, reverse the dynamics exactly: S marches
       back DOWN to its initial value and the blob reassembles
       (entropy decrease, on demand, legal). Then reverse again with
       ONE cell flipped: the conspiracy shatters at butterfly speed
       and the past becomes unreachable.
  [23] typicality — random states with the same matter count all sit
       at S_eq. The arrow is not in the dynamics; it is in how
       astronomically special the initial rendering was.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from engine.substrate import Substrate

SIZE, BLOCK = 128, 8
T_REV = 1500
EVERY = 6

LOG2C = np.array([math.lgamma(BLOCK ** 2 + 1) - math.lgamma(n + 1)
                  - math.lgamma(BLOCK ** 2 - n + 1)
                  for n in range(BLOCK ** 2 + 1)]) / math.log(2)

# validated dark-mode categorical palette (dataviz slots 1-3)
C_FWD, C_ORANGE, C_AQUA = (57, 135, 229), (217, 89, 38), (25, 158, 112)
INK, MUTED, GRID = (195, 194, 183), (122, 122, 130), (38, 38, 44)
BG = (14, 14, 18)


def frame_of(sub):
    return sub.grid ^ (sub.t % 2)


def coarse_entropy(frame):
    k = SIZE // BLOCK
    counts = frame.reshape(k, BLOCK, k, BLOCK).transpose(0, 2, 1, 3) \
                  .reshape(-1, BLOCK ** 2).sum(axis=1)
    return float(LOG2C[counts].sum())


def blob_universe(seed=5):
    rng = np.random.default_rng(seed)
    sub = Substrate(SIZE)
    blob = (rng.random((64, 64)) < 0.5).astype(np.uint8)
    sub.grid[32:96, 32:96] = blob
    return sub


def run_leg(sub, ticks, backward=False, record_frames=None, every_f=12):
    ss = []
    for i in range(ticks):
        (sub.unstep if backward else sub.step)()
        if i % EVERY == 0 or i == ticks - 1:
            ss.append(coarse_entropy(frame_of(sub)))
        if record_frames is not None and i % every_f == 0:
            record_frames.append(frame_of(sub).copy())
    return ss


def chart(s_fwd, s_exact, s_pert, s_init, s_eq, path):
    W, H, ml, mr, mt, mb = 1120, 540, 80, 30, 56, 52
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    t_max, s_max = 2 * T_REV, max(max(s_fwd), max(s_pert)) * 1.06

    def xy(t, s):
        return (ml + (W - ml - mr) * t / t_max,
                H - mb - (H - mt - mb) * s / s_max)

    for s in range(0, int(s_max), 2000):
        y = xy(0, s)[1]
        d.line([(ml, y), (W - mr, y)], fill=GRID)
        d.text((10, y - 7), f'{s}', fill=MUTED)
    for t in range(0, t_max + 1, 500):
        x = xy(t, 0)[0]
        d.text((x - 12, H - mb + 8), f'{t}', fill=MUTED)
    d.text((10, 12), 'coarse-grained entropy S (bits)', fill=INK)
    d.text((W // 2 - 14, H - 24), 'tick', fill=MUTED)

    xr = xy(T_REV, 0)[0]
    for y in range(mt, H - mb, 9):
        d.line([(xr, y), (xr, y + 4)], fill=MUTED)
    d.text((xr + 6, mt + 2), 'reversal', fill=MUTED)

    def series(ss, t0, color):
        pts = [xy(t0 + i * EVERY, s) for i, s in enumerate(ss)]
        d.line(pts, fill=color, width=2)

    series(s_fwd, 0, C_FWD)
    series(s_pert, T_REV, C_ORANGE)
    series(s_exact, T_REV, C_AQUA)

    for label, color, y in (('forward', C_FWD, mt + 6),
                            ('reversal, one cell flipped', C_ORANGE, mt + 26),
                            ('exact reversal', C_AQUA, mt + 46)):
        d.line([(ml + 14, y + 6), (ml + 40, y + 6)], fill=color, width=3)
        d.text((ml + 48, y), label, fill=INK)
    d.text((xy(60, s_init)[0], xy(0, s_init)[1] - 18),
           f'S(blob) = {s_init:.0f}', fill=MUTED)
    d.text((xy(650, 0)[0], xy(0, s_eq)[1] + 14),
           f'S(equilibrium) ~ {s_eq:.0f}', fill=MUTED)
    img.save(path)


def film(frames_l, frames_r, path):
    scale = 3
    n = min(len(frames_l), len(frames_r))
    imgs = []
    for i in range(n):
        canvas = np.full((SIZE + 14, 2 * SIZE + 8, 3), 14, dtype=np.uint8)
        for k, f in enumerate((frames_l[i], frames_r[i])):
            x0 = k * (SIZE + 8)
            canvas[14:, x0:x0 + SIZE][f > 0] = (225, 225, 232)
        im = Image.fromarray(np.kron(canvas, np.ones((scale, scale, 1),
                                                     dtype=np.uint8)), 'RGB')
        dr = ImageDraw.Draw(im)
        dr.text((10, 6), 'exact reversal', fill=(25, 198, 142))
        dr.text((SIZE * scale + 34, 6), 'one cell flipped', fill=(240, 120, 60))
        imgs.append(im)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=45, loop=0)


def main():
    print('=' * 68)
    print("PART 9: THE ARROW OF TIME (the engine has none; the render does)")
    print('=' * 68)

    sub = blob_universe()
    start = sub.grid.copy()
    s_init = coarse_entropy(frame_of(sub))
    n_matter = int(frame_of(sub).sum())

    s_fwd = [s_init] + run_leg(sub, T_REV)
    s_eq = float(np.mean(s_fwd[-50:]))
    print(f'[21] the rise: {n_matter} particles start as a blob,')
    print(f'     S(0) = {s_init:.0f} bits -> S({T_REV}) = {s_fwd[-1]:.0f} '
          f'(equilibrium ~ {s_eq:.0f})')
    print('     the engine is bijective the whole way: not one bit of')
    print('     microstate was created or destroyed while S climbed.')
    print()

    g_rev = sub.grid.copy()
    t_rev = sub.t

    frames_exact = []
    s_exact = [s_fwd[-1]] + run_leg(sub, T_REV, backward=True,
                                    record_frames=frames_exact)
    restored = np.array_equal(sub.grid, start)
    print(f'[22] Loschmidt:')
    print(f'     exact reversal: S falls {s_fwd[-1]:.0f} -> '
          f'{s_exact[-1]:.0f}; microstate restored EXACTLY: {restored}')

    sub2 = Substrate(SIZE)
    sub2.grid = g_rev.copy()
    sub2.t = t_rev
    sub2.grid[64, 64] ^= 1  # one flipped bit, in the thick of the gas
    frames_pert = []
    s_pert = [s_fwd[-1]] + run_leg(sub2, T_REV, backward=True,
                                   record_frames=frames_pert)
    ham = int((sub2.grid != start).sum())
    print(f'     flip ONE cell in the gas first: S never follows the')
    print(f'     descent — it clings to equilibrium, ending at '
          f'{s_pert[-1]:.0f}; {ham} cells ({ham / SIZE ** 2:.0%} of the '
          f'torus) corrupted — the past is gone.')

    sub3 = Substrate(SIZE)
    sub3.grid = g_rev.copy()
    sub3.t = t_rev
    sub3.grid[0, 0] ^= 1  # one flipped bit, in empty space
    for _ in range(T_REV):
        sub3.unstep()
    ham3 = int((sub3.grid != start).sum())
    print(f'     flip ONE cell in VACUUM instead: only {ham3} cell(s)')
    print(f'     differ at the end — the perturbation formed a bound')
    print(f'     state and never touched anything. Chaos here is')
    print(f'     matter-borne: vacuum protects the past; collisions')
    print(f'     destroy it. The arrow is fragility, not law.')
    print()

    rng = np.random.default_rng(3)
    s_typ = []
    for _ in range(40):
        f = np.zeros(SIZE * SIZE, dtype=np.uint8)
        f[rng.choice(SIZE * SIZE, n_matter, replace=False)] = 1
        s_typ.append(coarse_entropy(f.reshape(SIZE, SIZE)))
    s_typ_m = float(np.mean(s_typ))
    print(f'[23] typicality: 40 random states with the same matter count:')
    print(f'     S = {s_typ_m:.0f} +/- {np.std(s_typ):.0f} — every one at '
          f'equilibrium.')
    print(f'     the blob is atypical by {s_typ_m - s_init:.0f} bits: about')
    print(f'     1 in 2^{s_typ_m - s_init:.0f} microstates look like our past.')
    print('     The arrow is not in the laws (we ran them backwards).')
    print('     It is in the initial condition — and in the coarse')
    print('     description doing the looking. The render forgets;')
    print('     the engine never did.')

    chart(s_fwd, s_exact, s_pert, s_init, s_eq, 'films/arrow.png')
    film(frames_exact, frames_pert, 'films/loschmidt.gif')
    print('\n     films/arrow.png, films/loschmidt.gif')


if __name__ == '__main__':
    main()
