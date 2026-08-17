"""Part 2 of the experimental program: entanglement cartography.

Part 1 established c = 1.0 and located two 'doors' from eight anomalous
trajectory events (vanish/reappear endpoint pairs):

    door D1: screen rows 14..28, cols 14..30
    door D2: screen rows 94..108, cols 30..46

and their action: matter entering D1 exits D2 and vice versa. The
simplest chart consistent with this: the mechanism-content of the two
rectangles is swapped on screen. That theory makes a further, testable
prediction: the mechanism-content DISPLAYED inside D2 was cut out of
the sky at D1's location — so its mechanism-neighbors are the pixels
just OUTSIDE D1. In other words:

    screen strip left of D1  and  the interior of D2
    are mechanism-adjacent, despite being ~40+ cells apart on screen.

Two experiments test this, both timed so that no influence at speed c
could connect the stations through screen space:

  pixel_chsh_game — the CHSH game played with raw pixel matter.
      Each station pokes one of two patterns (its 'setting') and later
      reads the parity of its own patch (its 'outcome'). If outcomes
      are produced locally in screen space, |S| <= 2. Always.

  deep_chsh — the same game with deep-matter pairs and analyzer
      angles, plus a no-signaling audit of the marginals.

Everything below sees only frames and instrument handles.
"""
import math

import numpy as np

# Physicist's inferred chart (from part-1 survey events)
RECT = 12
DOOR1 = (16, 16)   # screen corner of door 1
DOOR2 = (96, 32)   # screen corner of door 2

# CHSH stations, main placement: strip just left of D1, interior of D2
GAME_L = (16, 2)
GAME_R = DOOR2
# Control placement: same screen separation, ordinary sky
CTRL_L = (16, 66)
CTRL_R = (96, 96)

GAME_T = 28  # readout tick; screen distance between rects is 37 > 28*c


def _screen_gap(a, b, n=128):
    """Chebyshev gap between two RECTxRECT patches on the torus."""
    def axgap(u, v):
        lo, hi = sorted((u, v))
        return max(0, min(hi - (lo + RECT - 1), (lo + n) - (hi + RECT - 1)))
    return max(axgap(a[0], b[0]), axgap(a[1], b[1]))


def _parity(frame, corner):
    y, x = corner
    return 1 if int(frame[y:y + RECT, x:x + RECT].sum()) % 2 == 0 else -1


def _chsh(E):
    """Max |S| over the four sign conventions (relabeling freedom)."""
    combos = []
    for flip in ((0, 0), (0, 1), (1, 0), (1, 1)):
        s = sum(E[(x, y)] * (-1 if (x, y) == flip else 1)
                for x in (0, 1) for y in (0, 1))
        combos.append(s)
    return max(combos, key=abs)


def _play_pixel_game(lab, place_l, place_r, pl, pr, ticks):
    """Run all four setting combinations; outcomes are patch parities."""
    out = {}
    for x in (0, 1):
        for y in (0, 1):
            u = lab()
            u.poke(place_l[0], place_l[1], pl[x])
            u.poke(place_r[0], place_r[1], pr[y])
            u.tick(ticks)
            f = u.frame()
            out[(x, y)] = (_parity(f, place_l), _parity(f, place_r))
    E = {xy: a * b for xy, (a, b) in out.items()}
    return _chsh(E), out


def pixel_chsh_game(lab, max_configs=300, seed=5):
    """Search over setting-pattern choices for a game that beats the
    locality bound at the predicted placement, then rerun the winning
    game at the control placement."""
    rng = np.random.default_rng(seed)
    pool = [(rng.random((6, 6)) < 0.5).astype(np.uint8) for _ in range(12)]
    for trial in range(max_configs):
        idx = rng.choice(len(pool), size=4, replace=False)
        pl = (pool[idx[0]], pool[idx[1]])
        pr = (pool[idx[2]], pool[idx[3]])
        s, out = _play_pixel_game(lab, GAME_L, GAME_R, pl, pr, GAME_T)
        if abs(s) == 4:
            s_ctrl, _ = _play_pixel_game(lab, CTRL_L, CTRL_R, pl, pr, GAME_T)
            return {'S': s, 'S_control': s_ctrl, 'outcomes': out,
                    'configs_tried': trial + 1,
                    'gap_main': _screen_gap(GAME_L, GAME_R),
                    'gap_ctrl': _screen_gap(CTRL_L, CTRL_R)}
    return None


# --- deep matter -----------------------------------------------------------

ANGLES_A = (0.0, math.pi / 2)
ANGLES_B = (math.pi / 4, 3 * math.pi / 4)

# analyzer positions: main placement rides the predicted hidden adjacency
DEEP_L, DEEP_R = (20, 8), (98, 34)
DEEP_CTRL_L, DEEP_CTRL_R = (20, 70), (98, 98)


def _read_flash(frame, loc):
    y, x = loc
    left = int(frame[y:y + 2, x:x + 2].sum())
    right = int(frame[y:y + 2, x + 4:x + 6].sum())
    if left == right:
        return 0  # no reading
    return 1 if left > right else -1


def deep_chsh(lab, loc_l, loc_r, trials=12000, seed=17):
    """CHSH with deep pairs: free random setting choice each trial,
    one pair per fresh universe, outcomes read from screen flashes.
    Returns S, per-combination statistics, and the no-signaling audit."""
    rng = np.random.default_rng(seed)
    tally = {(x, y): [] for x in (0, 1) for y in (0, 1)}
    for _ in range(trials):
        u = lab()
        x = int(rng.integers(2))
        y = int(rng.integers(2))
        u.emit_pair(loc_l, loc_r)
        u.set_analyzer(loc_l, ANGLES_A[x])
        u.set_analyzer(loc_r, ANGLES_B[y])
        u.tick(1)
        f = u.frame()
        a, b = _read_flash(f, loc_l), _read_flash(f, loc_r)
        if a and b:
            tally[(x, y)].append((a, b))
    E = {xy: float(np.mean([a * b for a, b in v])) for xy, v in tally.items()}
    counts = {xy: len(v) for xy, v in tally.items()}
    s = _chsh(E)
    # no-signaling audit: does MY marginal depend on YOUR setting?
    def marg(side, own, other):
        vals = []
        for xy, v in tally.items():
            if xy[0 if side == 'A' else 1] == own and \
               xy[1 if side == 'A' else 0] == other:
                vals += [ab[0 if side == 'A' else 1] for ab in v]
        return float(np.mean(vals)) if vals else 0.0
    leak = max(abs(marg(side, own, 0) - marg(side, own, 1))
               for side in ('A', 'B') for own in (0, 1))
    return {'S': s, 'E': E, 'counts': counts, 'signal_leak': leak,
            'n_used': sum(counts.values())}
