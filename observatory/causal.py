"""Geometry from causal structure.

The probe: prepare a thermal reference state, flip one cell, and run
both copies. The set of cells whose state differs at tick t is the
causal ball of radius t around the flip. From arrival times we get:

  - butterfly speed  : how fast the causal ball's radius grows
  - dimension        : how fast its VOLUME grows (slope of log V vs
                       log r) — the emergent dimension of space
  - distance fields  : d(source, x) = first tick the flip can matter
                       at x, in units of causal time
"""
import numpy as np

from engine.substrate import Substrate


def arrival_field(size, lut, source, ticks, rng, wires=(), refs=3, states=2):
    """Mean first-influence time from `source` to every cell.
    Cells never reached get ticks+1."""
    total = np.zeros((size, size))
    for _ in range(refs):
        if states == 2:
            ref = (rng.random((size, size)) < 0.5).astype(np.uint8)
        else:
            ref = rng.integers(0, states, (size, size)).astype(np.uint8)
        a = Substrate(size, lut=lut, wires=wires, states=states)
        b = Substrate(size, lut=lut, wires=wires, states=states)
        a.grid = ref.copy()
        b.grid = ref.copy()
        b.grid[source] = (b.grid[source] + 1) % states
        arrive = np.full((size, size), ticks + 1, dtype=float)
        arrive[source] = 0
        for t in range(1, ticks + 1):
            a.step()
            b.step()
            fresh = (a.grid != b.grid) & (arrive > ticks)
            arrive[fresh] = t
        total += arrive
    return total / refs


def butterfly_and_dimension(size, lut, ticks=56, rng=None, states=2):
    """Probe from the center; return (speed, dimension, converged).

    Speed: max Chebyshev radius of the causal ball / elapsed time.
    Dimension: local slope of log(ball volume) vs log(causal radius)
    in the final window before torus saturation. The slope is only a
    dimension if growth has STABILIZED — a front still accelerating
    when the probe ends yields a number that measures the transient,
    not the geometry. We compare the last two windows and report
    converged=False when they disagree; the honest verdict there is
    'probe too small', never a dimension.
    """
    rng = rng or np.random.default_rng(0)
    src = (size // 2, size // 2)
    arr = arrival_field(size, lut, src, ticks, rng, refs=3, states=states)
    yy, xx = np.mgrid[0:size, 0:size]
    dy = np.minimum(np.abs(yy - src[0]), size - np.abs(yy - src[0]))
    dx = np.minimum(np.abs(xx - src[1]), size - np.abs(xx - src[1]))
    cheb = np.maximum(dy, dx)

    reached = arr <= ticks
    if reached.sum() <= 1:
        return 0.0, 0.0, True  # nothing spreads: converged on 'frozen'
    speed = float((cheb[reached] / np.maximum(arr[reached], 1)).max())

    rs, vols = [], []
    for r in range(3, ticks):
        v = int((arr <= r).sum())
        if v >= size * size * 0.6:
            break
        rs.append(np.log(r))
        vols.append(np.log(v))
    if len(rs) < 16:
        return speed, 0.0, False
    w = 8
    s_prev = float(np.polyfit(rs[-2 * w:-w], vols[-2 * w:-w], 1)[0])
    s_last = float(np.polyfit(rs[-w:], vols[-w:], 1)[0])
    return speed, s_last, abs(s_last - s_prev) < 0.4
