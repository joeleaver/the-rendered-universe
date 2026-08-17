"""The chart: which engine cell each screen pixel displays.

The chart is the identity almost everywhere — screen position equals
engine position — except for a set of rectangular patch *swaps*: two
regions of engine space whose screen positions are exchanged. Matter
crossing such a region in the engine (where nothing unusual happens)
appears to an observer to vanish at one place and reappear at another.

The engine does not know the chart exists. Observers cannot see past it.
"""
import numpy as np


class Renderer:
    def __init__(self, substrate, swaps=()):
        """swaps: list of ((y1, x1), (y2, x2), (h, w)) — engine rectangles
        whose screen positions are exchanged."""
        self.sub = substrate
        n = substrate.n
        src_y, src_x = np.mgrid[0:n, 0:n]
        src_y, src_x = src_y.copy(), src_x.copy()
        for (y1, x1), (y2, x2), (h, w) in swaps:
            a = (slice(y1, y1 + h), slice(x1, x1 + w))
            b = (slice(y2, y2 + h), slice(x2, x2 + w))
            for arr in (src_y, src_x):
                tmp = arr[a].copy()
                arr[a] = arr[b]
                arr[b] = tmp
        self.src_y, self.src_x = src_y, src_x

    def frame(self):
        """The rendered screen: phase-corrected so vacuum is dark and
        matter is bright, then projected through the chart."""
        g = self.sub.grid
        if self.sub.t % 2 == 1:
            g = 1 - g
        return g[self.src_y, self.src_x]

    def poke(self, y, x, pattern):
        """Write matter into the universe at *screen* coordinates.

        The inverse chart routes the write to wherever those pixels
        actually live in the engine.
        """
        pattern = np.asarray(pattern, dtype=np.uint8)
        n = self.sub.n
        h, w = pattern.shape
        ys = np.arange(y, y + h) % n
        xs = np.arange(x, x + w) % n
        sy = self.src_y[np.ix_(ys, xs)]
        sx = self.src_x[np.ix_(ys, xs)]
        phase = self.sub.t % 2
        self.sub.grid[sy, sx] = pattern ^ phase
