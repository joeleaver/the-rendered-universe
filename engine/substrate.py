"""The substrate and its rule.

A rule is a bijection on the 16 possible states of a 2x2 block,
applied to every block of the grid, with the partition offset
alternating each tick. Every such rule is exactly reversible: the
universe never loses information.

The default rule is 'Critters' (Toffoli & Margolus). Its entire
physics: with n = number of live cells in the block —
    n == 2 : leave the block unchanged
    n == 3 : invert every cell, then rotate the block 180 degrees
    else   : invert every cell

Block state encoding: cells [(0,0),(0,1),(1,0),(1,1)] = bits
[8,4,2,1], so state index = 8a + 4b + 2c + d.

Optional `wires`: pairs of engine rectangles whose contents are
exchanged after every block update. This is MECHANISM wiring — a
modification of which cells are dynamically adjacent — not rendering.
"""
import numpy as np


def critters_lut():
    lut = np.zeros(16, dtype=np.uint8)
    for s in range(16):
        bits = [(s >> k) & 1 for k in (3, 2, 1, 0)]  # a, b, c, d
        n = sum(bits)
        if n == 2:
            out = bits
        else:
            out = [1 - v for v in bits]
            if n == 3:
                out = out[::-1]  # 180-degree rotation of a 2x2
        lut[s] = out[0] * 8 + out[1] * 4 + out[2] * 2 + out[3]
    return lut


class Substrate:
    def __init__(self, size, lut=None, wires=(), states=2):
        assert size % 2 == 0
        self.n = size
        self.k = states
        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.t = 0
        if lut is None:
            assert states == 2
            lut = critters_lut()
        self.lut = np.asarray(lut)
        self.inv_lut = np.argsort(self.lut).astype(self.lut.dtype)
        self.wires = list(wires)

    def _apply(self, lut, offset):
        g = np.roll(self.grid, (-offset, -offset), axis=(0, 1))
        n, k = self.n, self.k
        b = g.reshape(n // 2, 2, n // 2, 2).transpose(0, 2, 1, 3) \
             .reshape(-1, 4).astype(np.int64)
        idx = ((b[:, 0] * k + b[:, 1]) * k + b[:, 2]) * k + b[:, 3]
        out = lut[idx].astype(np.int64)
        b2 = np.stack([(out // k ** 3) % k, (out // k ** 2) % k,
                       (out // k) % k, out % k], axis=1).astype(np.uint8)
        g2 = b2.reshape(n // 2, n // 2, 2, 2).transpose(0, 2, 1, 3).reshape(n, n)
        self.grid = np.roll(g2, (offset, offset), axis=(0, 1))

    def _run_wires(self):
        for (y1, x1), (y2, x2), (h, w) in self.wires:
            a = self.grid[y1:y1 + h, x1:x1 + w].copy()
            self.grid[y1:y1 + h, x1:x1 + w] = self.grid[y2:y2 + h, x2:x2 + w]
            self.grid[y2:y2 + h, x2:x2 + w] = a

    def step(self):
        self._apply(self.lut, self.t % 2)
        self._run_wires()
        self.t += 1

    def unstep(self):
        """Exact inverse of step(). Time's arrow is a convention here."""
        self.t -= 1
        self._run_wires()
        self._apply(self.inv_lut, self.t % 2)
