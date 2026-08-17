"""Everything the physicist is allowed to touch.

An Instruments object is a telescope and a prod: you can look at the
screen, and you can put matter onto the screen. That's it. No engine
handles ever pass through here.
"""


class Instruments:
    def __init__(self, frame, poke, tick, emit_pair=None, set_analyzer=None):
        self._frame = frame
        self._poke = poke
        self._tick = tick
        self._emit_pair = emit_pair
        self._set_analyzer = set_analyzer

    def frame(self):
        """Current rendered screen as a 2D array (0 = vacuum, 1 = matter)."""
        return self._frame()

    def poke(self, y, x, pattern):
        """Place a pattern of matter at screen coordinates (y, x)."""
        self._poke(y, x, pattern)

    def tick(self, k=1):
        """Let the universe run for k ticks."""
        self._tick(k)

    def emit_pair(self, loc_a, loc_b):
        """Fire the deep-matter source: one pair, its two ends delivered
        to the given screen locations."""
        self._emit_pair(loc_a, loc_b)

    def set_analyzer(self, loc, theta):
        """Point an analyzer (angle theta) at a screen location. The
        outcome appears on screen as a flash: lit 2x2 at the location
        itself means +1, lit 2x2 four cells to the right means -1."""
        self._set_analyzer(loc, theta)
