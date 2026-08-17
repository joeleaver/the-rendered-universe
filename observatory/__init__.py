"""The observatory: renderer v2.

Version 1's renderer was handed a chart — the geometry was an input.
The observatory inverts that: it takes a universe and DERIVES its
geometry, from causal structure alone. Distance between two cells is
defined by how quickly a disturbance at one can influence the other
(damage spreading in a thermal medium). A classical theorem (Malament
1977) licenses this: causal order determines the metric up to scale.

Nothing here reads cell indices as positions. Position is an output.
"""
