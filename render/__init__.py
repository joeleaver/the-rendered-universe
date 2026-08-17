"""The renderer: projects engine state into screen space.

Screen space is where observers live. Pixels are not the mechanism;
they are the output. The mapping from engine cells to screen pixels
(the 'chart') is a fixed bijection — but nothing requires it to
preserve adjacency, and in this universe it doesn't quite.
"""
