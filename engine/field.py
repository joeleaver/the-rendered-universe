"""Dynamics, at last: the scalar field evolving in time.

Everything before part 6 studied ground states — statics. Here the
same field obeys its equation of motion (lattice Klein-Gordon with a
position-dependent mass):

    d(phi)/dt = pi
    d(pi)/dt  = laplacian(phi) - m(x)^2 phi

integrated by leapfrog. A sponge layer near the boundary absorbs
outgoing waves so the torus doesn't echo.
"""
import numpy as np


class Field:
    def __init__(self, ny, nx, mass, dt=0.2, sponge=12):
        self.ny, self.nx = ny, nx
        self.m2 = np.asarray(mass, dtype=float) ** 2
        self.phi = np.zeros((ny, nx))
        self.pi = np.zeros((ny, nx))
        self.dt = dt
        self.t = 0.0
        yy, xx = np.mgrid[0:ny, 0:nx]
        d = np.minimum.reduce([yy, ny - 1 - yy, xx, nx - 1 - xx])
        ramp = np.clip((sponge - d) / sponge, 0, 1)
        self.damp = 1 - 0.06 * ramp ** 2

    def _lap(self, f):
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)

    def step(self, k=1):
        for _ in range(k):
            self.pi += self.dt * (self._lap(self.phi) - self.m2 * self.phi)
            self.phi += self.dt * self.pi
            self.pi *= self.damp
            self.phi *= self.damp
            self.t += self.dt

    def add_packet(self, y0, x0, k_x, wy=10, wx=6, m0=0.2):
        """A right-moving wave packet: carrier cos(k x - w t) under a
        Gaussian envelope. Returns the carrier frequency (lattice
        dispersion)."""
        yy, xx = np.mgrid[0:self.ny, 0:self.nx]
        env = np.exp(-((yy - y0) ** 2 / (2 * wy ** 2)
                       + (xx - x0) ** 2 / (2 * wx ** 2)))
        om = np.sqrt(m0 ** 2 + 4 * np.sin(k_x / 2) ** 2)
        self.phi += env * np.cos(k_x * xx)
        self.pi += env * om * np.sin(k_x * xx)
        return om

    def energy(self):
        return self.pi ** 2 + self.phi ** 2
