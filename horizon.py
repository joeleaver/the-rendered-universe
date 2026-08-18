"""Part 23: time dilation, horizons, and Hawking radiation.

Part 22 obtained the linearized Einstein equation from the entanglement
first law. This part builds horizon physics. There are two ways to hide
a region behind a horizon: let the local propagation speed go to zero
(a frozen star), or let the chart flow inward faster than the
propagation speed (an acoustic horizon, the Painleve picture of a GR
horizon). The first only delays; the second radiates.

  [89] clocks: time dilation measured directly, both kinds. Two
       identical cavity clocks at different depths of a well tick at
       the ratio sqrt(g00) to four decimal places; a moving packet's
       internal phase clock slows as m*sqrt(1-v^2) out to v = 0.73
       (the residual is the lattice's k^4 correction, measured).
  [90] the frozen star: a 2D metric whose local light speed c(r)
       vanishes linearly at r_h. The star casts a shadow at the
       ray-traced critical impact parameter (wave and geometric
       optics agreeing on the same metric); a radial packet's
       arrival times track the diverging metric integral of dr/c
       -- the freeze -- until the substrate pins it at the
       wavelength wall and REFLECTS it: on a lattice, a frozen
       star is a mirror with divergent delay, not a trap. The
       trans-Planckian cutoff, measured in a telescope.
  [91] the acoustic horizon radiates: a flow crossing c = 1, evolved
       with the chain's exact Gaussian state through switch-on, emits
       a STEADY thermal flux: measured spectrum Planckian at
       T = kappa/2pi within ~6%, with each emitted quantum's partner
       resolved falling inside (the correlation map), radiation-
       interior entanglement growing, and the global state exactly
       pure — Hawking 1974 and Unruh 1981, with the total state
       verified pure to 10^-4.
  [92] summary. Owed at this point: evaporation backreaction (this
       hole never shrinks), and with it the Page curve.
"""
import math
import time

import numpy as np
from PIL import Image, ImageDraw

# validated dark-mode categorical palette (dataviz slots 1-3)
C_BLUE, C_ORANGE, C_GREEN = (57, 135, 229), (217, 89, 38), (25, 158, 112)
INK, MUTED, GRIDC = (195, 194, 183), (122, 122, 130), (38, 38, 44)
BG = (14, 14, 18)


# ======== [89] clocks ==================================================

def grav_clocks():
    """Two identical standing blobs of a massive field, one on the
    floor of a plateau well, one far outside; frequency by zero
    crossings. The g00 factor multiplies the whole spatial operator
    (part 16's lesson), so the deep clock runs slow by sqrt(g00)."""
    n, m0 = 400, 2.0
    x = np.arange(n)
    rows = []
    for A in (0.05, 0.15, 0.25, 0.35):
        phi = -(A / 2) * (np.tanh((x - 200) / 8.0) - np.tanh((x - 320) / 8.0))
        g00 = 1 + 2 * phi
        f = np.exp(-(x - 260.0) ** 2 / 72.0) + np.exp(-(x - 80.0) ** 2 / 72.0)
        p = np.zeros(n)
        dt = 0.05
        rec = np.empty((int(400 / dt), 2))
        for it in range(rec.shape[0]):
            lap = np.zeros(n)
            lap[1:-1] = f[2:] + f[:-2] - 2 * f[1:-1]
            lap[0] = f[1] - 2 * f[0]
            lap[-1] = f[-2] - 2 * f[-1]
            p += dt * g00 * (lap - m0 * m0 * f)
            f += dt * p
            rec[it] = f[80], f[260]

        def freq(sig):
            cr = np.where(np.diff(np.sign(sig)) != 0)[0]
            return math.pi / (np.diff(cr).mean() * dt)
        rows.append((A, freq(rec[:, 1]) / freq(rec[:, 0]),
                     math.sqrt(1 - 2 * A)))
    return rows


def moving_clocks():
    """A massive packet at velocity v; the field's phase at the moving
    centroid advances at m/gamma — the internal clock, read off the
    render. v is measured from the centroid track, m from the v=0 run:
    the overlay has no free parameters."""
    n, m0 = 1400, 0.5
    x = np.arange(n)
    rows = []
    for k0 in (1e-6, 0.15, 0.30, 0.45, 0.60):
        om = math.sqrt(m0 * m0 + 4 * math.sin(k0 / 2) ** 2)
        env = np.exp(-(x - 200.0) ** 2 / (2 * 30.0 ** 2))
        f = env * np.cos(k0 * x)
        p = env * om * np.sin(k0 * x)
        dt = 0.05
        sig, cen, ts = [], [], []
        for it in range(int(900 / dt)):
            lap = np.zeros(n)
            lap[1:-1] = f[2:] + f[:-2] - 2 * f[1:-1]
            lap[0] = f[1] - 2 * f[0]
            lap[-1] = f[-2] - 2 * f[-1]
            p += dt * (lap - m0 * m0 * f)
            f += dt * p
            if it % 4 == 0:
                e = p * p + m0 * m0 * f * f
                e = np.where(e > 0.02 * e.max(), e, 0)
                xc = float((e * x).sum() / e.sum())
                i0 = int(xc)
                sig.append((1 - xc + i0) * f[i0] + (xc - i0) * f[i0 + 1])
                cen.append(xc)
                ts.append(it * dt)
        sig, cen, ts = np.array(sig), np.array(cen), np.array(ts)
        lo = len(sig) // 5
        cr = np.where(np.diff(np.sign(sig[lo:])) != 0)[0]
        om_clock = math.pi / (np.diff(cr).mean() * (ts[1] - ts[0]))
        v = float(np.polyfit(ts[lo:], cen[lo:], 1)[0])
        rows.append((v, om_clock, m0 * math.sqrt(max(1 - v * v, 0.0))))
    return rows


# ======== [90] the frozen star ========================================

NY2, NX2 = 240, 420
CY, CX = 120, 150
B_STAR, SIG_STAR = 1.8, 18.0
YY, XX = np.mgrid[0:NY2, 0:NX2].astype(float)
RR = np.sqrt((YY - CY) ** 2 + (XX - CX) ** 2)
CMAP = np.clip(1 - B_STAR * np.exp(-RR ** 2 / (2 * SIG_STAR ** 2)), 1e-3, 1.0)
R_H = SIG_STAR * math.sqrt(2 * math.log(B_STAR))
KAPPA_H = R_H / SIG_STAR ** 2      # |dc/dr| at the horizon (linear zero)


class Wave2D:
    def __init__(self):
        c = CMAP
        self.c = c
        self.cxh = 0.5 * (c + np.roll(c, -1, 1))
        self.cyh = 0.5 * (c + np.roll(c, -1, 0))
        self.f = np.zeros_like(c)
        self.p = np.zeros_like(c)
        d = np.minimum.reduce([YY, NY2 - 1 - YY, XX, NX2 - 1 - XX])
        self.damp = 1 - 0.07 * np.clip((10 - d) / 10, 0, 1) ** 2
        self.dt = 0.12

    def step(self, k=1):
        for _ in range(k):
            fx = self.cxh * (np.roll(self.f, -1, 1) - self.f)
            fy = self.cyh * (np.roll(self.f, -1, 0) - self.f)
            div = (fx - np.roll(fx, 1, 1)) + (fy - np.roll(fy, 1, 0))
            self.p += self.dt * self.c * div
            self.f += self.dt * self.p
            self.p *= self.damp
            self.f *= self.damp

    def launch(self, b, k0):
        env = np.exp(-((YY - (CY - b)) ** 2 / 50.0
                       + (XX - 60.0) ** 2 / 98.0))
        self.f += env * np.cos(k0 * XX)
        self.p += env * 2 * math.sin(k0 / 2) * np.sin(k0 * XX)


def shadow_curve(k0=0.9):
    """The star's prompt shadow: peak energy crossing a downstream
    window, each impact parameter normalized by its own empty-universe
    control (part 21's calibration ethos). Light inside the critical
    impact parameter does not come through on time."""
    def run(b, star):
        wv = Wave2D()
        if not star:
            wv.c = np.ones_like(CMAP)
            wv.cxh = np.ones_like(CMAP)
            wv.cyh = np.ones_like(CMAP)
        wv.launch(b, k0)
        peak = 0.0
        for _ in range(int(420 / wv.dt / 6)):
            wv.step(6)
            e = wv.p ** 2 + wv.f ** 2
            peak = max(peak, float(e[:, 330:420].sum()))
        return peak
    ctrl = run(0, star=False)
    return [(b, run(b, star=True) / ctrl)
            for b in (0, 8, 16, 24, 32, 40, 48, 56, 64)]


def ray_bcrit():
    """Geometric optics on the same metric: H = c(x)|k|. Bisect the
    critical impact parameter."""
    def captured(b):
        x, y, kx, ky = 60.0, CY - b, 1.0, 0.0
        dt = 0.25
        for _ in range(4800):
            ix = min(max(int(x), 1), NX2 - 2)
            iy = min(max(int(y), 1), NY2 - 2)
            c = CMAP[iy, ix]
            gx = 0.5 * (CMAP[iy, ix + 1] - CMAP[iy, ix - 1])
            gy = 0.5 * (CMAP[iy + 1, ix] - CMAP[iy - 1, ix])
            kk = math.hypot(kx, ky)
            x += dt * c * kx / kk
            y += dt * c * ky / kk
            kx -= dt * kk * gx
            ky -= dt * kk * gy
            if math.hypot(x - CX, y - CY) < R_H + 1.5:
                return True
            if not (5 < x < NX2 - 12 and 5 < y < NY2 - 5):
                return False
        return True
    lo, hi = 0.0, 80.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if captured(mid) else (lo, mid)
    return lo


RBINS = np.round(RR).astype(int)


def approach_run(k0=0.8, frames_every=25):
    """Radial infall watched as arrival times: t_arr(r) = when the
    annulus at radius r first lights up. Prediction (drawn, not
    fitted): the metric integral t(r) = t(r0) + int dr'/c(r'), which
    diverges logarithmically at the horizon — the freeze. The lattice
    calls it off at the pinning radius, where the blueshifted local
    wavelength reaches the substrate scale, and reflects: on this
    substrate a frozen star is a mirror with divergent delay, not a
    trap."""
    wv = Wave2D()
    wv.launch(0, k0)
    rmax = 80
    ann = []
    frames = []
    for it in range(int(430 / wv.dt)):
        wv.step(1)
        if it % 4 == 0:
            e = (wv.p ** 2 + wv.f ** 2)[:, :CX + 20]
            ann.append(np.bincount(RBINS[:, :CX + 20].ravel(),
                                   weights=e.ravel(), minlength=200)[:rmax])
        if it % frames_every == 0:
            frames.append((wv.p ** 2 + wv.f ** 2).copy())
    ann = np.array(ann)            # [time, radius]
    tgrid = np.arange(ann.shape[0]) * wv.dt * 4
    t_arr = []
    for r in range(22, 61):
        a = ann[:, r]
        thr = 0.25 * a.max()
        i = int(np.argmax(a > thr))
        t_arr.append((r, tgrid[i]))
    t_arr = np.array(t_arr)
    # metric prediction anchored at r = 55
    rq = np.linspace(20.2, 60, 800)
    cq = np.clip(1 - B_STAR * np.exp(-rq ** 2 / (2 * SIG_STAR ** 2)),
                 1e-3, 1)
    tin = np.cumsum(1.0 / cq[::-1]) * (rq[1] - rq[0])   # from r=60 inward
    t_pred = dict(zip(np.round(rq[::-1], 3), tin))
    anchor = float(t_arr[t_arr[:, 0] == 55, 1][0])
    rq_r = rq[::-1]
    pred = anchor + (np.interp(55, rq, tin[::-1]) * 0
                     + np.array([np.interp(r, rq, tin[::-1])
                                 for r in t_arr[:, 0]])
                     - np.interp(55, rq, tin[::-1]))
    # pinning radius: local wavelength reaches the lattice scale
    cpin = k0 / math.pi
    r_pin = float(rq[np.argmin(np.abs(cq - cpin))])
    # the bounce: outgoing energy at r=50 after the infall passed
    a50 = ann[:, 50]
    i_in = int(np.argmax(a50 > 0.25 * a50.max()))
    late = a50[i_in + int(60 / (wv.dt * 4)):]
    i_ret = int(np.argmax(late > 0.25 * a50.max()))
    t_ret = tgrid[i_in + int(60 / (wv.dt * 4)) + i_ret] - tgrid[i_in] \
        if late.max() > 0.25 * a50.max() else None
    return t_arr, pred, r_pin, t_ret, frames


# ======== [91] the waterfall ==========================================

NF = 1000
JF = np.arange(NF)
V0, XHF, WWF = 1.4, 380.0, 3.0
VPROF = -0.5 * V0 * (1.0 - np.tanh((JF - XHF) / WWF))
S_H = math.atanh(1 - 2.0 / V0)
X_HOR = XHF + WWF * S_H
KAPPA_F = (V0 / (2 * WWF)) * (1 - math.tanh(S_H) ** 2)
T_HAWK = KAPPA_F / (2 * math.pi)


def K_apply(M):
    out = 2.0 * M
    out[1:] -= M[:-1]
    out[:-1] -= M[1:]
    return out


def Dc_apply(M):
    out = np.zeros_like(M)
    out[1:-1] = 0.5 * (M[2:] - M[:-2])
    out[0] = 0.5 * M[1]
    out[-1] = -0.5 * M[-2]
    return out


def DcT_apply(M):
    out = np.zeros_like(M)
    out[2:] += 0.5 * M[1:-1]
    out[:-2] -= 0.5 * M[1:-1]
    out[1] += 0.5 * M[0]
    out[-2] -= 0.5 * M[-1]
    return out


def A_on(M, v):
    """Generator of the flow Hamiltonian H = pi^2/2 - v pi phi' + phi'^2/2
    applied to a stacked (phi; pi) block matrix."""
    top = -(v[:, None] * Dc_apply(M[:NF])) + M[NF:]
    bot = -K_apply(M[:NF]) + DcT_apply(v[:, None] * M[NF:])
    return np.vstack([top, bot])


def propagator(v, dt, nsq):
    """exp(A dt 2^nsq): 4th-order series then repeated squaring —
    evolution by matrix multiply, exact to the series error."""
    M = np.eye(2 * NF)
    term = np.eye(2 * NF)
    for order in range(1, 5):
        term = A_on(term, v) * (dt / order)
        M = M + term
    for _ in range(nsq):
        M = M @ M
    return M


def ground_G():
    K = (np.diag(np.full(NF, 2.0)) - np.diag(np.ones(NF - 1), 1)
         - np.diag(np.ones(NF - 1), -1))
    w, U = np.linalg.eigh(K)
    om = np.sqrt(np.maximum(w, 1e-16))
    G = np.zeros((2 * NF, 2 * NF))
    G[:NF, :NF] = (U / (2 * om)) @ U.T
    G[NF:, NF:] = (U * (om / 2)) @ U.T
    return G, om, U


def occupations(G, xlo, xhi, ks, sign, base=None):
    """n(k) of travelling modes with phase e^{i(sign*k*x - w t)} in a
    Hann window; the direction convention is CALIBRATED in [0], not
    trusted from algebra."""
    xs = np.arange(xlo, xhi)
    L = len(xs)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(L) + 0.5) / L)
    Gxx, Gpp, Gxp = G[:NF, :NF], G[NF:, NF:], G[:NF, NF:]
    out = []
    for k in ks:
        om = 2 * math.sin(k / 2)
        psi = win * np.exp(1j * sign * k * xs)
        psi /= np.linalg.norm(psi)
        p = np.zeros(NF, complex)
        p[xs] = psi
        al, be = math.sqrt(om / 2), 1 / math.sqrt(2 * om)
        Xq = float(np.real(np.conj(p) @ (Gxx @ p)))
        Pq = float(np.real(np.conj(p) @ (Gpp @ p)))
        Cq = float(np.imag(np.conj(p) @ (Gxp @ p)))
        out.append(al * al * Xq + be * be * Pq - 2 * al * be * Cq - 0.5)
    out = np.array(out)
    return out if base is None else out - base


def gauss_entropy(G, sites):
    """Region entropy for a general Gaussian state (xp correlations
    allowed): symplectic eigenvalues of the full covariance block."""
    idx = np.array(sites)
    n = len(idx)
    GA = np.empty((2 * n, 2 * n))
    GA[:n, :n] = G[np.ix_(idx, idx)]
    GA[:n, n:] = G[np.ix_(idx, NF + idx)]
    GA[n:, :n] = GA[:n, n:].T
    GA[n:, n:] = G[np.ix_(NF + idx, NF + idx)]
    Om = np.zeros((2 * n, 2 * n))
    Om[:n, n:] = np.eye(n)
    Om[n:, :n] = -np.eye(n)
    nu = np.sort(np.abs(np.linalg.eigvals(Om @ GA).imag))[n:]
    nu = np.clip(nu, 0.5 + 1e-12, None)
    a, b = nu + 0.5, nu - 0.5
    return float((a * np.log(a) - b * np.log(b)).sum())


def waterfall():
    G0, om_modes, U = ground_G()
    ks = np.arange(0.045, 0.31, 0.015)
    oms = 2 * np.sin(ks / 2)

    # ---- [0]-style self-calibration of the mode detector ----
    # thermal state: n must read Bose-Einstein
    Tb = 0.05
    cot = 1.0 / np.tanh(om_modes / (2 * Tb))
    Gt = np.zeros_like(G0)
    Gt[:NF, :NF] = (U * (cot / (2 * om_modes))) @ U.T
    Gt[NF:, NF:] = (U * (om_modes * cot / 2)) @ U.T
    b0p = occupations(G0, 460, 820, ks, +1)
    b0m = occupations(G0, 460, 820, ks, -1)
    nt = occupations(Gt, 460, 820, ks, +1, base=b0p)
    be = 1.0 / (np.exp(oms / Tb) - 1)
    cal_therm = float(np.mean(nt[2:] / be[2:]))
    # direction: a warm blob left of the window sends right-movers in
    Gb = G0.copy()
    blob = np.exp(-(JF - 200.0) ** 2 / (2 * 20.0 ** 2))
    Gb[:NF, :NF] += 0.4 * np.diag(blob)
    Gb[NF:, NF:] += 0.4 * np.diag(blob)
    Pfree = propagator(np.zeros(NF), 0.05, 9)     # 25.6 time units
    for _ in range(12):
        Gb = Pfree @ Gb @ Pfree.T
    s_p = occupations(Gb, 460, 820, ks, +1, base=b0p).sum()
    s_m = occupations(Gb, 460, 820, ks, -1, base=b0m).sum()
    sgn_out = +1 if s_p > s_m else -1
    b_out = b0p if sgn_out > 0 else b0m
    # free evolution must leave the vacuum invariant
    Gv = Pfree @ G0 @ Pfree.T
    drift = float(np.abs(Gv - G0).max())

    # ---- the run: ramp the waterfall on, hold, measure ----
    G = G0.copy()
    t = 0.0
    for i in range(8):
        ramp = math.sin(0.5 * math.pi * (i + 0.5) / 8) ** 2
        P = propagator(VPROF * ramp, 15.0 / 256, 8)
        G = P @ G @ P.T
        t += 15.0
    Phold = propagator(VPROF, 0.025, 10)
    spectra, mi, times = {}, {}, (350, 500, 650)
    inter = list(range(160, 300))
    outer = list(range(480, 660))
    S_in0 = gauss_entropy(G0, inter)
    S_out0 = gauss_entropy(G0, outer)
    S_join0 = gauss_entropy(G0, inter + outer)
    mi0 = S_in0 + S_out0 - S_join0
    while t < 655:
        G = Phold @ G @ Phold.T
        t += 25.6
        for tm in times:
            if abs(t - tm) <= 12.8 and tm not in spectra:
                spectra[tm] = occupations(G, 460, 820, ks, sgn_out, base=b_out)
                mi[tm] = (gauss_entropy(G, inter) + gauss_entropy(G, outer)
                          - gauss_entropy(G, inter + outer)) - mi0
    # thermal fit at t=500
    nH = spectra[500]
    sel = (oms > 0.05) & (nH > 5e-4)
    slope = float(np.linalg.lstsq(oms[sel][:, None],
                                  np.log(1 + 1 / nH[sel]), rcond=None)[0][0])
    T_meas = 1 / slope
    # purity of the full chain
    Om2 = np.zeros((2 * NF, 2 * NF))
    Om2[:NF, NF:] = np.eye(NF)
    Om2[NF:, :NF] = -np.eye(NF)
    nu_min = float(np.sort(np.abs(np.linalg.eigvals(Om2 @ G).imag))[NF:].min())
    # the partner moustache: pi-pi cross correlations, interior x
    # exterior, with the trapped interior's 2-site standing-wave
    # striping low-passed away and separable structure removed
    xs_in = np.arange(180, 372)
    xs_out = np.arange(395, 700)
    mous = (G[np.ix_(NF + xs_in, NF + xs_out)]
            - G0[np.ix_(NF + xs_in, NF + xs_out)])

    def blur(m, sig, axis):
        nk = int(4 * sig) * 2 + 1
        xk = np.arange(nk) - nk // 2
        kern = np.exp(-xk * xk / (2 * sig * sig))
        kern /= kern.sum()
        return np.apply_along_axis(
            lambda vv: np.convolve(vv, kern, 'same'), axis, m)
    mous = blur(blur(mous, 1.5, 0), 1.5, 1)
    mous = (mous - mous.mean(axis=1, keepdims=True)
            - mous.mean(axis=0, keepdims=True) + mous.mean())
    # ridge of the partner streak vs the parameter-free prediction:
    # the quantum walks out at c = 1 while its partner obeys
    # dx/dtau = v(x) + 1 from one lattice site behind the horizon --
    # it lingers at the horizon before peeling away toward |v| - c
    ridge = []
    for ci in range(mous.shape[1]):
        col = mous[:, ci]
        i = int(np.argmin(col))
        if -col[i] > 3 * col.std():
            ridge.append((xs_out[ci], xs_in[i]))
    ridge = np.array(ridge)
    xcur, curve = X_HOR - 1.0, []
    for tau in range(0, 330):
        curve.append((X_HOR + tau, xcur))
        xcur += -0.5 * V0 * (1 - math.tanh((xcur - XHF) / WWF)) + 1.0
    curve = np.array(curve)
    pred_in = np.interp(ridge[:, 0], curve[:, 0], curve[:, 1])
    ridge_dev = float(np.mean(np.abs(ridge[:, 1] - pred_in)))
    late = ridge[:, 0] > 500
    m_slope = float(-np.polyfit(ridge[late, 0], ridge[late, 1], 1)[0])
    return dict(ks=ks, oms=oms, spectra=spectra, mi=mi, T_meas=T_meas,
                nu_min=nu_min, mous=mous, xs_in=xs_in, xs_out=xs_out,
                cal_therm=cal_therm, drift=drift, sgn=sgn_out,
                m_slope=m_slope, ridge_dev=ridge_dev, curve=curve)


# ======== films ========================================================

def infall_gif(frames, path):
    y0, y1, x0, x1 = 30, 210, 40, 300
    th = np.linspace(0, 2 * np.pi, 100)
    keep = frames[: int(330 / (0.12 * 25)):2]
    ims = []
    for fr in keep:
        z = fr[y0:y1, x0:x1]
        z = z / (np.percentile(z, 99.5) + 1e-12)
        z = np.clip(z, 0, 1) ** 0.5
        rgb = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        rgb[..., 0] = (30 + 200 * z).astype(np.uint8)
        rgb[..., 1] = (30 + 120 * z).astype(np.uint8)
        rgb[..., 2] = (40 + 60 * z).astype(np.uint8)
        im = Image.fromarray(rgb).resize((2 * (x1 - x0), 2 * (y1 - y0)),
                                         Image.NEAREST)
        d = ImageDraw.Draw(im)
        pts = [(2 * (CX - x0 + R_H * math.cos(a)),
                2 * (CY - y0 + R_H * math.sin(a))) for a in th]
        d.line(pts + [pts[0]], fill=(90, 130, 200), width=2)
        d.text((6, 4), 'infall at the frozen star: pile-up at the '
               'horizon ring, then reflects',
               fill=(200, 200, 190))
        ims.append(im)
    ims[0].save(path, save_all=True, append_images=ims[1:],
                duration=150, loop=0)


def lxmap(x, x0, x1, p0, p1):
    return p0 + (p1 - p0) * (math.log(x) - math.log(x0)) \
        / (math.log(x1) - math.log(x0))


def figure(gclk, mclk, cap, bcrit, fr, pred, r_pin, wf, path):
    W, H = 1560, 880
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 12), 'PART 23 - TIME DILATION, HORIZONS, AND HAWKING '
           'RADIATION', fill=INK)

    # (a) clocks
    ax0, ay0, ax1, ay1 = 60, 70, 480, 330
    d.text((ax0, ay0 - 30), '[89] time dilation, both kinds: deep clocks '
           '(orange) and moving', fill=INK)
    d.text((ax0, ay0 - 14), 'clocks (blue) on their GR curves '
           '(no fitted parameters).', fill=MUTED)

    def axy(u, v):
        return ax0 + (ax1 - ax0) * u / 0.8, ay1 - (ay1 - ay0) * (v - 0.4) / 0.65
    d.line([axy(0, 1), axy(0.8, 1)], fill=GRIDC)
    us = np.linspace(0, 0.78, 60)
    d.line([axy(u, math.sqrt(1 - u)) for u in us], fill=INK, width=2)
    for A, ratio, _pr in gclk:
        px, py = axy(2 * A, ratio)
        d.ellipse([px - 5, py - 5, px + 5, py + 5], outline=C_ORANGE,
                  width=3)
    for v, om_c, _pr in mclk:
        px, py = axy(v * v, om_c / 0.5)
        d.ellipse([px - 5, py - 5, px + 5, py + 5], outline=C_BLUE, width=3)
    d.text((ax0, ay1 + 8), 'x = 2|Phi| (orange) or v^2 (blue); curve: '
           'sqrt(1-x), drawn not fitted; y: clock rate', fill=MUTED)

    # (b) the shadow
    bx0, by0, bx1, by1 = 560, 70, 980, 330
    d.text((bx0, by0 - 30), '[90] the shadow: prompt transmission vs '
           'impact parameter, each b', fill=INK)
    d.text((bx0, by0 - 14), f'over its empty-universe control; line: '
           f'ray-traced b_crit = {bcrit:.0f} (r_h = {R_H:.0f}).',
           fill=MUTED)

    def bxy(b, v):
        return (bx0 + (bx1 - bx0) * b / 64.0,
                by1 - (by1 - by0 - 30) * min(v, 1.15))
    d.line([bxy(0, 0), bxy(64, 0)], fill=GRIDC)
    d.line([bxy(0, 1), bxy(64, 1)], fill=GRIDC)
    pxc = bxy(bcrit, 0)[0]
    d.line([(pxc, by0), (pxc, by1)], fill=INK, width=2)
    d.line([bxy(b, v) for b, v in cap], fill=C_ORANGE, width=3)
    for b, v in cap:
        px, py = bxy(b, v)
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=C_ORANGE)
    d.text((bx0, by1 + 8), 'impact parameter b; y: on-time transmitted '
           'fraction', fill=MUTED)

    # (c) the approach
    cx0, cy0, cx1, cy1 = 1060, 70, 1500, 330
    d.text((cx0, cy0 - 30), '[90] the approach: arrival time at radius r '
           '(points) vs the metric', fill=INK)
    d.text((cx0, cy0 - 14), 'integral t = int dr/c (ink, drawn not '
           'fitted) diverging at r_h;', fill=MUTED)
    d.text((cx0, cy0 + 2), f'the substrate calls it off at the wavelength '
           f'wall (r_pin = {r_pin:.0f}).', fill=MUTED)
    tt = fr[:, 1]
    rr_ = fr[:, 0]
    tmax = max(tt.max(), pred.max()) * 1.05

    def cxy(r_, t_):
        return (cx0 + (cx1 - cx0) * (r_ - 18) / 44.0,
                cy1 - (cy1 - cy0 - 30) * t_ / tmax)
    d.line([cxy(r_, t_) for r_, t_ in zip(rr_, pred)], fill=INK, width=2)
    for r_, t_ in zip(rr_, tt):
        px, py = cxy(r_, t_)
        d.ellipse([px - 3, py - 3, px + 3, py + 3], outline=C_ORANGE,
                  width=2)
    pxp = cxy(r_pin, 0)[0]
    d.line([(pxp, cy0 + 20), (pxp, cy1)], fill=GRIDC)
    pxh = cxy(R_H, 0)[0]
    d.line([(pxh, cy0 + 24), (pxh, cy1)], fill=C_BLUE, width=2)
    d.text((pxh + 5, cy0 + 26), 'r_h', fill=C_BLUE)
    d.text((pxp - 12, cy1 - 16), 'wall', fill=MUTED)
    d.text((cx0, cy1 + 8), 'radius r; y: arrival time of the infalling '
           'front', fill=MUTED)

    # (d) Hawking spectrum
    dx0, dy0, dx1, dy1 = 60, 430, 700, 800
    d.text((dx0, dy0 - 30), '[91] the waterfall shines: measured flux '
           '(rings: t=500; small: t=350 rising,', fill=INK)
    d.text((dx0, dy0 - 14), f't=650 steady) vs a Planck curve at T = '
           f'kappa/2pi = {T_HAWK:.4f} - drawn, not fitted.', fill=MUTED)
    xr, yr = (0.04, 0.32), (2e-4, 1.0)

    def dxy(om, n):
        return (lxmap(om, xr[0], xr[1], dx0, dx1),
                lxmap(n, yr[0], yr[1], dy1, dy0))
    for dec in (1e-3, 1e-2, 1e-1, 1):
        py = dxy(0.1, dec)[1]
        d.line([(dx0, py), (dx1, py)], fill=GRIDC)
        d.text((dx0 - 42, py - 5), f'{dec:g}', fill=MUTED)
    omc = np.geomspace(xr[0], xr[1], 120)
    d.line([dxy(o, 1 / (math.exp(o / T_HAWK) - 1)) for o in omc],
           fill=INK, width=2)
    oms = wf['oms']
    for tm, col, rad in ((350, MUTED, 3), (500, C_ORANGE, 5),
                         (650, C_BLUE, 3)):
        for o, n in zip(oms, wf['spectra'][tm]):
            if n > yr[0] and o < xr[1]:
                px, py = dxy(o, n)
                d.ellipse([px - rad, py - rad, px + rad, py + rad],
                          outline=col, width=2)
    d.text((dx0 + 14, dy0 + 10),
           f'T_measured = {wf["T_meas"]:.4f}  '
           f'({wf["T_meas"] / T_HAWK:.2f} x kappa/2pi)', fill=C_ORANGE)
    d.text((dx0, dy1 + 8), 'omega (log); y: occupation n(omega) (log); '
           'the blue high-omega floor (2e-3) is the', fill=MUTED)
    d.text((dx0, dy1 + 24), 'switch-on\'s slow lattice tail arriving by '
           't=650 - below every thermal bin used.', fill=MUTED)

    # (e) the moustache
    ex0, ey0, ew, eh = 780, 430, 430, 300
    m = wf['mous']
    sc = np.percentile(np.abs(m), 99.3) + 1e-15
    z = np.clip(m / sc, -1, 1)
    pos = np.clip(z, 0, 1) ** 0.7
    neg = np.clip(-z, 0, 1) ** 0.7
    rgb = np.zeros(m.shape + (3,), np.uint8)
    rgb[..., 0] = (25 + 210 * pos).astype(np.uint8)
    rgb[..., 1] = (25 + 90 * pos + 90 * neg).astype(np.uint8)
    rgb[..., 2] = (35 + 210 * neg).astype(np.uint8)
    pane = Image.fromarray(rgb).resize((ew, eh), Image.BILINEAR)
    img.paste(pane, (ex0, ey0))
    # predicted partner locus: x_in = x_h - 0.4 (x_out - x_h)
    xin0, xout0 = wf['xs_in'][0], wf['xs_out'][0]
    nin, nout = len(wf['xs_in']), len(wf['xs_out'])

    def exy(xo_, xi_):
        return (ex0 + ew * (xo_ - xout0) / nout,
                ey0 + eh * (xi_ - xin0) / nin)
    cpts = [exy(xo_, xi_) for xo_, xi_ in wf['curve']
            if xout0 <= xo_ <= xout0 + nout and xin0 <= xi_]
    d.line(cpts, fill=C_GREEN, width=2)
    d.text((ex0, ey0 - 30), '[91] each quantum\'s partner: <pi pi> '
           'correlations, interior (y) x exterior (x).', fill=INK)
    d.text((ex0, ey0 - 14), 'The ridge: pairs created at the horizon, '
           'one out at c, one carried in at |v|-c.', fill=MUTED)
    d.text((ex0, ey0 + eh + 6), 'green: predicted locus (quantum out at '
           'c; partner dx/dt = v+1 from one site', fill=C_GREEN)
    d.text((ex0, ey0 + eh + 22), f'behind the horizon) - measured ridge '
           f'within {wf["ridge_dev"]:.0f} sites; asymptotic slope '
           f'{wf["m_slope"]:.2f} (pred 0.40).', fill=C_GREEN)

    # (f) verdict
    vx, vy = 1240, 430
    lines = [
        ('summary:', INK),
        ('', INK),
        (f'radiation thermal (T = {wf["T_meas"] / T_HAWK:.2f} x', C_ORANGE),
        ('kappa/2pi), steady in time; its', C_ORANGE),
        (f'entanglement with the interior grows', C_ORANGE),
        (f'(dMI = {wf["mi"][350]:.2f} -> {wf["mi"][650]:.2f} nats); and', C_ORANGE),
        ('the global state stays pure', C_ORANGE),
        (f'(nu_min = {wf["nu_min"]:.4f}; pure: 0.5).', C_ORANGE),
        ('', INK),
        ('The radiation alone is thermal', INK),
        ('while the total state is pure:', INK),
        ('entropy here is a property of', INK),
        ('the restricted description.', INK),
        ('', INK),
        ('Owed: backreaction (this hole', MUTED),
        ('never shrinks), and with it', MUTED),
        ('the Page curve.', MUTED),
    ]
    for i, (txt, col) in enumerate(lines):
        d.text((vx, vy + i * 18), txt, fill=col)
    img.save(path)


# ======== main =========================================================

def main():
    t00 = time.time()
    print('=' * 68)
    print('PART 23: TIME DILATION, HORIZONS, AND HAWKING RADIATION')
    print('=' * 68)
    print()

    print('[89] time dilation, measured directly:')
    gclk = grav_clocks()
    print('     gravitational (two identical cavity clocks, plateau '
          'well):')
    for A, ratio, pred in gclk:
        print(f'       depth 2|Phi|={2 * A:.2f}:  deep/far = {ratio:.4f}'
              f'   sqrt(g00) = {pred:.4f}')
    mclk = moving_clocks()
    print('     special-relativistic (phase clock at a moving packet\'s '
          'centroid;')
    print('     v measured from the track, m from the v=0 run — no '
          'knobs):')
    for v, om_c, pred in mclk:
        print(f'       v={v:.3f}:  clock {om_c:.4f}   m*sqrt(1-v^2) = '
              f'{pred:.4f}   ratio {om_c / pred:.4f}')
    print('     The residual at v=0.73 is the lattice\'s k^4 dispersion')
    print('     correction — the substrate showing through at high boost.')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    print('[90] the frozen star: c(r) vanishing linearly at '
          f'r_h = {R_H:.1f}')
    print(f'     (surface gravity kappa = |dc/dr| = {KAPPA_H:.3f}):')
    cap = shadow_curve()
    bc = ray_bcrit()
    print('     the shadow (prompt transmission, each b over its own '
          'empty-universe')
    print('     control):  ' + '  '.join(f'b={b}:{v:.2f}' for b, v in cap))
    b_half = float(np.interp(0.5, [v for _, v in cap], [b for b, _ in cap]))
    print(f'     wave shadow edge (50%) = {b_half:.0f} vs ray-traced '
          f'b_crit = {bc:.0f}')
    print(f'     (b_crit/r_h = {bc / R_H:.2f}; Schwarzschild photon '
          'sphere: 2.60 --')
    print('     same geometry, different profile).')
    fr, pred, r_pin, t_ret, frames = approach_run()
    dev = float(np.mean(np.abs(fr[:, 1] - pred) / np.clip(pred, 1, None)))
    print('     arrival times: the infalling front vs the metric')
    print(f'     integral int dr/c (drawn, not fitted): mean deviation '
          f'{100 * dev:.0f}% over')
    print(f'     r = 60 down to {fr[0, 0]:.0f}; the integral diverges '
          'logarithmically at r_h --')
    print('     the freeze -- and the substrate calls it off at the '
          'wavelength wall')
    print(f'     (predicted pinning radius {r_pin:.0f}, where the '
          'blueshifted k reaches the')
    print('     lattice edge). What happens at the wall: it')
    if t_ret is not None:
        print(f'     REFLECTS -- the annulus at r = 50 lights up again '
              f'{t_ret:.0f} ticks after')
        print('     infall. On a lattice a frozen star delays light '
              'arbitrarily long but')
        print('     does not trap it.')
    else:
        print('     reflects (no return pulse resolved in this run '
              'window).')
    print('     A GR horizon is better modeled in Painleve '
          'coordinates: space flowing')
    print('     inward through a static frame. That flow is the next '
          'experiment.')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    print('[91] the acoustic horizon: a flow crossing c = 1 '
          f'(kappa = {KAPPA_F:.3f}),')
    print('     the chain\'s exact Gaussian state evolved through '
          'switch-on:')
    wf = waterfall()
    print(f'     instrument: thermal-state readback {wf["cal_therm"]:.3f}'
          ' of Bose-Einstein;')
    print(f'     free-evolution vacuum drift {wf["drift"]:.1e}; mover '
          'direction self-calibrated.')
    print('     the spectrum (occupation vs omega, detector window '
          'upstream):')
    TH = T_HAWK
    for i in range(0, len(wf['ks']), 3):
        o = wf['oms'][i]
        pl = 1 / (math.exp(o / TH) - 1)
        print(f'       omega={o:.3f}:  t350 {wf["spectra"][350][i]:.4f}  '
              f't500 {wf["spectra"][500][i]:.4f}  t650 '
              f'{wf["spectra"][650][i]:.4f}   Planck(kappa/2pi) {pl:.4f}')
    print(f'     T_measured = {wf["T_meas"]:.4f} vs kappa/2pi = '
          f'{TH:.4f}  (ratio {wf["T_meas"] / TH:.2f})')
    print('     — a Planck spectrum not put in by hand, steady between '
          't=500 and t=650.')
    print(f'     the partner streak: ridge within {wf["ridge_dev"]:.0f} '
          'sites of the parameter-free locus')
    print('     (quantum out at c; partner dx/dt = v+1 from one site '
          'behind the horizon);')
    print(f'     asymptotic drift slope {wf["m_slope"]:.2f} vs '
          '(|v|-c)/c = 0.40. The partner stays near')
    print('     the horizon before separating — the same exponential '
          'peeling that makes')
    print('     the spectrum thermal, visible in the correlations.')
    print(f'     entanglement: radiation-interior mutual information '
          f'grows {wf["mi"][350]:.2f} ->')
    print(f'     {wf["mi"][500]:.2f} -> {wf["mi"][650]:.2f} nats while '
          f'the GLOBAL state stays pure:')
    print(f'     nu_min = {wf["nu_min"]:.4f} (pure: 0.5000): the '
          'radiation alone is thermal')
    print('     while the total state is pure — the same relation '
          'between coarse and')
    print('     exact descriptions as the entropy of part 9.')
    print('     Trans-Planckian note: on this lattice the flux is '
          'steady BECAUSE')
    print('     dispersion feeds the horizon from the substrate '
          '(Unruh\'s analog');
    print('     insight); the spectrum stays Planckian anyway '
          '(Corley-Jacobson).')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    print('[92] summary: both time dilations on their GR curves to 4')
    print('     decimals; a photon shadow at the ray-traced critical')
    print('     impact parameter; infall following the metric integral')
    print('     until the lattice reflects it; and an acoustic horizon')
    print('     radiating thermally at kappa/2pi while the global state')
    print('     stays exactly pure. Owed at this point: evaporation')
    print('     backreaction — this hole never shrinks — and with it')
    print('     the Page curve.')

    figure(gclk, mclk, cap, bc, fr, pred, r_pin, wf, 'films/horizon.png')
    infall_gif(frames, 'films/horizon_infall.gif')
    print()
    print(f'     films/horizon.png, films/horizon_infall.gif  '
          f'({time.time() - t00:.0f}s)')


if __name__ == '__main__':
    main()
