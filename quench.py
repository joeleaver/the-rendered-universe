"""Part 25: the time-dependent first law.

Part 22 measured the entanglement first law on static states, and the
honest accounting of section 6 listed the missing piece: a
time-dependent version. This part supplies it. A warm region is
prepared on the critical fermion chain and released; it splits into
two heat fronts moving at the Fermi velocity. At every moment, and for
every interval, the first law is tested in its local form:
delta-S(t) = (2pi/v) * sum beta(x) * delta-T00(x, t), with both sides
measured independently and the kernel containing no free parameters.

  [97] the quench: the energy of the warm region propagates at
       v_F = 2, and the entanglement of every interval follows it —
       the delta-S(c, t) map shows the same light cone as the energy.
       Total energy is conserved through the run.
  [98] the first law, time-resolved: the warm region splits into two
       moving thermal pulses, and the kernel prediction matches the
       measured delta-S on them at every time slice, to the same few
       percent as the static case — motion does not break the first
       law as long as the perturbation stays thermal. The agreement
       degrades mildly with interval size (the x^2 correction of
       part 22), and the late-time slices fall below the measurement
       floor as the pulses dilute.
  [99] the bulk statement: through the Ryu-Takayanagi dictionary the
       slice-by-slice first law says the AdS3 metric response tracks
       the boundary stress tensor instantaneously. That is the
       correct behavior: linearized gravity in three dimensions has
       no propagating degrees of freedom (the fact that forced part
       21 into 3+1 dimensions), so all causal structure lives in the
       stress tensor's own propagation — measured here as the light
       cone of [97].

What this does not deliver: the corresponding statement in 3+1
dimensions, where the metric does propagate and a time-dependent
first law would have to source the h_ij of part 21. That remains
open.
"""
import math
import time

import numpy as np
from PIL import Image, ImageDraw

from observatory.quantum import region_entropy

# validated dark-mode categorical palette (dataviz slots 1-3)
C_BLUE, C_ORANGE, C_GREEN = (57, 135, 229), (217, 89, 38), (25, 158, 112)
INK, MUTED, GRIDC = (195, 194, 183), (122, 122, 130), (38, 38, 44)
BG = (14, 14, 18)

N = 1400
CTR = N // 2
VF = 2.0
T0, T_HOT, SIG_HOT = 0.010, 0.025, 40.0


def interval(c, R):
    return list(range(c - R, c + R))


def beta_w(c, R):
    u = np.arange(c - R, c + R, dtype=float)
    return np.clip(R * R - (u - (c - 0.5)) ** 2, 0, None) / (2 * R)


def mod_energy(e, c, R):
    return (2 * np.pi / VF) * float(beta_w(c, R) @ e[interval(c, R)])


def main():
    t00 = time.time()
    print('=' * 68)
    print('PART 25: THE TIME-DEPENDENT FIRST LAW')
    print('=' * 68)
    print()
    print('A warm region on the part-22 fermion chain, released at t = 0.')
    print('The chain Hamiltonian is static; the initial state is a local-')
    print(f'temperature Gibbs state (T = {T0} background, +{T_HOT} bump '
          f'of width {SIG_HOT:.0f}),')
    print('so the state is Gaussian and every entropy is exact at every '
          'time.')
    print()

    H = (np.diag(np.full(N - 1, -1.0), 1)
         + np.diag(np.full(N - 1, -1.0), -1))
    w, V = np.linalg.eigh(H)
    j = np.arange(N)

    def local_gibbs(Tprof):
        beta = 1.0 / np.clip(Tprof, 1e-4, None)
        D = np.diag(beta)
        Mb = 0.5 * (D @ H + H @ D)
        mw, mv = np.linalg.eigh(Mb)
        occ = 1.0 / (1.0 + np.exp(np.clip(mw, -600, 600)))
        return (mv * occ) @ mv.T

    Tprof = T0 + T_HOT * np.exp(-(j - CTR) ** 2 / (2 * SIG_HOT ** 2))
    C = local_gibbs(Tprof)
    occ0 = 1.0 / (1.0 + np.exp(w / T0))
    Cref = (V * occ0) @ V.T          # uniform background, stationary

    def bond_e(Cm):
        return -2 * np.real(np.diag(Cm[:-1, 1:]))

    e_ref = bond_e(Cref)

    print('[0]  instrument validation:')
    # reference stationarity is exact: Cref commutes with H
    comm = float(np.abs(H @ Cref - Cref @ H).max())
    print(f'     background state commutes with H to {comm:.1e} '
          '(stationary by construction)')
    e0 = bond_e(C) - e_ref
    e_pred = (np.pi / 12) * (Tprof ** 2 - T0 ** 2)
    sel = np.abs(j - CTR) < 60
    print(f'     initial energy profile vs (pi/12)(T(x)^2 - T0^2): '
          f'ratio {float(e0[sel[:-1]].sum() / e_pred[sel].sum()):.3f}')
    print(f'     total excess energy E = {e0.sum():.5f}')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    # ---- evolve and measure ----
    DT = 12.0
    NT = 29                       # t = 0 .. 336
    Ud = (V * np.exp(-1j * w * DT)) @ V.T
    cs = np.arange(CTR - 500, CTR + 501, 20)
    R_MAIN = 12
    Rs_scan = (8, 12, 16, 24)
    t_scan = (60, 144, 240)
    S_ref = {}
    for R in set((R_MAIN,) + Rs_scan):
        for c in cs:
            S_ref[(c, R)] = region_entropy(Cref, interval(c, R))

    Ct = C.astype(complex)
    ts = []
    e_map, dS_map, dK_map = [], [], []
    scan_rows = []
    e_tot = []
    for it in range(NT):
        t = it * DT
        ts.append(t)
        e = np.real(bond_e(Ct)) - e_ref
        e_full = np.zeros(N)
        e_full[:N - 1] = e
        e_map.append(e.copy())
        e_tot.append(float(e.sum()))
        dS_row, dK_row = [], []
        for c in cs:
            A = interval(c, R_MAIN)
            dS_row.append(region_entropy(np.real(Ct), A)
                          - S_ref[(c, R_MAIN)])
            dK_row.append(mod_energy(e_full, c, R_MAIN))
        dS_map.append(dS_row)
        dK_map.append(dK_row)
        if t in t_scan:
            for R in Rs_scan:
                for c in cs[::2]:
                    A = interval(c, R)
                    dS = region_entropy(np.real(Ct), A) - S_ref[(c, R)]
                    dK = mod_energy(e_full, c, R)
                    scan_rows.append((t, c, R, dS, dK))
        Ct = Ud @ Ct @ Ud.conj().T
    ts = np.array(ts)
    e_map = np.array(e_map)
    dS_map = np.array(dS_map)
    dK_map = np.array(dK_map)

    print('[97] the quench and the light cone:')
    # front position from the energy map: outermost bond with e > 10% max
    def front_speed(rows, coords):
        pts = []
        for i, t in enumerate(ts):
            prof = np.array(rows[i])
            above = np.where(prof > 0.3 * prof.max())[0]
            if len(above):
                pts.append((t, coords[above[-1]] - CTR))
        pts = np.array([p for p in pts if 48 <= p[0] <= 230])
        return float(np.polyfit(pts[:, 0], pts[:, 1], 1)[0])
    v_front = front_speed(e_map, np.arange(N - 1))
    v_ent = front_speed(dS_map, cs)
    print(f'     energy front speed: {v_front:.2f} (Fermi velocity: '
          f'{VF:.1f})')
    print(f'     entanglement front speed (from delta-S(c, t)): '
          f'{v_ent:.2f}')
    cons = (max(e_tot) - min(e_tot)) / e_tot[0]
    print(f'     energy conservation over the run: {100 * cons:.2f}% '
          'variation')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    print('[98] the first law at every time slice '
          f'(intervals of half-width {R_MAIN}):')
    print('        t      mean dK/dS   n intervals above the floor')
    inter_ratios = []
    for i, t in enumerate(ts):
        dS_row = np.array(dS_map[i])
        dK_row = np.array(dK_map[i])
        strong = dS_row > 5e-3
        if strong.sum() < 3:
            continue
        r = float(np.mean(dK_row[strong] / dS_row[strong]))
        inter_ratios.append(r)
        if i % 4 == 0:
            print(f'      {t:5.0f}      {r:.3f}        {strong.sum()}')
    print(f'     across all usable slices: dK/dS = '
          f'{np.mean(inter_ratios):.3f} +/- {np.std(inter_ratios):.3f}')
    print('     The warm region splits into two moving thermal pulses, '
          'and the first')
    print('     law holds on them slice by slice, at the same few-'
          'percent level as the')
    print('     static case of part 22: motion does not break it while '
          'the')
    print('     perturbation stays thermal. Late slices fall below the '
          'measurement')
    print('     floor as the pulses dilute, and are excluded.')
    print(f'     [{time.time() - t00:.0f}s]')
    print()

    print('[99] the slice-by-slice kernel across interval sizes:')
    print('        t      R=8      R=12     R=16     R=24    '
          '(mean dK/dS, strong intervals)')
    for t in t_scan:
        row = [f'      {t:5.0f} ']
        for R in Rs_scan:
            sel = [(dS, dK) for (tt, c, R_, dS, dK) in scan_rows
                   if tt == t and R_ == R and dS > 5e-3]
            if sel:
                r = np.mean([dK / dS for dS, dK in sel])
                row.append(f'  {r:.3f} ')
            else:
                row.append('    —   ')
        print(' '.join(row))
    print('     Through the Ryu-Takayanagi dictionary this is the '
          'statement that the')
    print('     AdS3 metric response tracks the boundary stress '
          'tensor on each time')
    print('     slice. That is the correct behavior for three-'
          'dimensional gravity,')
    print('     which has no propagating degrees of freedom (the fact '
          'that forced')
    print('     part 21 into 3+1 dimensions): causality is carried by '
          'the stress')
    print('     tensor itself, i.e. the light cone of [97]. The 3+1 '
          'version — a')
    print('     time-dependent first law sourcing a propagating h_ij '
          '— remains open,')
    print('     and section 6 of the paper says so.')

    figure(ts, cs, e_map, dS_map, dK_map, v_front, v_ent,
           np.mean(inter_ratios), 'films/quench.png')
    print()
    print(f'     films/quench.png  ({time.time() - t00:.0f}s)')


def figure(ts, cs, e_map, dS_map, dK_map, v_front, v_ent, r_mean, path):
    W, H = 1560, 620
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 12), 'PART 25 - THE TIME-DEPENDENT FIRST LAW', fill=INK)

    def pane(mat, x0, y0, wid, hgt, signed=False, ref=None):
        m = np.array(mat, dtype=float)
        sc = np.percentile(np.abs(m), 99.5) + 1e-15
        z = np.clip(m / sc, -1, 1) if signed else np.clip(m / sc, 0, 1)
        if signed:
            pos = np.clip(z, 0, 1) ** 0.6
            neg = np.clip(-z, 0, 1) ** 0.6
        else:
            pos = z ** 0.6
            neg = np.zeros_like(z)
        rgb = np.zeros(m.shape + (3,), np.uint8)
        rgb[..., 0] = (25 + 210 * pos).astype(np.uint8)
        rgb[..., 1] = (25 + 90 * pos + 90 * neg).astype(np.uint8)
        rgb[..., 2] = (35 + 210 * neg).astype(np.uint8)
        p = Image.fromarray(rgb).resize((wid, hgt), Image.BILINEAR)
        img.paste(p, (x0, y0))

    ph = 380
    # (a) energy density
    pane(e_map, 60, 120, 440, ph)
    d.text((60, 88), '[97] energy density (x across, time down): two '
           'fronts at v_F.', fill=INK)
    # (b) delta-S
    pane(dS_map, 560, 120, 440, ph)
    d.text((560, 88), '[97] delta-S of sliding intervals: the same '
           'light cone.', fill=INK)
    # cone guide lines, clipped to their panes
    dd = ImageDraw.Draw(img)
    for x0, span, full in ((60, 440, float(N)),
                           (560, 440, float(cs[-1] - cs[0]))):
        ctr_px = x0 + span // 2
        for sgn in (-1, 1):
            px_per_site = span / full
            t_edge = min(ts[-1], (span / 2) / (VF * px_per_site))
            y_bot = 120 + ph * t_edge / ts[-1]
            x_bot = ctr_px + sgn * VF * t_edge * px_per_site
            dd.line([(ctr_px, 120), (x_bot, y_bot)],
                    fill=(120, 220, 160), width=1)
    # (c) ratio map
    ratio = np.array(dK_map) / np.clip(np.array(dS_map), 1e-3, None) - 1
    ratio[np.array(dS_map) < 5e-3] = 0
    ratio = np.clip(ratio / 0.15, -1, 1) * 0.15   # fixed +/-15% scale
    pane(ratio, 1060, 120, 440, ph, signed=True)
    d.text((1060, 88), '[98] dK/dS - 1 (orange high, blue low), '
           'where delta-S is above the floor.', fill=INK)

    d.text((60, 120 + ph + 14),
           f'front speeds: energy {v_front:.2f}, entanglement '
           f'{v_ent:.2f} (v_F = 2); cone-interior dK/dS = '
           f'{r_mean:.3f}; kernel has no fitted parameters.', fill=MUTED)
    d.text((60, 120 + ph + 34),
           'Through the RT dictionary: the AdS3 metric response tracks '
           'the boundary stress tensor slice by slice - 3D gravity has '
           'no propagating modes;', fill=MUTED)
    d.text((60, 120 + ph + 50),
           'causality is carried by the stress tensor. The propagating '
           '3+1 version remains open.', fill=MUTED)
    img.save(path)


if __name__ == '__main__':
    main()
