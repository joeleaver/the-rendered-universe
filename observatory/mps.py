"""A matrix-product-state engine: MPO compiler and two-site DMRG.

numpy only, in the repository's tradition. Hamiltonians are supplied
as sums of operator strings; the MPO compiler assembles them by
batched direct-sum and exact SVD compression, so long-range terms
(Coulomb tails, tangent-fermion hopping) cost nothing special — the
compressor discovers the efficient representation by itself. Ground
states by two-site DMRG with a hand-rolled Lanczos; excited states by
re-running with an overlap penalty against the states already found.

Index conventions:
  MPS tensor  m[Dl, s, Dr]
  MPO tensor  w[Wl, Wr, s_bra, s_ket]
  left env    E[ket_bond, mpo_bond, bra_bond]
  right env   R[ket_bond, mpo_bond, bra_bond]

Validated at import on a 6-site transverse-field Ising chain (ground
AND first excited energy vs dense diagonalization).
"""
import numpy as np

I2 = np.eye(2)
CR = np.array([[0.0, 0.0], [1.0, 0.0]])   # c+ : |0> -> |1>
AN = np.array([[0.0, 1.0], [0.0, 0.0]])   # c  : |1> -> |0>
NUM = np.array([[0.0, 0.0], [0.0, 1.0]])
ZF = np.array([[1.0, 0.0], [0.0, -1.0]])  # (-1)^n
X = np.array([[0.0, 1.0], [1.0, 0.0]])
OPS = {'I': I2, 'c+': CR, 'c': AN, 'n': NUM, 'Z': ZF, 'X': X}


# ---- MPO compiler ------------------------------------------------------


def term_mpo(L, term, coef):
    """Bond-dimension-1 MPO for coef * prod_j op_j.
    term: dict {site: opname or matrix}; sites absent get identity."""
    ws = []
    for j in range(L):
        o = term.get(j, 'I')
        op = (OPS[o] if isinstance(o, str) else o).astype(complex)
        if j == 0:
            op = op * coef
        ws.append(op.reshape(1, 1, 2, 2).copy())
    return ws


def mpo_add(A, B):
    L = len(A)
    out = []
    for j in range(L):
        a, b = A[j], B[j]
        al, ar = a.shape[0], a.shape[1]
        bl, br = b.shape[0], b.shape[1]
        Dl = 1 if j == 0 else al + bl
        Dr = 1 if j == L - 1 else ar + br
        w = np.zeros((Dl, Dr, 2, 2), dtype=complex)
        if L == 1:
            w[0, 0] = a[0, 0] + b[0, 0]
        elif j == 0:
            w[0, :ar] = a[0]
            w[0, ar:] = b[0]
        elif j == L - 1:
            w[:al, 0] = a[:, 0]
            w[al:, 0] = b[:, 0]
        else:
            w[:al, :ar] = a
            w[al:, ar:] = b
        out.append(w)
    return out


def mpo_compress(W, tol=1e-11):
    L = len(W)
    W = [w.copy() for w in W]
    for j in range(L - 1):
        Dl, Dr = W[j].shape[0], W[j].shape[1]
        m = W[j].transpose(0, 2, 3, 1).reshape(Dl * 4, Dr)
        u, s, vt = np.linalg.svd(m, full_matrices=False)
        r = max(1, int((s > tol * max(s[0], 1e-30)).sum()))
        W[j] = u[:, :r].reshape(Dl, 2, 2, r).transpose(0, 3, 1, 2)
        sv = (s[:r, None] * vt[:r])
        W[j + 1] = np.einsum('ij,jklm->iklm', sv, W[j + 1])
    for j in range(L - 1, 0, -1):
        Dl, Dr = W[j].shape[0], W[j].shape[1]
        m = W[j].reshape(Dl, Dr * 4)
        u, s, vt = np.linalg.svd(m, full_matrices=False)
        r = max(1, int((s > tol * max(s[0], 1e-30)).sum()))
        W[j] = vt[:r].reshape(r, Dr, 2, 2)
        us = u[:, :r] * s[None, :r]
        W[j - 1] = np.einsum('iklm,kj->ijlm', W[j - 1], us)
    return W


def build_mpo(L, terms, batch=48, tol=1e-11):
    total = None
    for start in range(0, len(terms), batch):
        chunk = terms[start:start + batch]
        acc = term_mpo(L, chunk[0][1], chunk[0][0])
        for coef, t in chunk[1:]:
            acc = mpo_add(acc, term_mpo(L, t, coef))
        total = acc if total is None else mpo_add(total, acc)
        total = mpo_compress(total, tol)
    return total


def mpo_bond_max(W):
    return max(w.shape[0] for w in W)


# ---- MPS basics --------------------------------------------------------


def random_mps(L, chi, rng):
    ms = []
    for j in range(L):
        Dl = int(min(chi, 2 ** j, 2 ** (L - j)))
        Dr = int(min(chi, 2 ** (j + 1), 2 ** (L - j - 1)))
        ms.append(rng.normal(size=(Dl, 2, Dr)).astype(complex))
    return ms


def product_mps(bits, noise=1e-3, rng=None):
    """Bond-dimension-2 MPS near the product state |bits>, with a
    little noise so two-site DMRG can grow out of it."""
    if rng is None:
        rng = np.random.default_rng(0)
    L = len(bits)
    ms = []
    for j, b in enumerate(bits):
        Dl = 1 if j == 0 else 2
        Dr = 1 if j == L - 1 else 2
        m = noise * rng.normal(size=(Dl, 2, Dr)).astype(complex)
        m[0, b, 0] += 1.0
        ms.append(m)
    return ms


def right_canonicalize(ms):
    L = len(ms)
    for j in range(L - 1, 0, -1):
        Dl, d, Dr = ms[j].shape
        u, s, vt = np.linalg.svd(ms[j].reshape(Dl, d * Dr),
                                 full_matrices=False)
        ms[j] = vt.reshape(-1, d, Dr)
        ms[j - 1] = np.einsum('lda,ak->ldk', ms[j - 1],
                              u * s[None, :])
    ms[0] = ms[0] / np.linalg.norm(ms[0])
    return ms


def envL_step(E, m, w, mc):
    t = np.einsum('awb,asc->wbsc', E, m, optimize=True)
    t = np.einsum('wbsc,wvts->bcvt', t, w, optimize=True)
    return np.einsum('bcvt,btd->cvd', t, np.conj(mc), optimize=True)


def envR_step(R, m, w, mc):
    t = np.einsum('asc,cvd->asvd', m, R, optimize=True)
    t = np.einsum('asvd,wvts->awtd', t, w, optimize=True)
    return np.einsum('awtd,btd->awb', t, np.conj(mc), optimize=True)


def overlap(ms, ps):
    """<p|m>."""
    e = np.ones((1, 1), dtype=complex)
    for m, p in zip(ms, ps):
        e = np.einsum('ab,asc,bsd->cd', e, m, np.conj(p),
                      optimize=True)
    return complex(e[0, 0])


def expect_mpo(ms, W):
    E = np.ones((1, 1, 1), dtype=complex)
    for m, w in zip(ms, W):
        E = envL_step(E, m, w, m)
    return float(np.real(E[0, 0, 0]))


# ---- DMRG --------------------------------------------------------------


def _lanczos(matvec, v0, iters=30):
    V = [v0 / np.linalg.norm(v0)]
    al, be = [], []
    e_prev = None
    for it in range(iters):
        w = matvec(V[-1])
        a = float(np.real(np.vdot(V[-1], w)))
        al.append(a)
        w = w - a * V[-1] - (be[-1] * V[-2] if be else 0)
        for u in V:
            w = w - np.vdot(u, w) * u
        b = float(np.linalg.norm(w))
        if b < 1e-13:
            break
        if it >= 5 and it % 2 == 1:
            T = np.diag(al) + np.diag(be, 1) + np.diag(be, -1)
            e_now = float(np.linalg.eigvalsh(T)[0])
            if e_prev is not None and abs(e_now - e_prev) < 1e-11 * (
                    1 + abs(e_now)):
                break
            e_prev = e_now
        be.append(b)
        V.append(w / b)
    T = np.diag(al) + np.diag(be[:len(al) - 1], 1) \
        + np.diag(be[:len(al) - 1], -1)
    ev, evec = np.linalg.eigh(T)
    vmin = np.zeros_like(v0)
    for i, u in enumerate(V[:len(al)]):
        vmin = vmin + evec[i, 0] * u
    return float(ev[0]), vmin / np.linalg.norm(vmin)


def dmrg(W, L, chi, sweeps=10, rng=None, penalty=(), pw=25.0,
         tol=1e-9, lanc=30, verbose=False, init=None):
    """Two-site DMRG. Returns (energy, mps). penalty: MPS list to
    orthogonalize against via +pw|p><p|. init: starting MPS."""
    if rng is None:
        rng = np.random.default_rng(0)
    ms = right_canonicalize([m.copy() for m in init] if init
                            else random_mps(L, chi, rng))
    lenv = [None] * (L + 1)
    renv = [None] * (L + 1)
    lenv[0] = np.ones((1, 1, 1), dtype=complex)
    renv[L] = np.ones((1, 1, 1), dtype=complex)
    for j in range(L - 1, -1, -1):
        renv[j] = envR_step(renv[j + 1], ms[j], W[j], ms[j])
    pen = []
    for p in penalty:
        pL = [None] * (L + 1)
        pR = [None] * (L + 1)
        pL[0] = np.ones((1, 1), dtype=complex)
        pR[L] = np.ones((1, 1), dtype=complex)
        for j in range(L - 1, -1, -1):
            # pR[j][ket_bond, p_bond] over sites j..L-1
            pR[j] = np.einsum('asc,bsd,cd->ab', ms[j],
                              np.conj(p[j]), pR[j + 1], optimize=True)
        pen.append({'mps': p, 'L': pL, 'R': pR})
    e = None
    e_last = None
    for sw in range(sweeps):
        for going_right in (True, False):
            sites = range(L - 1) if going_right else \
                range(L - 2, -1, -1)
            for j in sites:
                Dl = ms[j].shape[0]
                Dr = ms[j + 1].shape[2]
                Le, Re = lenv[j], renv[j + 2]
                w1, w2 = W[j], W[j + 1]
                # precontract per bond: LW[a,b,v,x,s], RW[e,d,v,y,t]
                LW = np.tensordot(Le, w1, axes=([1], [0]))
                RW = np.tensordot(Re, w2, axes=([1], [1]))
                pvs = []
                for p in pen:
                    blk = np.einsum('ab,bsc->asc', p['L'][j],
                                    p['mps'][j], optimize=True)
                    blk = np.einsum('asc,ctd->astd', blk,
                                    p['mps'][j + 1], optimize=True)
                    blk = np.einsum('astd,ed->aste', blk,
                                    p['R'][j + 2], optimize=True)
                    # indices: a = psi left bond, e = psi right bond
                    pvs.append(blk.reshape(-1))
                shape = (Dl, 2, 2, Dr)

                def matvec(v):
                    t = v.reshape(shape)
                    # t[a,s,t,e] x LW[a,b,v,x,s] over (a,s)
                    #   -> (t,e,b,v,x)
                    r = np.tensordot(t, LW, axes=([0, 1], [0, 4]))
                    # r[t,e,b,v,x] x RW[e,d,v,y,t] over (t,e,v)
                    #   -> (b,x,d,y)
                    r = np.tensordot(r, RW, axes=([0, 1, 3],
                                                  [4, 0, 2]))
                    out = r.transpose(0, 1, 3, 2).reshape(-1)
                    for pv in pvs:
                        out = out + pw * np.conj(pv) * np.vdot(
                            np.conj(pv), v)
                    return out

                v0 = np.einsum('asb,btc->astc', ms[j],
                               ms[j + 1]).reshape(-1)
                nv = np.linalg.norm(v0)
                if nv < 1e-12:
                    v0 = rng.normal(size=v0.shape) + 0j
                e, vmin = _lanczos(matvec, v0, lanc)
                t = vmin.reshape(Dl * 2, 2 * Dr)
                u, s, vt = np.linalg.svd(t, full_matrices=False)
                keep = int(min(chi, max(1, (s > 1e-10).sum())))
                u, s, vt = u[:, :keep], s[:keep], vt[:keep]
                s = s / np.linalg.norm(s)
                if going_right:
                    ms[j] = u.reshape(Dl, 2, keep)
                    ms[j + 1] = (s[:, None] * vt).reshape(keep, 2, Dr)
                    lenv[j + 1] = envL_step(lenv[j], ms[j], W[j],
                                            ms[j])
                    for p in pen:
                        p['L'][j + 1] = np.einsum(
                            'ab,asc,bsd->cd', p['L'][j], ms[j],
                            np.conj(p['mps'][j]), optimize=True)
                else:
                    ms[j] = (u * s[None, :]).reshape(Dl, 2, keep)
                    ms[j + 1] = vt.reshape(keep, 2, Dr)
                    renv[j + 1] = envR_step(renv[j + 2], ms[j + 1],
                                            W[j + 1], ms[j + 1])
                    for p in pen:
                        p['R'][j + 1] = np.einsum(
                            'asc,bsd,cd->ab', ms[j + 1],
                            np.conj(p['mps'][j + 1]), p['R'][j + 2],
                            optimize=True)
        if verbose:
            print(f'      sweep {sw}: E = {e:.10f}')
        if e_last is not None and abs(e - e_last) < tol:
            break
        e_last = e
    return e, ms


# ---- fermion helpers ---------------------------------------------------


def jw_term(coef, ops):
    """Fermionic term -> spin term with Jordan-Wigner strings.
    ops: list of (jw_site, 'c+'|'c'), applied left to right as
    written (leftmost operator acts last). Returns (coef, {site: mat})
    with Z-strings included. Sites must be distinct."""
    # normal-order bookkeeping: build operator as product over sites.
    # For each fermion op at site m: string Z_0..Z_{m-1} (x) op_m.
    # Multiply matrices site by site in the given operator order.
    L_needed = max(m for m, _ in ops) + 1
    mats = {}
    sign = 1.0
    for (m, kind) in ops:
        # apply Z string to sites < m
        for k in range(m):
            mats[k] = (mats.get(k, I2) @ ZF)
        base = CR if kind == 'c+' else AN
        mats[m] = (mats.get(m, I2) @ base)
    out = {}
    for k, mat in mats.items():
        if np.allclose(mat, I2):
            continue
        out[k] = mat
    return (coef * sign, out)


# ---- import-time validation -------------------------------------------


def _validate():
    L, h = 6, 1.0
    terms = [(-1.0, {j: 'X', j + 1: 'X'}) for j in range(L - 1)]
    terms += [(-h, {j: 'Z'}) for j in range(L)]
    W = build_mpo(L, terms)
    Hd = np.zeros((2 ** L, 2 ** L), dtype=complex)
    for coef, t in terms:
        op = np.array([[1.0]], dtype=complex) * coef
        for j in range(L):
            o = t.get(j, 'I')
            op = np.kron(op, OPS[o] if isinstance(o, str) else o)
        Hd += op
    w_ex = np.linalg.eigvalsh(Hd)
    e0, ms0 = dmrg(W, L, chi=16, sweeps=8)
    assert abs(e0 - w_ex[0]) < 1e-8, ('mps validate E0', e0, w_ex[0])
    e1, _ = dmrg(W, L, chi=16, sweeps=10, penalty=[ms0], pw=20.0,
                 rng=np.random.default_rng(5))
    assert abs(e1 - w_ex[1]) < 1e-6, ('mps validate E1', e1, w_ex[1])


_validate()
