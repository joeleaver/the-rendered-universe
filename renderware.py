"""Part 15: the render-aware simulator.

Thesis, in one line: the classical cost of a quantum computation is
the size of its smallest ENGINE, not its qubit count.

The simulator keeps the global state as a product of independent
blocks and gives each block the cheapest faithful representation it
can get away with:

  - stabilizer tableau (Aaronson-Gottesman CHP) while the block's
    history is Clifford: polynomial cost, even at maximal
    entanglement;
  - raw statevector (2^n amplitudes) the moment a non-Clifford gate
    ("magic") arrives: exponential cost, paid only by the block that
    earned it.

Blocks fuse when a gate entangles them (exponents multiply); a
measured qubit factors out of its block (exponents shrink). The
engine-size meter runs live. Three consequences, measured:

  [54] the fusion cascade — many small quantum computers cost the
       SUM of their exponentials, not the exponential of their sum;
       only entangling them fuses the ledgers. (This is why a world
       full of small quantum devices is cheap for a bounded classical
       engine, and one large fault-tolerant entangled block is not.)
  [55] entanglement is cheap; magic is expensive — a maximally
       entangled GHZ state costs kilobytes in the tableau; one T gate
       forces the amplitudes to materialize.
  [56] decoherence is the engine's garbage collector — measuring
       qubits splits them out of the block and the ledger halves,
       step by step.

Validated at import: tableau simulation must agree with direct
statevector simulation on random Clifford circuits, or we refuse
to run.
"""
import math

import numpy as np

VEC_LIMIT_BYTES = 512 * 1024 * 1024


# ------------------------------------------------ statevector blocks

H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * math.pi / 4)])
GATES1 = {'H': H, 'S': S, 'T': T}


def vec_apply1(psi, U, q, n):
    psi = psi.reshape(2 ** (n - q - 1), 2, 2 ** q)
    return np.einsum('ij,ajb->aib', U, psi).reshape(-1)


def vec_cnot(psi, c, t, n):
    idx = np.arange(len(psi))
    sel = ((idx >> c) & 1) == 1
    out = psi.copy()
    out[idx[sel]] = psi[idx[sel] ^ (1 << t)]
    return out


# ------------------------------------------------ CHP tableau

class Tableau:
    """Aaronson-Gottesman stabilizer simulation."""

    def __init__(self, n):
        self.n = n
        self.x = np.zeros((2 * n, n), dtype=np.uint8)
        self.z = np.zeros((2 * n, n), dtype=np.uint8)
        self.r = np.zeros(2 * n, dtype=np.uint8)
        for i in range(n):
            self.x[i, i] = 1          # destabilizers X_i
            self.z[n + i, i] = 1      # stabilizers  Z_i

    def h(self, a):
        self.r ^= self.x[:, a] & self.z[:, a]
        self.x[:, a], self.z[:, a] = (self.z[:, a].copy(),
                                      self.x[:, a].copy())

    def s(self, a):
        self.r ^= self.x[:, a] & self.z[:, a]
        self.z[:, a] ^= self.x[:, a]

    def cnot(self, a, b):
        self.r ^= self.x[:, a] & self.z[:, b] & (self.x[:, b]
                                                 ^ self.z[:, a] ^ 1)
        self.x[:, b] ^= self.x[:, a]
        self.z[:, a] ^= self.z[:, b]

    def _rowsum(self, h, i):
        x1, z1 = self.x[i].astype(int), self.z[i].astype(int)
        x2, z2 = self.x[h].astype(int), self.z[h].astype(int)
        g = np.where((x1 == 1) & (z1 == 1), z2 - x2, 0)
        g += np.where((x1 == 1) & (z1 == 0), z2 * (2 * x2 - 1), 0)
        g += np.where((x1 == 0) & (z1 == 1), x2 * (1 - 2 * z2), 0)
        tot = (2 * self.r[h] + 2 * self.r[i] + g.sum()) % 4
        self.r[h] = tot // 2
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]

    def measure(self, a, rng):
        n = self.n
        ps = [p for p in range(n, 2 * n) if self.x[p, a]]
        if ps:
            p = ps[0]
            for i in range(2 * n):
                if i != p and self.x[i, a]:
                    self._rowsum(i, p)
            self.x[p - n] = self.x[p].copy()
            self.z[p - n] = self.z[p].copy()
            self.r[p - n] = self.r[p]
            self.x[p] = 0
            self.z[p] = 0
            self.z[p, a] = 1
            self.r[p] = k = int(rng.integers(2))
            return k
        # deterministic outcome
        sx = np.zeros(self.n, dtype=np.uint8)
        sz = np.zeros(self.n, dtype=np.uint8)
        sr = 0
        for i in range(n):
            if self.x[i, a]:
                # rowsum(scratch, i+n): source row is the stabilizer
                x1 = self.x[i + n].astype(int)
                z1 = self.z[i + n].astype(int)
                x2, z2 = sx.astype(int), sz.astype(int)
                g = np.where((x1 == 1) & (z1 == 1), z2 - x2, 0)
                g += np.where((x1 == 1) & (z1 == 0), z2 * (2 * x2 - 1), 0)
                g += np.where((x1 == 0) & (z1 == 1), x2 * (1 - 2 * z2), 0)
                # sr already holds the doubled (mod-4) phase: add, don't re-double
                sr = (sr + 2 * int(self.r[i + n]) + int(g.sum())) % 4
                sx ^= self.x[i + n]
                sz ^= self.z[i + n]
        return int(sr // 2)

    def nbytes(self):
        return (2 * self.n * self.n * 2 + 2 * self.n) // 8 + 1


# ------------------------------------------------ the engine

class Block:
    def __init__(self, qubits):
        self.qubits = list(qubits)      # global ids, order = local index
        self.log = []                   # gate history for replay
        self.tab = Tableau(len(qubits))
        self.vec = None                 # set when magic materializes

    def local(self, q):
        return self.qubits.index(q)

    def is_vec(self):
        return self.vec is not None

    def materialize(self):
        n = len(self.qubits)
        need = 16 * 2 ** n
        if need > VEC_LIMIT_BYTES:
            raise MemoryError(
                f'block of {n} qubits needs {need / 1e9:.1f} GB')
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1.0
        for (g, qs) in self.log:
            if g == 'CNOT':
                psi = vec_cnot(psi, qs[0], qs[1], n)
            else:
                psi = vec_apply1(psi, GATES1[g], qs[0], n)
        self.vec = psi
        self.tab = None

    def nbytes(self):
        if self.is_vec():
            return 16 * 2 ** len(self.qubits)
        return self.tab.nbytes()


class Engine:
    def __init__(self, n_qubits, seed=0):
        self.rng = np.random.default_rng(seed)
        self.blocks = [Block([q]) for q in range(n_qubits)]
        self.owner = {q: self.blocks[q] for q in range(n_qubits)}
        self.classical = {}

    def _merge(self, ba, bb):
        if ba is bb:
            return ba
        merged = Block(ba.qubits + bb.qubits)
        off = len(ba.qubits)
        merged.log = list(ba.log) + [
            (g, tuple(q + off for q in qs)) for (g, qs) in bb.log]
        if ba.is_vec() or bb.is_vec():
            merged.materialize()
        else:
            merged.tab = Tableau(len(merged.qubits))
            for (g, qs) in merged.log:
                self._tab_apply(merged.tab, g, qs)
        self.blocks.remove(ba)
        self.blocks.remove(bb)
        self.blocks.append(merged)
        for q in merged.qubits:
            self.owner[q] = merged
        return merged

    @staticmethod
    def _tab_apply(tab, g, qs):
        if g == 'H':
            tab.h(qs[0])
        elif g == 'S':
            tab.s(qs[0])
        elif g == 'CNOT':
            tab.cnot(qs[0], qs[1])
        else:
            raise ValueError(g)

    def gate(self, g, *qubits):
        if g == 'CNOT':
            b = self._merge(self.owner[qubits[0]], self.owner[qubits[1]])
        else:
            b = self.owner[qubits[0]]
        loc = tuple(b.local(q) for q in qubits)
        b.log.append((g, loc))
        if g == 'T' and not b.is_vec():
            b.materialize()
        if b.is_vec():
            n = len(b.qubits)
            if g == 'CNOT':
                b.vec = vec_cnot(b.vec, loc[0], loc[1], n)
            else:
                b.vec = vec_apply1(b.vec, GATES1[g], loc[0], n)
        else:
            self._tab_apply(b.tab, g, loc)

    def measure(self, q):
        b = self.owner[q]
        if not b.is_vec():
            k = b.tab.measure(b.local(q), self.rng)
            b.log = None  # tableau state is no longer replayable
            self.classical[q] = k
            return k
        n = len(b.qubits)
        a = b.local(q)
        psi = b.vec.reshape(2 ** (n - a - 1), 2, 2 ** a)
        p1 = float((np.abs(psi[:, 1, :]) ** 2).sum())
        k = int(self.rng.random() < p1)
        keep = psi[:, k, :].reshape(-1)
        keep /= np.linalg.norm(keep)
        self.classical[q] = k
        # the measured qubit factors out of the block
        b.qubits.pop(a)
        if b.qubits:
            b.vec = keep
            b.log = None
        else:
            self.blocks.remove(b)
        del self.owner[q]
        return k

    def nbytes(self):
        return sum(b.nbytes() for b in self.blocks)

    def largest(self):
        return max((len(b.qubits) for b in self.blocks), default=0)


def fmt(nb):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB', 'PB'):
        if nb < 1024:
            return f'{nb:.0f} {unit}'
        nb /= 1024
    return f'{nb:.1e} EB'


# ------------------------------------------------ import-time validation

def _validate():
    rng = np.random.default_rng(5)
    for trial in range(30):
        n = 4
        ops = []
        for _ in range(25):
            g = rng.choice(['H', 'S', 'CNOT'])
            if g == 'CNOT':
                a, b = rng.choice(n, 2, replace=False)
                ops.append(('CNOT', (int(a), int(b))))
            else:
                ops.append((g, (int(rng.integers(n)),)))
        # tableau path with fixed random outcomes vs statevector path
        tab = Tableau(n)
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1.0
        for g, qs in ops:
            Engine._tab_apply(tab, g, qs)
            if g == 'CNOT':
                psi = vec_cnot(psi, qs[0], qs[1], n)
            else:
                psi = vec_apply1(psi, GATES1[g], qs[0], n)
        # compare full measurement distribution: sample tableau many
        # times (fresh copies) vs statevector probabilities
        probs = np.abs(psi) ** 2
        counts = np.zeros(2 ** n)
        shots = 400
        for s in range(shots):
            t2 = Tableau(n)
            for g, qs in ops:
                Engine._tab_apply(t2, g, qs)
            r2 = np.random.default_rng(1000 + s)
            bits = [t2.measure(a, r2) for a in range(n)]
            counts[sum(b << i for i, b in enumerate(bits))] += 1
        tv = 0.5 * np.abs(counts / shots - probs).sum()
        assert tv < 0.15, f'CHP disagrees with statevector: TV={tv:.2f}'


_validate()


# ------------------------------------------------ experiments

def machine(engine, qubits, magic=True):
    """A small quantum computer: GHZ across its qubits (+ one T)."""
    engine.gate('H', qubits[0])
    for a, b in zip(qubits, qubits[1:]):
        engine.gate('CNOT', a, b)
    if magic:
        engine.gate('T', qubits[0])


def main():
    print('=' * 68)
    print('PART 15: THE RENDER-AWARE SIMULATOR')
    print('     (validated at import: CHP tableau vs statevector)')
    print('=' * 68)

    print('[54] the fusion cascade — 8 quantum computers, 10 qubits each,')
    print('     80 qubits in total, every machine entangled internally')
    print('     and carrying magic:')
    eng = Engine(80)
    for m in range(8):
        machine(eng, list(range(10 * m, 10 * m + 10)))
    print(f'     {"stage":<28} {"largest block":>13} {"engine size":>12}')
    print(f'     {"8 independent machines":<28} {eng.largest():>13} '
          f'{fmt(eng.nbytes()):>12}')
    for m in range(0, 8, 2):
        eng.gate('CNOT', 10 * m, 10 * (m + 1))
    print(f'     {"entangled in pairs":<28} {eng.largest():>13} '
          f'{fmt(eng.nbytes()):>12}')
    for stage, nblk in (('pairs entangled again', 2),
                        ('one fully fused block', 1)):
        n = 80 // nblk
        proj = nblk * 16 * 2 ** n
        print(f'     {stage:<28} {n:>13} {fmt(proj):>12}  (projected — '
              f'refused to materialize)')
    print('     Total qubits never changed. Only the entanglement')
    print('     structure did. Exponentials ADD across independent')
    print('     blocks and MULTIPLY when blocks fuse: a world full of')
    print('     small quantum devices is cheap for a bounded classical')
    print('     engine; one large sustained entangled block is the only')
    print('     thing that is not. (One 400-qubit block: '
          f'{fmt(16 * 2.0 ** 400)} ~ the holographic budget.)')
    print()

    print('[55] entanglement is cheap; magic is expensive:')
    eng2 = Engine(22)
    machine(eng2, list(range(22)), magic=False)
    tab_size = eng2.nbytes()
    print(f'     GHZ-22, maximally entangled, Clifford only: '
          f'{fmt(tab_size)} (tableau)')
    eng2.gate('T', 0)
    vec_size = eng2.nbytes()
    print(f'     the same state after ONE T gate:            '
          f'{fmt(vec_size)} (amplitudes forced)')
    print(f'     one gate of magic multiplied the engine by '
          f'{vec_size / tab_size:,.0f}x.')
    print('     Entanglement was never the expensive resource.')
    print()

    print('[56] decoherence is the engine\'s garbage collector:')
    eng3 = Engine(18)
    machine(eng3, list(range(18)))
    print(f'     18-qubit entangled block with magic: {fmt(eng3.nbytes())}')
    for q in range(0, 12, 3):
        for qq in range(q, q + 3):
            eng3.measure(qq)
        print(f'     after measuring {q + 3:>2} qubits: '
              f'{fmt(eng3.nbytes())}')
    print('     Every qubit the environment (or a detector) collapses')
    print('     factors out of the block and the ledger halves. NISQ-era')
    print('     devices decohere themselves back into cheapness; only')
    print('     fault tolerance sustains the block a bounded engine')
    print('     cannot afford.')


if __name__ == '__main__':
    main()
