# The Rendered Universe

Fourteen runnable experiments on one hypothesis: **the physical universe is not
a simulation — it is the *render output* of one.** Particles are pixels: you
can observe them and infer the rules of the rendering, but they are not the
mechanical parts of the machine.

📄 **The paper:** [`writeup/the-rendered-universe.pdf`](writeup/the-rendered-universe.pdf)
— a research-report treatment of everything below, with figures, a claim
taxonomy, and an honest accounting of what is measured, what is reproduced,
and what is conjecture.

## The idea in three layers

- **The engine** — the mechanism: a state and an update rule. It knows nothing
  about observers or display.
- **The chart** — a fixed projection from engine degrees of freedom to observed
  ones. It is a bijection but *not* an isometry: adjacency on the screen need
  not be adjacency in the engine.
- **The render** — the observed layer: particles, fields, spacetime. Observers
  are patterns in the render.

On this reading, quantum nonlocality is a chart artifact (far apart on screen
≠ far apart in the engine), particles are render events sampled from engine
amplitudes, geometry is the shape of the entanglement pattern, and the arrow
of time is a property of coarse render descriptions of a reversible engine.
Every one of those sentences is demonstrated by code in this repository.

## Quickstart

Requirements: Python 3.12+, `numpy`, `pillow`. Nothing else. Every experiment
is a standalone script that prints a lab-notebook narrative and (usually)
writes figures/films to `films/`.

```bash
python3 run.py        # start here: a universe, a firewall, and the doors
```

## The experiments

Run them in order — each part builds on discoveries of the previous ones.

| Part | Script | The question it asks | Headline result | ~time |
|-----:|--------|----------------------|-----------------|------:|
| 1 | `run.py` | What laws does a physicist confined to pixels discover? | c = 1.000, exact conservation, a particle zoo — and doors in the sky where particles jump at 48× light speed | 15s |
| 2 | `bell.py` | Can chart geometry produce Bell violations? | Pixel CHSH \|S\| = 4 from raw CA dynamics across screen-causally isolated stations; singlet statistics S = 2.84 with no signaling when the pair is engine-adjacent (ER=EPR in 128px) | 6s |
| 3 | `corner.py` | Do structural constraints corner rule space? | 2.1×10¹³ candidate rules → 32 by pure combinatorics; geometry derived from causal structure grows a wormhole nobody drew | 7s |
| 3 | `census.py` | Which surviving rules make universes? | 9 of 32 (space + matter + particles) | 6s |
| 3 | `frontier.py` | Does cornering scale with richer alphabets? | Ternary: 2²⁶ survivors, 83% of samples are universes — universes are generic; constraints are the scarce thing | 13s |
| 3 | `ledger3d.py` | …and with dimension? | Same 256 states, cube-group isotropy: 2²⁸ survivors where 2D left 10³³ — dimension is a constraint multiplier | 1s |
| 4 | `entangle.py` | Is distance just entanglement? | Mutual-information ruler tracks true distance at r = 0.986 (scalar); r = 0.61 for fermions — **geometry belongs to the matter**; wormhole and Van Raamsdonk tear demonstrated | 1s |
| 5 | `curve.py` | Does matter curve derived space? | Vacuum-calibrated rulers stretch 2.69× where matter sits; geodesics bend around the well; 3D embedding is a literal funnel | 21s |
| 6 | `tempo.py` | Does the static entanglement metric predict dynamics? | Shapiro-delay race confirms it — and gravity-as-medium is chromatic: the **equivalence principle is violated**, naming exactly what real gravity does differently | 2s |
| 6 | `selfgrav.py` | Can the matter→geometry loop close? | Energy-sourced Poisson potential: parallel beams fall together and merge (capture); entanglement entropy scales with **perimeter** (r = 1.0000) — the holographic area law | 1s |
| 7 | `duality.py` | What is wave/particle duality? | The wave is the engine object; the particle is the render event. One field through both slits; Born-sampled clicks assemble the fringes dot by dot | 1s |
| 8 | `eraser.py` | What is measurement? | Which-path coupling kills visibility 0.90 → 0.49; sorting the *same* clicks by the detector's record restores V = 0.89–0.93. Correlation, not collapse | 8s |
| 9 | `arrow.py` | Where does time's arrow live? | Entropy 3,831 → 7,160 bits, then *exactly* reversed to 3,831 (bit-perfect); one flipped cell in the gas destroys the past, in vacuum it survives — chaos is matter-borne; the blob is 1-in-2⁴²⁹⁵ | 3s |
| 10 | `sprinkle.py` | Can discreteness hide from Lorentz? | A boosted lattice shifts its statistics 29% (an absolute rest frame); a boosted Poisson sprinkle shifts 1% (none) | 15s |
| 10 | `nogo.py` | Where exactly do classical engines die? | Both-slit arrivals at 4.8% of the classical P₁+P₂ floor; the amplitude wall measured at 2.14× per qubit | 8s |
| 10 | `insider.py` | Can an observer be made of pixels? | 21% of 3,192 glider collisions leave a stable changed record — a 1-bit detector inside the universe, intact 400+ ticks | 40s |
| 10 | `chiral.py` | Why is the Standard Model hard to host on a lattice? | Fermion doubling watched happening (2 species in 1D, 4 in 2D, net chirality always 0); Wilson's cure breaks the symmetry chirality needs | 1s |
| 11 | `axioms.py` | Can quantum mechanics be cornered? | Classical fails continuity, real QM fails local tomography, complex QM alone passes (15 = 15); Born rule from branch-swap symmetry | 1s |
| 11 | `generation.py` | Can the Standard Model's charges be cornered? | Anomaly cancellation leaves exactly one chiral hypercharge ray — the SM's; atoms exactly neutral; one generation = a 5-bit register | 1s |
| 12 | `collider.py` | What is the method itself? | Physics as constraint satisfaction: 15 axiom collisions catalogued in [`COLLIDER.md`](COLLIDER.md), 12 backed by in-repo measurements | 1s |
| 13 | `fingerprint.py` | Can the universe's seed be detected? | Detectability of a seed's generator dies at ~40 bytes of program; statistical fingerprints launder under dynamics, **exact seed symmetries survive forever** | 3s |
| 14 | `sky.py` | What should the CMB show, then? | Injecting the three observed low-ℓ anomalies at published amplitudes: individually ~2σ, **jointly p = 4.3×10⁻³** under the common-origin hypothesis; an exact seed symmetry is a spectral selection rule (odd-(ℓ+m) power = 0 vs null 0.5) | 1s |
| 15 | `renderware.py` | What does a quantum computation cost a classical engine? | The cost is the smallest *engine*, not the qubit count: independent machines' exponentials add (8×10 qubits = 128 KB; fused = 10⁷ EB projected); GHZ-22 costs 248 B until one T gate ×270,600s it; measurement halves the ledger per qubit | 3s |
| 16 | `universal.py` | Can gravity be universal instead of chromatic? | Coupling through an optical **metric** (not a medium) restores the equivalence principle: attraction, achromatic lensing (1.9-cell spread per octave), matched Shapiro delays, and an Eötvös pass — different masses at the same velocity fall within 2.4 cells | 2s |
| 17 | `genesis.py` | Do the three postulates actually run? | Universes from ~50 lofted bytes, space never given as input: the renderer sees only anonymous sensor IDs, yet recovers 2D geometry (dim 2.2, hidden-wiring r = 0.9) — by *intervention*, because watching fails (period-locked matter fakes wormholes, r = 0.1): in a deterministic world, space is causation | 5s |

## Repository layout

```
engine/       the mechanism: reversible block CA (any k-state rule), scalar
              field dynamics, deep matter — knows nothing about screens
render/       the chart: engine → screen projection (identity almost
              everywhere, and that "almost" is the whole story)
physicist/    the observer: forbidden by tests/test_firewall.py from
              importing engine or render — knows only frames
observatory/  research instruments: causal geometry, Gaussian-state
              entanglement, curved-space rulers, spherical harmonics
rulespace/    exact orbit combinatorics over spaces of possible physics
films/        every figure and film, regenerated by the scripts
writeup/      the paper (HTML source + PDF)
COLLIDER.md   the axiom collision matrix
```

## How honest is this?

Deliberately. Three rules were enforced throughout, and the paper tags every
claim accordingly:

1. **The epistemic firewall** — observer code cannot import the engine; all
   "discoveries" come from rendered frames (`tests/test_firewall.py` fails the
   build otherwise).
2. **Calibrate on a known universe** — every instrument (dimension estimators,
   visibility, detection significance) is validated against a system whose
   answer is known before being trusted. Several results were retracted and
   corrected mid-program when calibration failed; the scripts contain the
   fixed versions and the paper narrates the failures.
3. **Installed ≠ emergent** — where quantum statistics were installed rather
   than derived (singlet law in part 2, Born sampling in part 7), the code
   and paper say so. What those parts demonstrate is architecture, not
   emergence.

What is *not* claimed: QM derived from beneath (Bell/PBR close that road — we
corner it from axioms instead), the Standard Model derived (we corner its
charges), or the seed identified (only its class — compressible — is argued).

## License

MIT — see [`LICENSE`](LICENSE).
