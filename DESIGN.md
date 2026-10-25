# DESIGN.md — quantum-number-blocked contraction for DMRG

Status: rung 1 built (generator, dense oracle, per-block cuBLAS baseline,
gate 0 green — see README status). This note remains the design record: it
fixes the data layout, works the sector combinatorics for the MVP target,
draws the fusion boundary, and pre-writes the gates and kill criteria so
that later rungs start from this file without re-deciding anything.

The MVP, when built, is a standalone block-sparse contraction microkernel fed
synthetic block structures with realistic sector statistics. No DMRG code
exists locally to attach to, and none is claimed. See the scoping line in
section 5.

## 1. Tensor/block data layout

**Decision: single contiguous device arena, blocks packed dense, with a
flat structure-of-arrays schedule. Per-block device allocation is rejected.**

Layout:

- One arena allocation per tensor per precision. Blocks are packed back to
  back, each block dense row-major, each block start aligned to 256 bytes,
  each leading dimension padded so rows start on a 32-byte boundary (4
  consecutive FP64 elements per 32-byte transaction).
- The symmetry index lives in a host-side open-addressing hash map
  `(q_left, q_right) -> (offset, m, n)`, built once per tensor and rebuilt
  after each truncation. Charge labels are small integer tuples packed into
  a 64-bit key (four signed 16-bit fields cover U(1)xU(1) and 4-charge
  multi-orbital models).
- The contraction schedule — the matched block list of section 3 — is a set
  of parallel arrays (`m[], k[], n[], offA[], offB[], offC[]`), built on the
  host, uploaded once, reused for every Lanczos/Davidson iteration on that
  bond.

Why not per-block allocation: a batched or fused execution needs the whole
block list addressable from one base pointer plus integer offsets. Per-block
`cudaMalloc` gives pointer-chasing on the device, allocation jitter in the
timing loop, and fragmentation across a sweep as block sizes change at every
truncation. It also forecloses the vendor comparison path — cuBLAS batched
GEMM wants a pointer array, which is cheap to build from an arena and
pointless to build over scattered allocations. The arena is what makes both
the fused kernel and the vendor baseline possible from the same data; that
is the property the layout must have, and per-block allocation does not have
it.

SoA versus an array of per-block structs: the schedule is consumed by index
on the device (a tile scheduler reads entry `i`, not a linked list), so
parallel arrays win on every access that matters and lose nowhere. The hash
map stays on the host; the device never does charge lookup — by the time a
kernel runs, matching has already happened and what remains is offsets.

## 2. Sector analysis

MVP target: 1D Hubbard or t–J chain, U(1)xU(1) (N_up, N_down).

**Combinatorics.** At a bond cutting the chain into blocks of `ell` and
`L - ell` sites, the left-block basis splits into symmetry sectors labeled
`q = (n_up, n_down)`, `0 <= n_sigma <= ell`: at most `(ell+1)^2` labels, with
sector Hilbert-space dimension

```
d(q) = C(ell, n_up) * C(ell, n_down).
```

The ground state at fixed total `(N_up, N_down)` puts Schmidt weight in
sector `q` proportional to how likely the cut is to realize those particle
numbers. For a gapless 1D phase, number fluctuations across a cut are
logarithmic: with `ell_eff = (L/pi) sin(pi*ell/L)`,

```
Var(n_sigma^left) ~ sigma^2 = ((1 + K_rho)/2) * ln(ell_eff) / pi^2,
```

where the spin mode contributes K_s = 1 (fixed by SU(2)) and the charge mode
is scaled by the charge Luttinger parameter K_rho. The sector-weight
distribution is then taken as a product of independent Gaussians in
`Delta n_up`, `Delta n_down` with that sigma. This is a model, labeled as an
estimate: factor-of-two convention choices in the Luttinger-liquid variance
are absorbed into sigma, and a Mott-insulating charge sector (half-filled
Hubbard) has short-ranged charge fluctuations, modeled by a smaller,
L-independent charge sigma. Gate 1 below replaces this model with a measured
histogram before any kernel decision is made.

Block sizes: sector `q` retains `m_q ~ chi * W(q)` states, truncated below
at `m_q >= 1`. Contraction cost per matched block triple scales like the
product of three block dimensions (section 3), so the concentration measure
used here is the FLOP share of the largest blocks under the proxy
`F ~ sum_q m_q^3` — exact for the peaked distributions below up to the
neighbor-sector coupling, and stated as a proxy.

Computed histograms (discrete Gaussian weights, `m_q = round(chi * W_q)`,
entries with weight < 1e-6 dropped; every number from this model is an
estimate):

t–J chain, L = 64, 1/8 doping, K_rho ~ 0.6, sigma = 0.55 per species:

| chi | blocks | top-1 FLOP share | top-4 | top-9 | blocks >= 64 | blocks < 16 |
|--:|--:|--:|--:|--:|--:|--:|
| 100 | 21 | 0.97 | 0.99 | 1.00 | 0 | 20 |
| 500 | 21 | 0.97 | 0.99 | 1.00 | 1 | 16 |
| 1000 | 21 | 0.97 | 0.99 | 1.00 | 5 | 12 |
| 2000 | 21 | 0.97 | 0.99 | 1.00 | 5 | 12 |

t–J chain, L = 256, sigma = 0.67:

| chi | blocks | top-1 | top-4 | top-9 | >= 64 | < 16 |
|--:|--:|--:|--:|--:|--:|--:|
| 1000 | 37 | 0.87 | 0.97 | 1.00 | 5 | 28 |
| 4000 | 37 | 0.87 | 0.97 | 1.00 | 9 | 24 |

Half-filled Hubbard, L = 64 (charge sigma 0.30, spin sigma 0.55):

| chi | blocks | top-1 | top-4 | blocks >= 64 |
|--:|--:|--:|--:|--:|
| 1000 | 15 | 0.99 | 1.00 | 3 |

For contrast, the regimes where the block count grows: a two-orbital model
(4 U(1) charges, sigma = 0.55 each) gives 297 blocks at L = 64 with the same
peaked weight concentration (top-1 share 0.95, but 264–288 blocks below size
16); a width-4 t–J cylinder (charge fluctuations wider by ~sqrt(W),
sigma = 1.1) gives 89 blocks with top-1 share only 0.39 and top-9 at 0.98 —
the first profile where FLOPs genuinely spread.

**Conclusion, stated plainly.** The MVP target — a single-band 1D chain — is
the *hardest* case for this kernel, not the easiest. The sector structure is
few-large-blocks with a long tail of tiny ones: at chi = 1000, one block
carries 87–99% of the FLOPs and a dozen blocks are smaller than 16x16.
cuBLAS handles the head trivially; the fused kernel can only win the tail,
and the tail is a small fraction of FLOPs. The win, if there is one, is
wall-clock: per-launch overhead and index handling for ~10–30 small GEMMs
per matvec, amortized over every Lanczos iteration and every bond of every
sweep. Whether that wall-clock share justifies the kernel is exactly what
gate 2 measures. The natural win regime — many mid-size blocks — appears
with cylinder width or orbital multiplicity, i.e. precisely the iron-based
and nickelate payloads, but those are not the MVP. The MVP target is kept
because it is the cheapest honest falsification: if the kernel cannot beat
the vendor loop where the deck is stacked against it, the idea dies before
anyone spends a month on multi-orbital sector generators.

## 3. The fusion boundary

**Target contraction.** In the two-site update, the dominant cost is the
effective-Hamiltonian matvec inside the eigensolver, specifically the
environment-application step. One-site notation for clarity: with MPS tensor
`A[l, s, r]`, MPO `W[w_in, w_out, s', s]`, and environment `L[l', w, l]`,
the first and most expensive contraction is

```
T[l', w, s, r] = sum_l L[l', w, l] A[l, s, r]      cost ~ chi^3 * D_w * d
```

(D_w the MPO bond dimension, d the physical dimension; D_w ~ 5–10 for
nearest-neighbor Hubbard/t–J MPOs). For each fixed MPO bond sector `w` this
is a GEMM over `l` with `(s, r)` flattened. The later steps (physical-index
contraction, right environment) are chi^2 D_w^2 d^2 and chi^3 D_w — the
chi^3 terms share the block structure described here, so one schedule serves
all of them.

**Charge matching.** `A`'s blocks carry labels `(q_l, q_r)` with
`q_r = Q_total - q_l`; the environment block `L[q_l', w, q_l]` exists only
when `q_l' = q_l + delta(w)`, where `delta(w)` is the charge carried by MPO
bond sector `w`. The matched list is therefore:

```
for each w-sector delta:
    for each left sector q_l with a block in both A (at q_l) and L (at q_l+delta):
        emit (m = m_{q_l+delta}, k = m_{q_l}, n = d * m_{Q-q_l}, offsets...)
```

Matching is integer tuple arithmetic over a few dozen sectors. It costs
nothing next to one chi^3 GEMM, it is identical for every eigensolver
iteration on the bond, and it never touches the device.

**The boundary, explicitly:**

- Host, always, both paths: charge matching, schedule construction, arena
  offset bookkeeping. This is the "block-index lookup" and it is *not* fused
  into any kernel — fusing integer scheduling into a GEMM kernel buys
  nothing because the schedule is reused thousands of times per build.
- Vendor path (baseline): the matched list executed as dense GEMMs — either
  one cuBLAS call per block (the naive loop), blocks bucketed by `(m, k, n)`
  into cuBLAS batched calls, or CUTLASS grouped GEMM for heterogeneous
  shapes in one launch. All FLOPs stay here.
- Fused path (the candidate kernel): one persistent-kernel launch over the
  matched list. A device-side tile scheduler walks the SoA schedule; blocks
  below a size threshold are executed by the fused kernel's own GEMM tiles;
  blocks above the threshold are peeled off and still go to cuBLAS, because
  hand-written tiles will not beat cuBLAS at 500x500x500. What the fused
  kernel absorbs is therefore narrow and honest: launch consolidation and
  tail-block execution. It does not absorb matching (host), does not absorb
  the head blocks (vendor), and does not attempt to be a better large GEMM.

That narrowness is the point: the kernel exists to remove per-block launch
and scheduling latency from the tail. If the tail doesn't matter (section
2's open question), the kernel doesn't either.

## 4. What the MVP measures

- **Correctness gate first.** Fused-path output vs a dense reference:
  assemble the full dense tensors (zeros outside matched blocks), contract
  with dense cuBLAS, compare against the block-sparse result. Tolerance
  `||diff||_F <= 1e-12 * ||ref||_F` in FP64, matching the sibling repos'
  tolerance discipline. No timing number is recorded before this passes.
- **Performance metrics.** Blocks/sec and effective GFLOPS (sum of
  `2*m*k*n` over the matched list, divided by measured time) for: per-block
  cuBLAS loop, shape-bucketed batched cuBLAS, CUTLASS grouped GEMM, fused
  kernel. Wall-clock per full matvec is the headline number; GFLOPS is the
  diagnostic.
- **Sweep.** chi in {64, 128, 256, 512, 1024, 2048}; sector profiles:
  t–J L = 64, t–J L = 256, half-filled Hubbard, and the width-4 cylinder
  profile as the many-block control. System size enters only through the
  sector statistics, so "system-size sweep" means the profile sweep.
- **Benchmark discipline** (as in the sibling repos, e.g.
  https://github.com/FireHorseAutumn/kpm-tightbinding and
  https://github.com/FireHorseAutumn/pauli-hamiltonian): raw per-repetition
  timings, median with [min, max] spreads, and a JSON record carrying
  device, clocks, driver, seed, timestamp, and toolchain. Tables report the
  vendor paths whether or not the fused kernel wins.

## 5. Honest scoping line

No DMRG code exists locally, and this project does not produce any. The MVP
is a standalone block-sparse contraction microkernel fed synthetic block
structures whose sector statistics follow section 2's model (and, after gate
1, a measured histogram from wherever an external DMRG code can supply one).
Claims are limited to: contraction throughput on DMRG-shaped block
structures, and correctness against a dense reference contraction. There are
no accuracy claims about any physical observable, no ground states, no
sweeps. It is a contraction kernel with a DMRG-shaped workload, and the
README and any future publication of this repo must keep that wording.

## 6. Gates and kill criteria

Pre-written, so an implementation window starts in five minutes:

- **Gate 0 — correctness before timing.** Fused path matches the dense
  reference contraction at `1e-12 * ||ref||_F` in FP64 on every sector
  profile and every chi in the sweep. Failure stops all performance work
  until fixed. No timing is logged against a build that fails this.
- **Gate 1 — the histogram decides.** Measure (not model) the block-size
  histogram for the MVP profile. Pre-committed decision rule: if the top 8
  blocks carry >= 95% of contraction FLOPs *and* the per-block cuBLAS loop
  spends < 20% of matvec wall-clock in tail blocks (< 64), the kernel idea
  dies for the 1D single-band profile. Record the histogram, record the
  death, and re-scope to the cylinder/multi-orbital profile or close.
  Section 2's model already predicts the 1D profile sits near this line —
  that is why the rule is written now, before anyone is invested.
- **Gate 2 — beat the vendor on wall-clock or die.** On profiles that
  survive gate 1: the fused kernel must beat the per-block cuBLAS loop by
  >= 2x on full-matvec wall-clock at chi >= 512, *and* must beat shape-
  bucketed batched cuBLAS (or CUTLASS grouped GEMM, whichever is stronger)
  by any margin, on the same matched lists. Beating only the naive loop is
  not a result — the batched vendor call is the real incumbent.
- **Honest death conditions, enumerated:** gate-1 rule triggered on all
  profiles available; fused kernel loses to bucketed batched cuBLAS at every
  chi; or correctness gate unreachable without compromising the design
  (e.g. the tile scheduler forces reduction orders that break FP64
  reproducibility requirements). Any of these closes the project with the
  measured table published as the record.

## 7. TPU expansion record

Recorded as a direction on the six-month arc; not built, not prototyped.

The observation: everything dynamic about this kernel is dynamic only at
DMRG time, not at contraction time. The sector structure is known before the
matvec starts; charge matching is integer arithmetic over a static label
set; the matched block list is a fixed schedule reused for every eigensolver
iteration. That is exactly the regime TPU-style execution wants: static
shapes, no runtime block discovery, scheduling done at compile time. The
XLA-shaped version of this kernel is: pad every block to systolic-tile
multiples (128), emit the matched list as a statically unrolled batched
contraction, and let the compiler place it. Quantum-number matching becomes
a compile-time pass that produces a block-diagonal mask and a fixed op list.

The honest caveats: padding waste is the whole game — on the 1D profile,
padding a 12x12 block to 128x128 wastes ~99% of the systolic array, so the
TPU mapping is attractive *only* in the same many-mid-size-block regime
(cylinders, multi-orbital) where the GPU fused kernel is attractive, and it
competes there against simply padding less aggressively. What the TPU record
actually preserves is the structural insight: block-sparse QN contraction
has a fully static schedule, so it is a compile-and-broadcast workload on
any systolic/batched architecture, and the GPU persistent-kernel schedule
built here is the same artifact an XLA pass would emit. If this lead
survives its gates, a six-month-arc follow-up is: express the matched-block
schedule as a static XLA computation, measure padding overhead against the
GPU fused kernel, and publish the comparison. Until then this section is the
entire TPU effort.

## 8. First rung — BUILT 2026-10-25

Rung 1 is built and gate-0 green (see README status for the numbers): the
seeded profile generator, the dense reference oracle, the per-block cuBLAS
loop, and gate 0 wired between them, exactly the four items below. One
refinement the build surfaced: the exact matched-triple cost 2*m*k*n is
flatter than section 2's sum-of-m^3 proxy (the complement factor
n = d*m_{Q-q} keeps n large for near-center sectors), so the 1D profile is
head-dominated but less extremely than the proxy table suggests; the
generator uses unnormalized Gaussian weights, which shifts the published
block counts only in corner sectors (25 vs 21 for t–J L=64; 15 and 37
match; 101 vs 89 for the cylinder).

Original rung-1 specification (kept as the record):

1. A host-side synthetic sector-profile generator: seeded, deterministic,
   emits the arena image + SoA matched list for a named profile
   (t–J L = 64, etc.) at a given chi. Pure host code.
2. The dense reference oracle (assemble dense tensors, cuBLAS dense
   contraction).
3. The per-block cuBLAS loop baseline.
4. Gate 0 wired between 3 and 2.

Dependencies: CUDA toolkit + cuBLAS + CMake. Nothing local; nothing from the
sibling repos. No DMRG library, no MPO construction code — profiles are
parameter files `(charges, sigma per charge, chi)`.

Time estimate (labeled as estimate): 3–5 focused days for rung 1; the fused
kernel, the bucketed-batched comparison path, and the full sweep add 1–2
weeks. The June feasibility estimate of 2–3 months to anything *validated*
stands, because validation here means surviving gates 0–2 with measured
tables, and the multi-orbital/cylinder profiles are where the surviving
question lives.

## Open uncertainties

- The Luttinger-liquid sigma model in section 2 is an estimate with
  convention-level factors absorbed; real retained-sector counts from an
  actual DMRG code may differ by ~2x in block count. Gate 1 exists because
  of this.
- Per-launch overhead (~5–10 us per small GEMM, from portfolio experience,
  not measured here) is the number the whole idea rests on; it is
  machine-dependent and must be re-measured on the target GPU before gate 2.
- Whether the environment tensors' own block structure (not modeled here)
  changes the matched-list length materially. Assumed no; unchecked.
- The D_w used for cost scaling (5–10) is for single-band nearest-neighbor
  MPOs; multi-orbital MPO bond dimensions are much larger and shift the
  contraction-order optimum. Section 3's target is the single-band ordering.
