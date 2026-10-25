# qn-dmrg-contraction

Quantum-number-blocked tensor contraction for DMRG: fuse block-index lookup
with batched multiplication over symmetry sectors, instead of one dense GEMM
call per block.

## Vendor gap

cuBLAS sees only dense GEMMs. DMRG tensors are block-diagonal by Abelian
symmetry (U(1) particle number, S_z) with varying block sizes — structural
sparsity no vendor library exploits.

## Physics payload

The most immediate unconventional-superconductivity lane of this portfolio:
iron-based superconductors, bilayer nickelates (La₃Ni₂O₇), quantum spin
liquids.

## Design-first

This is an algorithm project with a kernel inside, not a kernel project:
the kernel idea lives or dies on the block-size statistics of the symmetry
sectors, so the analysis and the gates were written before any CUDA. The
design is done; rung 1 (generator, oracle, baseline, gate 0) is built.

## Deliverable

[`DESIGN.md`](DESIGN.md) — the design note: tensor/block data layout,
sector combinatorics for the MVP target (1D Hubbard / t–J chain,
U(1)xU(1)), the fusion boundary between vendor GEMM and the fused
block-schedule kernel, the MVP measurement plan, pre-written gates and kill
criteria, and the first-rung specification for any future implementation
window. It also records (without building) why the static sector structure
maps naturally to TPU-style systolic/batched execution.

Scope: the eventual MVP is a standalone block-sparse contraction microkernel
fed synthetic block structures with realistic sector statistics. No DMRG
code exists here and no physics accuracy claims are made — a contraction
kernel with a DMRG-shaped workload.

## Status

Rung 1 (DESIGN.md §8) landed 2026-10-25: `include/qnd/profile.hpp` is the
seeded sector-profile generator (Gaussian sector-weight model, presets for
t–J L=64/L=256, half-filled Hubbard, width-4 cylinder), single-arena block
packing (256 B block starts, 32 B padded leading dims), and the SoA
matched-block schedule; `src/qnd_ref.cu` runs the per-block cuBLAS loop and
the dense reference contraction and applies gate 0. Gate 0 passes on all
four presets at chi = 32..256: worst ||diff||_F = 5.6e-14 against
tolerances of ~1e-9 (1e-12 * ||ref||_F), CTest 7/7.

Sector census at chi = 1000 (generator convention: unnormalized Gaussian
weights, cutoff 1e-6 — DESIGN.md's histograms used normalized weights, so
counts differ only in corner sectors): t–J L=64 25 sectors/105 triples,
t–J L=256 37/157, half-filled Hubbard 15/59, cylinder 101/461. Under the
exact matched-triple cost 2*m*k*n the top-1 FLOP shares are 0.51, 0.33,
0.68, 0.10 — flatter than DESIGN.md's sum-of-m^3 proxy (0.97, 0.87, 0.99,
0.39) because the complement factor n = d*m_{Q-q} stays large for
near-center sectors. That is the neighbor-sector coupling the note flagged,
measured instead of modeled: the 1D profile is still head-dominated, but
less extremely than the proxy claimed.

Rung 2 — the bucketed-batched cuBLAS comparison path and the fused
persistent kernel — has not started. Gate 0 must stay green before any
timing number exists.

Sibling repos in the same kernel portfolio:
[kpm-tightbinding](https://github.com/FireHorseAutumn/kpm-tightbinding),
[pauli-hamiltonian](https://github.com/FireHorseAutumn/pauli-hamiltonian),
[lcao-grid-kernel](https://github.com/FireHorseAutumn/lcao-grid-kernel),
[batched-eigensolver](https://github.com/FireHorseAutumn/batched-eigensolver),
[afqmc-determinants](https://github.com/FireHorseAutumn/afqmc-determinants).
