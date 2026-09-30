# CUDA backend strategy assessment

## Decision

Do **not** replace C-DISORT with a literal pure-Python port. Retain the current
C-DISORT backend as the CPU implementation, compatibility route, and numerical
oracle. If CUDA performance remains a product requirement, develop a separate,
restricted **batched tensor backend** whose Python layer orchestrates PyTorch
operations and whose numerical work runs in CUDA kernels, cuBLAS/cuSOLVER, or a
small native CUDA extension.

This is not a choice between C and Python syntax. A Python `for`-loop
translation of C-DISORT would be slower than the current C CPU backend and
would give CUDA the same one-scalar-solve-per-lane structure. A PyTorch
implementation is useful only after the algorithm is expressed as operations
on batches of layers and independent wavelength/geometry solves.

## Evidence from the current backend

The general C-DISORT implementation is about 14,700 lines of mechanically
split C/C++ headers. One solve owns mutable pointer-based `disort_state` and
`disort_output` structures, uses roughly 70 work-array allocations, and calls
serial numerical routines for layer eigenproblems and a banded boundary solve.
The CUDA wrapper currently copies that model directly: one `(wavelength,
column)` solve per CUDA lane, private pmem-backed state/output, then a serial
`c_disort()` call.

For the 100-layer, 32-stream TP9 production workload, the C-derived CUDA kernel
is 16.50 s for 1,024 wavelengths and 17 columns, while the pydisort CPU batch
is 9.58 s. Transfers cannot explain this difference: the steady-state solve has
no D2H transfer and only 272 KiB of internal H2D grids per forward. The
resident-workspace experiment also found no repeatable improvement after
increasing scalar concurrency from two to four warps per SM.

The C source remains very valuable. Its individual stages—`c_disort_set`,
`c_solve_eigen`, `c_set_matrix`, `c_solve0`, and `c_fluxes`—give a precise
algorithmic reference, and its CPU outputs provide a differential test at each
step. It is not, however, a source-to-source conversion path: one-based macros,
mutable aliasing, in-place allocation, and branch-heavy feature handling must
be redesigned before tensors can represent them efficiently.

## Options

| Option | Performance outlook | Correctness and maintenance | Recommendation |
| --- | --- | --- | --- |
| Keep the scalar C-DISORT CUDA wrapper and tune launches/transfers | Low: the known cache and residency experiments showed no production gain. | Lowest implementation risk, but CPU remains faster. | Retain as supported compatibility path; stop micro-tuning. |
| Literal pure-Python port of C-DISORT | Worse than C on CPU; unchanged scalar mapping if put on CUDA. | Very high parity and maintenance cost; duplicates the solver. | Reject. |
| Tensorized PyTorch port of all DISORT features | Potentially high, but it is a new solver rather than a port. | High cost: broad feature matrix, numerical behavior, and two implementations. | Do not start as a full rewrite. |
| Restricted batched tensor/CUDA backend | Best chance of a GPU win on large spectral/geometry batches. | Bounded risk if explicitly opt-in and validated against C/Fortran. | Recommended investigation. |

## Recommended staged design

The target is an optional backend, initially limited to the physics required by
the production TP9 family. It must be selected explicitly and fall back to the
C backend for every unsupported feature. It does not change the existing
`backend="cuda"` contract until it passes the gates below.

1. **Specify the first supported subset.** Freeze a TP9-derived,
   plane-parallel, float64 flux contract: streams, layer properties, direct
   beam, Lambertian surface, delta-M behavior, user optical depths and angles,
   and exactly which radiance-related requests remain unsupported. Add a
   capability guard and a Fortran v4 fixture before implementation.
2. **Build a tensor reference for one stage at a time.** Start with batched
   layer setup and the reduced eigenproblem across `[batch, layer]`. Use
   explicit structure-of-arrays tensors rather than `disort_state` copies.
   Compare each intermediate and final flux against C-DISORT for tiny batches
   before using CUDA.
3. **Test the hard boundary solve separately.** The global layer-coupling
   system is the principal redesign risk. Prototype a batched block-banded
   solve, initially through stable PyTorch linear algebra; only introduce a
   custom CUDA kernel if profiling shows that library calls are inadequate.
4. **Measure the complete restricted path.** Require repeated H100 timings for
   TP9 1,024 x 17 against the current 16.50 s CUDA and 9.58 s CPU baselines,
   with device-resident tensors. Keep only a material, repeatable gain after
   CPU/CUDA and Fortran validation.
5. **Expand only after success.** Add thermal emission, pseudo-spherical
   geometry, BRDFs, general source, Fourier/radiance output, and special
   boundary conditions as separate capabilities. The C backend remains the
   default reference and fallback throughout.

## Decision gates

Proceed beyond design only if a small tensor prototype can reproduce the
restricted C-DISORT stage in float64 and demonstrates batching across many
solves. Proceed to a full restricted solver only if it improves the production
H100 case by enough to beat the current C CUDA route repeatedly without
regressing C/Fortran parity. If batched eigensystems or the boundary solve do
not show useful throughput, stop: the C CPU backend is the better production
route, and a full Python rewrite is not justified.

## What the C baseline provides

The C code supports a smooth *verification* conversion: immutable fixtures,
intermediate values, branch behavior, and final fluxes can be compared stage by
stage. It does not make the performance conversion smooth automatically. The
new backend must deliberately change data layout from per-solve pointer graphs
to batched tensors; that data-layout change is the source of its possible CUDA
benefit and the main engineering work.
