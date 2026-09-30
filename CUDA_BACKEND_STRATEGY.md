# CUDA backend strategy assessment

## Decision

Do not replace C-DISORT with a *literal scalar Python-loop translation*.
Retain the current C-DISORT backend as a compatibility route and numerical
oracle. Develop a separate, restricted **Python/PyTorch batched backend** as
the first CUDA redesign experiment. Its Python code expresses the algorithm,
while PyTorch dispatches optimized CPU BLAS/LAPACK or CUDA/cuBLAS/cuSOLVER
kernels for tensor operations.

This is not a choice between Python and C execution speed. Python orchestration
over large tensor operations can approach or exceed handwritten C because the
hot loops still run in compiled numerical libraries. It also allows rapid
iteration on algorithm and layout without rebuilding the pydisort extension.
Only a line-by-line `for`-loop translation would remain interpreter-bound and
preserve the current one-scalar-solve-per-lane GPU mapping.

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
| Literal scalar Python-loop port of C-DISORT | Worse than C on CPU; unchanged scalar mapping if put on CUDA. | Very high parity and maintenance cost; duplicates the solver. | Reject. |
| Vectorized Python/PyTorch port of all DISORT features | Potentially high, but it is a new solver rather than a direct translation. | High cost: broad feature matrix, numerical behavior, and two implementations. | Do not start as a full rewrite. |
| Restricted Python/PyTorch batched backend | Best chance of a GPU win on large spectral/geometry batches, without extension rebuilds during exploration. | Bounded risk if explicitly opt-in and validated against C/Fortran. | Recommended investigation. |

## Recommended staged design

The target is an optional Python/PyTorch backend, initially limited to the
physics required by the production TP9 family. It must be selected explicitly
and fall back to the C backend for every unsupported feature. Early experiments
use ordinary eager PyTorch tensor operations, so changing Python code requires
no pydisort extension rebuild. It does not change the existing `backend="cuda"`
contract until it passes the gates below.

1. **Specify the first supported subset.** Freeze a TP9-derived,
   plane-parallel, float64 flux contract: streams, layer properties, direct
   beam, Lambertian surface, delta-M behavior, user optical depths and angles,
   and exactly which radiance-related requests remain unsupported. Add a
   capability guard and a Fortran v4 fixture before implementation.
2. **Build an eager Python/PyTorch reference one stage at a time.** Start with
   batched layer setup and the reduced eigenproblem across `[batch, layer]`.
   Use explicit structure-of-arrays tensors rather than `disort_state` copies.
   First measure the vectorized CPU path, then move the same tensors to CUDA.
   Compare each intermediate and final flux against C-DISORT for tiny batches
   before optimizing or compiling anything.
3. **Test the hard boundary solve separately.** The global layer-coupling
   system is the principal redesign risk. Prototype a batched block-banded
   solve, initially through stable PyTorch linear algebra; only introduce a
   custom CUDA kernel if profiling shows that library calls are inadequate.
4. **Measure the complete restricted path.** Require repeated H100 timings for
   TP9 1,024 x 17 against the current 16.50 s CUDA and 9.58 s CPU baselines,
   with device-resident tensors. Keep only a material, repeatable gain after
   CPU/CUDA and Fortran validation. Introduce `torch.compile` or a small custom
   CUDA kernel only for a measured eager-PyTorch bottleneck.
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
stage. A Python/PyTorch implementation can be a practical execution backend
once its hot work is tensorized. The main engineering work is deliberately
changing data layout from per-solve pointer graphs to batched tensors; that
change is the source of both rapid Python iteration and possible CUDA benefit.
