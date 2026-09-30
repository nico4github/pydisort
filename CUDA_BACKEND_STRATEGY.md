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

The replacement phase is **parity first, performance second**. A complete,
validated tensor flow is retained even when its first end-to-end time is similar
to, or moderately worse than, the C CUDA route. Reject it only for incorrect or
non-finite results, an unsupported feature presented as supported, or an
architecture that cannot express the required batch operations. Performance
optimization begins only after the restricted flow is complete and measured as
a whole; otherwise every slow intermediate step would trigger an unnecessary
redesign.

Instrument every tensor-backend stage with `@pydisort.timed` and run it inside
`pydisort.TimingCollector`. The decorator is inactive outside that context. Its
summary reports inclusive wall time and CUDA-event time per named function,
synchronizing only when the summary is requested. Append that summary to
`$DISORT_REPORT_DIR/testproblem09_tensor_reconstruction_timing.txt` after each
run, alongside the JSON benchmark artifacts, before optimizing the completed
flow.

1. **Specify the first supported subset.** Freeze a TP9-derived,
   plane-parallel, float64 flux contract: streams, layer properties, direct
   beam, Lambertian surface, delta-M behavior, user optical depths and angles,
   and exactly which radiance-related requests remain unsupported. Add a
   capability guard and a Fortran v4 fixture before implementation.
2. **Port data and stages for parity, not speed.** Build eager Python/PyTorch
   layer setup and the reduced eigenproblem across `[batch, layer]`, using
   explicit structure-of-arrays tensors rather than `disort_state` copies.
   Preserve the C calculation order where practical. Compare intermediates and
   final fluxes against C-DISORT on tiny batches before moving on. A slower
   stage is accepted while it makes the final batched flow more explicit.
3. **Port the boundary solve separately.** The global layer-coupling system is
   the principal redesign risk. Implement a batched block-banded formulation,
   initially through stable PyTorch linear algebra. Validate it independently
   before composing it with the eigen stage; do not optimize either stage yet.
4. **Accept a complete restricted reference flow.** Run the full TP9 subset on
   CPU and CUDA with device-resident tensors. Require machine-precision
   numerical agreement with C-DISORT and the Fortran reference fixture. Record
   repeated H100 timings against the current 16.50 s CUDA and 9.58 s CPU
   baselines, but retain the correct flow when it is similar or moderately
   slower. This is the replacement milestone, not a performance gate.
5. **Optimize only after replacement parity.** Profile the complete tensor
   flow, then introduce `torch.compile`, library-specific layouts, fusion, or a
   small custom CUDA kernel for a measured bottleneck. Each optimization keeps
   the same parity checks and reports before/after complete-flow timing.
6. **Expand only after the restricted flow is stable.** Add thermal emission,
   pseudo-spherical geometry, BRDFs, general source, Fourier/radiance output,
   and special boundary conditions as separate capabilities. The C backend
   remains the default reference and fallback throughout.

## Decision gates

Proceed beyond design when a small tensor prototype reproduces the restricted
C-DISORT stage in float64 and exposes a batched representation across many
solves. Proceed to the complete restricted solver when every stage passes the
numerical gate, even if its initial timing is similar to or moderately worse
than the current C CUDA route. Then profile and optimize the complete flow.

For this project, “machine precision” means a documented float64 comparison
against C-DISORT and the Fortran fixture with errors at the scale of floating
point rounding. Bitwise identity is not required where a batched library uses a
different but numerically equivalent operation order; any tolerance is stated
per stage and justified from the observed C/Fortran rounding envelope.

## Reconstruction progress

The first replacement component is implemented in
`pydisort.tensor_backend.prepare_atmosphere`. It vectorizes the exact
property-to-state preparation from `disort_impl`: optical-depth and
single-scattering-albedo extraction, normalized zeroth Legendre moment,
zero-filling of absent moments, and upward layer reversal. It preserves
float64 tensors and supports the full `(nwave, ncol)` batch shape. Its parity
tests use exact tensor equality because this stage performs copies and fills,
not reordered floating-point arithmetic. The function is instrumented as
`tensor_backend.prepare_atmosphere` for `TimingCollector` summaries.

The next replacement component is the batched layer setup/eigenproblem input;
it must consume this structure-of-arrays state without rebuilding per-solve
C-DISORT pointer graphs.

## What the C baseline provides

The C code supports a smooth *verification* conversion: immutable fixtures,
intermediate values, branch behavior, and final fluxes can be compared stage by
stage. A Python/PyTorch implementation can be a practical execution backend
once its hot work is tensorized. The main engineering work is deliberately
changing data layout from per-solve pointer graphs to batched tensors; that
change is the source of both rapid Python iteration and possible CUDA benefit.

## Restricted TP9 reconstruction tracker

This is a 15-milestone implementation tracker. Milestones are not equal in
size; the remaining boundary and flux stages carry most of the work.

| # | Milestone | Status |
| ---: | --- | --- |
| 1 | Freeze restricted TP9 float64 contract and Fortran fixture | Complete |
| 2 | Opt-in nested wall/CUDA timing and incremental text log | Complete |
| 3 | Batched atmosphere state preparation | Complete |
| 4 | Delta-M layer optics | Complete |
| 5 | User output-depth grid | Complete |
| 6 | Device Gauss-Legendre quadrature | Complete |
| 7 | Batched reduced eigenproblem matrix | Complete |
| 8 | Batched eigensolve and full eigenvector reconstruction | Complete |
| 9 | Layer-continuity factors | Complete |
| 10 | Generic batched block-tridiagonal solver | Complete |
| 11 | C-DISORT TP9 boundary block and RHS assembly | Complete |
| 12 | Beam/source terms and constants of integration | Direct plane-parallel beam and finite-band thermal layer/surface/top sources complete; general source pending |
| 13 | Flux extraction on the user output grid | Complete for no-beam TP9a |
| 14 | Complete tensor flow C-DISORT/Fortran parity tests | Active: TP9a/TP9b, direct beam, C-DISORT thermal, and a Fortran-v4-validated finite-band thermal CPU/CUDA fixture |
| 15 | CPU/H100 end-to-end timing, incremental log, and optimization baseline | Pending |

## Earliest-reference rule

A reconstruction stage is not considered numerically established merely because
its tensors have the expected shapes or its linear algebra residual is small.
As soon as a newly connected subset can produce a derived physical quantity,
add a `pytest` reference case for that quantity before extending the solver.
Use the simplest applicable reference first: an analytic absorption-only flux
case, then a C-DISORT fixture, then the matching Fortran fixture for the
restricted TP9 flow. Keep each test permanently so later optimization cannot
silently break a previously reconstructed path.

For the first executable TP9 flux subset this means: compare upward and
downward fluxes at the configured user optical depths against C-DISORT and the
Fortran v4 fixture immediately after constants and flux extraction are
connected. Record the observed float64 tolerance in the test and benchmark
report. End-to-end timing is not a prerequisite for this numerical gate.
