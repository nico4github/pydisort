# Tensor parity findings

This log records evidence from the parity-first PyTorch reconstruction. Update
it whenever an experiment changes the supported tensor contract, narrows an
active divergence, or rejects a plausible implementation path. The detailed
plan and milestone status remain in `CUDA_BACKEND_STRATEGY.md`.

## Current state

- The focused tensor suite has 42 passing tests and two strict expected
  failures: the CPU and CUDA variants of the Problem 9c thermal-only source
  decomposition test.
- The source inputs and C-DISORT outputs are stored in
  `tests/fixtures/tensor_tp9c_source_decomposition_reference.json`.
- The diffuse-only six-layer Problem 9c component matches native C-DISORT to
  float64 rounding. This establishes the batched optics, delta-M transform,
  eigenproblem, boundary matrix, and diffuse Lambertian boundary path for this
  case.

## Accepted results

### Thermal and Lambertian boundary terms

- Finite-band Planck integration reproduces `c_planck_func2`, including the
  narrow-band Simpson branch.
- One-layer thermal, thermal surface/top emission, Lambertian direct-beam, and
  combined thermal-Lambertian fixtures pass on CPU and CUDA.
- The corresponding committed fixtures are covered by the regular tensor test
  suite.

### Problem 9c source decomposition

- `376f017` added durable native C-DISORT source fixtures and strict expected
  failures for the unresolved components.
- Diffuse-only and beam-only fluxes are exact to float64 rounding for all five
  output depths on CPU and CUDA.
- The direct transmitted beam and diffuse beam particular solution are both
  validated by required fixtures.

### Incident beam angle

- `ae4d3af` corrected the beam Legendre argument to `-umu0`, matching
  C-DISORT's call to `c_legendre_poly` for the incident direction.
- In the six-layer, black-surface, anisotropic beam probe, this reduced the
  top upward-flux error from about `+5.84e-2` to `-6.95e-4`.

### C-DISORT beam-source trace

- `0002-upbeam-trace.patch` adds a CPU-only, opt-in entry and `c_upbeam`
  trace. `make rebuild` is now the required local workflow because the Python
  binding inlines the C-DISORT solver as well as linking its shared libraries.
- On the black-surface Problem 9c beam probe, C-DISORT reports `fbeam = pi`,
  `umu0 = 0.5`, eight streams, six layers, plane-parallel geometry, and
  Planck mode. These are the intended probe inputs.
- For the flux-relevant `mazim=0` component, all six C `YLM0` vectors,
  pre-solve `ZJ` RHS vectors, and post-solve vectors match the tensor
  calculation to float64 rounding. The largest observed difference is below
  `5e-15`.
- The remaining beam-only flux gap is therefore downstream of `c_upbeam`:
  it is in `ZZ` consumption while assembling/matching the boundary system or
  reconstructing output fluxes. The beam source and its linear solve are no
  longer candidates.

## Active divergence

The active probe is Problem 9c with diffuse and beam sources disabled and a
black lower surface. The remaining CPU/CUDA strict expected failures are
thermal-only output fluxes. The largest observed difference is about `6.04e-2`
in diffuse downward flux at output optical depth 1.05. The top boundary
downward flux already agrees to float64 rounding.

Do not relax the thermal fixture tolerance or convert either test to a normal
pass until the C-DISORT thermal source equation is reproduced.

## Rejected hypotheses

- **Delta-M mismatch:** running the tensor probe without delta-M left the
  beam discrepancy effectively unchanged. Delta-M is not the first divergent
  stage.
- **Alternative `ZZ` storage permutations:** the remaining simple permutations
  produced negative fluxes or errors much larger than the current mapping.
  The current storage mapping is retained.
- **Native C-direction-order beam system:** rebuilding the temporary beam
  system in reversed C quadrature order reproduced a rejected storage
  permutation and was decisively worse.

## Next investigation

Trace the first downstream use of the verified beam vectors with the same
black-surface probe:

1. Compare C-DISORT's `c_solve0` boundary RHS entries that consume `ZZ` with
   the tensor boundary RHS before its dense solve. This catches direction and
   layer-interface indexing independently of the beam source.
2. If that matches, compare the solved integration constants and then the
   `c_fluxes` beam contribution at each output depth.
3. Trace C-DISORT `c_upisot` thermal `ZPLK0`/`ZPLK1` values and compare them
   directly with the tensor thermal source before changing output integration.
4. Correct the first divergent thermal term, promote the thermal fixture to
   required CPU/CUDA parity, then add the full Problem 9c flux fixture.

## Apple MPS note

The tensor layout is device-generic, but the parity path deliberately uses
float64. PyTorch MPS cannot allocate float64 tensors, so Apple GPU support
requires a separate float32 execution profile and tolerance suite after the
float64 parity gates pass. It is not a second solver rewrite.

### Boundary trace result

- The `c_solve0` trace confirms that the tensor beam boundary RHS, including
  every layer-interface term, matches C-DISORT to float64 rounding.
- C-DISORT's solved integration constants then diverge from the tensor dense
  boundary solve. The active defect is therefore the tensor representation of
  `c_set_matrix` / its constant ordering, not beam source construction or RHS
  assembly.
- A direct eigenvalue reorder was rejected because it broke all established
  diffuse, thermal, and CUDA parity gates. C `GC` columns agree with tensor
  eigenvectors up to arbitrary column scaling, while C's `KK` ordering must be
  represented inside the boundary matrix rather than changing the public
  eigensystem contract.
- A single C trace of the raw `c_set_matrix` band storage established that
  tensor continuity rows had both homogeneous blocks reversed in sign. The
  C RHS already used the correct source sign, so the error changed the solved
  constants rather than merely rescaling an equation. Reversing those two
  tensor matrix signs yields float64 beam-only TP9c parity on CPU and CUDA.
- The trace now also captures `PYDISORT_TRACE_CBAND`, allowing the C band
  storage to be decoded as a dense reference without further speculative
  ordering changes.
