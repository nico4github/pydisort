# Tensor parity findings

This log records evidence from the parity-first PyTorch reconstruction. Update
it whenever an experiment changes the supported tensor contract, narrows an
active divergence, or rejects a plausible implementation path. The detailed
plan and milestone status remain in `CUDA_BACKEND_STRATEGY.md`.

## Current state

- The focused tensor suite contains 66 test instances: 45 CPU passes, two
  documented skips, and 19 CUDA gates exercised on the H100. They include
  required CPU/CUDA parity for full Problem 9c, all six published Test Problem 1 flux
  configurations, all four Rayleigh Test Problem 2 cases, both high-order
  Henyey-Greenstein Test Problem 3 cases, and the two azimuth-independent
  Haze-L Test Problem 4 beam cases, and both 48-stream Cloud C.1 Test Problem 5 beam cases.
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
- Diffuse-only, beam-only, and thermal-only fluxes are exact to float64
  rounding for all five output depths on CPU and CUDA.
- The direct transmitted beam, diffuse beam particular solution, and thermal
  particular solution are all validated by required fixtures.

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

## Current validated scope

Problem 9c source decomposition and its full combined calculation are complete:
diffuse-only, beam-only, thermal-only, and combined runs each match their stored
native C-DISORT reference at float64 rounding on CPU and CUDA. The combined gate
caught and fixed a lower-boundary beam assignment that overwrote thermal RHS
terms; it now accumulates both source contributions as C-DISORT does.

All six published Test Problem 1 flux configurations are now stored as native
float64 fixtures and pass on CPU and CUDA. They cover thin/thick slabs, beam
and diffuse illumination, and conservative and near-conservative scattering.

The four Rayleigh Test Problem 2 cases and two high-order Henyey-Greenstein
Test Problem 3 cases are also stored and pass on CPU and CUDA. They establish
that the reconstructed angular scattering path covers non-isotropic phase
moments, delta-M scaling, and conservative transport.

The two azimuth-independent Test Problem 4 Haze-L cases now provide 32-stream,
high-order tabulated phase-moment beam coverage. Their self-contained native
C-DISORT fixtures pass on CPU and CUDA within 2e-6; the observed independent
solver difference is below 9.4e-7.

The two azimuth-independent Test Problem 5 Cloud C.1 cases extend that gate
to 48 streams and 299 tabulated phase moments through a 64-optical-depth cloud.
Their native fixtures pass on CPU and CUDA at 1e-8; the observed maximum
difference is below 3.2e-10.

The tensor radiance path now has its first direct user-angle gate. It ports
C-DISORT `c_interp_eigenvec`, the one-layer homogeneous m=0 ray integral, and
the attenuated diffuse top boundary. The `mu=(-0.5, 0.5)` native C-DISORT
fixture at optical depths 0, 0.35, and 0.7 passes on CPU and CUDA. Multilayer
ray transport, source terms, and Fourier orders remain pending.

The one-layer gate deliberately covers both directions: the downward value
combines the homogeneous integral with the attenuated diffuse top boundary,
while the upward value has a black lower boundary. The next fixture extends
only the homogeneous m=0 transport to two nonuniform layers. It will separate
the complete-layer and partial-layer contributions before beam, thermal, or
surface terms are introduced.

The native input and float64 output for that two-layer gate are stored in
`tests/fixtures/tensor_user_ray_two_layer_reference.json`: optical thicknesses
are 0.2 and 0.5, single-scattering albedos are 0.4 and 0.7, and requested
depths cover both an interface and an interior point in the second layer. The
fixture is present; the tensor implementation and CPU/H100 assertion remain
pending.

`0003-user-ray-trace.patch` retains an opt-in CPU diagnostic for the
post-constant user-angle modes in `c_user_intensities`. Set
`PYDISORT_TRACE_USER_RAY` only while investigating a user-ray mismatch; it
does not affect ordinary solver runs. It also records the final homogeneous
ray integral, boundary term, and output value at each traced user point.

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

1. Complete the two-layer source-free m=0 user-ray fixture, preserving
   explicit CPU/CUDA parity for complete and partial ray segments.
2. Add user-angle beam, thermal, and general-source terms one at a time, with
   a source-only C-DISORT fixture before any combined-source fixture.
3. Generalize to Fourier orders and reconstruct azimuth-dependent radiance
   against TP4c before exposing that public capability.
4. Add Hapke surface coupling only after its fixed-parameter C-backend
   reference is selected; RPV, CAM, and AMB are explicitly deferred.
5. Establish the representative 100-layer / 1000-channel tensor workload,
   then begin the end-to-end timing and CUDA optimization baseline.

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
- The tensor flux path now supports the Fourier-zero computational general
  source array in C-DISORT's `(nwave, ncol, nstr, nlyr, nstr)` layout. Its
  zero-source reduction and the C-DISORT one-layer nonzero reference pass on
  CPU and CUDA at `1e-12`; user-angle sources remain deferred with radiance.
- The thermal trace shows all six C `ZPLK0` and `ZPLK1` vectors match the
  tensor source arrays to float64 rounding. Its `c_solve0` RHS then exposed
  the missing five thermal continuity terms in the tensor system. Adding the
  exact C expression restored RHS parity below `5e-15` and promoted thermal
  TP9c CPU/CUDA fixtures to required passes.

- A nonuniform two-layer general-source trace confirms both C ZZG vectors and every source-bearing c_solve0 boundary RHS entry agree with the tensor path at float64 rounding. The independent C and torch eigensystem/boundary solves produce final fluxes within 5e-8; the stored two-layer fixture therefore uses 1e-7 absolute tolerance. The one-layer analytical source gate remains at 1e-12.
