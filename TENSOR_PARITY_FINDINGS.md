# Tensor parity findings

This log records evidence from the parity-first PyTorch reconstruction. Update
it whenever an experiment changes the supported tensor contract, narrows an
active divergence, or rejects a plausible implementation path. The detailed
plan and milestone status remain in `CUDA_BACKEND_STRATEGY.md`.

## Current state

- The focused tensor suite has 40 passing tests and four strict expected
  failures. The failures are the CPU and CUDA variants of the Problem 9c
  beam-only and thermal-only source decomposition tests.
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
- Diffuse-only flux is exact to numerical rounding for all five output depths.
- The direct transmitted beam flux is exact. The remaining beam-only error is
  in the diffuse beam particular solution, not direct attenuation.

### Incident beam angle

- `ae4d3af` corrected the beam Legendre argument to `-umu0`, matching
  C-DISORT's call to `c_legendre_poly` for the incident direction.
- In the six-layer, black-surface, anisotropic beam probe, this reduced the
  top upward-flux error from about `+5.84e-2` to `-6.95e-4`.

## Active divergence

The active probe is Problem 9c with thermal sources disabled, diffuse top
illumination disabled, and a black lower surface. Native C-DISORT and the
tensor reconstruction currently give:

| Output optical depth | C-DISORT upward | Tensor upward | C-DISORT diffuse downward | Tensor diffuse downward |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.351301568 | 0.350606379 | 0.000000000 | 0.000000000 |
| 1.05 | 0.100917382 | 0.100367431 | 0.463211326 | 0.455737311 |
| 2.10 | 0.035832362 | 0.034779043 | 0.176581240 | 0.171175381 |
| 6.00 | 0.002268712 | 0.002192343 | 0.011961739 | 0.011564688 |
| 21.0 | ~0 | ~0 | 0.000055681 | 0.000053873 |

The thermal-only expected failure shares the remaining lower-boundary/source
composition gap. Do not relax the fixture tolerance or convert either test to
a normal pass until the C-DISORT source equation is reproduced.

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

Trace `c_upbeam` term by term using the established black-surface beam probe:

1. Compare the C `ZJ` right-hand side constructed from `GL`, `YLMC`, and
   `YLM0` against the tensor source before solving.
2. Compare the solved particular vectors before the `ZZ` storage transform.
3. Correct the first divergent term, promote the beam fixture from strict
   expected-fail to required CPU/CUDA parity, then repeat for thermal-only.
4. Only then add the full Problem 9c flux fixture.

## Apple MPS note

The tensor layout is device-generic, but the parity path deliberately uses
float64. PyTorch MPS cannot allocate float64 tensors, so Apple GPU support
requires a separate float32 execution profile and tolerance suite after the
float64 parity gates pass. It is not a second solver rewrite.
