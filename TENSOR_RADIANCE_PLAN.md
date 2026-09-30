# Tensor radiance reconstruction plan

This is the parity-first expansion from the completed tensor flux path to
user-angle and azimuth-dependent radiance. It is deliberately staged: each
connected physical result receives a native C-DISORT fixture and pytest gate
before the next source or Fourier feature is added.

## Scope and order

1. **Native trace baseline — complete.** `PYDISORT_TRACE_INTERP` records C
   `GU` after `c_interp_eigenvec`; the Makefile now copies the real extension
   artifact before installation. The trace confirmed the tensor `GC` matrix
   differs only by per-column eigenvector scaling.
2. **Homogeneous user-angle ray path — in progress.** The one-layer,
   plane-parallel m=0 C-DISORT ray integral and diffuse top-boundary term pass
   CPU/CUDA parity at `mu=(-0.5, 0.5)` and three output depths. The next
   accepted increment is a two-layer, source-free C-DISORT fixture covering
   complete traversed layers and the partial target layer. It will retain a
   black lower boundary; beam, thermal, and surface reflection remain outside
   this increment.
3. **User-angle particular sources.** Add beam, thermal, and Fourier-zero
   general-source interpolation one source at a time. Each adds a source-only
   native fixture before a combined-source fixture.
4. **Fourier m>0.** Generalize the eigen/source/boundary flow by Fourier order,
   reconstruct radiance over phi, and validate TP4c before calling azimuthal
   radiance supported.
5. **Hapke.** Freeze the existing C parameters in self-contained fixtures,
   add its Fourier surface coupling, and validate CPU/CUDA radiance. RPV, CAM,
   and AMB remain deferred.

## Validation and rebuild rules

- CPU and CUDA tests run for every accepted tensor increment. The standard
  gate is the stored native float64 result; any tolerance records the measured
  independent-solver envelope.
- Trace all required C intermediates in one batch before `make rebuild`.
  Current interpolation tracing is complete, so step 2 requires no rebuild.
- Do not commit a partial public capability. Commit a trace/reproducibility
  change separately from a passing tensor feature.
- Run `ruff`, `mypy`, focused pytest, `make install`, full pytest, then commit.

## Current state

Step 1 is complete in commit `2edbdde`. The accepted step-2 increments now
cover exact one-layer and two-layer homogeneous m=0 ray integration with the
attenuated diffuse top boundary, tested against native C-DISORT references on
CPU and CUDA. The next step-2 evidence gate is a five-layer, nonuniform,
anisotropic source-free fixture with both symmetric and asymmetric user angles.
Its native baseline is emitted by the dedicated diagnostic driver, rather than
a wrapper result. A retained native trace has located the current mismatch
upstream of ray integration in the interpolated-eigenvector/solved-constant
product. Compare `GC` and the boundary-system solution by layer, correct that
mapping, and only then promote the five-layer fixture to a required CPU/CUDA
pytest gate. Source and Fourier stages remain after this gate.
