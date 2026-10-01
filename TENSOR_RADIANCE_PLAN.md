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

Step 1 is complete in commit `2edbdde`. Step 2 is complete for
source-free m=0 user rays: one-, two-, and five-layer native C-DISORT fixtures
pass on CPU and CUDA. The five-layer gate uses nonuniform optical thicknesses,
single-scattering albedos, Henyey-Greenstein moments, and both symmetric and
asymmetric user angles. It also verifies the automatic delta-M selection used
by C-DISORT when `PMOM(nstr)` is nonzero.

Step 3 has accepted its beam-only m=0 user-angle fixture on CPU and CUDA.
The next source-only increments are thermal, then general source. Each fixture must be emitted by a dedicated native diagnostic with its
input contract stored alongside the values before a combined-source gate is
introduced. Fourier m>0 work remains after those source gates.

## Progress update (2026-10-01)

Steps 2 and 3 are complete: five-layer m=0 source-free, beam, thermal,
general-source, combined, and Lambertian user-ray fixtures all pass on CPU and
H100 CUDA. Step 4 has completed the component-level reconstruction for the
beam-only five-layer diagnostic: m=0 through m=3 each have native float64
fixtures and CPU/CUDA pytest gates. The active increment is final cosine
summation at arbitrary azimuth using the retained native final-radiance trace;
only after that gate passes will the implementation move to an
azimuth-dependent published reference case.
