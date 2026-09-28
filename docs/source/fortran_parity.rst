Fortran parity plan
===================

This is the authoritative plan for extending the Python interface while
preserving numerical behaviour of the C-DISORT core and comparison with the
DISORT Fortran implementations in the sibling ``../disort-pyf`` checkout.
Each item is an additive, focused change. The implementation order follows
the project contract in ``AGENTS.md``.

Status at the H100 baseline
---------------------------

The validated baseline is plane-parallel DISORT with beam and isotropic
illumination, thermal emission, Lambertian reflection, user optical depths and
angles, batching, and explicit CPU/CUDA backend selection. Published reference
coverage includes Problems 1, 2, 3, 6a-c, 9, and 10. Problem 9b already
checks a tabulated phase function represented by explicit Legendre moments.
On 2026-09-28 the local H100 build passed 193 pytest tests and 21 CTest tests;
the CUDA selector and CPU-versus-CUDA suites also passed.

The C-DISORT source has implementations for pseudo-spherical direct-beam
geometry, general-source arrays, several BRDFs, and special-boundary
albedo/transmissivity. These are not complete Python capabilities yet:

* ``ibcnd`` reaches a generic runtime check rather than raising
  ``NotImplementedError``.
* ``spher``, ``general_source``, and ``output_uum`` can be parsed, but their
  required inputs or result accessors are not public.
* BRDF selection is absent from the Python API; the state uses
  ``BRDF_NONE``.
* There is no feature-specific H100 test for any of these gaps.

Work item 0: capability guards
------------------------------

**Status:** planned; first implementation change.

Before exposing a new feature, centralize validation of every currently
unsupported public flag and backend combination. Each must raise
``NotImplementedError`` with the feature name, the unsupported condition, and
the supported alternative. In particular, cover ``ibcnd``, ``spher``,
``general_source``, and ``output_uum``. Do not silently ignore a flag, choose
another backend, or substitute a plane-parallel or Lambertian calculation.

Validation:

* Add dedicated pytest cases for each rejected flag and unsupported
  CPU/CUDA combination.
* Run the full CPU suite and CTest.
* Run the same rejection tests on the H100 CUDA build to prove that GPU
  dispatch cannot bypass the guard.

Work item 1: pseudo-spherical direct beam
-----------------------------------------

**Status:** planned; first physical capability after the guard work.

Expose a narrow typed configuration for body radius and level altitudes, with
validation that the values have common units, contain ``nlyr + 1`` levels, and
have the ordering required by C-DISORT. Keep the result contract unchanged:
the solver continues to return the usual flux and radiance outputs.

The CPU reference must be a documented C-DISORT or Fortran pseudo-spherical
case. It must include a plane-parallel limiting case so an invalid geometry
cannot appear plausible. The H100 work is not complete until the same case
runs on CUDA and agrees with the CPU reference within a recorded tolerance.
If CUDA cannot support the configuration, its request must raise
``NotImplementedError`` until the CUDA implementation and agreement test are
added.

Work item 2: phase-function and general-source inputs
------------------------------------------------------

**Status:** tabulated Legendre moments are covered by Problem 9b; general
source arrays are planned.

Preserve the existing moment input. Define a typed general-source input with
unambiguous tensor shapes for computational and user angles, and validate it
against ``nstr``, ``nlyr``, ``numu``, and the spectral/column batch axes.
Document which combinations of user-angle and intensity output are supported.

Validation requires a C-DISORT or Fortran general-source reference, a zero
source reduction to the existing solution, shape and dtype rejection tests,
and CPU/H100 CUDA agreement on the same nonzero source.

Work item 3: BRDF surface models
--------------------------------

**Status:** planned.

Expose each supported model through a typed surface option rather than raw
state access. Start with a single C-DISORT model and its required parameters;
do not claim other models from their presence in C headers. The first chosen
model needs a Fortran/C-DISORT reference such as DISORT Problem 6d-h, an
explicit Lambertian limiting test where applicable, and CPU/H100 agreement.
Unsupported model names and parameter combinations must raise
``NotImplementedError``.

Work item 4: special-boundary calculations
-------------------------------------------

**Status:** planned.

``ibcnd`` changes the calculation and output contract: it returns medium
albedo and transmissivity rather than the normal flux/radiance result. Give it
its own method and typed result instead of overloading ``forward``. Validate
against the special-boundary C-DISORT/Fortran cases, include invalid
combinations such as user angles, and add CPU/H100 agreement before declaring
CUDA support.

Deferred items
--------------

``output_uum`` needs a separate Fourier-component result design after its
input and indexing contract are documented. ``DELTAMPLUS`` requires
solver-core work or a documented C-DISORT upgrade or patch; it is not a hidden
binding to expose. Neither may be enabled through a silent fallback.

Definition of done for every capability
----------------------------------------

1. Add a small public option or typed input/result, update ``python/pydisort.pyi``
   and user documentation, and reject all unsupported combinations explicitly.
2. Add dedicated CPU pytest coverage based on a published DISORT reference,
   an analytic limit, conservation law, or direct Fortran comparison.
3. On the H100, run the same capability on CPU and CUDA, synchronize CUDA
   timing where measured, and assert numerical agreement with stated
   tolerances. A GPU limitation remains an explicit ``NotImplementedError``;
   it is not a CUDA-supported feature.
4. Run the focused tests, ``pytest tests/ -v -rs``,
   ``ctest --test-dir build --output-on-failure``, and the relevant
   documentation and pre-commit checks. Record the reference source, H100
   device, Torch/CUDA versions, tolerance, and result in the change.

Performance phase: host--H100 transfers
----------------------------------------

Do this only after the relevant capability has passed its CPU, CUDA, and
Fortran/reference checks. Measure end-to-end time separately from solver time
for representative wavelength/column batches, recording device, dtype,
stream count, batch shape, transfer direction, synchronization points, and
warm-up policy. Then evaluate persistent device inputs, batching strategy,
and overlap of transfers with computation without changing numerical results
or implicit device placement.

Status reporting
----------------

Update this page in the same commit as each capability: mark its status, name
the reference case and tolerances, state CPU and H100 CUDA results, and list
any remaining unsupported combinations. Do not mark an item complete from a
successful build alone.
