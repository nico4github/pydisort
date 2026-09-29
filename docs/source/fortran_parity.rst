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
On 2026-09-28 the local H100 build passed 215 pytest tests and 23 CTest tests;
the CUDA selector and CPU-versus-CUDA suites also passed.

The C-DISORT source has implementations for pseudo-spherical direct-beam
geometry, general-source arrays, several BRDFs, and special-boundary
albedo/transmissivity. These are not complete Python capabilities yet:

* ``ibcnd`` and ``output_uum`` have no typed Python inputs or result
  accessors yet. Raw configuration raises ``NotImplementedError``.
* ``general_source`` has a typed CPU-only input. CUDA requests raise
  ``NotImplementedError`` until source arrays are implemented and validated
  in the CUDA dispatch path.
* Pseudo-spherical direct-beam geometry is supported through the typed
  ``pseudo_spherical`` option. Raw ``spher`` configuration raises
  ``NotImplementedError`` so it cannot appear to work without its required
  radius and level-altitude inputs.
* BRDF selection is absent from the Python API; the state uses ``BRDF_NONE``.

Tracked remaining capability count and priority
-----------------------------------------------

There are **five remaining capability families**, comprising **eight
individually countable targets**. The BRDF family is counted as four targets
because C-DISORT exposes four distinct models. This is the current tracked
scope for C-DISORT-backed Fortran parity; it does not claim parity with
unported solver-core changes from other DISORT versions.

.. list-table:: Ordered parity backlog
   :header-rows: 1
   :widths: 8 24 14 54

   * - Priority
     - Target
     - Status
     - Why it comes next and required evidence
   * - 1
     - General source arrays
     - CPU-only
     - This is the smallest missing source-term input already implemented by
       C-DISORT. Add a typed input, shape validation, a zero-source reduction,
       a direct C-DISORT/Fortran nonzero reference, and CPU/H100 CUDA agreement.
   * - 2
     - Fourier-component output (``output_uum``)
     - CPU-only
     - A typed ``fourier_components()`` option and ``gather_fourier()`` accessor
       expose the C-DISORT buffer on CPU. CUDA requests explicitly raise
       ``NotImplementedError`` until result transfer and CPU/H100 agreement are
       implemented.
   * - 3
     - RPV BRDF
     - Unsupported
     - First non-Lambertian lower-boundary model. Expose typed parameters and
       validate against its C-DISORT/Fortran reference and the Lambertian limit.
   * - 4
     - CAM BRDF
     - Unsupported
     - Add only after the shared typed BRDF option and RPV test harness exist.
   * - 5
     - AMB BRDF
     - Unsupported
     - Add only after the shared typed BRDF option and RPV test harness exist.
   * - 6
     - Hapke BRDF
     - CPU-only
     - ``hapke_brdf()`` exposes C-DISORT's fixed-parameter Hapke model
       (``B0=1``, ``HH=0.06``, ``W=0.6``). Its Problem 6d direct reference
       passes on CPU; CUDA requests explicitly raise ``NotImplementedError``.
   * - 7
     - Special boundary (``ibcnd``)
     - Unsupported
     - It changes outputs to albedo/transmissivity, so it needs a separate
       result type after ordinary flux/radiance and surface contracts are firm.
   * - 8
     - ``DELTAMPLUS``
     - Deferred
     - This requires solver-core work or a documented C-DISORT upgrade/patch;
       it is intentionally last and must not be emulated in the wrapper.

The count is reduced only when a target has a public typed contract, a
dedicated reference-based pytest, and CPU/H100 CUDA agreement. A guarded
``NotImplementedError`` remains unsupported and is still included in this
count. Update this table and its five-family/eight-target totals in the same
commit that changes a target's status.


Work item 0: capability guards
------------------------------

**Status:** complete. Construction and dispatch reject each listed flag before
CPU or CUDA work; the dedicated test passes all 16 CPU/CUDA construction and
dispatch cases on the H100, checking the built-in exception and an actionable
alternative.

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

**Status:** complete. ``DisortOptions.pseudo_spherical(radius,
level_altitudes)`` maps the radius and strictly top-to-bottom levels to the
C-DISORT state without exposing mutable raw state. The raw ``spher`` flag still
raises ``NotImplementedError``. The direct-beam C-DISORT reference is a clear
100 km layer over a 6371 km body with optical depth 1 and ``umu0=0.1``: bottom
downward flux is ``4.4428045259949897e-05``. Its large-radius limit agrees
with plane parallel at ``2e-6`` relative tolerance. On the H100, CPU and CUDA
agree at ``1e-10`` relative and ``1e-12`` absolute tolerances.

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

**Status:** tabulated Legendre moments are covered by Problem 9b. General
source arrays are CPU-only: ``DisortOptions.general_source(computational,
user)`` accepts CPU ``float64`` tensors with shapes ``(nwave, ncol, nstr,
nlyr, nstr)`` and ``(nwave, ncol, nstr, nlyr, numu)``. CUDA requests raise
``NotImplementedError`` until the CUDA backend has equivalent source-term
support. The CPU reference is the bundled C-DISORT general-source calculation
with one layer, ``nstr=4``, ``dtauc=0.2``, ``ssalb=0.5``, and unit Fourier-zero
source arrays: its flux output is ``[[[[1.0891838249147526, 0.0], [0.0,
1.0891838249147526]]]]``. Zero arrays reduce to the existing no-source
solution. A matching Fortran general-source case has not yet been added to the
sibling reference driver.

Preserve the existing moment input. Define a typed general-source input with
unambiguous tensor shapes for computational and user angles, and validate it
against ``nstr``, ``nlyr``, ``numu``, and the spectral/column batch axes.
Document which combinations of user-angle and intensity output are supported.

Validation requires a C-DISORT or Fortran general-source reference, a zero
source reduction to the existing solution, shape and dtype rejection tests,
and CPU/H100 CUDA agreement on the same nonzero source.

Work item 3: BRDF surface models
--------------------------------

**Status:** Hapke is CPU-only. ``DisortOptions.hapke_brdf()`` selects the
fixed C-DISORT Hapke model without exposing raw state or accepting a Fortran
``brdf_type`` integer. The CPU reference is C-DISORT Problem 6d: direct flux
is ``[100.0, 36.7879, 13.5335]`` and upward flux is
``[0.670783, 1.39084, 3.31655]`` at optical depths ``[0, 0.5, 1]``. The
published values are rounded, so the dedicated test uses ``5e-6`` relative
tolerance. CUDA requests raise ``NotImplementedError`` pending a CUDA
implementation and CPU/H100 agreement.

Expose each later model through a typed surface option rather than raw state
access. Do not claim other models from their presence in C headers.
Unsupported model names and parameter combinations must raise
``NotImplementedError``.

Work item 4: special-boundary calculations
-------------------------------------------

**Status:** unsupported after a CPU prototype investigation on 2026-09-29.

A typed prototype produced Problem 13a's intermediate ``albedo=0.0378`` and
``transmissivity=0.9425`` at ``mu=0.5``, but normal ``forward()`` cleanup
failed with ``free(): invalid pointer`` and an isolated direct C-DISORT route
segfaulted. The prototype was removed. A standalone C reproduction of the
C-DISORT Problem 13 allocation/run/free sequence must pass before this API is
reintroduced.

``ibcnd`` changes the calculation and output contract: it returns medium
albedo and transmissivity rather than the normal flux/radiance result. Give it
its own method and typed result instead of overloading ``forward``. Validate
against the special-boundary C-DISORT/Fortran cases, include invalid
combinations such as user angles, and add CPU/H100 agreement before declaring
CUDA support.

Deferred items
--------------

``output_uum`` is CPU-only pending CUDA result transfer and H100 agreement.
``DELTAMPLUS`` requires
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
successful build alone. Validation reports use exclusive outcomes: ``PASS``
for a completed validated calculation, ``SKIP`` only for a documented
unsupported capability with its reason, and ``FAIL`` for an error or numerical
check failure in a declared supported calculation. Summaries must count each
outcome separately.
