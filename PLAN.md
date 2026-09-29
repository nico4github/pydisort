# pydisort development plan

The authoritative Fortran-parity plan, current H100 validation status, and
definition of done for each capability live in
[`docs/source/fortran_parity.rst`](docs/source/fortran_parity.rst).

Start with explicit unsupported-capability guards, then add pseudo-spherical
geometry, general-source inputs, BRDFs, and special-boundary outputs. Each
capability requires CPU reference validation and H100 CUDA execution and
agreement before performance work begins.
## Validation outcomes

Validation reports use three exclusive outcomes: `PASS` for a completed
validated calculation, `SKIP` for a documented unsupported capability with an
explicit reason, and `FAIL` for an error or a failed numerical check in a
declared supported calculation. Summaries must count them separately.

The current C-DISORT-backed Fortran-parity backlog is **five capability
families / eight individual targets**. The authoritative ordered inventory is
the table in `docs/source/fortran_parity.rst`: general-source arrays,
Fourier-component output, four BRDF models, special-boundary output, and
`DELTAMPLUS`. Update both totals whenever a target moves status.

## Special-boundary (`ibcnd`) status — 2026-09-29

`Disort.medium_albedo_transmissivity(prop, albedo=None)` is a CPU-only typed
special-boundary calculation. It returns a `SpecialBoundaryResult` whose
`albedo` and `transmissivity` tensors have shape `(nwave, ncol, numu)`, with
positive incidence-angle cosines from `DisortOptions.user_mu()`. The ordinary
`forward()` contract remains separate and raw `ibcnd` configuration continues
to raise `NotImplementedError`.

The method allocates an isolated C-DISORT `SPECIAL_BC` state per solve; it does
not reuse ordinary flux/radiance buffers. Its dedicated pytest compares
Problem 13a/13c atmospheres against ordinary unit-flux beam solutions
(13b/13d), including batched wave/column inputs, several albedos, and two
incidence cosines at `1e-6` relative plus `1e-8` absolute tolerance. Invalid
cosines are rejected and CUDA requests raise `NotImplementedError`.

The standalone regression in `tests/cdisort213/test_cdisort_special_boundary.c`
passes 16 cases under AddressSanitizer/UndefinedBehaviorSanitizer. The
documented patch fixes undersized internally doubled angle buffers, an
overlapping angle reversal, and a legacy uvspec overwrite of the first beam
result. See `cdisort_patches/README.md` for provenance and reproduction.

The sibling bridge validates the Fortran v4 Problem 13 pairs: 13a/13b gives
`0.5452584/0.8449987` and 13c/13d gives `0.2762007/0.5033189` for
albedo/transmissivity at `1e-6` relative plus `1e-8` absolute tolerance.
The v4 driver does not print ALBMED/TRNMED for 13a/13c, so the paired unit-flux
beam cases provide their direct reference values. Remaining work is CPU/H100
CUDA agreement before special-boundary CUDA support can be declared. The
five-family, eight-target inventory is unchanged while the target is CPU-only.
