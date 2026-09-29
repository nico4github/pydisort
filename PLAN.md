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

## Special-boundary (`ibcnd`) investigation — 2026-09-29

`ibcnd=1` remains an explicit unsupported capability. It is not compatible with
pydisort's ordinary `forward()` contract: C-DISORT returns medium albedo and
transmissivity by user cosine rather than the usual flux/radiance grids.

An attempted CPU-only typed prototype configured C-DISORT's special-boundary
state and read its `albmed`/`trnmed` buffers. For the one-layer Problem 13a
configuration at `mu=0.5`, C-DISORT computed the plausible intermediate values
`albedo=0.0378` and `transmissivity=0.9425`. Two execution routes were then
rejected:

- Calling the normal `forward()` route caused `free(): invalid pointer` during
  cleanup. Its ordinary radiance/flux buffers do not share the special-boundary
  allocation layout.
- A direct C-DISORT call that bypassed `forward()` still segfaulted before it
  could safely return. The prototype was removed and the stable CMake library
  rebuilt; no public API or bridge support was retained.

Next step: create a minimal standalone C reproduction of C-DISORT Problem 13a
using this fork's exact `c_disort_state_alloc`, `c_disort_out_alloc`,
`c_disort`, and free sequence. Compare it with the upstream self-test's
Problem 13 setup to identify the required state flags and allocation invariants.
Only after that program exits cleanly and matches 13a/13c should pydisort add a
typed CPU API, dedicated pytest coverage, bridge integration, and an explicit
CUDA `NotImplementedError` guard.
