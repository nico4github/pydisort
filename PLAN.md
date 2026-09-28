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
