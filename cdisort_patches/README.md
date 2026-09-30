# C-DISORT patches

`0001-special-boundary.patch` is already applied to the split C-DISORT 2.1.3
headers. Its base is pydisort commit `b122afb`; do not apply it twice.

The standalone Problem 13 investigation found three defects:

- `c_disort` doubles user angles after output allocation. Allocate `uu`,
  `u0u`, optional `uum`, `albmed`, and `trnmed` for the internal angle count.
  AddressSanitizer reproduced an out-of-bounds `memset` of `albmed` with two
  incident angles; one angle hides it because allocations include an extra slot.
- The negative-angle reversal read the same lower half that it overwrote.
  Read the saved positive angles in the upper half instead.
- A legacy uvspec customization overwrote the first beam result with spherical
  albedo/transmissivity, which were uninitialized for a black surface. Remove
  that overwrite to honor the documented `albmed`/`trnmed` beam-angle contract.
  Consumers relying on that undocumented spherical output must adapt.

No new Python capability or CUDA support is declared. No Fortran source changes
are involved. The regression uses C-DISORT Problem 13a/13c atmospheres and
compares with ordinary beam solutions (13b/13d), with unit incident flux and
`1e-6` relative plus `1e-8` absolute tolerance. It covers one/two layers,
one/three incidence angles, zero/two azimuths, black/reflecting surfaces, and
repeated calls with the same state. The sibling Fortran fixtures use different
angle/output conventions and are not claimed as direct validation here.

Run the standalone regression without Torch or the Python extension:

```bash
c++ -std=c++17 -g -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer -Icdisort213 \
  tests/cdisort213/test_cdisort_special_boundary.c -o /tmp/test_special_boundary
/tmp/test_special_boundary
```

The `.c` driver is compiled as C++ because the fork's solver is header-only C++.
It is also registered with CTest as `test_cdisort_special_boundary.release`.

## 0002-upbeam-trace.patch

This applied diagnostic patch adds an opt-in CPU C-DISORT entry trace and a trace around `c_upbeam`.
Set `PYDISORT_TRACE_UPBEAM=1` before a small CPU C-DISORT run to emit the
the C-DISORT call configuration, per-layer `YLM0`, pre-solve `ZJ` right-hand side,
and post-solve `ZJ` vector to standard error. It is compiled out of CUDA device
code and is inactive unless
the environment variable is set. The tensor reconstruction uses this trace to
compare the first divergent beam-source quantity without repeated speculative
solver rebuilds.
