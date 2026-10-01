# Development Instructions

## Mission and scope

This fork extends pydisort toward a documented, reproducible comparison with
the DISORT Fortran implementations in the sibling `../disort-pyf` checkout.
Preserve pydisort's existing plane-parallel API and numerical behaviour while
adding missing cdisort capabilities through explicit, backward-compatible APIs.

Treat the existing public API as stable. Do not expose a mutable raw
`disort_state` to Python: its allocations and pointer ownership depend on the
configured solver dimensions and flags. New configuration belongs in validated
`DisortOptions` methods or narrow, typed option objects; a feature with a
different result contract gets its own method and result type.

Public interface contract: an unsupported configuration, backend, dtype, or
feature combination must raise `NotImplementedError` with an actionable
message. Never silently select another backend, drop inputs, approximate a
missing feature, or change the calculation. A bridge or validation report must
separate `PASS`, `SKIP`, and `FAIL`: `SKIP` is only for a documented unsupported
capability with its reason, while a supported calculation error is `FAIL`.
Every additive capability requires
a dedicated pytest that proves its stated contract; use a published DISORT
reference, analytic limit, conservation law, or direct Fortran comparison when
one is available.
Do not claim Apple MPS acceleration. The project has a CUDA implementation;
Apple Silicon support currently means native CPU wheels and CPU batching.

## Parity work

Read `TENSOR_RADIANCE_PLAN.md` and `TENSOR_PARITY_FINDINGS.md` before
continuing tensor reconstruction work. Update the plan with the active
milestone and acceptance boundary, and update the findings with accepted
results, rejected hypotheses, measured gaps, and the next evidence-based
investigation step in the same commit as the related parity change.

For source-free m=0 user-angle radiance, establish the nonuniform two-layer
C-DISORT fixture first. After it passes on CPU and CUDA, add and pass a
nonuniform five-layer fixture before adding beam, thermal, general-source, or
surface terms to the user-angle path.

Every user-angle reference test must exercise both the symmetric direction
pair `(-0.5, 0.5)` and the asymmetric pair `(-0.3, 0.4)`. Apply this to every
new and existing radiance fixture as it is touched, and require CPU/H100 CUDA
parity for both pairs. Do not use mirrored-angle coverage as the sole evidence
for either sign branch.

Every azimuth-dependent reference or diagnostic must include at least one
nontrivial azimuth. Retain published angles where applicable, and add values
such as 12, 36, 80, or 100 degrees; 0, 90, and 180 degrees alone are not
sufficient evidence for Fourier reconstruction.

Reference tests and their published inputs are immutable baselines. Never
modify a reference test case for debugging, tracing, benchmarking, or a new
parity scenario. Add a separate, clearly named diagnostic or regression test
with its own fixture instead.

Before implementing a missing feature, inspect the relevant Fortran input
flag/output convention in `../disort-pyf`, the cdisort 2.1.3 state and test
driver, current pydisort wrapper/bindings, and neighbouring tests. Record the
feature in `docs/source/fortran_parity.rst` when parity work begins, including:

- Fortran name and supported input/output modes.
- cdisort availability and any semantic difference.
- pydisort status: unsupported, CPU-only, CUDA-supported, or intentionally
  out of scope.
- Reference test case, tolerances, and unsupported combinations.

Use a separate additive change for each capability, in this order unless the
task specifies otherwise:

1. Pseudo-spherical geometry: validated body radius and level altitudes.
2. Tabulated phase functions and general-source arrays.
3. BRDF surface models.
4. Special-boundary (`ibcnd`) calculations, with a dedicated result contract.

`DELTAMPLUS` is not a hidden cdisort binding. It requires solver-core work or
an intentional cdisort upgrade/port, with its own design note and references.

Never silently alter `cdisort213/`. Preserve upstream provenance and represent
any required cdisort change as a documented patch, following the repository's
`cdisort_patches` practice. Keep wrapper, binding, and core-solver changes
clearly separated in the commit history.

## Clean implementation loop

Before editing, inspect the relevant files, tests, examples, public type stub,
and project conventions. Reuse existing abstractions; do not create a helper,
class, dependency, or layer of indirection unless it removes meaningful
duplication or complexity.

For every change:

1. Implement the smallest behaviourally complete change.
2. Add or update a regression test. Prefer a published DISORT reference,
   analytic limit, conservation law, or a direct Fortran comparison over a
   snapshot of current output.
3. Run the relevant checks.
4. Review the diff as a strict senior engineer for duplication, dead code,
   needless abstractions, overly complex control flow, poor names,
   inconsistent patterns, needless dependencies, and swallowed errors.
5. Simplify without changing behaviour, then rerun the affected checks.

Match surrounding style and preserve readable layout. Do not reformat unrelated
files. Keep errors precise: reject unsupported flag/device/dtype combinations
with an actionable message rather than silently falling back or changing the
calculation.

## CUDA and H100 work

The H100 target is CUDA compute capability 9.0. Build CUDA explicitly and
confirm the PyTorch and extension CUDA versions agree before benchmarking:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
  -DCUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --parallel
python -m pip install --no-build-isolation .
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Do not mark a new feature CUDA-supported until it has CPU reference coverage
and a CPU-versus-CUDA agreement test on the H100. Keep unsupported features
CPU-only with an explicit error; do not copy MPS tensors to CPU implicitly.
Benchmark only after numerical agreement passes, and report device, PyTorch,
CUDA, compiler, stream count, batch shape, precision, timings, and validation
tolerances.

## Required validation

Use one isolated environment for CMake and pip so the compiled extension links
against the same `torch`. CMake builds the C++ library; `pip install` does not
replace that step.

```bash
python -m pip install pytest pre-commit
pre-commit install
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build --parallel
python -m pip install --no-build-isolation .
python -m pytest tests/ -v -rs
ctest --test-dir build --output-on-failure
pre-commit run --all-files
git diff --check
```

Run the relevant focused test while iterating, then the full commands above
before handoff when feasible. Pre-commit formats/lints only; it never replaces
solver tests. Reconfigure CMake after adding a Python test so CTest registers
and copies it.

For public API changes, update `python/pydisort.pyi` and the applicable Sphinx
pages. Validate documentation with:

```bash
python -m pip install -r docs/requirements.txt
python -m unittest discover -s docs/_ext -p 'test_*.py'
python -m sphinx -E -b html -W --keep-going docs/source docs/_build/html
python -m sphinx -b doctest -W docs/source docs/_build/doctest
```

Run benchmark tooling only when it changes:

```bash
python -m pip install PythonicDISORT threadpoolctl
python -m pytest benchmarks/tests/ -v -rs
```

## Git and handoff

Use a focused branch, stage only task-relevant files, inspect the final diff,
and make a focused commit once the work is ready. In the handoff, state the
feature contract, Fortran/cdisort comparison basis, exact commands and results,
and every check that was skipped with its reason.

## Counted parity inventory

Maintain the counted C-DISORT-backed Fortran-parity inventory in
``docs/source/fortran_parity.rst``. Update its family and target totals, the
target status, reference basis, and CPU/H100 CUDA result in the same commit as
each parity change. A guarded unsupported target remains in the count.

## Approved-plan continuity

When the user has approved a plan, execute its next unblocked task immediately
after every edit, test, benchmark, report update, and commit. A routine commit
or successful check is not a handoff point. Do not end a turn to report
intermediate progress. End only for a completed approved objective, a material
decision outside the plan, or a genuine blocker that cannot be resolved from
repository evidence. Keep the plan checklist current and commit ready
increments without interrupting execution.

## Persistent parity divergence

Do a bounded source-level audit first. If a C-DISORT/tensor parity divergence
remains unresolved after that audit, add a narrowly scoped, opt-in CPU trace to
the relevant C stage before making further speculative changes. Record the
trace in `cdisort_patches/`, document it in `TENSOR_PARITY_FINDINGS.md`, and
prepare all trace points before one rebuild. Compare the traced C intermediate
values with the tensor stage. Retain the opt-in diagnostic and its patch after
the investigation: it is dormant outside its environment variable and avoids
repeating a future trace edit/rebuild. Do not rebuild repeatedly for
uninstrumented guesses.

## Local extension rebuild

For any C-DISORT header, C++, CUDA, or binding change, use `make rebuild`.
It is the authoritative local development workflow: it builds the CMake
libraries, rebuilds the inline C-DISORT Python extension with the venv's
Ninja executable on `PATH`, installs the matching artifacts into the same
venv, and verifies the imported package location. Do not test a partial
shared-library copy: the binding contains inline solver specializations.
