# CUDA benchmark and optimization plan

This plan defines the performance work for the H100 after numerical parity is
established. It complements [`PLAN.md`](PLAN.md): that file orders missing
DISORT capabilities; this file defines how supported CPU and CUDA paths are
measured, compared, and optimized without weakening numerical checks.

The current CUDA implementation runs one independent `(wavelength, column)`
DISORT solve per GPU thread. It has two specialized flux-only kernels for 4
and 8 streams, a general kernel for other supported configurations, and a
per-`Disort` CUDA workspace cache. The dispatch currently creates and copies
per-call device arrays for wavenumber bounds and user grids in
`disort/disort_dispatch.cu`. That is a likely small-batch cost, but it is only
a hypothesis until the measurements below separate it from kernel execution,
input transfer, and output transfer.

## Rules and definition of a valid result

1. Every benchmark case is defined once in code and has a stable identifier,
   physical inputs, requested outputs, dtype, and a Fortran v4 DP reference
   fixture. The legacy Fortran result is the physical source of truth.
2. A result has four independent checks: Fortran v4 DP CPU steady-state time,
   pydisort CPU steady-state time, pydisort GPU steady-state throughput, and
   pydisort end-to-end application time. Do not quote one as another.
3. Timed CUDA regions use CUDA events on the active stream and synchronize
   only at the measurement boundary. CPU regions use a monotonic wall clock.
   Include warm-ups and report the median plus minimum/maximum of at least five
   measured repetitions.
4. Steady-state GPU measurements keep `prop`, boundary conditions,
   temperatures, and output on CUDA. End-to-end measurements begin with CPU
   inputs and end after the required result is on CPU. Setup and first-call
   measurements are reported separately.
5. CPU and CUDA use `float64` until a separately designed precision mode has a
   documented physical error budget. There is no implicit lower-precision
   speed path.
6. A supported case must first agree with its Fortran v4 DP reference. CUDA
   must then agree with the corresponding CPU result before its timing is
   retained. Existing fast 4/8-stream cases use their dedicated tolerance;
   ordinary shared-algorithm cases use the tighter CPU/CUDA tolerance. Each
   new case records maximum absolute error, relative RMS error for upward,
   downward, and net flux, and the agreed thresholds.
7. A failure, non-finite value, or unsupported configuration is never a speed
   result. Report it respectively as `FAIL` or `SKIP` with the exact reason.
   Unsupported CPU-only features must continue to raise `NotImplementedError`
   on a CUDA request.

## Reproducible harness and stored evidence

Add a single `benchmarks/compare_backends.py` harness, a Fortran-reference
adapter, and a small test module. The harness will construct all workloads
from named factories, obtain the Fortran v4 DP reference before timing, then
perform CPU/CUDA agreement checks, emit JSON Lines, and optionally write one
human-readable summary. It must provide these timing modes:

| Mode | Included work | Purpose |
| --- | --- | --- |
| `fortran_cpu_steady` | legacy Fortran v4 DP solve loop timed inside its benchmark driver | authoritative CPU baseline, excluding process startup and file output |
| `python_cpu_steady` | pydisort `forward()` with CPU-resident tensors | Python/C++ CPU throughput and thread scaling |
| `python_gpu_steady` | CUDA-resident pydisort `forward()` measured with CUDA events | CUDA kernel plus dispatch throughput |
| `gpu_first_call` | construction and first CUDA `forward()` | allocation/cache warm-up cost |
| `gpu_h2d` | pinned host-to-device input transfer only | transfer bandwidth and staging overhead |
| `gpu_d2h` | required output transfer only | result-return bandwidth |
| `gpu_end_to_end` | host inputs through host result | application-visible latency |

The harness must expose `--case`, `--nwave`, `--ncol`, `--nlyr`, `--nstr`,
`--repeat`, `--warmup`, `--threads`, `--timing-mode`, `--fortran-root`, and
`--output`. `--fortran-root` defaults to the sibling
`$HOME/disort-pyf` checkout. It must record the git revision and dirty state
of both repositories; Fortran executable and output-log hash; host name; CPU
model and thread count; GPU model, memory, driver, and compute capability;
PyTorch version and CUDA runtime; CMake CUDA compiler and architecture; dtype;
stream count; all shape parameters; timing samples; and accuracy metrics.

Store reviewed H100 records in
`benchmarks/results/<hostname>/`, one JSONL file per run plus a concise
checked-in Markdown summary. The hostname directory keeps runs from different
servers separate. Raw profiler traces, temporary CMake products, and repeated
unreviewed runs stay untracked. A result file is never copied after the run:
the harness writes directly to the requested host directory.

The baseline command must be runnable with the sole development environment:

```bash
/home/ngorius/pydisort/.venv/bin/python \
  benchmarks/compare_backends.py --case tp9_flux --output \
  benchmarks/results/$(hostname)/tp9_flux.jsonl
```

The harness test suite uses small shapes and CPU-only structural checks. The
H100 benchmark matrix remains a manual, recorded gate; it does not become a
slow CI job.

### Fortran-reference adapter

The adapter must run the sibling bridge's double-precision legacy reference,
not a pydisort result and not a hand-copied table. It first ensures the bridge
has generated `reports/si/disotest/disotest_v4_dp.out.txt` and its per-case
`*_v4_dp.log` records using:

```bash
cd /home/ngorius/disort-pyf
DISORT_REPORT_DIR=reports ./make_run_all_fortran.sh
```

Each benchmark factory declares a canonical Fortran fixture: its DISOTEST case
identifier, required input fields, output fields, and tolerance. The adapter
extracts the v4 DP input dump and output metrics, verifies that the benchmark's
base CPU inputs match that dump, and compares the base CPU result to those
Fortran outputs. It writes the selected case name and hashes to the benchmark
record. A missing, incomplete, or nonzero Fortran run is `FAIL`, never an
assumed reference.

For each fixture, add a companion non-legacy Fortran timing driver in
`disort-pyf`. It calls the unmodified v4 DP solver repeatedly with the same
validated inputs, measures only the solve loop with `system_clock` or
`cpu_time`, and prints elapsed seconds, solve count, and flux/radiance checksum
in machine-readable form. Configuration, allocation policy, warm-up count, and
repetition count must be recorded. The driver emits no diagnostic files inside
the timed loop. Its final checksum is compared to the ordinary v4 DP run, so a
fast loop cannot silently omit a source term or output calculation.

The three-way steady-state table reports the same scalar physics in:

| Label | Work compared |
| --- | --- |
| `fortran_cpu_steady` | v4 DP Fortran solve loop, one independent column/wavelength at a time |
| `python_cpu_steady` | pydisort CPU `forward()` for the identical scalar fixture, plus its batched sweep |
| `python_gpu_steady` | pydisort CUDA `forward()` for the identical scalar fixture and batches, with device-resident tensors |

The table reports seconds per scalar solve and solved `(wavelength, column)`
elements per second. Fortran is deliberately not charged Python process startup
or file logging; pydisort is likewise timed after construction and warm-up.
Pydisort's batch rows additionally show speed-up over the Fortran scalar
per-solve baseline, labelled as a throughput comparison rather than a claim
that Fortran supports the same batch interface.

The large `(nwave, ncol)` timing sweep repeats or combines independent copies
of that validated scalar fixture. Legacy Fortran is scalar, so it is run on the
base fixture rather than included in the throughput timing loop. Before timing
a batch, the harness verifies pydisort CPU batch output against independent
CPU scalar solves; CUDA then compares to that already Fortran-validated CPU
batch. This preserves the Fortran source of truth while measuring the batching
that pydisort adds.

If a challenging configuration has no existing v4 DP DISOTEST fixture, add a
separate non-legacy reference driver in `disort-pyf` that calls the unmodified
v4 DP solver and emits the same parseable dump/metrics. Add its Fortran run and
Python comparison test before adding the configuration to a performance chart.
Do not edit legacy Fortran sources merely to create a benchmark.

## Benchmark cases

The set deliberately spans cheap latency-sensitive calls, throughput-sized
batches, physical source terms, output costs, and numerical corner cases. A
case is not added merely to create a favorable speed-up.

| ID | Configuration | Shape sweep | Required comparison and purpose |
| --- | --- | --- | --- |
| `latency_flux_4` | Plane-parallel shortwave, Lambertian, `onlyfl`, 4 streams, ordinary scattering | `(nwave, ncol)=(1,1),(8,1),(32,4)`; 20 and 100 layers | CPU/CUDA flux agreement; exposes launch, state-copy, and allocation costs. |
| `latency_flux_8` | Same, 8 streams | Same | Exercises the other specialized CUDA kernel. |
| `tp9_flux` | Existing Test Problem 9-derived inhomogeneous scattering, flux only | `nwave=1,8,64,256,1024`; `ncol=1,4,16`; 32/100 layers; 8/16/32 streams | Fortran v4 DP base fixture first, C-DISORT as wrapper-overhead diagnostic, then CPU/CUDA agreement; primary realistic throughput curve. |
| `tp9_radiance_cpu` | Test Problem 9 with user tau, mu, and phi | Same spectral/column sweep, 8/16/32 streams | Fortran v4 DP base fixture first, then CPU/C-DISORT. CUDA currently returns flux but not radiance access, so record a documented `SKIP` for CUDA result retrieval rather than inventing a proxy. |
| `shortwave_beam` | Direct beam, oblique solar angles `umu0=0.1,0.5,0.9`, dark and bright Lambertian lower boundaries | 40/100/300 layers; 4/8/16 streams | CPU/CUDA flux agreement; catches beam and surface costs. |
| `longwave_thermal` | Planck emission, non-isothermal temperature profile, absorptive and scattering media | 40/100/300 layers; 4/8/16 streams; spectral batches 1–1024 | CPU/CUDA agreement and nonzero-emission physical check; measures the thermal path separately. |
| `anisotropic_high_order` | Strong forward scattering with Henyey–Greenstein-like moments and optical depth 0.1, 1, 20 | 16/32/64 streams; 100/300 layers | CPU/CUDA agreement; measures stream-order growth and conditioning. |
| `near_conservative` | `ssalb` around the fast-path routing threshold, including mixed batches | 4/8 streams; `nwave*ncol=1,32,256` | CPU/CUDA agreement with fast/general path tolerances; detects a performance change that breaks routing accuracy. |
| `batch_shape` | Distinct optical properties and boundary conditions in every wavelength and column | Fixed work counts with `(nwave,ncol)=(256,1),(64,4),(16,16),(1,256)` | Exact CPU batch-vs-single check plus CPU/CUDA agreement; identifies scheduling sensitivity to batch shape. |
| `transfer_boundary` | `tp9_flux` and `longwave_thermal` with 64/256/1024 work items | Vary property/output sizes | H2D, D2H, steady, and end-to-end decomposition; determines whether transfer optimization is justified. |

The CPU-only parity capabilities are included in correctness coverage but not
CUDA speed charts until they acquire explicit CUDA support and agreement tests:
`general_source`, Hapke BRDF, Fourier-component output, special-boundary
results, and `DELTAMPLUS`. Their benchmark row must say `SKIP` and name the
capability, rather than silently executing a CPU fallback.

## Accuracy gates

For each benchmark factory, add a dedicated pytest before measuring it:

1. Run the declared Fortran v4 DP fixture and compare the base CPU benchmark
   output against its logged output metrics and inputs. Test Problem 9 also
   uses the existing C-DISORT baseline verification as a secondary diagnostic.
2. Compare every supported CUDA result to CPU for the exact benchmark inputs.
   Preserve the existing `1e-6` relative-RMSE envelope for fast 4/8-stream
   flux paths and `1e-8` for general-path conservative routing unless a new,
   documented case demonstrates a justified tighter or looser bound.
3. Require finite output and physical assertions appropriate to the case:
   incident top-of-atmosphere beam, nonzero thermal emission, and sensible
   flux signs/conservation where applicable.
4. Check batched output against independent single solves before timing the
   batch. Benchmark shape changes must not change the calculation.
5. Re-run the complete affected CPU and CUDA pytest suites after each
   optimization. Compare the saved pre- and post-change JSONL accuracy fields;
   a timing improvement with worse agreement is rejected.

## Optimization sequence

### Phase 0 — establish the H100 baseline

Implement the Fortran-reference adapter, then the harness and small benchmark
factories. Run each declared v4 DP fixture first and retain its input/output
hashes alongside the resulting benchmark record. Run the full matrix with the
Release `sm_90` build and record CPU thread sweeps (1, physical core count,
and the chosen production count), CUDA steady-state, and end-to-end results.
Capture a Nsight Systems timeline for `latency_flux_4`,
`tp9_flux`, and `transfer_boundary`; use Nsight Compute only on a representative
small and throughput-sized kernel. This phase establishes a numerical and
performance baseline; it changes no solver code.

Exit criterion: every currently supported GPU case passes accuracy gates, the
results contain all provenance fields, and the charts distinguish first-call,
steady-state, and end-to-end time.

### Phase 1 — remove measured dispatch overhead

Profile the known per-forward `cudaMalloc`/`cudaMemcpy`/`cudaFree` operations
for `wvnmlo`, `wvnmhi`, `utau`, `umu`, `phi`, and `zd` in
`disort_dispatch.cu`. If they materially affect latency or medium batches,
cache immutable grid arrays and resize/reuse per-wave arrays with the owning
`Disort` instance, following the existing workspace lifetime. Update only
changed values, bind memory to the current device and stream safely, and make
`release_cuda_workspace()` release all associated cache safely.

Validate cache invalidation after option/grid changes, device changes, solver
destruction, and shape growth/shrink. Re-run `latency_flux_*`, `batch_shape`,
and `transfer_boundary`; retain the change only if it improves its intended
metric without regressing `tp9_flux` or accuracy.

#### Phase 1 checkpoint — 2026-09-29

A prototype retained the immutable CUDA copies of `wvnmlo`, `wvnmhi`, `utau`,
`umu`, `phi`, and `zd` on the owning `Disort` instance. It invalidated them on
`reset()` and `release_cuda_workspace()`, and used transient copies for the
conservative-scattering subset because that path owns a temporary state array.
The focused CUDA agreement suite passed **98/98**.

The TP9 geometry sweep did not show a material steady-state improvement: the
clean event-timed rows were 0.329495 s for `(nwave, ncol)=(1,1)`, 1.112660 s
for `(1,17)`, and 1.542806 s for `(256,1)`, within normal measurement noise of
the Phase 0 values. The grid host-to-device transfer is microseconds to a few
tenths of a millisecond, while each CUDA solve is 0.33–1.54 s. The `(256,17)`
prototype sample ran concurrently with an accidental duplicate benchmark worker
and had contention outliers, so it is deliberately excluded from comparison.

The prototype is not retained: it fails this phase's required material-gain
criterion and retains device allocations without improving TP9 throughput. The
next optimization starts with Phase 2 profiler evidence for occupancy, per-thread
workspace, stack, and register pressure.

### Phase 2 — improve GPU occupancy and memory behavior

Use profiler evidence to decide whether the one-warp-per-solve policy,
resident-warp cap, workspace chunking, register pressure, stack size, or
coalescing limits throughput at 16/32/64 streams and 100/300 layers. Change
one parameter or allocation layout at a time. Candidate experiments include
an empirically selected resident-warp limit by work size, avoiding host-side
conservative index handling when its mixed-batch cost is measurable, and
reducing repeated per-thread setup only if it preserves each independent
C-DISORT state.

Do not fuse physically different columns or reuse mutable solver state across
threads. Keep the general solver as the reference route for near-conservative
scattering. The acceptance chart is throughput versus `nwave*ncol`, with the
same output and agreement values as the baseline.

#### Phase 2 checkpoint — 2026-09-29

Nsight Systems traces on the H100 isolate the device kernel as the bottleneck.
For TP9 `(256,1)`, the sole DISORT kernel has an `8 x 1 x 1` grid of
`32 x 1 x 1` blocks and runs for about 1.536 s; dispatch allocation and host
copy APIs are negligible beside synchronization. For `(256,17)`, the traces
show 32-thread, one-solve-per-thread blocks in grids of 46, 68, and 136 blocks,
with DISORT kernels taking 1.97–3.07 s each.

This establishes that more caching or transfer work cannot improve steady-state
TP9. The current kernel gives each independent solve one lane and at most one
warp per block; small batches leave most H100 SMs idle, while the 4,352-solve
geometry batch still has only roughly one long-running warp per SM. The next
experiment must change only the launch/workspace residency policy and retain the
existing independent C-DISORT state contract. It needs CPU/CUDA agreement and
the same TP9 event timing before and after the rebuild.


The first launch-policy trial changed the scalar block from 32 to 16 threads
to double the number of schedulable blocks. It immediately segfaulted in
`tests/cuda/test_cuda_backend_selection.py`: the pmem workspace mapping depends
on full-warp lane behavior. The source was reverted, a clean CUDA rebuild was
completed, and that smoke test passed again. Do not retry sub-warp blocks; the
next experiment must preserve 32-thread warp blocks and vary only a
full-warp-safe residency or workspace policy.


The first full-warp-safe trial raised `kResidentWarpsPerSm` from two to four.
Focused CUDA backend and CPU/CUDA agreement tests passed (**69 passed**). On the
100-layer production `(1024,17)` case, CUDA event time improved from **16.493 s**
to **14.876 s** (9.8%). It raises cached workspace use from roughly 22 GB to
52 GB on the H100, which remains within the device budget. Retain this setting
and regenerate the benchmark artifact; it reduces chunking from three launches
to two but does not yet beat the 9.732 s CPU result.

### Phase 3 — optimize transfer and application integration

Act only if `transfer_boundary` shows transfers dominate end-to-end time.
Benchmark pinned host staging, nonblocking copies on an explicit stream, and
producer/consumer overlap with a second batch. Preserve an unambiguous
synchronization point before returning results and offer explicit APIs or
examples rather than hidden asynchronous behavior. The normal device-resident
path remains the recommended high-throughput interface; an explicit
`backend="cuda"` must not conceal copies in a benchmark labelled
steady-state.

This phase evaluates host–H100 back-and-forth only after solver parity and
on-device throughput are stable, consistent with the parity plan.

### Phase 4 — CPU optimization in parallel

Use the same factories and correctness gates for CPU. Sweep `torch` threads
and batch shape, compare single-thread throughput with the hoisted C-DISORT
baseline, and use a CPU profiler to identify copying, state reset, and solver
hot spots. Prefer better batching/state reuse and elimination of demonstrated
wrapper overhead. Do not alter cdisort algorithmic semantics or use compiler
flags that weaken floating-point correctness. Record CPU gains against one
thread and against the pre-change production thread count.

### Phase 5 — feature-specific CUDA work

Only after a CPU parity feature is complete, give it its own CUDA design note,
output/result contract, reference case, CPU/CUDA pytest, and one row in this
matrix. Start with a feature whose result can remain in the current flux
output tensor; radiance, Fourier, and special-boundary outputs require result
transfer design before they can be called CUDA-supported. Unsupported requests
continue to raise `NotImplementedError` until that work is complete.

## Reporting and review checklist

A performance change is ready for review only when its commit includes:

- the benchmark-factory or harness change and focused tests;
- before/after H100 records for affected cases in
  `benchmarks/results/<hostname>/`;
- CPU and GPU timing modes, batch shape, dtype, thread/stream settings, and
  full toolchain provenance;
- Fortran v4 DP fixture identifier, log hashes, CPU/Fortran and CPU/CUDA
  accuracy metrics, and the reference basis;
- profiler evidence that supports the claimed bottleneck; and
- the full relevant pytest and CTest results.

The report describes Fortran CPU, pydisort CPU, and pydisort GPU absolute
latency; throughput in solved `(wavelength,column)` elements per second; and
pydisort end-to-end time. Relative speed-ups are secondary and always name the
baseline and whether it is scalar or batched. No benchmark result is published
from a `SKIP` or `FAIL` case.

## Phase 5A — special-boundary CUDA result support

**Status:** implementation scaffold committed; validation has not yet begun.

This phase promotes `Disort.medium_albedo_transmissivity()` from CPU-only to
CPU/CUDA support without changing its public result contract. It is separate
from ordinary flux dispatch because C-DISORT `SPECIAL_BC` allocates doubled
user-angle work arrays and produces `ALBMED`/`TRNMED`, not flux/radiance
buffers.

1. Audit the dedicated CUDA dispatcher against the CPU implementation. One
   GPU element represents one `(nwave, ncol)` solve, constructs an isolated
   `SPECIAL_BC` state, copies positive user cosines once, populates optical
   properties, and writes `(numu, 2)` albedo/transmissivity output. It must
   use `c_disort_work_size()` and the existing per-thread workspace cache.
2. Add CUDA-gated tests before compiling: Problem 13a and 13c, multiple user
   cosines, batched waves/columns, nonzero surface albedo, output device and
   shape, finite values, and CPU/H100 agreement at `rtol=1e-6`, `atol=1e-8`.
   The former CUDA-rejection test changes only after this evidence passes.
3. Compile once in two checkpoints. Build `disort_cuda_release` first; only
   after it succeeds rebuild the pybind extension once with
   `CUDA=ON CMAKE_CUDA_ARCHITECTURES=90 python -m pip install
   --no-build-isolation .`. Never run a second build while the first exists.
4. Run a one-case CPU/CUDA smoke test after installation. Verify result device,
   shape, finiteness, synchronization, and agreement before running pytest.
5. Run focused special-boundary CPU/CUDA pytest, CUDA backend tests, the
   C-DISORT lifecycle CTest, C/C++ hooks, Ruff, mypy, and `git diff --check`.
   A failure remains `FAIL`; no CPU fallback may be reported as CUDA support.
6. Only after all checks pass, update README, PLAN, and Fortran-parity docs
   with H100 architecture 90, exact tolerance, and PASS/SKIP/FAIL counts.
   Commit implementation corrections, tests, and documentation separately.

**Decision gate:** retain the feature only if the direct device implementation
passes the smoke test and agreement suite. If the per-thread C-DISORT path
cannot do so safely within the existing workspace model, revert its scaffold
and retain the explicit CUDA `NotImplementedError`.
