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
   physical inputs, requested outputs, dtype, and reference source.
2. A result has three independent checks: correctness, steady-state device
   throughput, and end-to-end application time. Do not quote one as another.
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
6. A supported CUDA case must agree with its CPU reference before its timing
   is retained. Existing fast 4/8-stream cases use their dedicated tolerance;
   ordinary shared-algorithm cases use the tighter CPU/CUDA tolerance. Each
   new case records maximum absolute error, relative RMS error for upward,
   downward, and net flux, and the agreed threshold.
7. A failure, non-finite value, or unsupported configuration is never a speed
   result. Report it respectively as `FAIL` or `SKIP` with the exact reason.
   Unsupported CPU-only features must continue to raise `NotImplementedError`
   on a CUDA request.

## Reproducible harness and stored evidence

Add a single `benchmarks/compare_backends.py` harness and small test module.
It will construct all workloads from named factories, perform the agreement
check before timing, emit JSON Lines, and optionally write one human-readable
summary. It must provide these timing modes:

| Mode | Included work | Purpose |
| --- | --- | --- |
| `cpu_steady` | `forward()` with CPU-resident tensors | CPU solver throughput and thread scaling |
| `gpu_steady` | CUDA-resident `forward()` measured with CUDA events | kernel plus CUDA dispatch throughput |
| `gpu_first_call` | construction and first CUDA `forward()` | allocation/cache warm-up cost |
| `gpu_h2d` | pinned host-to-device input transfer only | transfer bandwidth and staging overhead |
| `gpu_d2h` | required output transfer only | result-return bandwidth |
| `gpu_end_to_end` | host inputs through host result | application-visible latency |

The harness must expose `--case`, `--nwave`, `--ncol`, `--nlyr`, `--nstr`,
`--repeat`, `--warmup`, `--threads`, `--timing-mode`, and `--output`. It must
record the git revision and dirty state; host name; CPU model and thread count;
GPU model, memory, driver, and compute capability; PyTorch version and CUDA
runtime; CMake CUDA compiler and architecture; dtype; stream count; all shape
parameters; timing samples; and accuracy metrics.

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

## Benchmark cases

The set deliberately spans cheap latency-sensitive calls, throughput-sized
batches, physical source terms, output costs, and numerical corner cases. A
case is not added merely to create a favorable speed-up.

| ID | Configuration | Shape sweep | Required comparison and purpose |
| --- | --- | --- | --- |
| `latency_flux_4` | Plane-parallel shortwave, Lambertian, `onlyfl`, 4 streams, ordinary scattering | `(nwave, ncol)=(1,1),(8,1),(32,4)`; 20 and 100 layers | CPU/CUDA flux agreement; exposes launch, state-copy, and allocation costs. |
| `latency_flux_8` | Same, 8 streams | Same | Exercises the other specialized CUDA kernel. |
| `tp9_flux` | Existing Test Problem 9-derived inhomogeneous scattering, flux only | `nwave=1,8,64,256,1024`; `ncol=1,4,16`; 32/100 layers; 8/16/32 streams | C-DISORT verification on the base shape, then CPU/CUDA agreement; primary realistic throughput curve. |
| `tp9_radiance_cpu` | Test Problem 9 with user tau, mu, and phi | Same spectral/column sweep, 8/16/32 streams | CPU and C-DISORT only. CUDA currently returns flux but not radiance access, so record a documented `SKIP` for CUDA result retrieval rather than inventing a proxy. |
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

1. Check a published DISORT reference or C-DISORT result at a small base shape
   when one exists. Test Problem 9 uses the existing C baseline verification.
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

Implement the harness and the small benchmark factories first. Run the full
matrix with the Release `sm_90` build and record CPU thread sweeps (1, physical
core count, and the chosen production count), CUDA steady-state, and
end-to-end results. Capture a Nsight Systems timeline for `latency_flux_4`,
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
- CPU/CUDA accuracy metrics and the reference basis;
- profiler evidence that supports the claimed bottleneck; and
- the full relevant pytest and CTest results.

The report describes absolute latency, throughput in solved
`(wavelength,column)` elements per second, and end-to-end time. Relative
speed-ups are secondary and always name the baseline. No benchmark result is
published from a `SKIP` or `FAIL` case.
