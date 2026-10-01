# CUDA optimization log

This log records completed CUDA performance experiments. `PLAN_CUDA.md` defines
future work; this file records evidence, validation, and the retain/reject
decision for each attempt.

## Measurement contract

All entries use the H100 TP9-scaled fixture unless stated otherwise: 100
atmospheric layers, 32 streams, float64, device-resident pydisort inputs, and
CUDA event timing. Production throughput is 1,024 wavelength channels with one
or 17 five-degree geometry columns. CPU and CUDA outputs must agree before a
timing is considered valid.

## Results

| Date | Change | Evidence | Result | Decision |
| --- | --- | --- | --- | --- |
| 2026-09-29 | Cache immutable CUDA grids (`wvnm`, user grids, altitude) across forwards | 98 focused CUDA tests passed. Clean TP9 rows were within timing noise; host-to-device grid copies were microseconds to tenths of a millisecond while solves took 0.33–1.54 s. | No material steady-state gain. | Rejected and reverted. |
| 2026-09-30 | Use 16 scalar solves per block | First CUDA backend test segfaulted because pmem requires full-warp lane mapping. | Invalid implementation. | Rejected and reverted; no sub-warp blocks. |
| 2026-09-30 | Raise resident full-warps per SM from 2 to 4 | 69 focused CUDA tests passed. One direct `(1024,17)` event sample fell from 16.493 s to 14.876 s, but the regenerated complete production report measured 16.502 s. Workspace grew from about 22 GB to 52 GB. | The apparent 9.8% gain did not reproduce. | Rejected and reverted. |

## Baseline that remains

The retained CUDA launch policy uses 32 threads (one full warp) per block and
two resident warps per H100 SM. The current production report is at
`../disort-pyf/benchmarks/PLATO-Ganymede/testproblem09_production_summary.txt`:
CPU remains faster than CUDA for both 1,024-channel workloads.

## Next investigation

Nsight Systems shows the C-DISORT element kernel dominates device time. Further
work must keep the full-warp pmem contract and focus on a repeatable way to
raise useful per-solve parallelism or reduce per-thread solver work; cache and
chunk-count changes alone have not improved the production result.

## Host/device transfer audit — 2026-09-30

The TP9 benchmark keeps its atmospheric inputs and results device-resident while
CUDA events time repeated `forward()` calls. The only bulk host-to-device
transfer is the initial placement of seven tensors: `prop`, `umu0`, `phi0`,
`fbeam`, `fisot`, `fluor`, and `albedo`. There is no device-to-host transfer in
a steady-state `forward()`; one result tensor is copied for each host consumer.
The benchmark's extra validation downloads are separately identified and are
not part of the application transfer contract.

The CUDA dispatch still creates and frees five small host-derived arrays per
TP9 forward: `wvnmlo`, `wvnmhi`, `utau`, `umu`, and `phi`. This is five H2D
copies, with no corresponding D2H copy. It is a real source-level transfer
cost, but it is small relative to the solver kernel:

| TP9 production case | Initial H2D (7 copies) | Per-forward internal H2D (5 copies) | Per-forward D2H | One output D2H after timing |
| --- | ---: | ---: | ---: | ---: |
| 1,024 wavelengths × 1 column | 27,885,584 B (26.59 MiB) | 16,464 B (16.08 KiB) | 0 B | 81,920 B (80 KiB) |
| 1,024 wavelengths × 17 columns | 474,054,928 B (452.09 MiB) | 278,848 B (272.31 KiB) | 0 B | 1,392,640 B (1.33 MiB) |

The earlier immutable-grid cache experiment targeted those five internal
uploads. It removed less than 0.3 MiB per 17-column forward and did not change
end-to-end timing against a 16.49 s kernel-dominated solve, so it remains
rejected. The benchmark now records this transfer contract and measures
`cuda_d2h_output_seconds` using the actual CUDA result, replacing the old
misleading metric that copied device inputs back to the host.

## CUDA mapping assessment — 2026-09-30

The lack of a production CUDA gain is consistent with the current mapping being
host-solver-friendly rather than GPU-friendly. `disort_dispatch.cu` assigns one
independent `(wavelength, column)` C-DISORT solve to each CUDA lane. Before
calling `c_disort`, every lane allocates a complete private state/output from
its pmem slice and copies the 100-layer property profile into that state. The
solver's control flow and dense work then remain serial within that lane.

`gpu_chunk_kernel()` groups 32 scalar lanes in a block and deliberately caps
resident work at two such warps per SM. Thus the H100 runs at most 64 scalar
solves concurrently per SM under this policy, even for the 17,408-solve
production batch. The four-warp trial used about 52 GB of workspace and showed
no reproducible benefit, so scalar-concurrency tuning is exhausted for the
current ownership model.

This is source-level and Nsight Systems evidence, not an occupancy/register
measurement: Nsight Compute is not installed on the H100 environment. The
next CUDA investigation is therefore a feasibility design, not another
transfer or residency tweak. It must identify a numerically safe stage of the
32-stream general solver that can run cooperatively within a warp/block or as a
batched operation across solves, with structure-of-arrays work storage. A
prototype must first retain float64 CPU/CUDA agreement and be timed against the
1,024 x 17 TP9 baseline. Replacing the current lane-local C-DISORT call in one
unmeasured rewrite would be too high-risk.

## Tensor reconstruction baseline — 2026-10-01

The parity-complete pure-PyTorch TP9 flux path now has a reproducible
`benchmarks/tensor_tp9_baseline.py` runner. It constructs inputs directly on
the requested device, warms up outside the measured region, times the complete
`solve_tp9_flux` call with `TimingCollector`, and appends each stage and its
root `TOTAL` to the bridge report directory. It makes no host/device transfer
inside the steady-state region.

The initial H100 float64 diffuse TP9-scaled measurements are recorded in
`../disort-pyf/benchmarks/PLATO-Ganymede/testproblem09_tensor_reconstruction_timing.txt`:

| Shape | Total CUDA event time | Dominant stage | Evidence |
| --- | ---: | ---: | --- |
| 1 wavelength x 1 column x 100 layers x 32 streams | 0.100825 s | reduced eigensolve, 0.080604 s | one warmed run |
| 10 wavelengths x 1 column x 100 layers x 32 streams | 0.8897--0.8962 s | reduced eigensolve, 0.7789 s; boundary solve, 0.1024 s | five warmed runs |

This is a first correctness-preserving baseline, not a throughput result for
1,000 channels. The present representation materializes a dense
`(nlyr*nstr)^2` boundary system per batch element: at 100 layers and 32 streams
that is about 81.9 MiB of float64 coefficients per channel before solver
workspace. A direct 1,000-channel batch would therefore exceed H100 memory
once matrix and factorization workspace are included. The next optimization
must retain the existing parity gates while using the existing block-banded
structure and bounded channel chunks; transfer caching is explicitly not the
bottleneck and will not be retried.

A matching five-run CPU baseline for one channel is 0.0468--0.0835 s
(median 0.0479 s), versus the warmed H100 event time of 0.1008 s. The tensor
implementation is therefore currently about 2.1x slower on H100 for this
single 100-layer column. This reinforces that the next experiment is an
algorithm/layout change, not transfer tuning or a device-selection change.
The reduced matrix is not symmetric (measured maximum antisymmetric entry
107.55), so a substitution of `torch.linalg.eigh` for the required general
eigensolve is rejected on correctness grounds.
