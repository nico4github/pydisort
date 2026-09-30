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
