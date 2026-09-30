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
