"""Time the device-resident tensor TP9 flux reconstruction reproducibly."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pydisort.tensor_backend import solve_tp9_flux
from pydisort.timing import TimingCollector


def build_inputs(
    *, nwave: int, ncol: int, nlyr: int, nstr: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the 100-layer TP9-scaled diffuse workload on ``device``."""
    prop = torch.zeros(
        (nwave, ncol, nlyr, nstr + 2), dtype=torch.float64, device=device
    )
    layer = torch.arange(1, nlyr + 1, dtype=torch.float64, device=device)
    prop[..., 0] = layer * (6.0 / nlyr)
    prop[..., 1] = 0.6 + layer * (0.3 / nlyr)
    utau = torch.tensor(
        [0.0, 1.05, 2.1, 6.0, 21.0], dtype=torch.float64, device=device
    )
    fisot = torch.full(
        (nwave, ncol), 1.0 / torch.pi, dtype=torch.float64, device=device
    )
    return prop, utau, fisot


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    prop, utau, fisot = build_inputs(
        nwave=args.nwave,
        ncol=args.ncol,
        nlyr=args.nlyr,
        nstr=args.nstr,
        device=device,
    )
    keyword = {"nstr": args.nstr, "nmom": args.nstr, "deltam": False}
    for _ in range(args.warmup):
        solve_tp9_flux(prop, utau, fisot, **keyword)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = Path(args.report)
    case = (
        f"tensor_tp9_flux_nwave={args.nwave}_ncol={args.ncol}_"
        f"nlyr={args.nlyr}_nstr={args.nstr}"
    )
    for repetition in range(args.repeat):
        with TimingCollector(
            cuda_device=device if device.type == "cuda" else None
        ) as timing:
            result = solve_tp9_flux(prop, utau, fisot, **keyword)
        timing.append_text_report(
            report, case=f"{case}_repeat={repetition}", backend=str(device)
        )
        if not torch.isfinite(result).all():
            raise RuntimeError(
                "tensor TP9 baseline produced non-finite fluxes"
            )
        print(
            f"{case} repeat={repetition} result_shape={tuple(result.shape)} "
            f"report={report}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nwave", type=int, default=1)
    parser.add_argument("--ncol", type=int, default=1)
    parser.add_argument("--nlyr", type=int, default=100)
    parser.add_argument("--nstr", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Append timing rows directly to this report file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
