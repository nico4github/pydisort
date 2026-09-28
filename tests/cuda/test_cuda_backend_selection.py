"""The explicit CUDA selector must agree with the CPU selector."""

import math

import torch

from conftest import requires_cuda
from pydisort import Disort, DisortOptions

pytestmark = requires_cuda


def build_solver(backend):
    """Create the same minimal flux-only calculation for both selectors."""
    options = DisortOptions().flags("onlyfl,lamber,quiet").backend(backend)
    options.ds().nlyr = 1
    options.ds().nstr = options.ds().nmom = options.ds().nphase = 4
    return Disort(options)


def test_backend_selector_moves_a_complete_cpu_input_set_to_cuda():
    """CPU source tensors can be used unchanged for a numerical comparison."""
    prop = torch.zeros((1, 1, 1, 6), dtype=torch.float64)
    prop[..., 0] = 0.1
    boundary = {"fbeam": torch.tensor([[math.pi]], dtype=torch.float64)}

    cpu = build_solver("cpu").forward(prop, **boundary)
    cuda = build_solver("cuda").forward(prop, **boundary)
    torch.cuda.synchronize()

    assert cuda.device.type == "cuda"
    torch.testing.assert_close(cuda.cpu(), cpu, rtol=1.0e-10, atol=1.0e-12)
