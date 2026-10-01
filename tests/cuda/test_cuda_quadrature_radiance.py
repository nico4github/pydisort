"""CUDA quadrature radiance access is explicitly unsupported."""

import math

import pytest
import torch
from conftest import requires_cuda
from pydisort import Disort, DisortOptions, scattering_moments

pytestmark = requires_cuda


def _solve(backend):
    options = DisortOptions().flags("usrtau,lamber,quiet").backend(backend)
    options.ds().nlyr = 1
    options.ds().nstr = options.ds().nmom = options.ds().nphase = 4
    options.user_tau([0.0, 1.0])
    # C-DISORT replaces this requested count with nstr when usrang is clear.
    options.user_mu([0.0])
    options.user_phi([0.0, 36.0, 90.0])
    options.ncol(1)
    options.nwave(1)

    disort = Disort(options)
    prop = torch.zeros((1, 1, 1, 6), dtype=torch.float64)
    prop[..., 0] = 1.0
    prop[..., 1] = 0.5
    prop[..., 2:] = scattering_moments(4, "isotropic")
    disort.forward(
        prop,
        umu0=torch.tensor([0.5], dtype=torch.float64),
        phi0=torch.tensor([0.0], dtype=torch.float64),
        fbeam=torch.tensor([[math.pi]], dtype=torch.float64),
        fisot=torch.tensor([[1.0]], dtype=torch.float64),
        albedo=torch.tensor([[0.3]], dtype=torch.float64),
    )
    return disort.gather_rad()


def test_cuda_quadrature_radiance_is_explicitly_unsupported():
    """CPU exposes nstr angles; CUDA fails clearly because it retains no radiance grid."""
    cpu = _solve("cpu")
    assert cpu.shape == (1, 1, 3, 2, 4)

    with pytest.raises(
        NotImplementedError, match="gather_rad is not implemented for CUDA"
    ):
        _solve("cuda")
