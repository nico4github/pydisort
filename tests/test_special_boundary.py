"""CPU contract tests for special-boundary albedo/transmissivity output."""

import pytest
import torch
from pydisort import Disort, DisortOptions, scattering_moments

DTYPE = torch.float64
NSTR = 16


def options(backend="cpu", nwave=1, ncol=1, angles=(0.25, 0.5)):
    configured = DisortOptions().flags("quiet").backend(backend)
    configured.ds().nlyr = 1
    configured.ds().nstr = configured.ds().nmom = configured.ds().nphase = NSTR
    configured.nwave(nwave).ncol(ncol).user_mu(list(angles))
    return configured


def properties(nwave=1, ncol=1):
    prop = torch.zeros((nwave, ncol, 1, 2 + NSTR), dtype=DTYPE)
    prop[..., 0] = 1.0
    prop[..., 1] = 0.99
    prop[..., 2:] = scattering_moments(NSTR, "henyey-greenstein", 0.8)
    return prop


def ordinary_beam(prop, albedo, mu):
    configured = options()
    configured.flags("onlyfl,lamber,quiet")
    solver = Disort(configured)
    flux = solver.forward(
        prop,
        umu0=torch.tensor([mu], dtype=DTYPE),
        fbeam=torch.tensor([[1.0 / mu]], dtype=DTYPE),
        albedo=albedo,
    )
    return torch.stack((flux[0, 0, 0, 0], flux[0, 0, -1, 1]))


def test_special_boundary_matches_problem_13a_beam_references():
    prop = properties()
    albedo = torch.tensor([[0.5]], dtype=DTYPE)
    result = Disort(options()).medium_albedo_transmissivity(prop, albedo)

    assert result.albedo.shape == (1, 1, 2)
    assert result.transmissivity.shape == (1, 1, 2)
    for angle, albedo_value, transmissivity_value in zip(
        (0.25, 0.5), result.albedo[0, 0], result.transmissivity[0, 0]
    ):
        actual = torch.stack((albedo_value, transmissivity_value))
        torch.testing.assert_close(
            actual, ordinary_beam(prop, albedo, angle), rtol=1e-6, atol=1e-8
        )


def test_special_boundary_batches_each_wave_and_column():
    prop = properties(nwave=2, ncol=2)
    albedo = torch.tensor([[0.0, 0.25], [0.5, 0.75]], dtype=DTYPE)
    result = Disort(options(nwave=2, ncol=2)).medium_albedo_transmissivity(
        prop, albedo
    )

    assert result.albedo.shape == (2, 2, 2)
    assert result.transmissivity.shape == (2, 2, 2)
    assert torch.isfinite(result.albedo).all()
    assert torch.isfinite(result.transmissivity).all()
    for wave in range(2):
        for column in range(2):
            for angle_index, angle in enumerate((0.25, 0.5)):
                expected = ordinary_beam(
                    prop[wave, column], albedo[wave, column], angle
                )
                torch.testing.assert_close(
                    torch.stack(
                        (
                            result.albedo[wave, column, angle_index],
                            result.transmissivity[wave, column, angle_index],
                        )
                    ),
                    expected,
                    rtol=1e-6,
                    atol=1e-8,
                )


def test_special_boundary_rejects_thermal_options_changed_after_construction():
    configured = options()
    solver = Disort(configured)
    configured.flags("planck,quiet")

    with pytest.raises(RuntimeError, match="thermal"):
        solver.medium_albedo_transmissivity(properties())


@pytest.mark.parametrize("angles", [(-0.5,), (0.0,), (1.1,)])
def test_special_boundary_rejects_nonpositive_or_out_of_range_cosines(angles):
    with pytest.raises(RuntimeError, match="positive finite"):
        Disort(options(angles=angles)).medium_albedo_transmissivity(
            properties()
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)
def test_special_boundary_cuda_matches_cpu_for_batched_problem_13_cases():
    prop = properties(nwave=2, ncol=2)
    prop[1, :, 0, 0] = 0.5
    prop[1, :, 0, 1] = 0.5
    albedo = torch.tensor([[0.0, 0.25], [0.5, 0.75]], dtype=DTYPE)

    cpu = Disort(options(nwave=2, ncol=2)).medium_albedo_transmissivity(
        prop, albedo
    )
    cuda = Disort(
        options(backend="cuda", nwave=2, ncol=2)
    ).medium_albedo_transmissivity(prop.cuda(), albedo.cuda())
    torch.cuda.synchronize()

    assert cuda.albedo.device.type == "cuda"
    assert cuda.transmissivity.device.type == "cuda"
    assert cuda.albedo.shape == (2, 2, 2)
    assert cuda.transmissivity.shape == (2, 2, 2)
    assert torch.isfinite(cuda.albedo).all()
    assert torch.isfinite(cuda.transmissivity).all()
    torch.testing.assert_close(
        cuda.albedo.cpu(), cpu.albedo, rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(
        cuda.transmissivity.cpu(), cpu.transmissivity, rtol=1e-6, atol=1e-8
    )
