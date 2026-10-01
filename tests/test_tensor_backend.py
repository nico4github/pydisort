"""Parity tests for the first vectorized C-DISORT state-preparation stage."""

import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch

SOURCE_ROOT = Path(__file__).parents[1] / "python"
PACKAGE = types.ModuleType("pydisort")
PACKAGE.__path__ = [str(SOURCE_ROOT)]
sys.modules.setdefault("pydisort", PACKAGE)
from pydisort.tensor_backend import prepare_atmosphere
from pydisort.timing import TimingCollector


def test_prepare_atmosphere_matches_downward_disort_property_rules():
    prop = torch.tensor(
        [[[[1.0, 0.2, 0.3, 0.4], [2.0, 0.5, 0.6, 0.7]]]],
        dtype=torch.float64,
    )

    state = prepare_atmosphere(prop, nstr=2, nmom=3)

    assert torch.equal(state.dtauc, prop[..., 0])
    assert torch.equal(state.ssalb, prop[..., 1])
    expected = torch.tensor(
        [[[[1.0, 0.3, 0.4, 0.0], [1.0, 0.6, 0.7, 0.0]]]],
        dtype=torch.float64,
    )
    assert torch.equal(state.pmom, expected)


def test_prepare_atmosphere_matches_upward_layer_reversal_and_zero_fill():
    prop = torch.tensor([[[[1.0], [2.0], [3.0]]]], dtype=torch.float64)

    state = prepare_atmosphere(prop, nstr=2, nmom=2, upward=True)

    assert torch.equal(
        state.dtauc, torch.tensor([[[3.0, 2.0, 1.0]]], dtype=torch.float64)
    )
    assert torch.equal(state.ssalb, torch.zeros_like(state.dtauc))
    assert torch.equal(state.pmom[..., 0], torch.ones_like(state.dtauc))
    assert torch.equal(
        state.pmom[..., 1:], torch.zeros_like(state.pmom[..., 1:])
    )


def test_prepare_atmosphere_is_batched_and_timing_instrumented():
    prop = torch.arange(2 * 3 * 4 * 4, dtype=torch.float64).reshape(2, 3, 4, 4)
    with TimingCollector() as timing:
        state = prepare_atmosphere(prop, nstr=2, nmom=2)

    assert state.dtauc.shape == (2, 3, 4)
    assert state.pmom.shape == (2, 3, 4, 3)
    assert timing.summary()["tensor_backend.prepare_atmosphere"]["calls"] == 1


@pytest.mark.parametrize(
    ("prop", "nstr", "nmom", "message"),
    [
        (torch.zeros(2, dtype=torch.float64), 2, 2, "shape"),
        (torch.zeros((1, 1, 1, 1)), 2, 2, "float64"),
        (torch.zeros((1, 1, 1, 1), dtype=torch.float64), 3, 3, "even"),
        (torch.zeros((1, 1, 1, 1), dtype=torch.float64), 4, 2, "at least"),
    ],
)
def test_prepare_atmosphere_rejects_incompatible_contract(
    prop, nstr, nmom, message
):
    with pytest.raises(ValueError, match=message):
        prepare_atmosphere(prop, nstr=nstr, nmom=nmom)


def test_prepare_layer_optics_matches_delta_m_formulas():
    prop = torch.tensor(
        [[[[2.0, 0.5, 0.2, 0.4], [3.0, 0.25, 0.1, 0.8]]]], dtype=torch.float64
    )
    atmosphere = prepare_atmosphere(prop, nstr=2, nmom=2)

    from pydisort.tensor_backend import prepare_layer_optics

    layer = prepare_layer_optics(atmosphere, nstr=2, deltam=True)
    expected_f = torch.tensor([[[0.4, 0.8]]], dtype=torch.float64)
    expected_oprim = (
        prop[..., 1] * (1.0 - expected_f) / (1.0 - expected_f * prop[..., 1])
    )
    expected_dtaucpr = (1.0 - expected_f * prop[..., 1]) * prop[..., 0]
    assert torch.equal(layer.flyr, expected_f)
    assert torch.allclose(layer.oprim, expected_oprim, rtol=0.0, atol=0.0)
    assert torch.allclose(layer.dtaucpr, expected_dtaucpr, rtol=0.0, atol=0.0)
    assert torch.allclose(
        layer.tauc,
        torch.tensor([[[2.0, 5.0]]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_prepare_output_grid_matches_delta_m_layer_mapping():
    from pydisort.tensor_backend import (
        prepare_layer_optics,
        prepare_output_grid,
    )

    prop = torch.tensor(
        [[[[2.0, 0.5, 0.0, 0.4], [3.0, 0.25, 0.0, 0.8]]]], dtype=torch.float64
    )
    atmosphere = prepare_atmosphere(prop, nstr=2, nmom=2)
    optics = prepare_layer_optics(atmosphere, nstr=2, deltam=True)
    grid = prepare_output_grid(
        torch.tensor([0.0, 2.0, 3.0, 5.0], dtype=torch.float64),
        atmosphere,
        optics,
        deltam=True,
    )
    assert torch.equal(grid.layru, torch.tensor([[[1, 1, 2, 2]]]))
    assert torch.allclose(
        grid.utaupr[..., 0],
        torch.zeros((1, 1), dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    assert torch.allclose(
        grid.utaupr[..., -1], optics.taucpr[..., -1], atol=0, rtol=0
    )


def test_gaussian_quadrature_has_disort_symmetry_and_normalization():
    from pydisort.tensor_backend import gaussian_quadrature

    quadrature = gaussian_quadrature(32, device="cpu")
    assert quadrature.cmu.shape == (32,)
    assert torch.all(quadrature.cmu[:16] > 0)
    assert torch.equal(quadrature.cmu[16:], -quadrature.cmu[:16])
    assert torch.equal(quadrature.cwt[16:], quadrature.cwt[:16])
    assert torch.allclose(
        quadrature.cwt.sum(),
        torch.tensor(2.0, dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )


def test_reduced_eigen_matrix_is_batched_and_finite():
    from pydisort.tensor_backend import (
        build_reduced_eigen_matrix,
        gaussian_quadrature,
        prepare_layer_optics,
    )

    prop = torch.ones((2, 3, 4, 4), dtype=torch.float64)
    atmosphere = prepare_atmosphere(prop, nstr=2, nmom=2)
    optics = prepare_layer_optics(atmosphere, nstr=2, deltam=False)
    matrix = build_reduced_eigen_matrix(
        optics, gaussian_quadrature(2, device="cpu"), nstr=2
    )
    assert matrix.matrix.shape == (2, 3, 4, 1, 1)
    assert torch.isfinite(matrix.matrix).all()


def test_reduced_eigen_matrix_accepts_a_nonzero_fourier_order():
    from pydisort.tensor_backend import (
        build_reduced_eigen_matrix,
        gaussian_quadrature,
        prepare_layer_optics,
    )

    prop = torch.tensor([[[[1.0, 0.6, 0.3, 0.1, 0.03]]]], dtype=torch.float64)
    optics = prepare_layer_optics(
        prepare_atmosphere(prop, nstr=4, nmom=4), nstr=4, deltam=False
    )
    matrix = build_reduced_eigen_matrix(
        optics, gaussian_quadrature(4, device="cpu"), nstr=4, fourier_order=1
    )
    assert torch.isfinite(matrix.matrix).all()


def test_nakajima_tanaka_phase_matches_legendre_sum():
    from pydisort.tensor_backend import nakajima_tanaka_phase

    pmom = torch.tensor([1.0, 0.3, 0.2], dtype=torch.float64)
    cosine = torch.tensor(0.4, dtype=torch.float64)
    actual = nakajima_tanaka_phase(pmom, cosine)
    expected = 1.0 + 3.0 * 0.3 * 0.4 + 5.0 * 0.2 * (3.0 * 0.4**2 - 1.0) / 2.0
    assert torch.allclose(actual, torch.tensor(expected, dtype=torch.float64))


def test_nakajima_tanaka_layer_phase_preserves_layer_and_angle_axes():
    from pydisort.tensor_backend import nakajima_tanaka_layer_phase

    pmom = torch.tensor([[[1.0, 0.2], [1.0, 0.6]]], dtype=torch.float64)
    cosine = torch.tensor([[[0.25, 0.75]]], dtype=torch.float64)
    actual = nakajima_tanaka_layer_phase(pmom, cosine)
    expected = 1.0 + 3.0 * pmom[..., 1, None, None] * cosine[..., None, :, :]
    assert actual.shape == (1, 2, 1, 2)
    assert torch.allclose(actual, expected)


def test_nakajima_tanaka_single_scatter_matches_one_layer_formula():
    from pydisort.tensor_backend import nakajima_tanaka_single_scatter

    phase = torch.full((5, 1, 1), 1.5, dtype=torch.float64)
    omega = torch.full_like(phase, 0.4)
    tau = torch.ones_like(phase)
    user_tau = torch.tensor([0.25], dtype=torch.float64)
    user_mu = torch.tensor([0.4], dtype=torch.float64)
    umu0 = torch.tensor(
        [[0.2], [0.35], [0.5], [0.65], [0.8]], dtype=torch.float64
    )
    fbeam = torch.full_like(umu0, torch.pi)
    actual = nakajima_tanaka_single_scatter(
        phase, omega, tau, user_tau, user_mu, umu0, fbeam
    )
    exp0 = torch.exp(-user_tau[0] / umu0)
    exp1 = torch.exp(-((1.0 - user_tau[0]) / user_mu[0] + 1.0 / umu0))
    expected = (
        fbeam
        / (4.0 * torch.pi * (1.0 + user_mu[0] / umu0))
        * omega[..., 0]
        * phase[..., 0]
        * (exp0 - exp1)
    )
    assert torch.allclose(actual[..., 0, 0], expected, rtol=1e-14, atol=1e-14)


def test_reduced_eigensolve_is_batched_and_real():
    from pydisort.tensor_backend import solve_reduced_eigenproblem

    matrix = torch.tensor([[[[[4.0, 0.0], [0.0, 9.0]]]]], dtype=torch.float64)
    from pydisort.tensor_backend import TensorReducedEigenMatrix

    reduced = TensorReducedEigenMatrix(
        matrix=matrix,
        amb=torch.eye(2, dtype=torch.float64).reshape(1, 1, 1, 2, 2),
    )
    values, vectors = solve_reduced_eigenproblem(reduced)
    assert torch.equal(
        values, torch.tensor([[[[2.0, 3.0]]]], dtype=torch.float64)
    )
    assert vectors.shape == (1, 1, 1, 4, 4)


def test_layer_continuity_blocks_scale_eigenvector_columns():
    from pydisort.tensor_backend import (
        TensorLayerOptics,
        build_layer_continuity_blocks,
    )

    optics = TensorLayerOptics(
        *(torch.ones((1, 1, 1), dtype=torch.float64) for _ in range(5)),
        torch.ones((1, 1, 1, 2), dtype=torch.float64),
    )
    vectors = torch.arange(16, dtype=torch.float64).reshape(1, 1, 1, 4, 4)
    values = torch.log(torch.tensor([[[[2.0, 3.0]]]], dtype=torch.float64))
    upper, lower = build_layer_continuity_blocks(vectors, values, optics)
    assert torch.allclose(
        upper,
        vectors[..., :2] * torch.tensor([2.0, 3.0], dtype=torch.float64),
        rtol=1e-14,
        atol=0.0,
    )
    assert torch.allclose(
        lower,
        vectors[..., 2:] * torch.tensor([2.0, 3.0], dtype=torch.float64),
        rtol=1e-14,
        atol=0.0,
    )


def test_block_tridiagonal_solver_matches_dense_batched_system():
    from pydisort.tensor_backend import solve_block_tridiagonal

    lower = torch.tensor([[[[1.0, 2.0], [0.0, 1.0]]]], dtype=torch.float64)
    diagonal = torch.tensor(
        [[[[4.0, 1.0], [2.0, 5.0]], [[6.0, 2.0], [1.0, 7.0]]]],
        dtype=torch.float64,
    )
    upper = torch.tensor([[[[2.0, 0.0], [3.0, 1.0]]]], dtype=torch.float64)
    rhs = torch.tensor([[[6.0, 5.0], [7.0, 4.0]]], dtype=torch.float64)
    solution = solve_block_tridiagonal(lower, diagonal, upper, rhs)
    dense = torch.cat(
        (
            torch.cat((diagonal[..., 0, :, :], upper[..., 0, :, :]), dim=-1),
            torch.cat((lower[..., 0, :, :], diagonal[..., 1, :, :]), dim=-1),
        ),
        dim=-2,
    )
    expected = torch.linalg.solve(dense, rhs.reshape(1, 4, 1)).reshape(1, 2, 2)
    assert torch.allclose(solution, expected, atol=1e-14, rtol=0)


def test_tp9_boundary_system_matches_explicit_c_set_matrix_equations():
    from pydisort.tensor_backend import (
        TensorLayerOptics,
        build_tp9_boundary_system,
        solve_tp9_boundary_system,
    )

    vectors = torch.tensor(
        [[[[[1.0, 2.0], [3.0, 5.0]], [[7.0, 11.0], [13.0, 17.0]]]]],
        dtype=torch.float64,
    )
    values = torch.log(torch.tensor([[[[2.0], [3.0]]]], dtype=torch.float64))
    optics = TensorLayerOptics(
        *(torch.ones((1, 1, 2), dtype=torch.float64) for _ in range(5)),
        torch.ones((1, 1, 2, 2), dtype=torch.float64),
    )
    system = build_tp9_boundary_system(
        vectors, values, optics, torch.tensor([[4.0]], dtype=torch.float64)
    )
    expected = torch.tensor(
        [
            [
                [
                    [0.5, 2.0, 0.0, 0.0],
                    [1.0, 1.0, -7.0 / 3.0, -11.0],
                    [3.0, 2.5, -13.0 / 3.0, -17.0],
                    [0.0, 0.0, 13.0, 17.0 / 3.0],
                ]
            ]
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(system.matrix, expected, atol=1e-14, rtol=0)
    assert torch.equal(
        system.rhs, torch.tensor([[[4.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    )
    solution = solve_tp9_boundary_system(system)
    assert torch.allclose(
        system.matrix @ solution.unsqueeze(-1),
        system.rhs.unsqueeze(-1),
        atol=1e-14,
        rtol=0,
    )


def test_tp9_boundary_system_is_batched():
    from pydisort.tensor_backend import (
        TensorLayerOptics,
        build_tp9_boundary_system,
    )

    vectors = torch.eye(4, dtype=torch.float64).reshape(1, 1, 1, 4, 4)
    vectors = vectors.expand(2, 3, -1, -1, -1).clone()
    values = torch.ones((2, 3, 1, 2), dtype=torch.float64)
    optics = TensorLayerOptics(
        *(torch.ones((2, 3, 1), dtype=torch.float64) for _ in range(5)),
        torch.ones((2, 3, 1, 4), dtype=torch.float64),
    )
    system = build_tp9_boundary_system(
        vectors, values, optics, torch.ones((2, 3), dtype=torch.float64)
    )
    assert system.matrix.shape == (2, 3, 4, 4)
    assert system.rhs.shape == (2, 3, 4)


def test_absorption_only_flux_matches_the_discrete_ordinate_reference():
    """First derived-value parity gate required by CUDA_BACKEND_STRATEGY.md."""
    from pydisort.tensor_backend import (
        build_reduced_eigen_matrix,
        build_tp9_boundary_system,
        extract_tp9_fluxes,
        gaussian_quadrature,
        prepare_layer_optics,
        prepare_output_grid,
        solve_reduced_eigenproblem,
        solve_tp9_boundary_system,
    )

    prop = torch.zeros((1, 1, 1, 6), dtype=torch.float64)
    prop[..., 0] = 1.0
    atmosphere = prepare_atmosphere(prop, nstr=4, nmom=4)
    optics = prepare_layer_optics(atmosphere, nstr=4, deltam=False)
    quadrature = gaussian_quadrature(4, device="cpu")
    values, vectors = solve_reduced_eigenproblem(
        build_reduced_eigen_matrix(optics, quadrature, nstr=4)
    )
    grid = prepare_output_grid(
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        atmosphere,
        optics,
        deltam=False,
    )
    fisot = torch.full((1, 1), 1.0 / torch.pi, dtype=torch.float64)
    constants = solve_tp9_boundary_system(
        build_tp9_boundary_system(vectors, values, optics, fisot)
    )
    fluxes = extract_tp9_fluxes(
        vectors, values, optics, grid, quadrature, constants
    )
    mu = quadrature.cmu[:2]
    weight = quadrature.cwt[:2]
    expected_downward = torch.stack(
        (
            torch.tensor(1.0, dtype=torch.float64),
            2.0 * torch.sum(weight * mu * torch.exp(-1.0 / mu)),
        )
    )
    assert torch.allclose(
        fluxes[0, 0, :, 0],
        torch.zeros(2, dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    assert torch.allclose(
        fluxes[0, 0, :, 1], expected_downward, atol=1e-13, rtol=0
    )
    # C-DISORT gives the same bottom flux for this four-stream case.
    assert torch.allclose(
        fluxes[0, 0, 1, 1],
        torch.tensor(0.22380103757909353, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize("fixture_name", ["tp9a", "tp9b"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_tp9_flux_matches_self_contained_reference_fixture(
    fixture_name, device
):
    """Scattering references run without a C-DISORT or Fortran installation."""
    from pydisort.tensor_backend import (
        build_reduced_eigen_matrix,
        build_tp9_boundary_system,
        extract_tp9_fluxes,
        gaussian_quadrature,
        prepare_layer_optics,
        prepare_output_grid,
        solve_reduced_eigenproblem,
        solve_tp9_boundary_system,
    )

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / f"tensor_{fixture_name}_flux_reference.json"
        ).read_text()
    )
    nstr = fixture["nstr"]
    dtauc = torch.tensor(fixture["dtauc"], dtype=torch.float64)
    ssalb = torch.tensor(fixture["ssalb"], dtype=torch.float64)
    prop = torch.zeros(
        (1, 1, dtauc.numel(), 2 + nstr), dtype=torch.float64, device=device
    )
    prop[..., 0] = dtauc.to(device)
    prop[..., 1] = ssalb.to(device)
    if "moments" in fixture:
        prop[..., 2:] = torch.tensor(
            fixture["moments"], dtype=torch.float64, device=device
        )
    atmosphere = prepare_atmosphere(prop, nstr=nstr, nmom=nstr)
    optics = prepare_layer_optics(
        atmosphere, nstr=nstr, deltam=fixture["deltam"]
    )
    quadrature = gaussian_quadrature(nstr, device=device)
    values, vectors = solve_reduced_eigenproblem(
        build_reduced_eigen_matrix(optics, quadrature, nstr=nstr)
    )
    grid = prepare_output_grid(
        torch.tensor(fixture["user_tau"], dtype=torch.float64, device=device),
        atmosphere,
        optics,
        deltam=fixture["deltam"],
    )
    fisot = torch.full(
        (1, 1), fixture["fisot"], dtype=torch.float64, device=device
    )
    constants = solve_tp9_boundary_system(
        build_tp9_boundary_system(vectors, values, optics, fisot)
    )
    fluxes = extract_tp9_fluxes(
        vectors, values, optics, grid, quadrature, constants
    )
    assert torch.allclose(
        fluxes[0, 0].cpu(),
        torch.tensor(fixture["flux"], dtype=torch.float64),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


def test_solve_tp9_flux_matches_connected_stage_pipeline():
    from pydisort.tensor_backend import solve_tp9_flux

    prop = torch.zeros((2, 3, 1, 6), dtype=torch.float64)
    prop[..., 0] = 1.0
    utau = torch.tensor([0.0, 1.0], dtype=torch.float64)
    fisot = torch.full((2, 3), 1.0 / torch.pi, dtype=torch.float64)
    fluxes = solve_tp9_flux(prop, utau, fisot, nstr=4, nmom=4)

    assert fluxes.shape == (2, 3, 2, 2)
    assert torch.allclose(
        fluxes[..., 0], torch.zeros_like(fluxes[..., 0]), atol=1e-14, rtol=0
    )
    assert torch.allclose(
        fluxes[..., 0, 1],
        torch.ones((2, 3), dtype=torch.float64),
        atol=1e-13,
        rtol=0,
    )


def test_tp9_beam_source_is_zero_without_scattering_and_batched():
    from pydisort.tensor_backend import (
        build_tp9_beam_source,
        gaussian_quadrature,
        prepare_layer_optics,
    )

    prop = torch.zeros((2, 3, 4, 6), dtype=torch.float64)
    prop[..., 0] = 1.0
    atmosphere = prepare_atmosphere(prop, nstr=4, nmom=4)
    optics = prepare_layer_optics(atmosphere, nstr=4, deltam=False)
    source = build_tp9_beam_source(
        optics,
        gaussian_quadrature(4, device="cpu"),
        torch.full((2, 3), 0.5, dtype=torch.float64),
        torch.ones((2, 3), dtype=torch.float64),
        nstr=4,
    )
    assert source.shape == (2, 3, 4, 4)
    assert torch.equal(source, torch.zeros_like(source))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tp9_fourier_order_one_beam_source_matches_native_trace(device):
    """Gate the first nonzero Fourier beam solution against C-DISORT."""
    from pydisort.tensor_backend import (
        build_tp9_beam_source,
        gaussian_quadrature,
        prepare_layer_optics,
    )

    prop = torch.zeros((1, 1, 5, 6), dtype=torch.float64, device=device)
    prop[..., 0] = torch.tensor(
        [0.15, 0.4, 0.5, 0.7, 0.65], dtype=torch.float64, device=device
    )
    prop[..., 1] = torch.tensor(
        [0.2, 0.55, 0.75, 0.45, 0.85], dtype=torch.float64, device=device
    )
    asymmetry = torch.tensor(
        [0.0, 0.25, 0.55, 0.1, 0.7], dtype=torch.float64, device=device
    )
    prop[..., 2:] = asymmetry.view(1, 1, 5, 1).pow(
        torch.arange(1, 5, dtype=torch.float64, device=device)
    )
    optics = prepare_layer_optics(
        prepare_atmosphere(prop, nstr=4, nmom=4), nstr=4, deltam=True
    )
    source = build_tp9_beam_source(
        optics,
        gaussian_quadrature(4, device=device),
        torch.full((1, 1), 0.5, dtype=torch.float64, device=device),
        torch.full((1, 1), torch.pi, dtype=torch.float64, device=device),
        nstr=4,
        fourier_order=1,
    )
    # c_upbeam's ZJ order is (positive streams, negative streams); convert
    # the tensor ZZ order back before comparing the opt-in native trace.
    native_order = torch.cat(
        (source[..., 2:], source[..., :2].flip(dims=(-1,))), dim=-1
    )
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [
                0.058617081508026109,
                0.013679639354586926,
                0.17973449413972381,
                -0.14392546019576868,
            ],
            [
                0.18918872924579241,
                0.023842982980122218,
                0.62788519135547827,
                -0.48604298015713249,
            ],
            [
                0.019601862069725889,
                0.0058567487140942691,
                0.0533916177098979,
                -0.038095492278107823,
            ],
            [
                0.30896761029373676,
                0.026374657055707542,
                1.0001639795815007,
                -0.67841221607980373,
            ],
        ],
        dtype=torch.float64,
        device=device,
    )
    assert torch.allclose(native_order[0, 0], expected, rtol=5e-8, atol=5e-10)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_tp9_nonzero_fourier_user_rays_match_cdisort(device, order):
    """Gate each resolved nonzero Fourier component against C-DISORT."""
    from pydisort.tensor_backend import (
        build_reduced_eigen_matrix,
        build_tp9_beam_source,
        build_tp9_boundary_system,
        extract_tp9_user_intensity_m0,
        gaussian_quadrature,
        interpolate_tp9_eigenvectors,
        interpolate_tp9_user_beam_source,
        prepare_layer_optics,
        prepare_output_grid,
        solve_reduced_eigenproblem,
        solve_tp9_boundary_system,
    )

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_user_ray_beam_five_layer_fourier_reference.json"
        ).read_text()
    )
    prop = torch.zeros((1, 1, 5, 6), dtype=torch.float64, device=device)
    prop[..., 0] = torch.tensor(
        fixture["dtau"], dtype=torch.float64, device=device
    )
    prop[..., 1] = torch.tensor(
        fixture["ssalb"], dtype=torch.float64, device=device
    )
    asymmetry = torch.tensor(fixture["gg"], dtype=torch.float64, device=device)
    prop[..., 2:] = asymmetry.view(1, 1, 5, 1).pow(
        torch.arange(1, 5, dtype=torch.float64, device=device)
    )
    atmosphere = prepare_atmosphere(prop, nstr=4, nmom=4)
    optics = prepare_layer_optics(atmosphere, nstr=4, deltam=True)
    tau = torch.tensor(fixture["user_tau"], dtype=torch.float64, device=device)
    tau[-1] = optics.tauc[..., -1].max()
    grid = prepare_output_grid(tau, atmosphere, optics, deltam=True)
    quadrature = gaussian_quadrature(4, device=device)
    user_mu = torch.tensor(
        fixture["user_mu"], dtype=torch.float64, device=device
    )
    umu0 = torch.full(
        (1, 1), fixture["umu0"], dtype=torch.float64, device=device
    )
    fbeam = torch.full(
        (1, 1), fixture["fbeam"], dtype=torch.float64, device=device
    )
    fisot = torch.zeros((1, 1), dtype=torch.float64, device=device)
    eigenvalues, eigenvectors = solve_reduced_eigenproblem(
        build_reduced_eigen_matrix(
            optics, quadrature, nstr=4, fourier_order=order
        )
    )
    beam_source = build_tp9_beam_source(
        optics, quadrature, umu0, fbeam, nstr=4, fourier_order=order
    )
    constants = solve_tp9_boundary_system(
        build_tp9_boundary_system(
            eigenvectors, eigenvalues, optics, fisot, beam_source, None, umu0
        )
    )
    actual = extract_tp9_user_intensity_m0(
        interpolate_tp9_eigenvectors(
            eigenvectors,
            optics,
            quadrature,
            user_mu,
            fourier_order=order,
        ),
        eigenvalues,
        optics,
        grid,
        constants,
        user_mu,
        fisot,
        interpolate_tp9_user_beam_source(
            beam_source,
            optics,
            quadrature,
            user_mu,
            umu0,
            fbeam,
            fourier_order=order,
        ),
        umu0,
    )
    expected = torch.tensor(
        fixture["fourier_components"][order],
        dtype=torch.float64,
        device=device,
    ).reshape(6, 4)
    assert torch.allclose(actual[0, 0], expected, rtol=5e-8, atol=5e-10)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_beam_flux_matches_self_contained_cdisort_fixture(device):
    from pydisort.tensor_backend import solve_tp9_flux

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_beam_flux_reference.json"
        ).read_text()
    )
    nstr = fixture["nstr"]
    prop = torch.zeros((1, 1, 1, 2 + nstr), dtype=torch.float64, device=device)
    prop[..., 0] = fixture["dtauc"][0]
    prop[..., 1] = fixture["ssalb"][0]
    fluxes = solve_tp9_flux(
        prop,
        torch.tensor(fixture["user_tau"], dtype=torch.float64, device=device),
        torch.zeros((1, 1), dtype=torch.float64, device=device),
        nstr=nstr,
        nmom=nstr,
        umu0=torch.full(
            (1, 1), fixture["umu0"], dtype=torch.float64, device=device
        ),
        fbeam=torch.full(
            (1, 1), fixture["fbeam"], dtype=torch.float64, device=device
        ),
    )
    assert torch.allclose(
        fluxes[0, 0].cpu(),
        torch.tensor(fixture["flux"], dtype=torch.float64),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_lambertian_flux_matches_problem_6c_fixture(device):
    from pydisort.tensor_backend import solve_tp9_flux

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_lambertian_flux_reference.json"
        ).read_text()
    )
    nstr = fixture["nstr"]
    prop = torch.zeros((1, 1, 1, 2 + nstr), dtype=torch.float64, device=device)
    prop[..., 0] = fixture["dtauc"][0]
    prop[..., 1] = fixture["ssalb"][0]
    fluxes = solve_tp9_flux(
        prop,
        torch.tensor(fixture["user_tau"], dtype=torch.float64, device=device),
        torch.zeros((1, 1), dtype=torch.float64, device=device),
        nstr=nstr,
        nmom=nstr,
        umu0=torch.full(
            (1, 1), fixture["umu0"], dtype=torch.float64, device=device
        ),
        fbeam=torch.full(
            (1, 1), fixture["fbeam"], dtype=torch.float64, device=device
        ),
        surface_albedo=torch.full(
            (1, 1),
            fixture["surface_albedo"],
            dtype=torch.float64,
            device=device,
        ),
    )
    assert torch.allclose(
        fluxes[0, 0].cpu(),
        torch.tensor(fixture["flux"], dtype=torch.float64),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_thermal_flux_matches_self_contained_cdisort_fixture(device):
    from pydisort.tensor_backend import solve_tp9_flux

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_thermal_flux_reference.json"
        ).read_text()
    )
    nstr = fixture["nstr"]
    prop = torch.zeros((1, 1, 1, 2 + nstr), dtype=torch.float64, device=device)
    prop[..., 0] = fixture["dtauc"][0]
    prop[..., 1] = fixture["ssalb"][0]
    fluxes = solve_tp9_flux(
        prop,
        torch.tensor(fixture["user_tau"], dtype=torch.float64, device=device),
        torch.zeros((1, 1), dtype=torch.float64, device=device),
        nstr=nstr,
        nmom=nstr,
        temperature=torch.tensor(
            fixture["temperature"], dtype=torch.float64, device=device
        ).reshape(1, 1, -1),
        wavenumber_lower=torch.full(
            (1, 1),
            fixture["wavenumber_lower"],
            dtype=torch.float64,
            device=device,
        ),
        wavenumber_upper=torch.full(
            (1, 1),
            fixture["wavenumber_upper"],
            dtype=torch.float64,
            device=device,
        ),
    )
    assert torch.allclose(
        fluxes[0, 0].cpu(),
        torch.tensor(fixture["flux"], dtype=torch.float64),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_thermal_surface_flux_matches_fortran_validated_fixture(device):
    from pydisort.tensor_backend import solve_tp9_flux

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_thermal_surface_flux_reference.json"
        ).read_text()
    )
    nstr = fixture["nstr"]
    prop = torch.zeros((1, 1, 1, 2 + nstr), dtype=torch.float64, device=device)
    prop[..., 0] = fixture["dtauc"][0]
    prop[..., 1] = fixture["ssalb"][0]
    fluxes = solve_tp9_flux(
        prop,
        torch.tensor(fixture["user_tau"], dtype=torch.float64, device=device),
        torch.zeros((1, 1), dtype=torch.float64, device=device),
        nstr=nstr,
        nmom=nstr,
        temperature=torch.tensor(
            fixture["temperature"], dtype=torch.float64, device=device
        ).reshape(1, 1, -1),
        bottom_temperature=torch.full(
            (1, 1),
            fixture["bottom_temperature"],
            dtype=torch.float64,
            device=device,
        ),
        top_temperature=torch.full(
            (1, 1),
            fixture["top_temperature"],
            dtype=torch.float64,
            device=device,
        ),
        top_emissivity=torch.full(
            (1, 1),
            fixture["top_emissivity"],
            dtype=torch.float64,
            device=device,
        ),
        wavenumber_lower=torch.full(
            (1, 1),
            fixture["wavenumber_lower"],
            dtype=torch.float64,
            device=device,
        ),
        wavenumber_upper=torch.full(
            (1, 1),
            fixture["wavenumber_upper"],
            dtype=torch.float64,
            device=device,
        ),
    )
    assert torch.allclose(
        fluxes[0, 0].cpu(),
        torch.tensor(fixture["flux"], dtype=torch.float64),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_thermal_lambertian_flux_matches_reference_fixture(device):
    from pydisort.tensor_backend import solve_tp9_flux

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_thermal_lambertian_flux_reference.json"
        ).read_text()
    )
    nstr = fixture["nstr"]
    prop = torch.zeros((1, 1, 1, 2 + nstr), dtype=torch.float64, device=device)
    prop[..., 0] = fixture["dtauc"][0]
    prop[..., 1] = fixture["ssalb"][0]
    scalar_inputs = {
        name: torch.full(
            (1, 1), fixture[name], dtype=torch.float64, device=device
        )
        for name in (
            "bottom_temperature",
            "top_temperature",
            "top_emissivity",
            "surface_albedo",
            "wavenumber_lower",
            "wavenumber_upper",
        )
    }
    fluxes = solve_tp9_flux(
        prop,
        torch.tensor(fixture["user_tau"], dtype=torch.float64, device=device),
        torch.zeros((1, 1), dtype=torch.float64, device=device),
        nstr=nstr,
        nmom=nstr,
        temperature=torch.tensor(
            fixture["temperature"], dtype=torch.float64, device=device
        ).reshape(1, 1, -1),
        **scalar_inputs,
    )
    assert torch.allclose(
        fluxes[0, 0].cpu(),
        torch.tensor(fixture["flux"], dtype=torch.float64),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


def _tp9c_source_component_flux(
    device: str, component: str
) -> tuple[torch.Tensor, dict]:
    from pydisort.tensor_backend import solve_tp9_flux

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_tp9c_source_decomposition_reference.json"
        ).read_text()
    )
    nstr = fixture["nstr"]
    prop = torch.zeros(
        (1, 1, len(fixture["dtauc"]), 2 + nstr),
        dtype=torch.float64,
        device=device,
    )
    prop[..., 0] = torch.tensor(
        fixture["dtauc"], dtype=torch.float64, device=device
    )
    prop[..., 1] = torch.tensor(
        fixture["ssalb"], dtype=torch.float64, device=device
    )
    asymmetry = torch.tensor(
        fixture["asymmetry"], dtype=torch.float64, device=device
    )
    prop[0, 0, :, 2:] = torch.stack(
        [asymmetry.pow(degree) for degree in range(1, nstr + 1)], dim=-1
    )
    common = {
        "nstr": nstr,
        "nmom": nstr,
        "deltam": True,
        "surface_albedo": torch.full(
            (1, 1),
            fixture["surface_albedo"],
            dtype=torch.float64,
            device=device,
        ),
    }
    fisot = torch.zeros((1, 1), dtype=torch.float64, device=device)
    if component in {"diffuse_only", "combined"}:
        fisot.fill_(fixture["fisot"])
    if component in {"beam_only", "combined"}:
        common.update(
            umu0=torch.full(
                (1, 1), fixture["umu0"], dtype=torch.float64, device=device
            ),
            fbeam=torch.full(
                (1, 1), fixture["fbeam"], dtype=torch.float64, device=device
            ),
        )
    if component in {"thermal_only", "combined"}:
        common.update(
            temperature=torch.tensor(
                fixture["temperature"], dtype=torch.float64, device=device
            ).reshape(1, 1, -1),
            bottom_temperature=torch.full(
                (1, 1),
                fixture["bottom_temperature"],
                dtype=torch.float64,
                device=device,
            ),
            top_temperature=torch.full(
                (1, 1),
                fixture["top_temperature"],
                dtype=torch.float64,
                device=device,
            ),
            top_emissivity=torch.full(
                (1, 1),
                fixture["top_emissivity"],
                dtype=torch.float64,
                device=device,
            ),
            wavenumber_lower=torch.full(
                (1, 1),
                fixture["wavenumber_lower"],
                dtype=torch.float64,
                device=device,
            ),
            wavenumber_upper=torch.full(
                (1, 1),
                fixture["wavenumber_upper"],
                dtype=torch.float64,
                device=device,
            ),
        )
    if component not in {
        "diffuse_only",
        "beam_only",
        "thermal_only",
        "combined",
    }:
        raise ValueError(f"unknown source component: {component}")
    return (
        solve_tp9_flux(
            prop,
            torch.tensor(
                fixture["user_tau"], dtype=torch.float64, device=device
            ),
            fisot,
            **common,
        )[0, 0],
        fixture,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_tp9c_diffuse_component_matches_cdisort_fixture(device):
    fluxes, fixture = _tp9c_source_component_flux(device, "diffuse_only")

    assert torch.allclose(
        fluxes.cpu(),
        torch.tensor(fixture["diffuse_only_flux"], dtype=torch.float64),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_tp9c_beam_component_matches_cdisort_fixture(device):
    fluxes, fixture = _tp9c_source_component_flux(device, "beam_only")

    assert torch.allclose(
        fluxes.cpu(),
        torch.tensor(fixture["beam_only_flux"], dtype=torch.float64),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_tp9c_thermal_component_matches_cdisort_fixture(device):
    fluxes, fixture = _tp9c_source_component_flux(device, "thermal_only")

    assert torch.allclose(
        fluxes.cpu(),
        torch.tensor(fixture["thermal_only_flux"], dtype=torch.float64),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


def test_tp9_thermal_source_matches_nonscattering_c_upisot_solution():
    from pydisort.tensor_backend import (
        build_tp9_thermal_source,
        gaussian_quadrature,
        prepare_layer_optics,
    )

    prop = torch.zeros((1, 1, 1, 6), dtype=torch.float64)
    prop[..., 0] = 1.0
    atmosphere = prepare_atmosphere(prop, nstr=4, nmom=4)
    optics = prepare_layer_optics(atmosphere, nstr=4, deltam=False)
    xr0 = torch.tensor([[[2.0]]], dtype=torch.float64)
    xr1 = torch.tensor([[[3.0]]], dtype=torch.float64)
    z0, z1 = build_tp9_thermal_source(
        optics, gaussian_quadrature(4, device="cpu"), xr0, xr1, nstr=4
    )
    mu = gaussian_quadrature(4, device="cpu").cmu
    expected_z1 = torch.cat((torch.full((2,), 3.0), torch.full((2,), 3.0)))
    expected_z0 = torch.cat(
        ((2.0 - 3.0 * mu[:2]).flip(dims=(-1,)), 2.0 + 3.0 * mu[:2])
    )
    assert torch.equal(z1[0, 0, 0], expected_z1)
    assert torch.allclose(z0[0, 0, 0], expected_z0, atol=1e-14, rtol=0)


def test_thermal_particular_solution_contributes_to_user_fluxes():
    from pydisort.tensor_backend import (
        build_reduced_eigen_matrix,
        extract_tp9_fluxes,
        gaussian_quadrature,
        prepare_layer_optics,
        prepare_output_grid,
        solve_reduced_eigenproblem,
    )

    prop = torch.zeros((1, 1, 1, 6), dtype=torch.float64)
    prop[..., 0] = 1.0
    atmosphere = prepare_atmosphere(prop, nstr=4, nmom=4)
    optics = prepare_layer_optics(atmosphere, nstr=4, deltam=False)
    quadrature = gaussian_quadrature(4, device="cpu")
    values, vectors = solve_reduced_eigenproblem(
        build_reduced_eigen_matrix(optics, quadrature, nstr=4)
    )
    grid = prepare_output_grid(
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        atmosphere,
        optics,
        deltam=False,
    )
    thermal0 = torch.full((1, 1, 1, 4), 2.0, dtype=torch.float64)
    thermal1 = torch.zeros_like(thermal0)
    fluxes = extract_tp9_fluxes(
        vectors,
        values,
        optics,
        grid,
        quadrature,
        torch.zeros((1, 1, 4), dtype=torch.float64),
        thermal0=thermal0,
        thermal1=thermal1,
    )
    expected = torch.full((2, 2), 2.0 * torch.pi, dtype=torch.float64)
    assert torch.allclose(fluxes[0, 0], expected, atol=1e-14, rtol=0)


def test_solve_tp9_flux_accepts_connected_thermal_coefficients():
    from pydisort.tensor_backend import solve_tp9_flux

    prop = torch.zeros((1, 1, 1, 6), dtype=torch.float64)
    prop[..., 0] = 1.0
    fluxes = solve_tp9_flux(
        prop,
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        nstr=4,
        nmom=4,
        thermal_xr0=torch.full((1, 1, 1), 2.0, dtype=torch.float64),
        thermal_xr1=torch.zeros((1, 1, 1), dtype=torch.float64),
    )
    assert torch.allclose(
        fluxes[0, 0, 0, 1],
        torch.zeros((), dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    assert torch.allclose(
        fluxes[0, 0, 1, 0],
        torch.zeros((), dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    assert fluxes[0, 0, 0, 0] > 0
    assert torch.allclose(
        fluxes[0, 0, 0, 0], fluxes[0, 0, 1, 1], atol=1e-14, rtol=0
    )


def test_thermal_coefficients_preserve_an_isothermal_planck_source():
    from pydisort.tensor_backend import (
        planck_band_radiance,
        prepare_layer_optics,
        prepare_thermal_coefficients,
    )

    prop = torch.zeros((1, 1, 2, 6), dtype=torch.float64)
    prop[..., 0] = torch.tensor([1.0, 2.0], dtype=torch.float64)
    atmosphere = prepare_atmosphere(prop, nstr=4, nmom=4)
    optics = prepare_layer_optics(atmosphere, nstr=4, deltam=False)
    temperature = torch.full((1, 1, 3), 600.0, dtype=torch.float64)
    lower = torch.full((1, 1), 999.0, dtype=torch.float64)
    upper = torch.full((1, 1), 1000.0, dtype=torch.float64)
    xr0, xr1 = prepare_thermal_coefficients(temperature, optics, lower, upper)
    expected = planck_band_radiance(temperature[..., :1], lower, upper)
    assert torch.equal(xr1, torch.zeros_like(xr1))
    assert torch.allclose(xr0, expected.expand_as(xr0), atol=1e-14, rtol=0)


def test_solve_tp9_flux_accepts_temperature_and_wavenumber_inputs():
    from pydisort.tensor_backend import solve_tp9_flux

    prop = torch.zeros((1, 1, 1, 6), dtype=torch.float64)
    prop[..., 0] = 1.0
    common = {
        "nstr": 4,
        "nmom": 4,
        "temperature": torch.full((1, 1, 2), 600.0, dtype=torch.float64),
        "wavenumber_lower": torch.full((1, 1), 999.0, dtype=torch.float64),
        "wavenumber_upper": torch.full((1, 1), 1000.0, dtype=torch.float64),
    }
    fluxes = solve_tp9_flux(
        prop,
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        **common,
    )
    assert torch.isfinite(fluxes).all()
    assert torch.allclose(
        fluxes[0, 0, 0, 1],
        torch.zeros((), dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    assert torch.allclose(
        fluxes[0, 0, 1, 0],
        torch.zeros((), dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_tp9c_combined_sources_match_cdisort_fixture(device):
    fluxes, fixture = _tp9c_source_component_flux(device, "combined")

    assert torch.allclose(
        fluxes.cpu(),
        torch.tensor(fixture["combined_flux"], dtype=torch.float64),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_tp1_flux_cases_match_cdisort_fixtures(device):
    from pydisort.tensor_backend import solve_tp9_flux

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_tp1_flux_reference.json"
        ).read_text()
    )
    for case in fixture["cases"]:
        nlyr = len(case["dtauc"])
        nstr = case["nstr"]
        prop = torch.zeros(
            (1, 1, nlyr, 2 + nstr), dtype=torch.float64, device=device
        )
        prop[..., 0] = torch.tensor(
            case["dtauc"], dtype=torch.float64, device=device
        )
        prop[..., 1] = torch.tensor(
            case["ssalb"], dtype=torch.float64, device=device
        )
        fluxes = solve_tp9_flux(
            prop,
            torch.tensor(case["user_tau"], dtype=torch.float64, device=device),
            torch.full(
                (1, 1), case["fisot"], dtype=torch.float64, device=device
            ),
            nstr=nstr,
            nmom=nstr,
            deltam=True,
            umu0=torch.full(
                (1, 1), case["umu0"], dtype=torch.float64, device=device
            ),
            fbeam=torch.full(
                (1, 1), case["fbeam"], dtype=torch.float64, device=device
            ),
            surface_albedo=torch.full(
                (1, 1),
                case["surface_albedo"],
                dtype=torch.float64,
                device=device,
            ),
        )
        assert torch.allclose(
            fluxes[0, 0].cpu(),
            torch.tensor(case["flux"], dtype=torch.float64),
            atol=fixture["atol"],
            rtol=fixture["rtol"],
        ), case["label"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_tp2_tp3_flux_cases_match_cdisort_fixtures(device):
    from pydisort.tensor_backend import solve_tp9_flux

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_tp2_tp3_flux_reference.json"
        ).read_text()
    )
    for case in fixture["cases"]:
        prop = torch.tensor(case["prop"], dtype=torch.float64, device=device)
        prop = prop.unsqueeze(0).unsqueeze(0)
        fluxes = solve_tp9_flux(
            prop,
            torch.tensor(case["user_tau"], dtype=torch.float64, device=device),
            torch.full(
                (1, 1), case["fisot"], dtype=torch.float64, device=device
            ),
            nstr=case["nstr"],
            nmom=case["nmom"],
            deltam=True,
            umu0=torch.full(
                (1, 1), case["umu0"], dtype=torch.float64, device=device
            ),
            fbeam=torch.full(
                (1, 1), case["fbeam"], dtype=torch.float64, device=device
            ),
            surface_albedo=torch.full(
                (1, 1),
                case["surface_albedo"],
                dtype=torch.float64,
                device=device,
            ),
        )
        assert torch.allclose(
            fluxes[0, 0].cpu(),
            torch.tensor(case["flux"], dtype=torch.float64),
            atol=fixture["atol"],
            rtol=fixture["rtol"],
        ), case["label"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_general_source_matches_cdisort_reference(device):
    from pydisort.tensor_backend import solve_tp9_flux

    nstr = 4
    prop = torch.zeros((1, 1, 1, 2 + nstr), dtype=torch.float64, device=device)
    prop[..., 0] = 0.2
    prop[..., 1] = 0.5
    source = torch.ones(
        (1, 1, nstr, 1, nstr), dtype=torch.float64, device=device
    )
    zero = torch.zeros((1, 1), dtype=torch.float64, device=device)
    result = solve_tp9_flux(
        prop,
        torch.tensor([0.0, 0.2], dtype=torch.float64, device=device),
        zero,
        nstr=nstr,
        nmom=nstr,
        general_source_computational=source,
    )
    expected = torch.tensor(
        [[1.0891838249147526, 0.0], [0.0, 1.0891838249147526]],
        dtype=torch.float64,
    )
    assert torch.allclose(result[0, 0].cpu(), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_zero_general_source_recovers_no_source_flux(device):
    from pydisort.tensor_backend import solve_tp9_flux

    nstr = 4
    prop = torch.zeros((1, 1, 1, 2 + nstr), dtype=torch.float64, device=device)
    prop[..., 0] = 0.2
    prop[..., 1] = 0.5
    zero = torch.zeros((1, 1), dtype=torch.float64, device=device)
    common = {
        "nstr": nstr,
        "nmom": nstr,
    }
    plain = solve_tp9_flux(
        prop,
        torch.tensor([0.0, 0.2], dtype=torch.float64, device=device),
        zero,
        **common,
    )
    sourced = solve_tp9_flux(
        prop,
        torch.tensor([0.0, 0.2], dtype=torch.float64, device=device),
        zero,
        general_source_computational=torch.zeros(
            (1, 1, nstr, 1, nstr), dtype=torch.float64, device=device
        ),
        **common,
    )
    assert torch.allclose(sourced, plain, atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_multilayer_general_source_matches_cdisort_fixture(device):
    from pydisort.tensor_backend import solve_tp9_flux

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_general_source_two_layer_reference.json"
        ).read_text()
    )
    nstr = fixture["nstr"]
    prop = torch.tensor(fixture["prop"], dtype=torch.float64, device=device)
    result = solve_tp9_flux(
        prop.unsqueeze(0).unsqueeze(0),
        torch.tensor(fixture["utau"], dtype=torch.float64, device=device),
        torch.zeros((1, 1), dtype=torch.float64, device=device),
        nstr=nstr,
        nmom=nstr,
        general_source_computational=torch.tensor(
            fixture["computational"], dtype=torch.float64, device=device
        ),
    )
    assert torch.allclose(
        result[0, 0].cpu(),
        torch.tensor(fixture["flux"], dtype=torch.float64),
        atol=fixture["atol"],
        rtol=fixture["rtol"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_tp4_haze_flux_cases_match_cdisort_fixtures(device):
    """Keep high-order Haze-L beam fluxes aligned with native C-DISORT."""
    from pydisort.tensor_backend import solve_tp9_flux

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_tp4_flux_reference.json"
        ).read_text()
    )
    for case in fixture["cases"]:
        prop = torch.tensor(
            [*case["dtauc"], *case["ssalb"], *case["moments"]],
            dtype=torch.float64,
            device=device,
        ).reshape(1, 1, 1, -1)
        fluxes = solve_tp9_flux(
            prop,
            torch.tensor(case["user_tau"], dtype=torch.float64, device=device),
            torch.full(
                (1, 1), case["fisot"], dtype=torch.float64, device=device
            ),
            nstr=case["nstr"],
            nmom=case["nmom"],
            deltam=True,
            umu0=torch.full(
                (1, 1), case["umu0"], dtype=torch.float64, device=device
            ),
            fbeam=torch.full(
                (1, 1), case["fbeam"], dtype=torch.float64, device=device
            ),
            surface_albedo=torch.full(
                (1, 1),
                case["surface_albedo"],
                dtype=torch.float64,
                device=device,
            ),
        )
        assert torch.allclose(
            fluxes[0, 0].cpu(),
            torch.tensor(case["flux"], dtype=torch.float64),
            atol=fixture["atol"],
            rtol=fixture["rtol"],
        ), case["label"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_tp5_cloud_flux_cases_match_cdisort_fixtures(device):
    """Keep 48-stream Cloud C.1 beam fluxes aligned with native C-DISORT."""
    from pydisort.tensor_backend import solve_tp9_flux

    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_tp5_flux_reference.json"
        ).read_text()
    )
    for case in fixture["cases"]:
        prop = torch.tensor(
            [*case["dtauc"], *case["ssalb"], *case["moments"]],
            dtype=torch.float64,
            device=device,
        ).reshape(1, 1, 1, -1)
        fluxes = solve_tp9_flux(
            prop,
            torch.tensor(case["user_tau"], dtype=torch.float64, device=device),
            torch.full(
                (1, 1), case["fisot"], dtype=torch.float64, device=device
            ),
            nstr=case["nstr"],
            nmom=case["nmom"],
            deltam=True,
            umu0=torch.full(
                (1, 1), case["umu0"], dtype=torch.float64, device=device
            ),
            fbeam=torch.full(
                (1, 1), case["fbeam"], dtype=torch.float64, device=device
            ),
            surface_albedo=torch.full(
                (1, 1),
                case["surface_albedo"],
                dtype=torch.float64,
                device=device,
            ),
        )
        assert torch.allclose(
            fluxes[0, 0].cpu(),
            torch.tensor(case["flux"], dtype=torch.float64),
            atol=fixture["atol"],
            rtol=fixture["rtol"],
        ), case["label"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_quadrature_intensity_integrates_to_homogeneous_flux(device):
    """The first m=0 radiance increment must reproduce its flux parent."""
    from pydisort.tensor_backend import (
        build_reduced_eigen_matrix,
        build_tp9_boundary_system,
        extract_tp9_fluxes,
        extract_tp9_quadrature_intensity,
        gaussian_quadrature,
        prepare_atmosphere,
        prepare_layer_optics,
        prepare_output_grid,
        solve_reduced_eigenproblem,
        solve_tp9_boundary_system,
    )

    nstr = 4
    prop = torch.zeros((1, 1, 1, 2 + nstr), dtype=torch.float64, device=device)
    prop[..., 0] = 0.7
    prop[..., 1] = 0.4
    fisot = torch.full(
        (1, 1), 1.0 / torch.pi, dtype=torch.float64, device=device
    )
    atmosphere = prepare_atmosphere(prop, nstr=nstr, nmom=nstr)
    optics = prepare_layer_optics(atmosphere, nstr=nstr, deltam=False)
    grid = prepare_output_grid(
        torch.tensor([0.0, 0.7], dtype=torch.float64, device=device),
        atmosphere,
        optics,
        deltam=False,
    )
    quadrature = gaussian_quadrature(nstr, device=device)
    values, vectors = solve_reduced_eigenproblem(
        build_reduced_eigen_matrix(optics, quadrature, nstr=nstr)
    )
    constants = solve_tp9_boundary_system(
        build_tp9_boundary_system(vectors, values, optics, fisot)
    )
    intensity = extract_tp9_quadrature_intensity(
        vectors, values, optics, grid, constants
    )
    fluxes = extract_tp9_fluxes(
        vectors, values, optics, grid, quadrature, constants
    )
    nn = nstr // 2
    expected_upward = (
        2.0
        * torch.pi
        * torch.sum(
            intensity[..., nn:] * quadrature.cwt[:nn] * quadrature.cmu[:nn],
            dim=-1,
        )
    )
    expected_downward = (
        2.0
        * torch.pi
        * torch.sum(
            intensity[..., :nn]
            * (quadrature.cwt[:nn] * quadrature.cmu[:nn]).flip(0),
            dim=-1,
        )
    )
    assert torch.allclose(
        fluxes[..., 0], expected_upward, atol=1e-13, rtol=1e-12
    )
    assert torch.allclose(
        fluxes[..., 1], expected_downward, atol=1e-13, rtol=1e-12
    )
    reference = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_quadrature_radiance_reference.json"
        ).read_text()
    )
    assert torch.allclose(
        intensity[0, 0].cpu(),
        torch.tensor(reference["radiance"], dtype=torch.float64),
        atol=reference["atol"],
        rtol=reference["rtol"],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_top_boundary_user_intensity(device):
    from pydisort.tensor_backend import (
        TensorOutputGrid,
        top_boundary_user_intensity,
    )

    grid = TensorOutputGrid(
        layru=torch.ones((1, 1, 2), dtype=torch.long, device=device),
        utaupr=torch.tensor(
            [[[0.0, 0.7]]], dtype=torch.float64, device=device
        ),
    )
    result = top_boundary_user_intensity(
        grid,
        torch.tensor([-0.5, 0.5], dtype=torch.float64, device=device),
        torch.full((1, 1), 1.0 / torch.pi, dtype=torch.float64, device=device),
    )
    expected = torch.tensor(
        [[[[1.0 / torch.pi, 0.0], [math.exp(-1.4) / torch.pi, 0.0]]]],
        dtype=torch.float64,
        device=device,
    )
    assert torch.allclose(result, expected, atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_one_layer_user_ray_matches_cdisort(device):
    from pydisort.tensor_backend import (
        build_reduced_eigen_matrix,
        build_tp9_boundary_system,
        extract_tp9_user_intensity_one_layer_m0,
        gaussian_quadrature,
        interpolate_tp9_eigenvectors_m0,
        prepare_atmosphere,
        prepare_layer_optics,
        prepare_output_grid,
        solve_reduced_eigenproblem,
        solve_tp9_boundary_system,
    )

    prop = torch.zeros((1, 1, 1, 6), dtype=torch.float64, device=device)
    prop[..., 0] = 0.7
    prop[..., 1] = 0.4
    fisot = torch.full(
        (1, 1), 1.0 / torch.pi, dtype=torch.float64, device=device
    )
    atmosphere = prepare_atmosphere(prop, nstr=4, nmom=4)
    optics = prepare_layer_optics(atmosphere, nstr=4, deltam=False)
    grid = prepare_output_grid(
        torch.tensor([0.0, 0.35, 0.7], dtype=torch.float64, device=device),
        atmosphere,
        optics,
        deltam=False,
    )
    quadrature = gaussian_quadrature(4, device=device)
    values, vectors = solve_reduced_eigenproblem(
        build_reduced_eigen_matrix(optics, quadrature, nstr=4)
    )
    constants = solve_tp9_boundary_system(
        build_tp9_boundary_system(vectors, values, optics, fisot)
    )
    user_mu = torch.tensor(
        [-0.5, -0.3, 0.4, 0.5], dtype=torch.float64, device=device
    )
    actual = extract_tp9_user_intensity_one_layer_m0(
        interpolate_tp9_eigenvectors_m0(vectors, optics, quadrature, user_mu),
        values,
        optics,
        grid,
        constants,
        user_mu,
        fisot,
    )
    expected = torch.tensor(
        [
            [
                [
                    0.3183098861837907,
                    0.3183098861837907,
                    0.03747797124418063,
                    0.03314712755971079,
                ],
                [
                    0.1823428160669597,
                    0.1314565496137554,
                    0.016071063969282897,
                    0.013758364434119348,
                ],
                [0.10337447141158351, 0.0580759519444697, 0.0, 0.0],
            ]
        ],
        dtype=torch.float64,
        device=device,
    )
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "tensor_user_ray_two_layer_reference.json",
        "tensor_user_ray_five_layer_reference.json",
        "tensor_user_ray_beam_five_layer_reference.json",
        "tensor_user_ray_thermal_five_layer_reference.json",
        "tensor_user_ray_general_source_five_layer_reference.json",
        "tensor_user_ray_combined_five_layer_reference.json",
        "tensor_user_ray_lambertian_five_layer_reference.json",
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_multilayer_user_ray_matches_cdisort(device, fixture_name):
    from pydisort.tensor_backend import (
        build_reduced_eigen_matrix,
        build_tp9_beam_source,
        build_tp9_boundary_system,
        build_tp9_general_source,
        build_tp9_thermal_source,
        extract_tp9_user_intensity_m0,
        gaussian_quadrature,
        interpolate_tp9_eigenvectors_m0,
        interpolate_tp9_user_beam_source_m0,
        interpolate_tp9_user_general_source_m0,
        interpolate_tp9_user_thermal_source_m0,
        lambertian_user_boundary_intensity,
        prepare_atmosphere,
        prepare_layer_optics,
        prepare_output_grid,
        prepare_thermal_coefficients,
        solve_reduced_eigenproblem,
        solve_tp9_boundary_system,
    )

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / fixture_name).read_text()
    )
    layers = fixture.get("layers")
    if layers is None:
        layers = [
            {"dtau": dtau, "ssalb": ssalb, "gg": gg}
            for dtau, ssalb, gg in zip(
                fixture["dtau"], fixture["ssalb"], fixture["gg"]
            )
        ]
    nlyr = len(layers)
    prop = torch.zeros((1, 1, nlyr, 6), dtype=torch.float64, device=device)
    prop[..., 0] = torch.tensor(
        [layer["dtau"] for layer in layers], dtype=torch.float64, device=device
    )
    prop[..., 1] = torch.tensor(
        [layer["ssalb"] for layer in layers],
        dtype=torch.float64,
        device=device,
    )
    if any("gg" in layer for layer in layers):
        gg = torch.tensor(
            [layer.get("gg", 0.0) for layer in layers],
            dtype=torch.float64,
            device=device,
        )
        prop[..., 2:] = gg.view(1, 1, nlyr, 1).pow(
            torch.arange(1, 5, dtype=torch.float64, device=device)
        )
    fisot = torch.full(
        (1, 1), fixture["fisot"], dtype=torch.float64, device=device
    )
    atmosphere = prepare_atmosphere(prop, nstr=4, nmom=4)
    deltam = bool(torch.any(atmosphere.pmom[..., 4] != 0.0))
    optics = prepare_layer_optics(atmosphere, nstr=4, deltam=deltam)
    grid = prepare_output_grid(
        torch.tensor(fixture["user_tau"], dtype=torch.float64, device=device),
        atmosphere,
        optics,
        deltam=deltam,
    )
    quadrature = gaussian_quadrature(4, device=device)
    values, vectors = solve_reduced_eigenproblem(
        build_reduced_eigen_matrix(optics, quadrature, nstr=4)
    )
    user_mu = torch.tensor(
        fixture["user_mu"], dtype=torch.float64, device=device
    )
    user_thermal0 = user_thermal1 = user_general_source = None
    lambertian_boundary = None
    if "fbeam" in fixture:
        umu0 = torch.full(
            (1, 1), fixture["umu0"], dtype=torch.float64, device=device
        )
        fbeam = torch.full(
            (1, 1), fixture["fbeam"], dtype=torch.float64, device=device
        )
        beam_source = build_tp9_beam_source(
            optics, quadrature, umu0, fbeam, nstr=4
        )
        constants = solve_tp9_boundary_system(
            build_tp9_boundary_system(
                vectors,
                values,
                optics,
                fisot,
                beam_source,
                None,
                umu0,
                None,
                None,
                None,
                None,
                None,
                quadrature,
                fbeam,
            )
        )
    else:
        umu0 = None
        beam_source = None
        if "temperature" in fixture:
            temperature = torch.tensor(
                fixture["temperature"], dtype=torch.float64, device=device
            ).view(1, 1, -1)
            lower = torch.tensor(
                fixture["wavenumber_lower"], dtype=torch.float64, device=device
            )
            upper = torch.tensor(
                fixture["wavenumber_upper"], dtype=torch.float64, device=device
            )
            xr0, xr1 = prepare_thermal_coefficients(
                temperature, optics, lower, upper
            )
            thermal0, thermal1 = build_tp9_thermal_source(
                optics, quadrature, xr0, xr1, nstr=4
            )
            constants = solve_tp9_boundary_system(
                build_tp9_boundary_system(
                    vectors,
                    values,
                    optics,
                    fisot,
                    thermal0=thermal0,
                    thermal1=thermal1,
                )
            )
            (
                user_thermal0,
                user_thermal1,
            ) = interpolate_tp9_user_thermal_source_m0(
                thermal0, thermal1, optics, quadrature, user_mu, xr0, xr1
            )
        elif "computational_source" in fixture:
            computational = torch.zeros(
                (1, 1, 4, nlyr, 4), dtype=torch.float64, device=device
            )
            user_source = torch.zeros(
                (1, 1, nlyr, user_mu.numel()),
                dtype=torch.float64,
                device=device,
            )
            for layer in range(nlyr):
                for angle in range(4):
                    computational[0, 0, 0, layer, angle] = (
                        0.01 * (layer + 1) * (angle + 1)
                    )
                    user_source[0, 0, layer, angle] = (
                        0.02 * (layer + 1) * (angle + 1)
                    )
            general_source = build_tp9_general_source(
                optics, quadrature, computational, nstr=4
            )
            constants = solve_tp9_boundary_system(
                build_tp9_boundary_system(
                    vectors,
                    values,
                    optics,
                    fisot,
                    general_source=general_source,
                )
            )
            user_general_source = interpolate_tp9_user_general_source_m0(
                general_source, optics, quadrature, user_mu, user_source
            )
        else:
            constants = solve_tp9_boundary_system(
                build_tp9_boundary_system(vectors, values, optics, fisot)
            )
    if "fbeam" in fixture and "temperature" in fixture:
        temperature = torch.tensor(
            fixture["temperature"], dtype=torch.float64, device=device
        ).view(1, 1, -1)
        lower = torch.tensor(
            fixture["wavenumber_lower"], dtype=torch.float64, device=device
        )
        upper = torch.tensor(
            fixture["wavenumber_upper"], dtype=torch.float64, device=device
        )
        xr0, xr1 = prepare_thermal_coefficients(
            temperature, optics, lower, upper
        )
        thermal0, thermal1 = build_tp9_thermal_source(
            optics, quadrature, xr0, xr1, nstr=4
        )
        computational = torch.zeros(
            (1, 1, 4, nlyr, 4), dtype=torch.float64, device=device
        )
        user_source = torch.zeros(
            (1, 1, nlyr, user_mu.numel()), dtype=torch.float64, device=device
        )
        for layer in range(nlyr):
            for angle in range(4):
                computational[0, 0, 0, layer, angle] = (
                    0.01 * (layer + 1) * (angle + 1)
                )
                user_source[0, 0, layer, angle] = (
                    0.02 * (layer + 1) * (angle + 1)
                )
        general_source = build_tp9_general_source(
            optics, quadrature, computational, nstr=4
        )
        constants = solve_tp9_boundary_system(
            build_tp9_boundary_system(
                vectors,
                values,
                optics,
                fisot,
                beam_source,
                general_source,
                umu0,
                thermal0,
                thermal1,
                None,
                None,
                None,
                quadrature,
                fbeam,
            )
        )
        user_thermal0, user_thermal1 = interpolate_tp9_user_thermal_source_m0(
            thermal0, thermal1, optics, quadrature, user_mu, xr0, xr1
        )
        user_general_source = interpolate_tp9_user_general_source_m0(
            general_source, optics, quadrature, user_mu, user_source
        )
    if "surface_albedo" in fixture:
        surface_albedo = torch.full(
            (1, 1),
            fixture["surface_albedo"],
            dtype=torch.float64,
            device=device,
        )
        constants = solve_tp9_boundary_system(
            build_tp9_boundary_system(
                vectors,
                values,
                optics,
                fisot,
                surface_albedo=surface_albedo,
                quadrature=quadrature,
            )
        )
        lambertian_boundary = lambertian_user_boundary_intensity(
            vectors, values, optics, quadrature, constants, surface_albedo
        )
    actual = extract_tp9_user_intensity_m0(
        interpolate_tp9_eigenvectors_m0(vectors, optics, quadrature, user_mu),
        values,
        optics,
        grid,
        constants,
        user_mu,
        fisot,
        None
        if beam_source is None or umu0 is None
        else interpolate_tp9_user_beam_source_m0(
            beam_source, optics, quadrature, user_mu, umu0, fbeam
        ),
        umu0,
        user_thermal0,
        user_thermal1,
        user_general_source,
        lambertian_boundary,
    )
    expected = torch.tensor(
        fixture["radiance"], dtype=torch.float64, device=device
    )
    assert torch.allclose(actual[0, 0], expected, atol=2e-8, rtol=1e-8)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tp9_azimuthal_reconstruction_matches_native_trace(device):
    """Reconstruct C-DISORT's final phi=60-degree radiance from UUM terms."""
    from pydisort.tensor_backend import reconstruct_tp9_azimuthal_radiance

    component_fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_user_ray_beam_five_layer_fourier_reference.json"
        ).read_text()
    )
    reference = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "tensor_user_ray_beam_five_layer_azimuth_reference.json"
        ).read_text()
    )
    components = torch.tensor(
        component_fixture["fourier_components"],
        dtype=torch.float64,
        device=device,
    ).reshape(1, 1, 4, 6, 4)
    actual = reconstruct_tp9_azimuthal_radiance(
        components,
        torch.full(
            (1, 1),
            reference["phi_degrees"],
            dtype=torch.float64,
            device=device,
        ),
        phi0_degrees=reference["phi0_degrees"],
    )
    expected = torch.tensor(
        reference["radiance"], dtype=torch.float64, device=device
    )
    assert torch.allclose(actual[0, 0], expected, rtol=0.0, atol=2e-16)
