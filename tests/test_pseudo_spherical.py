"""Pseudo-spherical direct-beam geometry uses C-DISORT on CPU and CUDA."""

import pytest
import torch
from pydisort import Disort, DisortOptions

RADIUS_KM = 6371.0
LEVEL_ALTITUDES_KM = [100.0, 0.0]
DIRECT_BEAM_BOTTOM_REFERENCE = 4.4428045259949897e-05
DTYPE = torch.float64


def make_options(backend: str, radius: float = RADIUS_KM, altitudes=None):
    """Configure one clear, one-layer direct-beam C-DISORT calculation."""
    options = DisortOptions().flags("onlyfl,lamber,quiet").backend(backend)
    options.ds().nlyr = 1
    options.ds().nstr = options.ds().nmom = options.ds().nphase = 4
    options.pseudo_spherical(
        radius, LEVEL_ALTITUDES_KM if altitudes is None else altitudes
    )
    return options


def direct_beam_inputs(device: torch.device):
    """A non-scattering layer isolates C-DISORT's Chapman direct-beam path."""
    prop = torch.zeros((1, 1, 1, 6), device=device, dtype=DTYPE)
    prop[..., 0] = 1.0
    boundary = {
        "umu0": torch.tensor([0.1], device=device, dtype=DTYPE),
        "fbeam": torch.tensor([[1.0]], device=device, dtype=DTYPE),
    }
    return prop, boundary


def solve(backend: str, radius: float = RADIUS_KM, altitudes=None):
    """Solve the direct-beam reference configuration on one backend."""
    device = torch.device(backend)
    solver = Disort(make_options(backend, radius, altitudes))
    prop, boundary = direct_beam_inputs(device)
    result = solver(prop, **boundary)
    if backend == "cuda":
        torch.cuda.synchronize(device)
    return result.cpu()


def test_pseudo_spherical_direct_beam_changes_a_grazing_path():
    """Finite-radius Chapman geometry cannot silently use the planar path."""
    pseudo_spherical = solve("cpu")

    # C-DISORT's Chapman direct-beam calculation for a 6371 km body, a
    # 100 km layer, optical depth one, and umu0=0.1. The transparent medium
    # makes this a geometry-only reference rather than a solver snapshot.
    torch.testing.assert_close(
        pseudo_spherical[0, 0, -1, 1],
        torch.tensor(DIRECT_BEAM_BOTTOM_REFERENCE, dtype=DTYPE),
        rtol=1e-12,
        atol=1e-15,
    )

    planar_options = (
        DisortOptions().flags("onlyfl,lamber,quiet").backend("cpu")
    )
    planar_options.ds().nlyr = 1
    planar_options.ds().nstr = (
        planar_options.ds().nmom
    ) = planar_options.ds().nphase = 4
    prop, boundary = direct_beam_inputs(torch.device("cpu"))
    planar = Disort(planar_options)(prop, **boundary)

    # At a grazing incidence angle, curvature increases direct-beam
    # transmission relative to the plane-parallel secant approximation.
    assert pseudo_spherical[0, 0, -1, 1] > planar[0, 0, -1, 1]
    assert not torch.allclose(pseudo_spherical, planar, rtol=1e-4, atol=1e-12)


def test_large_radius_recovers_the_plane_parallel_limit():
    """A nearly flat body must reduce to the existing plane-parallel result."""
    near_planar = solve("cpu", radius=1.0e12)

    planar_options = (
        DisortOptions().flags("onlyfl,lamber,quiet").backend("cpu")
    )
    planar_options.ds().nlyr = 1
    planar_options.ds().nstr = (
        planar_options.ds().nmom
    ) = planar_options.ds().nphase = 4
    prop, boundary = direct_beam_inputs(torch.device("cpu"))
    planar = Disort(planar_options)(prop, **boundary)

    torch.testing.assert_close(near_planar, planar, rtol=2e-6, atol=1e-12)


@pytest.mark.parametrize(
    "radius, altitudes, message",
    [
        (0.0, LEVEL_ALTITUDES_KM, "radius must be finite and positive"),
        (RADIUS_KM, [0.0, 100.0], "strictly descending"),
        (RADIUS_KM, [100.0], r"nlyr \+ 1 values"),
    ],
)
def test_pseudo_spherical_geometry_is_validated(radius, altitudes, message):
    """Invalid geometry never reaches C-DISORT allocation or dispatch."""
    with pytest.raises(RuntimeError, match=message):
        Disort(make_options("cpu", radius, altitudes))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)
def test_pseudo_spherical_cuda_matches_cpu():
    """The H100 receives the same C-DISORT radius and level-altitude state."""
    reference = solve("cpu")
    value = solve("cuda")

    torch.testing.assert_close(value, reference, rtol=1e-10, atol=1e-12)
