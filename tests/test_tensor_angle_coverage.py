"""Guard the nontrivial-angle contract for tensor radiance fixtures."""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SYMMETRIC_PAIR = {-0.5, 0.5}
ASYMMETRIC_PAIR = {-0.3, 0.4}


def _is_nontrivial_azimuth(angle: float) -> bool:
    """Return whether an angle avoids the principal-axis shortcuts."""
    return angle % 90.0 != 0.0


@pytest.mark.parametrize(
    "path",
    sorted(FIXTURES.glob("tensor_user_ray_*_reference.json")),
    ids=lambda path: path.stem,
)
def test_tensor_user_ray_fixtures_cover_symmetric_and_asymmetric_angles(path):
    """Every polar-angle fixture must exercise both sign branches fully."""
    fixture = json.loads(path.read_text())
    if "user_mu" not in fixture:
        pytest.skip("azimuth-only companion fixture")
    angles = set(fixture["user_mu"])
    assert SYMMETRIC_PAIR <= angles
    assert ASYMMETRIC_PAIR <= angles
    if "umu0" in fixture:
        assert 0.0 < fixture["umu0"] < 1.0


def test_tensor_azimuth_fixture_includes_a_nontrivial_angle():
    """Do not let final Fourier reconstruction regress to axis-only checks."""
    fixture = json.loads(
        (
            FIXTURES / "tensor_user_ray_beam_five_layer_azimuth_reference.json"
        ).read_text()
    )
    assert _is_nontrivial_azimuth(fixture["phi_degrees"])
