"""Unsupported public flags must fail before CPU or CUDA dispatch."""

import pytest
import torch
from pydisort import Disort, DisortOptions

UNSUPPORTED_FLAGS = {
    "ibcnd": "standard flux and radiance calculations",
    "spher": "DisortOptions.pseudo_spherical(radius, level_altitudes)",
    "general_source": "beam, isotropic, or thermal inputs",
    "output_uum": "gather_flx or gather_rad",
}


def options(backend: str, flags: str = "onlyfl,lamber,quiet") -> DisortOptions:
    """Return a minimal valid configuration for one selected backend."""
    configured = DisortOptions().flags(flags).backend(backend)
    configured.ds().nlyr = 1
    configured.ds().nstr = configured.ds().nmom = configured.ds().nphase = 4
    return configured


def inputs() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prop = torch.zeros((1, 1, 1, 6), dtype=torch.float64)
    prop[..., 0] = 0.1
    return prop, {"fbeam": torch.tensor([[1.0]], dtype=torch.float64)}


BACKENDS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="no CUDA device available"
        ),
    ),
]


@pytest.mark.parametrize("flag, alternative", UNSUPPORTED_FLAGS.items())
@pytest.mark.parametrize("backend", BACKENDS)
def test_unsupported_flag_is_rejected_before_solver_allocation(
    flag: str, alternative: str, backend: str
) -> None:
    """Unsafe C-DISORT state must never be allocated for an unsupported flag."""
    with pytest.raises(NotImplementedError, match=flag) as error:
        Disort(options(backend, f"onlyfl,lamber,quiet,{flag}"))

    assert alternative in str(error.value)


@pytest.mark.parametrize("flag, alternative", UNSUPPORTED_FLAGS.items())
@pytest.mark.parametrize("backend", BACKENDS)
def test_unsupported_flag_is_rejected_after_solver_construction(
    flag: str, alternative: str, backend: str
) -> None:
    """Mutating shared options cannot bypass the guard before dispatch."""
    configured = options(backend)
    solver = Disort(configured)
    configured.flags(f"onlyfl,lamber,quiet,{flag}")
    prop, boundary = inputs()

    with pytest.raises(NotImplementedError, match=flag) as error:
        solver.forward(prop, **boundary)

    assert alternative in str(error.value)
