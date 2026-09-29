"""CPU contract tests for C-DISORT general-source arrays."""

import pytest
import torch
from pydisort import Disort, DisortOptions


DTYPE = torch.float64
NSTR = 4
NLYR = 1
NWAVE = 1
NCOL = 1
NUMU = 1


def make_options(computational=None, user=None, backend="cpu"):
    options = (
        DisortOptions().flags("usrtau,usrang,lamber,quiet").backend(backend)
    )
    options.ds().nlyr = NLYR
    options.ds().nstr = options.ds().nmom = options.ds().nphase = NSTR
    options.user_tau([0.0, 0.2]).user_mu([0.5]).user_phi([0.0])
    if computational is not None:
        options.general_source(computational, user)
    return options


def source_tensors(value=0.0):
    computational = torch.full(
        (NWAVE, NCOL, NSTR, NLYR, NSTR), value, dtype=DTYPE
    )
    user = torch.full((NWAVE, NCOL, NSTR, NLYR, NUMU), value, dtype=DTYPE)
    return computational, user


def inputs():
    prop = torch.zeros((NWAVE, NCOL, NLYR, 2 + NSTR), dtype=DTYPE)
    prop[..., 0] = 0.2
    prop[..., 1] = 0.5
    return prop, {"fbeam": torch.zeros((NWAVE, NCOL), dtype=DTYPE)}


def test_zero_general_source_recovers_existing_solution():
    prop, boundary = inputs()
    plain = Disort(make_options())(prop, **boundary)
    computational, user = source_tensors()
    sourced = Disort(make_options(computational, user))(prop, **boundary)

    torch.testing.assert_close(sourced, plain, rtol=1e-12, atol=1e-14)


def test_nonzero_general_source_changes_cpu_solution():
    prop, boundary = inputs()
    computational, user = source_tensors()
    computational[0, 0, 0, 0, :] = 1.0
    user[0, 0, 0, 0, :] = 1.0

    result = Disort(make_options(computational, user))(prop, **boundary)

    # Direct C-DISORT reference for this one-layer source: Fourier order zero
    # is one at every computational and user angle, with no boundary source.
    expected = torch.tensor(
        [[[[1.0891838249147526, 0.0], [0.0, 1.0891838249147526]]]],
        dtype=DTYPE,
    )
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "computational_shape, user_shape, message",
    [
        ((1, 1, 4, 1, 3), (1, 1, 4, 1, 1), "computational source"),
        ((1, 1, 4, 1, 4), (1, 1, 4, 1, 2), "user source"),
    ],
)
def test_general_source_shapes_are_checked(
    computational_shape, user_shape, message
):
    computational = torch.zeros(computational_shape, dtype=DTYPE)
    user = torch.zeros(user_shape, dtype=DTYPE)

    with pytest.raises(RuntimeError, match=message):
        Disort(make_options(computational, user))


def test_general_source_rejects_non_float64_input():
    computational, user = source_tensors()

    with pytest.raises(RuntimeError, match="float64"):
        make_options(computational.float(), user)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)
def test_general_source_cuda_is_explicitly_unsupported():
    computational, user = source_tensors()
    solver = Disort(make_options(computational, user, backend="cuda"))
    prop, boundary = inputs()

    with pytest.raises(NotImplementedError, match="CPU-only"):
        solver(
            prop.cuda(),
            **{key: value.cuda() for key, value in boundary.items()},
        )
