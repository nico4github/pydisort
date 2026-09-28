"""The explicit backend selector must preserve the device contract."""

import math

import pytest
import torch

from pydisort import Disort, DisortOptions


def build_solver(backend):
    """Return the smallest flux-only solver configured for one backend."""
    options = DisortOptions().flags("onlyfl,lamber,quiet").backend(backend)
    options.ds().nlyr = 1
    options.ds().nstr = options.ds().nmom = options.ds().nphase = 4
    return Disort(options)


def inputs():
    """Return CPU inputs whose device is deliberately selected by the solver."""
    prop = torch.zeros((1, 1, 1, 6), dtype=torch.float64)
    prop[..., 0] = 0.1
    return prop, {"fbeam": torch.tensor([[math.pi]], dtype=torch.float64)}


def test_backend_defaults_to_auto():
    """Existing callers retain input-device dispatch unless they opt in."""
    assert DisortOptions().backend() == "auto"


def test_cpu_backend_returns_cpu_result():
    """The CPU selector accepts an ordinary CPU input and returns CPU output."""
    prop, boundary = inputs()
    result = build_solver("cpu").forward(prop, **boundary)

    assert result.device.type == "cpu"
    assert torch.isfinite(result).all()


def test_unknown_backend_is_rejected_before_dispatch():
    """A typo must not silently select an unintended solver backend."""
    prop, boundary = inputs()
    with pytest.raises(RuntimeError, match="backend must be one of"):
        build_solver("not-a-backend").forward(prop, **boundary)
