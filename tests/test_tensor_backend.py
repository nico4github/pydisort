"""Parity tests for the first vectorized C-DISORT state-preparation stage."""

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
