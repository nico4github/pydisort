"""Parity-first tensor stages for the experimental DISORT reconstruction.

This module deliberately does not select a new solver backend yet.  It holds
small, independently validated stages whose tensors replace mutable C-DISORT
state arrays.  Each stage must reproduce the corresponding C setup rule before
the eigenproblem and boundary-solve stages are added.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .timing import timed


@dataclass(frozen=True)
class TensorAtmosphere:
    """Structure-of-arrays optical state for a batch of DISORT solves.

    The leading axes are ``(nwave, ncol)`` and the layer axis is next.  ``pmom``
    includes DISORT's normalized zeroth Legendre moment at its final axis index
    zero, unlike the public property tensor whose moments start at one.
    """

    dtauc: torch.Tensor
    ssalb: torch.Tensor
    pmom: torch.Tensor


def _validate_properties(prop: torch.Tensor, nstr: int, nmom: int) -> None:
    if prop.ndim != 4:
        raise ValueError("prop must have shape (nwave, ncol, nlyr, nprop)")
    if prop.dtype != torch.float64:
        raise ValueError("prop must use float64 for C-DISORT parity")
    if prop.shape[-1] < 1:
        raise ValueError("prop must contain optical depth at property index 0")
    if nstr < 2 or nstr % 2:
        raise ValueError("nstr must be a positive even DISORT stream count")
    if nmom < nstr:
        raise ValueError("nmom must be at least nstr")
    if prop.shape[-1] > nmom + 2:
        raise ValueError("prop has more scattering moments than nmom")


@timed(name="tensor_backend.prepare_atmosphere")
def prepare_atmosphere(
    prop: torch.Tensor, *, nstr: int, nmom: int, upward: bool = False
) -> TensorAtmosphere:
    """Vectorize the property-to-state copy in ``disort_impl`` exactly.

    ``prop[..., 0]`` is optical depth, ``prop[..., 1]`` is single-scattering
    albedo when present, and the remaining entries are Legendre moments one and
    above.  Missing albedo or moments use C-DISORT's zero-fill rule.  Upward
    calculations reverse the atmospheric layer order just as ``disort_impl``
    does before calling C-DISORT.
    """
    _validate_properties(prop, nstr, nmom)
    source = prop.flip(dims=(2,)) if upward else prop
    nmom_nstr = max(nmom, nstr)
    shape = (*source.shape[:-1], nmom_nstr + 1)
    pmom = torch.zeros(shape, dtype=source.dtype, device=source.device)
    pmom[..., 0] = 1.0

    moment_count = source.shape[-1] - 2
    if moment_count > 0:
        pmom[..., 1 : moment_count + 1] = source[..., 2:]
    ssalb = (
        source[..., 1]
        if source.shape[-1] > 1
        else torch.zeros_like(source[..., 0])
    )
    return TensorAtmosphere(dtauc=source[..., 0], ssalb=ssalb, pmom=pmom)
