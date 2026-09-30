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


@dataclass(frozen=True)
class TensorLayerOptics:
    """Delta-M-scaled layer quantities used by the eigen and boundary stages."""

    oprim: torch.Tensor
    dtaucpr: torch.Tensor
    tauc: torch.Tensor
    taucpr: torch.Tensor
    flyr: torch.Tensor
    gl: torch.Tensor


@timed(name="tensor_backend.prepare_layer_optics")
def prepare_layer_optics(
    atmosphere: TensorAtmosphere, *, nstr: int, deltam: bool
) -> TensorLayerOptics:
    """Vectorize the delta-M layer portion of ``c_disort_set``.

    This excludes cut-off and user-output-grid bookkeeping.  It exactly follows
    the C formulas for the arrays consumed by the subsequent eigen and boundary
    stages, including the unscaled cumulative optical depth used by delta-M.
    """
    if atmosphere.pmom.shape[-1] <= nstr:
        raise ValueError("pmom must include the nstr-th Legendre moment")
    if atmosphere.dtauc.shape != atmosphere.ssalb.shape:
        raise ValueError("dtauc and ssalb must have the same shape")

    tauc = torch.cumsum(atmosphere.dtauc, dim=-1)
    if deltam:
        flyr = atmosphere.pmom[..., nstr]
        denominator = 1.0 - flyr * atmosphere.ssalb
        oprim = atmosphere.ssalb * (1.0 - flyr) / denominator
        dtaucpr = denominator * atmosphere.dtauc
        taucpr = torch.cumsum(dtaucpr, dim=-1)
        gl = (
            torch.arange(
                nstr,
                dtype=atmosphere.dtauc.dtype,
                device=atmosphere.dtauc.device,
            )
            .mul(2)
            .add(1)
            * oprim.unsqueeze(-1)
            * (atmosphere.pmom[..., :nstr] - flyr.unsqueeze(-1))
            / (1.0 - flyr).unsqueeze(-1)
        )
    else:
        flyr = torch.zeros_like(atmosphere.dtauc)
        oprim = atmosphere.ssalb
        dtaucpr = atmosphere.dtauc
        taucpr = tauc
        gl = (
            torch.arange(
                nstr,
                dtype=atmosphere.dtauc.dtype,
                device=atmosphere.dtauc.device,
            )
            .mul(2)
            .add(1)
            * oprim.unsqueeze(-1)
            * atmosphere.pmom[..., :nstr]
        )
    return TensorLayerOptics(oprim, dtaucpr, tauc, taucpr, flyr, gl)


@dataclass(frozen=True)
class TensorOutputGrid:
    """User optical depths mapped onto the delta-M computational mesh."""

    layru: torch.Tensor
    utaupr: torch.Tensor


@timed(name="tensor_backend.prepare_output_grid")
def prepare_output_grid(
    utau: torch.Tensor,
    atmosphere: TensorAtmosphere,
    optics: TensorLayerOptics,
    *,
    deltam: bool,
) -> TensorOutputGrid:
    """Vectorize C-DISORT's user-depth layer lookup and delta-M transform."""
    if utau.ndim != 1 or utau.dtype != torch.float64:
        raise ValueError("utau must be a one-dimensional float64 tensor")
    if utau.device != optics.tauc.device:
        raise ValueError("utau and optics must be on the same device")
    if torch.any(utau < 0) or torch.any(utau > optics.tauc[..., -1:].max()):
        raise ValueError("utau must lie within the atmospheric optical depth")
    layru0 = torch.searchsorted(
        optics.tauc, utau.expand(*optics.tauc.shape[:-1], -1), right=False
    )
    layru0 = layru0.clamp(max=optics.tauc.shape[-1] - 1)
    previous_tauc = torch.cat(
        (torch.zeros_like(optics.tauc[..., :1]), optics.tauc[..., :-1]), dim=-1
    )
    previous_taucpr = torch.cat(
        (torch.zeros_like(optics.taucpr[..., :1]), optics.taucpr[..., :-1]),
        dim=-1,
    )
    gather = layru0
    if deltam:
        factor = 1.0 - atmosphere.ssalb * optics.flyr
        utaupr = torch.gather(previous_taucpr, -1, gather) + torch.gather(
            factor, -1, gather
        ) * (utau - torch.gather(previous_tauc, -1, gather))
    else:
        utaupr = utau.expand_as(gather)
    return TensorOutputGrid(layru=layru0 + 1, utaupr=utaupr)


@dataclass(frozen=True)
class TensorQuadrature:
    """DISORT computational angle cosines and weights."""

    cmu: torch.Tensor
    cwt: torch.Tensor


@timed(name="tensor_backend.gaussian_quadrature")
def gaussian_quadrature(
    nstr: int, *, device: torch.device | str
) -> TensorQuadrature:
    """Generate C-DISORT's positive/negative Gauss-Legendre angle grid."""
    if nstr < 2 or nstr % 2:
        raise ValueError("nstr must be a positive even DISORT stream count")
    nn = nstr // 2
    index = torch.arange(1, nn, dtype=torch.float64, device=device)
    offdiag = index / torch.sqrt(4.0 * index.square() - 1.0)
    matrix = torch.diag(offdiag, diagonal=1) + torch.diag(offdiag, diagonal=-1)
    roots, vectors = torch.linalg.eigh(matrix)
    weights = 2.0 * vectors[0].square()
    positive = 0.5 * (roots + 1.0)
    positive_weights = 0.5 * weights
    return TensorQuadrature(
        cmu=torch.cat((positive, -positive)),
        cwt=torch.cat((positive_weights, positive_weights)),
    )


@dataclass(frozen=True)
class TensorReducedEigenMatrix:
    matrix: torch.Tensor
    amb: torch.Tensor


@timed(name="tensor_backend.build_reduced_eigen_matrix")
def build_reduced_eigen_matrix(
    optics: TensorLayerOptics, quadrature: TensorQuadrature, *, nstr: int
) -> TensorReducedEigenMatrix:
    """Build TP9's batched mazim=0 reduced eigenproblem matrix (SS(12))."""
    nn = nstr // 2
    mu = quadrature.cmu
    ylm = torch.empty((nstr, nstr), dtype=mu.dtype, device=mu.device)
    ylm[0] = 1.0
    ylm[1] = mu
    for degree in range(2, nstr):
        ylm[degree] = (
            (2 * degree - 1) * mu * ylm[degree - 1]
            - (degree - 1) * ylm[degree - 2]
        ) / degree
    cc = 0.5 * torch.einsum(
        "...l,li,lj,j->...ij", optics.gl, ylm, ylm, quadrature.cwt
    )
    alpha = cc[..., :nn, :nn] / mu[:nn].view(1, 1, nn, 1)
    beta = cc[..., :nn, nn:] / mu[:nn].view(1, 1, nn, 1)
    eye = torch.eye(nn, dtype=mu.dtype, device=mu.device) / mu[:nn].view(nn, 1)
    amb = alpha - beta - eye
    apb = alpha + beta - eye
    return TensorReducedEigenMatrix(matrix=apb @ amb, amb=amb)


@timed(name="tensor_backend.solve_reduced_eigenproblem")
def solve_reduced_eigenproblem(
    reduced: TensorReducedEigenMatrix,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the batched real reduced eigenproblem used by the TP9 flux path."""
    matrix = reduced.matrix
    values, vectors = torch.linalg.eig(matrix)
    scale = values.real.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    if torch.any(
        values.imag.abs() > torch.finfo(matrix.dtype).eps * 128.0 * scale
    ):
        raise RuntimeError(
            "reduced DISORT eigenproblem produced complex eigenvalues"
        )
    eigenvalues = values.real.abs().sqrt()
    gpplgm = (reduced.amb @ vectors.real) / eigenvalues.unsqueeze(-2)
    gpmigm = vectors.real
    positive = torch.cat(
        (0.5 * (gpplgm + gpmigm), 0.5 * (gpplgm - gpmigm)), dim=-2
    )
    negative = torch.cat(
        (0.5 * (-gpplgm + gpmigm), 0.5 * (-gpplgm - gpmigm)), dim=-2
    )
    return eigenvalues, torch.cat(
        (negative.flip(dims=(-1,)), positive), dim=-1
    )
