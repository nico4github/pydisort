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
        optics.tauc,
        utau.expand(*optics.tauc.shape[:-1], -1).contiguous(),
        right=False,
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
    gplus = 0.5 * (gpplgm + gpmigm)
    gminus = 0.5 * (gpplgm - gpmigm)
    negative_gplus = 0.5 * (-gpplgm + gpmigm)
    negative_gminus = 0.5 * (-gpplgm - gpmigm)

    # This follows c_solve_eigen's four GC assignments exactly. The first
    # stream rows are positive cosines and the first mode columns carry the
    # negative eigenvalues in reverse eigenvalue order.
    positive = torch.cat((gminus.flip(dims=(-2,)), gplus), dim=-2)
    negative = torch.cat(
        (negative_gminus.flip(dims=(-2,)), negative_gplus), dim=-2
    ).flip(dims=(-1,))
    full = torch.cat((negative, positive), dim=-1)
    # Eigenvectors are defined only up to independent column scaling. Scale
    # each reconstructed full-stream mode before boundary assembly so the
    # integration system retains C-DISORT's numerically stable normalization.
    full = full / full.abs().amax(dim=-2, keepdim=True).clamp_min(
        torch.finfo(full.dtype).tiny
    )
    return eigenvalues, full


@timed(name="tensor_backend.build_layer_continuity_blocks")
def build_layer_continuity_blocks(
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    optics: TensorLayerOptics,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build batched adjacent-layer continuity factors for the flux boundary solve."""
    nstr = eigenvectors.shape[-2]
    nn = nstr // 2
    if eigenvalues.ndim != 4 or eigenvalues.shape[-1] != nn:
        raise ValueError(
            "eigenvalues must have shape (nwave, ncol, nlyr, nstr / 2)"
        )
    scaled = torch.exp(eigenvalues * optics.dtaucpr.unsqueeze(-1))
    upper = eigenvectors[..., :nn] * scaled.unsqueeze(-2)
    lower = eigenvectors[..., nn:] * scaled.unsqueeze(-2)
    return upper, lower


@timed(name="tensor_backend.solve_block_tridiagonal")
def solve_block_tridiagonal(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Solve batched block-tridiagonal systems with a block Thomas sweep.

    The final three axes are ``(nblock, block_size, block_size)`` for the
    matrices and ``(nblock, block_size)`` for the right-hand side.  Leading
    axes batch independent wavelength/column systems.
    """
    if diagonal.ndim < 3 or rhs.shape != diagonal.shape[:-1]:
        raise ValueError(
            "rhs must match diagonal's batch, block, and row axes"
        )
    nblock = diagonal.shape[-3]
    if lower.shape != (*diagonal.shape[:-3], nblock - 1, *diagonal.shape[-2:]):
        raise ValueError("lower has incompatible block-tridiagonal shape")
    if upper.shape != lower.shape:
        raise ValueError("upper has incompatible block-tridiagonal shape")

    reduced_diagonal = [diagonal[..., 0, :, :]]
    reduced_rhs = [rhs[..., 0, :]]
    for block in range(1, nblock):
        factor = torch.linalg.solve(
            reduced_diagonal[-1].transpose(-1, -2),
            lower[..., block - 1, :, :].transpose(-1, -2),
        ).transpose(-1, -2)
        reduced_diagonal.append(
            diagonal[..., block, :, :] - factor @ upper[..., block - 1, :, :]
        )
        reduced_rhs.append(
            rhs[..., block, :]
            - (factor @ reduced_rhs[-1].unsqueeze(-1)).squeeze(-1)
        )
    solution = [
        torch.linalg.solve(
            reduced_diagonal[-1], reduced_rhs[-1].unsqueeze(-1)
        ).squeeze(-1)
    ]
    for block in range(nblock - 2, -1, -1):
        value = reduced_rhs[block] - (
            upper[..., block, :, :] @ solution[-1].unsqueeze(-1)
        ).squeeze(-1)
        solution.append(
            torch.linalg.solve(
                reduced_diagonal[block], value.unsqueeze(-1)
            ).squeeze(-1)
        )
    return torch.stack(tuple(reversed(solution)), dim=-2)


@dataclass(frozen=True)
class TensorBoundarySystem:
    """Dense batched TP9 boundary system for homogeneous integration constants."""

    matrix: torch.Tensor
    rhs: torch.Tensor


@timed(name="tensor_backend.build_tp9_boundary_system")
def build_tp9_boundary_system(
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    optics: TensorLayerOptics,
    fisot: torch.Tensor,
    beam_source: torch.Tensor | None = None,
    general_source: torch.Tensor | None = None,
    umu0: torch.Tensor | None = None,
    thermal0: torch.Tensor | None = None,
    thermal1: torch.Tensor | None = None,
    thermal_top: torch.Tensor | None = None,
    thermal_bottom: torch.Tensor | None = None,
    surface_albedo: torch.Tensor | None = None,
    quadrature: TensorQuadrature | None = None,
    fbeam: torch.Tensor | None = None,
) -> TensorBoundarySystem:
    """Assemble TP9's plane-parallel, no-beam, black-surface system.

    This is the azimuth-independent ``fbeam == 0`` branch of ``c_set_matrix``
    and ``c_solve0`` with no thermal or general source. TP9's sole source is
    isotropic top illumination, so only the first ``nstr / 2`` RHS entries are
    nonzero. Beam and reflecting-surface terms are added in a later stage.
    """
    if (
        eigenvectors.ndim != 5
        or eigenvectors.shape[-2] != eigenvectors.shape[-1]
    ):
        raise ValueError(
            "eigenvectors must have shape (nwave, ncol, nlyr, nstr, nstr)"
        )
    nstr = eigenvectors.shape[-1]
    if nstr < 2 or nstr % 2:
        raise ValueError("eigenvectors require a positive even stream count")
    if eigenvalues.shape != (*eigenvectors.shape[:-2], nstr // 2):
        raise ValueError("eigenvalues are incompatible with eigenvectors")
    if optics.dtaucpr.shape != eigenvectors.shape[:-2]:
        raise ValueError(
            "optics and eigenvectors have incompatible layer shapes"
        )
    if fisot.shape != eigenvectors.shape[:2]:
        raise ValueError("fisot must have shape (nwave, ncol)")
    if (
        fisot.dtype != eigenvectors.dtype
        or fisot.device != eigenvectors.device
    ):
        raise ValueError("fisot must share eigenvector dtype and device")

    *batch, nlyr, _, _ = eigenvectors.shape
    nn = nstr // 2
    nrow = nlyr * nstr
    matrix = torch.zeros(
        (*batch, nrow, nrow),
        dtype=eigenvectors.dtype,
        device=eigenvectors.device,
    )
    rhs = torch.zeros(
        (*batch, nrow), dtype=eigenvectors.dtype, device=eigenvectors.device
    )
    factors = torch.exp(-eigenvalues * optics.dtaucpr.unsqueeze(-1))

    if surface_albedo is not None:
        if surface_albedo.shape != eigenvectors.shape[:2]:
            raise ValueError("surface_albedo must have shape (nwave, ncol)")
        if (
            surface_albedo.dtype != eigenvectors.dtype
            or surface_albedo.device != eigenvectors.device
        ):
            raise ValueError(
                "surface_albedo must share eigenvector dtype and device"
            )
        if quadrature is None:
            raise ValueError("surface_albedo requires quadrature")
        if quadrature.cmu.shape != (nstr,) or quadrature.cwt.shape != (nstr,):
            raise ValueError("quadrature is incompatible with eigenvectors")
        if quadrature.cmu.device != eigenvectors.device:
            raise ValueError("quadrature must share eigenvector device")
        reflection_weights = (quadrature.cwt[:nn] * quadrature.cmu[:nn]).view(
            *((1,) * len(batch)), nn
        )
    else:
        surface_albedo = torch.zeros_like(fisot)
        reflection_weights = None

    top = eigenvectors[..., 0, :nn, :].flip(dims=(-2,))
    matrix[..., :nn, :nn] = top[..., :nn] * factors[..., 0, :].flip(
        dims=(-1,)
    ).unsqueeze(-2)
    matrix[..., :nn, nn:nstr] = top[..., nn:]
    rhs[..., :nn] = fisot.unsqueeze(-1)
    if (thermal0 is None) != (thermal1 is None):
        raise ValueError("thermal0 and thermal1 must be supplied together")
    if thermal0 is not None and thermal1 is not None:
        if thermal0.shape != (*eigenvectors.shape[:-2], nstr):
            raise ValueError("thermal0 is incompatible with eigenvectors")
        if thermal1.shape != thermal0.shape:
            raise ValueError("thermal1 is incompatible with eigenvectors")
        rhs[..., :nn] -= thermal0[..., 0, :nn].flip(dims=(-1,))
        rhs[..., -nn:] -= thermal0[..., -1, nn:] + thermal1[
            ..., -1, nn:
        ] * optics.taucpr[..., -1].unsqueeze(-1)
        if reflection_weights is not None:
            thermal_downward = thermal0[..., -1, :nn].flip(
                dims=(-1,)
            ) + thermal1[..., -1, :nn].flip(dims=(-1,)) * optics.taucpr[
                ..., -1
            ].unsqueeze(
                -1
            )
            reflected_thermal = (
                2.0
                * surface_albedo
                * torch.sum(thermal_downward * reflection_weights, dim=-1)
            )
            rhs[..., -nn:] += reflected_thermal.unsqueeze(-1)
    if (thermal_top is None) != (thermal_bottom is None):
        raise ValueError(
            "thermal top and bottom sources must be supplied together"
        )
    if thermal_top is not None and thermal_bottom is not None:
        expected_shape = eigenvectors.shape[:2]
        if (
            thermal_top.shape != expected_shape
            or thermal_bottom.shape != expected_shape
        ):
            raise ValueError(
                "thermal boundary sources must have shape (nwave, ncol)"
            )
        if (
            thermal_top.dtype != eigenvectors.dtype
            or thermal_bottom.dtype != eigenvectors.dtype
            or thermal_top.device != eigenvectors.device
            or thermal_bottom.device != eigenvectors.device
        ):
            raise ValueError(
                "thermal boundary sources must share eigenvectors"
            )
        rhs[..., :nn] += thermal_top.unsqueeze(-1)
        rhs[..., -nn:] += thermal_bottom.unsqueeze(-1)
    if (beam_source is None) != (umu0 is None):
        raise ValueError("beam_source and umu0 must be supplied together")
    if beam_source is not None and umu0 is not None:
        if beam_source.shape != (*eigenvectors.shape[:-2], nstr):
            raise ValueError("beam_source is incompatible with eigenvectors")
        if umu0.shape != eigenvectors.shape[:2]:
            raise ValueError("umu0 must have shape (nwave, ncol)")
        expbea = torch.exp(-optics.taucpr / umu0.unsqueeze(-1))
        rhs[..., :nn] -= beam_source[..., 0, :nn].flip(dims=(-1,))
        rhs[..., -nn:] += -beam_source[..., -1, nn:] * expbea[
            ..., -1
        ].unsqueeze(-1)
        if fbeam is not None:
            if fbeam.shape != eigenvectors.shape[:2]:
                raise ValueError("fbeam must have shape (nwave, ncol)")
            reflected_beam = (
                surface_albedo * umu0 * fbeam / torch.pi * expbea[..., -1]
            )
            rhs[..., -nn:] += reflected_beam.unsqueeze(-1)
    else:
        expbea = None
    if general_source is not None:
        if general_source.shape != (*eigenvectors.shape[:-2], nstr):
            raise ValueError(
                "general_source is incompatible with eigenvectors"
            )
        rhs[..., :nn] -= general_source[..., 0, :nn].flip(dims=(-1,))
        rhs[..., -nn:] -= general_source[..., -1, nn:]

    for layer in range(nlyr - 1):
        row = nn + layer * nstr
        left = slice(layer * nstr, (layer + 1) * nstr)
        right = slice((layer + 1) * nstr, (layer + 2) * nstr)
        lower_gc = eigenvectors[..., layer, :, :]
        upper_gc = eigenvectors[..., layer + 1, :, :]
        matrix[..., row : row + nstr, left.start : left.start + nn] = lower_gc[
            ..., :nn
        ]
        matrix[..., row : row + nstr, left.start + nn : left.stop] = lower_gc[
            ..., nn:
        ] * factors[..., layer, :].unsqueeze(-2)
        matrix[
            ..., row : row + nstr, right.start : right.start + nn
        ] = -upper_gc[..., :nn] * factors[..., layer + 1, :].flip(
            dims=(-1,)
        ).unsqueeze(
            -2
        )
        matrix[
            ..., row : row + nstr, right.start + nn : right.stop
        ] = -upper_gc[..., nn:]
        if beam_source is not None and expbea is not None:
            rhs[..., row : row + nstr] += (
                beam_source[..., layer + 1, :] - beam_source[..., layer, :]
            ) * expbea[..., layer].unsqueeze(-1)
        if thermal0 is not None and thermal1 is not None:
            rhs[..., row : row + nstr] += (
                thermal0[..., layer + 1, :]
                - thermal0[..., layer, :]
                + (thermal1[..., layer + 1, :] - thermal1[..., layer, :])
                * optics.taucpr[..., layer].unsqueeze(-1)
            )
        if general_source is not None:
            rhs[..., row : row + nstr] += (
                general_source[..., layer + 1, :]
                - general_source[..., layer, :]
            )

    bottom_row = nrow - nn
    bottom = eigenvectors[..., -1, nn:, :]
    if quadrature is not None:
        weights = (quadrature.cwt[:nn] * quadrature.cmu[:nn]).view(
            *((1,) * len(batch)), nn, 1
        )
        reflected = (
            2.0
            * surface_albedo.unsqueeze(-1)
            * torch.sum(
                eigenvectors[..., -1, :nn, :].flip(dims=(-2,)) * weights,
                dim=-2,
            )
        )
        bottom = bottom - reflected.unsqueeze(-2)
    matrix[..., bottom_row:, -nstr:-nn] = bottom[..., :nn]
    matrix[..., bottom_row:, -nn:] = bottom[..., nn:] * factors[
        ..., -1, :
    ].unsqueeze(-2)
    return TensorBoundarySystem(matrix=matrix, rhs=rhs)


@timed(name="tensor_backend.solve_tp9_boundary_system")
def solve_tp9_boundary_system(system: TensorBoundarySystem) -> torch.Tensor:
    """Solve TP9 integration constants in C-DISORT layer/mode ordering."""
    return torch.linalg.solve(system.matrix, system.rhs.unsqueeze(-1)).squeeze(
        -1
    )


def _legendre_m0(mu: torch.Tensor, nstr: int) -> torch.Tensor:
    values = [torch.ones_like(mu), mu]
    for ell in range(2, nstr):
        values.append(
            ((2 * ell - 1) * mu * values[-1] - (ell - 1) * values[-2]) / ell
        )
    return torch.stack(values[:nstr], dim=-1)


@timed(name="tensor_backend.interpolate_tp9_eigenvectors_m0")
def interpolate_tp9_eigenvectors_m0(
    eigenvectors: torch.Tensor,
    optics: TensorLayerOptics,
    quadrature: TensorQuadrature,
    user_mu: torch.Tensor,
) -> torch.Tensor:
    """C-DISORT c_interp_eigenvec for m=0 user polar angles."""
    nstr = eigenvectors.shape[-1]
    nn = nstr // 2
    evecc = torch.empty_like(eigenvectors)
    for i in range(nn):
        for j in range(nn):
            evecc[..., i, j] = eigenvectors[..., nn + i, nn + j]
            evecc[..., nn + i, j] = eigenvectors[..., nn - i - 1, nn + j]
            evecc[..., i, nn + j] = eigenvectors[..., nn + i, nn - j - 1]
            evecc[..., nn + i, nn + j] = eigenvectors[
                ..., nn - i - 1, nn - j - 1
            ]
    inner = torch.einsum(
        "j,jl,...ji->...il",
        quadrature.cwt,
        _legendre_m0(quadrature.cmu, nstr),
        evecc,
    )
    raw = torch.einsum(
        "...il,ul->...ui",
        0.5 * inner * optics.gl.unsqueeze(-2),
        _legendre_m0(user_mu, nstr),
    )
    return torch.cat((raw[..., :, nn:].flip(-1), raw[..., :, :nn]), dim=-1)


@timed(name="tensor_backend.top_boundary_user_intensity")
def top_boundary_user_intensity(
    grid: TensorOutputGrid, user_mu: torch.Tensor, fisot: torch.Tensor
) -> torch.Tensor:
    """C-DISORT's attenuated diffuse top boundary term for m=0 radiance."""
    if fisot.shape != grid.utaupr.shape[:2]:
        raise ValueError("fisot and output grid are incompatible")
    downward = user_mu < 0
    result = torch.zeros(
        (*grid.utaupr.shape, user_mu.numel()),
        dtype=fisot.dtype,
        device=fisot.device,
    )
    if downward.any():
        result[..., downward] = fisot[..., None, None] * torch.exp(
            grid.utaupr[..., None] / user_mu[downward]
        )
    return result


@timed(name="tensor_backend.extract_tp9_user_intensity_one_layer_m0")
def extract_tp9_user_intensity_one_layer_m0(
    user_eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    optics: TensorLayerOptics,
    grid: TensorOutputGrid,
    constants: torch.Tensor,
    user_mu: torch.Tensor,
    fisot: torch.Tensor,
) -> torch.Tensor:
    """C-DISORT's homogeneous m=0 user-ray integral for one layer."""
    nstr = user_eigenvectors.shape[-1]
    nn = nstr // 2
    if optics.dtaucpr.shape[-1] != 1:
        raise ValueError("one-layer user-ray integrator requires one layer")
    kk = torch.cat(
        (-eigenvalues[..., 0, :].flip(-1), eigenvalues[..., 0, :]), dim=-1
    )
    gu = user_eigenvectors[..., 0, :, :] * constants[..., None, :]
    tau = grid.utaupr
    depth = optics.dtaucpr[..., 0, None]
    output = torch.zeros(
        *tau.shape, user_mu.numel(), dtype=tau.dtype, device=tau.device
    )
    for index, mu in enumerate(user_mu):
        denominator = 1.0 + mu * kk
        if mu < 0:
            exp1 = torch.exp(tau / mu)
            negative = (
                torch.exp(
                    -kk[..., :nn, None]
                    * (tau.unsqueeze(-2) - depth.unsqueeze(-2))
                )
                - torch.exp(kk[..., :nn, None] * depth.unsqueeze(-2))
                * exp1.unsqueeze(-2)
            ) / denominator[..., :nn].unsqueeze(-1)
            positive = (
                torch.exp(-kk[..., nn:, None] * tau.unsqueeze(-2))
                - exp1.unsqueeze(-2)
            ) / denominator[..., nn:].unsqueeze(-1)
            integral = (gu[..., index, :nn, None] * negative).sum(-2) + (
                gu[..., index, nn:, None] * positive
            ).sum(-2)
            output[..., index] = (
                torch.where(tau > 1e-6, integral, torch.zeros_like(integral))
                + fisot[..., None] * exp1
            )
        else:
            exp2 = torch.exp((tau - depth) / mu)
            negative = (
                torch.exp(
                    -kk[..., :nn, None]
                    * (tau.unsqueeze(-2) - depth.unsqueeze(-2))
                )
                - exp2.unsqueeze(-2)
            ) / denominator[..., :nn].unsqueeze(-1)
            positive = (
                torch.exp(-kk[..., nn:, None] * tau.unsqueeze(-2))
                - torch.exp(-kk[..., nn:, None] * depth.unsqueeze(-2))
                * exp2.unsqueeze(-2)
            ) / denominator[..., nn:].unsqueeze(-1)
            integral = (gu[..., index, :nn, None] * negative).sum(-2) + (
                gu[..., index, nn:, None] * positive
            ).sum(-2)
            output[..., index] = torch.where(
                tau < depth - 1e-6, integral, torch.zeros_like(integral)
            )
    return output


@timed(name="tensor_backend.extract_tp9_user_intensity_m0")
def interpolate_tp9_user_beam_source_m0(
    beam_source: torch.Tensor,
    optics: TensorLayerOptics,
    quadrature: TensorQuadrature,
    user_mu: torch.Tensor,
    umu0: torch.Tensor,
    fbeam: torch.Tensor,
) -> torch.Tensor:
    """Interpolate C-DISORT's plane-parallel m=0 beam source to user rays."""
    nstr = quadrature.cmu.numel()
    nn = nstr // 2
    degree = torch.arange(nstr, dtype=user_mu.dtype, device=user_mu.device)
    ylm_user = user_mu.unsqueeze(0).pow(degree.unsqueeze(-1))
    ylm_quadrature = quadrature.cmu.unsqueeze(0).pow(degree.unsqueeze(-1))
    ylm_beam = (-umu0).unsqueeze(-1).pow(degree)
    # Convert monomials to Legendre polynomials using the standard recurrence.
    ylm_user[0] = 1.0
    ylm_quadrature[0] = 1.0
    ylm_beam[..., 0] = 1.0
    if nstr > 1:
        ylm_user[1] = user_mu
        ylm_quadrature[1] = quadrature.cmu
        ylm_beam[..., 1] = -umu0
    for order in range(2, nstr):
        ylm_user[order] = (
            (2 * order - 1) * user_mu * ylm_user[order - 1]
            - (order - 1) * ylm_user[order - 2]
        ) / order
        ylm_quadrature[order] = (
            (2 * order - 1) * quadrature.cmu * ylm_quadrature[order - 1]
            - (order - 1) * ylm_quadrature[order - 2]
        ) / order
        ylm_beam[..., order] = (
            (2 * order - 1) * (-umu0) * ylm_beam[..., order - 1]
            - (order - 1) * ylm_beam[..., order - 2]
        ) / order
    c_disort_beam_source = torch.cat(
        (beam_source[..., nn:], beam_source[..., :nn].flip(dims=(-1,))), dim=-1
    )
    projected = torch.einsum(
        "...j,lj,j->...l",
        c_disort_beam_source,
        ylm_quadrature,
        quadrature.cwt,
    )
    psi = 0.5 * optics.gl * projected
    direct = fbeam.unsqueeze(-1) * optics.gl * ylm_beam / (4.0 * torch.pi)
    return torch.einsum("lu,...l->...u", ylm_user, psi + direct)


@timed(name="tensor_backend.interpolate_tp9_user_thermal_source_m0")
def interpolate_tp9_user_thermal_source_m0(
    thermal0: torch.Tensor,
    thermal1: torch.Tensor,
    optics: TensorLayerOptics,
    quadrature: TensorQuadrature,
    user_mu: torch.Tensor,
    xr0: torch.Tensor,
    xr1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project C-DISORT m=0 thermal particular solutions onto user angles."""
    nstr = quadrature.cmu.numel()
    nn = nstr // 2
    degree = torch.arange(nstr, dtype=user_mu.dtype, device=user_mu.device)
    ylm_user = user_mu.unsqueeze(0).pow(degree.unsqueeze(-1))
    ylm_quadrature = quadrature.cmu.unsqueeze(0).pow(degree.unsqueeze(-1))
    ylm_user[0] = 1.0
    ylm_quadrature[0] = 1.0
    if nstr > 1:
        ylm_user[1] = user_mu
        ylm_quadrature[1] = quadrature.cmu
    for order in range(2, nstr):
        ylm_user[order] = (
            (2 * order - 1) * user_mu * ylm_user[order - 1]
            - (order - 1) * ylm_user[order - 2]
        ) / order
        ylm_quadrature[order] = (
            (2 * order - 1) * quadrature.cmu * ylm_quadrature[order - 1]
            - (order - 1) * ylm_quadrature[order - 2]
        ) / order

    def project(source: torch.Tensor, linear: torch.Tensor) -> torch.Tensor:
        c_source = torch.cat(
            (source[..., nn:], source[..., :nn].flip(dims=(-1,))), dim=-1
        )
        projected = torch.einsum(
            "...j,lj,j->...l", c_source, ylm_quadrature, quadrature.cwt
        )
        return torch.einsum(
            "lu,...l->...u", ylm_user, 0.5 * optics.gl * projected
        ) + (1.0 - optics.oprim).unsqueeze(-1) * linear.unsqueeze(-1)

    return project(thermal0, xr0), project(thermal1, xr1)


@timed(name="tensor_backend.extract_tp9_user_intensity_m0")
def extract_tp9_user_intensity_m0(
    user_eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    optics: TensorLayerOptics,
    grid: TensorOutputGrid,
    constants: torch.Tensor,
    user_mu: torch.Tensor,
    fisot: torch.Tensor,
    user_beam_source: torch.Tensor | None = None,
    umu0: torch.Tensor | None = None,
    user_thermal0: torch.Tensor | None = None,
    user_thermal1: torch.Tensor | None = None,
) -> torch.Tensor:
    """C-DISORT source-free m=0 user rays, including all crossed layers."""
    nstr = user_eigenvectors.shape[-1]
    nn = nstr // 2
    nlyr = optics.dtaucpr.shape[-1]
    if constants.shape != (*user_eigenvectors.shape[:-3], nlyr * nstr):
        raise ValueError("constants are incompatible with user eigenvectors")
    if (user_beam_source is None) != (umu0 is None):
        raise ValueError("user_beam_source and umu0 must be supplied together")
    if user_beam_source is not None and user_beam_source.shape != (
        *optics.dtaucpr.shape,
        user_mu.numel(),
    ):
        raise ValueError("user_beam_source is incompatible with user angles")
    if (user_thermal0 is None) != (user_thermal1 is None):
        raise ValueError(
            "user_thermal0 and user_thermal1 must be supplied together"
        )
    constants = constants.reshape(*constants.shape[:-1], nlyr, nstr)
    kk = torch.cat((-eigenvalues.flip(-1), eigenvalues), dim=-1)
    starts = torch.cat(
        (torch.zeros_like(optics.taucpr[..., :1]), optics.taucpr[..., :-1]),
        dim=-1,
    )
    result = torch.zeros(
        *grid.utaupr.shape,
        user_mu.numel(),
        dtype=grid.utaupr.dtype,
        device=grid.utaupr.device,
    )
    for iu, mu in enumerate(user_mu):
        gu = user_eigenvectors[..., :, iu, :] * constants
        for lu in range(grid.utaupr.shape[-1]):
            tau = grid.utaupr[..., lu]
            target = grid.layru[..., lu] - 1
            value = torch.zeros_like(tau)
            for lc in range(nlyr):
                begin, end = starts[..., lc], optics.taucpr[..., lc]
                dtau = optics.dtaucpr[..., lc]
                mode = kk[..., lc, :]
                denom = 1.0 + mu * mode
                if mu < 0:
                    exp1 = torch.exp((tau - begin) / mu)
                    exp2 = torch.exp((tau - end) / mu)
                    full = -(
                        gu[..., lc, :nn]
                        * (
                            torch.exp(mode[..., :nn] * dtau.unsqueeze(-1))
                            * exp1.unsqueeze(-1)
                            - exp2.unsqueeze(-1)
                        )
                        / denom[..., :nn]
                    ).sum(-1) - (
                        gu[..., lc, nn:]
                        * (
                            exp1.unsqueeze(-1)
                            - torch.exp(-mode[..., nn:] * dtau.unsqueeze(-1))
                            * exp2.unsqueeze(-1)
                        )
                        / denom[..., nn:]
                    ).sum(
                        -1
                    )
                    value += torch.where(
                        lc < target, full, torch.zeros_like(full)
                    )
                    dtau1, dtau2 = tau - begin, tau - end
                    exp1 = torch.exp(dtau1 / mu)
                    partial = (
                        gu[..., lc, :nn]
                        * (
                            torch.exp(-mode[..., :nn] * dtau2.unsqueeze(-1))
                            - torch.exp(mode[..., :nn] * dtau.unsqueeze(-1))
                            * exp1.unsqueeze(-1)
                        )
                        / denom[..., :nn]
                    ).sum(-1) + (
                        gu[..., lc, nn:]
                        * (
                            torch.exp(-mode[..., nn:] * dtau1.unsqueeze(-1))
                            - exp1.unsqueeze(-1)
                        )
                        / denom[..., nn:]
                    ).sum(
                        -1
                    )
                    value += torch.where(
                        (target == lc) & (dtau1 > 1e-6),
                        partial,
                        torch.zeros_like(partial),
                    )
                else:
                    exp1 = torch.exp((tau - begin) / mu)
                    exp2 = torch.exp((tau - end) / mu)
                    full = (
                        gu[..., lc, :nn]
                        * (
                            torch.exp(mode[..., :nn] * dtau.unsqueeze(-1))
                            * exp1.unsqueeze(-1)
                            - exp2.unsqueeze(-1)
                        )
                        / denom[..., :nn]
                    ).sum(-1) + (
                        gu[..., lc, nn:]
                        * (
                            exp1.unsqueeze(-1)
                            - torch.exp(-mode[..., nn:] * dtau.unsqueeze(-1))
                            * exp2.unsqueeze(-1)
                        )
                        / denom[..., nn:]
                    ).sum(
                        -1
                    )
                    value += torch.where(
                        lc > target, full, torch.zeros_like(full)
                    )
                    dtau1, dtau2 = tau - begin, tau - end
                    exp2 = torch.exp(dtau2 / mu)
                    partial = (
                        gu[..., lc, :nn]
                        * (
                            torch.exp(-mode[..., :nn] * dtau2.unsqueeze(-1))
                            - exp2.unsqueeze(-1)
                        )
                        / denom[..., :nn]
                    ).sum(-1) + (
                        gu[..., lc, nn:]
                        * (
                            torch.exp(-mode[..., nn:] * dtau1.unsqueeze(-1))
                            - torch.exp(-mode[..., nn:] * dtau.unsqueeze(-1))
                            * exp2.unsqueeze(-1)
                        )
                        / denom[..., nn:]
                    ).sum(
                        -1
                    )
                    value += torch.where(
                        (target == lc) & (dtau1 < dtau - 1e-6),
                        partial,
                        torch.zeros_like(partial),
                    )
                if user_beam_source is not None and umu0 is not None:
                    exp0 = torch.exp(-tau / umu0)
                    exp1 = torch.exp((tau - begin) / mu)
                    exp2 = torch.exp((tau - end) / mu)
                    expbea_begin = torch.exp(-begin / umu0)
                    expbea_end = torch.exp(-end / umu0)
                    denominator = 1.0 + mu / umu0
                    sign = -1.0 if mu < 0 else 1.0
                    full_beam = (
                        user_beam_source[..., lc, iu]
                        * (exp1 * expbea_begin - exp2 * expbea_end)
                        * sign
                        / denominator
                    )
                    beam_limit = (
                        user_beam_source[..., lc, iu] * dtau / umu0 * exp0
                    )
                    full_beam = torch.where(
                        denominator.abs() < 1e-4, beam_limit, full_beam
                    )
                    crosses = lc < target if mu < 0 else lc > target
                    value += torch.where(
                        crosses, full_beam, torch.zeros_like(full_beam)
                    )
                    partial_beam = (
                        user_beam_source[..., lc, iu]
                        * (
                            exp0
                            - (
                                expbea_begin * exp1
                                if mu < 0
                                else expbea_end * exp2
                            )
                        )
                        / denominator
                    )
                    partial_limit = (
                        user_beam_source[..., lc, iu] * dtau1 / umu0 * exp0
                    )
                    partial_beam = torch.where(
                        denominator.abs() < 1e-4,
                        partial_limit,
                        partial_beam,
                    )
                    partial_valid = (target == lc) & (
                        dtau1 > 1e-6 if mu < 0 else dtau1 < dtau - 1e-6
                    )
                    value += torch.where(
                        partial_valid,
                        partial_beam,
                        torch.zeros_like(partial_beam),
                    )
                if user_thermal0 is not None and user_thermal1 is not None:
                    sign = -1.0 if mu < 0 else 1.0
                    exp1 = torch.exp((tau - begin) / mu)
                    exp2 = torch.exp((tau - end) / mu)
                    full0 = sign * (exp1 - exp2)
                    full1 = sign * ((begin + mu) * exp1 - (end + mu) * exp2)
                    crosses = lc < target if mu < 0 else lc > target
                    full_thermal = (
                        user_thermal0[..., lc, iu] * full0
                        + user_thermal1[..., lc, iu] * full1
                    )
                    value += torch.where(
                        crosses, full_thermal, torch.zeros_like(full_thermal)
                    )
                    partial_exp = exp1 if mu < 0 else exp2
                    partial_fact = begin + mu if mu < 0 else end + mu
                    partial0 = 1.0 - partial_exp
                    partial1 = tau + mu - partial_fact * partial_exp
                    partial_thermal = (
                        user_thermal0[..., lc, iu] * partial0
                        + user_thermal1[..., lc, iu] * partial1
                    )
                    partial_valid = (target == lc) & (
                        dtau1 > 1e-6 if mu < 0 else dtau1 < dtau - 1e-6
                    )
                    value += torch.where(
                        partial_valid,
                        partial_thermal,
                        torch.zeros_like(partial_thermal),
                    )
            if mu < 0:
                value += fisot * torch.exp(tau / mu)
            result[..., lu, iu] = value
    return result


@timed(name="tensor_backend.extract_tp9_quadrature_intensity")
def extract_tp9_quadrature_intensity(
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    optics: TensorLayerOptics,
    grid: TensorOutputGrid,
    constants: torch.Tensor,
    beam_source: torch.Tensor | None = None,
    umu0: torch.Tensor | None = None,
    thermal0: torch.Tensor | None = None,
    thermal1: torch.Tensor | None = None,
    general_source: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the m=0 intensity at quadrature directions.

    This is the first radiance reconstruction increment.  It retains the
    device-resident layer/mode convention used by flux extraction and returns
    ``(nwave, ncol, ntau, nstr)`` before source interpolation or Fourier
    summation is introduced.
    """
    nstr = eigenvectors.shape[-1]
    nn = nstr // 2
    nlyr = eigenvectors.shape[-3]
    if constants.shape != (*eigenvectors.shape[:-3], nstr * nlyr):
        raise ValueError("constants are incompatible with eigenvectors")
    if grid.layru.shape[:2] != eigenvectors.shape[:2]:
        raise ValueError(
            "output grid and eigenvectors have incompatible batches"
        )
    constants = constants.reshape(*eigenvectors.shape[:-3], nlyr, nstr)
    layer = grid.layru - 1
    gc = torch.gather(
        eigenvectors,
        -3,
        layer[..., None, None].expand(*layer.shape, nstr, nstr),
    )
    layer_constants = torch.gather(
        constants, -2, layer[..., None].expand(*layer.shape, nstr)
    )
    taucpr_end = torch.gather(optics.taucpr, -1, layer)
    taucpr_begin = torch.gather(
        torch.cat(
            (
                torch.zeros_like(optics.taucpr[..., :1]),
                optics.taucpr[..., :-1],
            ),
            dim=-1,
        ),
        -1,
        layer,
    )
    values = torch.gather(
        eigenvalues, -2, layer[..., None].expand(*layer.shape, nn)
    )
    factors = torch.cat(
        (
            torch.exp(
                values.flip(dims=(-1,))
                * (grid.utaupr - taucpr_end).unsqueeze(-1)
            ),
            torch.exp(-values * (grid.utaupr - taucpr_begin).unsqueeze(-1)),
        ),
        dim=-1,
    )
    intensity = (gc @ (layer_constants * factors).unsqueeze(-1)).squeeze(-1)
    if beam_source is not None and umu0 is not None:
        intensity = intensity + torch.gather(
            beam_source, -2, layer[..., None].expand(*layer.shape, nstr)
        ) * torch.exp(-grid.utaupr / umu0.unsqueeze(-1)).unsqueeze(-1)
    if thermal0 is not None and thermal1 is not None:
        intensity = (
            intensity
            + torch.gather(
                thermal0, -2, layer[..., None].expand(*layer.shape, nstr)
            )
            + torch.gather(
                thermal1, -2, layer[..., None].expand(*layer.shape, nstr)
            )
            * grid.utaupr.unsqueeze(-1)
        )
    if general_source is not None:
        intensity = intensity + torch.gather(
            general_source, -2, layer[..., None].expand(*layer.shape, nstr)
        )
    return intensity


@timed(name="tensor_backend.extract_tp9_fluxes")
def extract_tp9_fluxes(
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    optics: TensorLayerOptics,
    grid: TensorOutputGrid,
    quadrature: TensorQuadrature,
    constants: torch.Tensor,
    beam_source: torch.Tensor | None = None,
    umu0: torch.Tensor | None = None,
    fbeam: torch.Tensor | None = None,
    thermal0: torch.Tensor | None = None,
    thermal1: torch.Tensor | None = None,
    general_source: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract homogeneous TP9 diffuse fluxes at the requested optical depths.

    The returned final axis is ``(upward, downward)``. This reproduces the
    homogeneous portion of ``c_fluxes`` for the no-beam, non-thermal TP9
    subset. Particular beam and thermal source terms are introduced separately.
    """
    nstr = eigenvectors.shape[-1]
    nn = nstr // 2
    nlyr = eigenvectors.shape[-3]
    if constants.shape != (*eigenvectors.shape[:-3], nstr * nlyr):
        raise ValueError("constants are incompatible with eigenvectors")
    constants = constants.reshape(*eigenvectors.shape[:-3], nlyr, nstr)
    if grid.layru.shape != grid.utaupr.shape:
        raise ValueError("output-grid layer and depth tensors must match")
    if grid.layru.shape[:2] != eigenvectors.shape[:2]:
        raise ValueError(
            "output grid and eigenvectors have incompatible batches"
        )
    if quadrature.cmu.shape != (nstr,) or quadrature.cwt.shape != (nstr,):
        raise ValueError("quadrature is incompatible with eigenvectors")

    layer = grid.layru - 1
    gather_gc = layer[..., None, None].expand(*layer.shape, nstr, nstr)
    gc = torch.gather(eigenvectors, -3, gather_gc)
    layer_constants = torch.gather(
        constants,
        -2,
        layer[..., None].expand(*layer.shape, nstr),
    )
    taucpr_end = torch.gather(optics.taucpr, -1, layer)
    taucpr_begin = torch.gather(
        torch.cat(
            (
                torch.zeros_like(optics.taucpr[..., :1]),
                optics.taucpr[..., :-1],
            ),
            dim=-1,
        ),
        -1,
        layer,
    )
    values = torch.gather(
        eigenvalues,
        -2,
        layer[..., None].expand(*layer.shape, nn),
    )
    negative = values.flip(dims=(-1,))
    factors = torch.cat(
        (
            torch.exp(negative * (grid.utaupr - taucpr_end).unsqueeze(-1)),
            torch.exp(-values * (grid.utaupr - taucpr_begin).unsqueeze(-1)),
        ),
        dim=-1,
    )
    intensity = (gc @ (layer_constants * factors).unsqueeze(-1)).squeeze(-1)
    if (thermal0 is None) != (thermal1 is None):
        raise ValueError("thermal0 and thermal1 must be supplied together")
    if thermal0 is not None and thermal1 is not None:
        if thermal0.shape != (*eigenvectors.shape[:-2], nstr):
            raise ValueError("thermal0 is incompatible with eigenvectors")
        thermal_at_grid0 = torch.gather(
            thermal0, -2, layer[..., None].expand(*layer.shape, nstr)
        )
        thermal_at_grid1 = torch.gather(
            thermal1, -2, layer[..., None].expand(*layer.shape, nstr)
        )
        intensity = (
            intensity
            + thermal_at_grid0
            + thermal_at_grid1 * grid.utaupr.unsqueeze(-1)
        )
    if general_source is not None:
        if general_source.shape != (*eigenvectors.shape[:-2], nstr):
            raise ValueError(
                "general_source is incompatible with eigenvectors"
            )
        intensity = intensity + torch.gather(
            general_source, -2, layer[..., None].expand(*layer.shape, nstr)
        )
    direct = None
    if any(value is not None for value in (beam_source, umu0, fbeam)):
        if beam_source is None or umu0 is None or fbeam is None:
            raise ValueError(
                "beam_source, umu0, and fbeam must be supplied together"
            )
        if beam_source.shape != (*eigenvectors.shape[:-2], nstr):
            raise ValueError("beam_source is incompatible with eigenvectors")
        beam = torch.gather(
            beam_source, -2, layer[..., None].expand(*layer.shape, nstr)
        )
        attenuation = torch.exp(-grid.utaupr / umu0.unsqueeze(-1))
        intensity = intensity + beam * attenuation.unsqueeze(-1)
        direct = umu0.unsqueeze(-1) * fbeam.unsqueeze(-1) * attenuation
    positive_mu = quadrature.cmu[:nn]
    positive_weight = quadrature.cwt[:nn]
    downward = (
        2.0
        * torch.pi
        * torch.sum(
            intensity[..., :nn]
            * (positive_weight * positive_mu).flip(dims=(-1,)),
            dim=-1,
        )
    )
    upward = (
        2.0
        * torch.pi
        * torch.sum(
            intensity[..., nn:] * positive_weight * positive_mu,
            dim=-1,
        )
    )
    if direct is not None:
        downward = downward + direct
    return torch.stack((upward, downward), dim=-1)


@timed(name="tensor_backend.solve_tp9_flux")
def solve_tp9_flux(
    prop: torch.Tensor,
    utau: torch.Tensor,
    fisot: torch.Tensor,
    *,
    nstr: int,
    nmom: int,
    deltam: bool = False,
    umu0: torch.Tensor | None = None,
    fbeam: torch.Tensor | None = None,
    thermal_xr0: torch.Tensor | None = None,
    thermal_xr1: torch.Tensor | None = None,
    temperature: torch.Tensor | None = None,
    wavenumber_lower: torch.Tensor | None = None,
    wavenumber_upper: torch.Tensor | None = None,
    bottom_temperature: torch.Tensor | None = None,
    top_temperature: torch.Tensor | None = None,
    top_emissivity: torch.Tensor | None = None,
    surface_albedo: torch.Tensor | None = None,
    general_source_computational: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the connected pure-PyTorch TP9a flux subset end to end.

    Supported physics is plane-parallel azimuth-independent diffuse top
    illumination, a black Lambertian lower boundary, and no thermal, beam, or
    general source. All tensors remain on ``prop.device`` throughout the
    calculation. The function is deliberately an experimental entry point;
    backend selection remains unchanged until broader parity is complete.
    """
    if utau.device != prop.device or fisot.device != prop.device:
        raise ValueError("prop, utau, and fisot must share a device")
    atmosphere = prepare_atmosphere(prop, nstr=nstr, nmom=nmom)
    optics = prepare_layer_optics(atmosphere, nstr=nstr, deltam=deltam)
    grid = prepare_output_grid(utau, atmosphere, optics, deltam=deltam)
    quadrature = gaussian_quadrature(nstr, device=prop.device)
    eigenvalues, eigenvectors = solve_reduced_eigenproblem(
        build_reduced_eigen_matrix(optics, quadrature, nstr=nstr)
    )
    if (umu0 is None) != (fbeam is None):
        raise ValueError("umu0 and fbeam must be supplied together")
    beam_source = (
        None
        if umu0 is None or fbeam is None
        else build_tp9_beam_source(optics, quadrature, umu0, fbeam, nstr=nstr)
    )
    general_source = (
        None
        if general_source_computational is None
        else build_tp9_general_source(
            optics, quadrature, general_source_computational, nstr=nstr
        )
    )
    if surface_albedo is not None:
        if surface_albedo.shape != prop.shape[:2]:
            raise ValueError("surface_albedo must have shape (nwave, ncol)")
        if (
            surface_albedo.dtype != prop.dtype
            or surface_albedo.device != prop.device
        ):
            raise ValueError("surface_albedo must share prop dtype and device")
    if (thermal_xr0 is None) != (thermal_xr1 is None):
        raise ValueError(
            "thermal_xr0 and thermal_xr1 must be supplied together"
        )
    if (bottom_temperature is None) != (top_temperature is None):
        raise ValueError(
            "bottom_temperature and top_temperature must be supplied together"
        )
    if top_emissivity is not None and top_temperature is None:
        raise ValueError(
            "top_emissivity requires thermal boundary temperatures"
        )
    thermal_boundary_requested = bottom_temperature is not None
    if temperature is not None:
        if thermal_xr0 is not None:
            raise ValueError(
                "supply either thermal coefficients or temperature inputs"
            )
        if wavenumber_lower is None or wavenumber_upper is None:
            raise ValueError(
                "temperature requires lower and upper wavenumber bounds"
            )
        thermal_xr0, thermal_xr1 = prepare_thermal_coefficients(
            temperature, optics, wavenumber_lower, wavenumber_upper
        )
    elif (wavenumber_lower is None) != (wavenumber_upper is None):
        raise ValueError("both lower and upper wavenumber bounds are required")
    elif (wavenumber_lower is not None) and not thermal_boundary_requested:
        raise ValueError("wavenumber bounds require temperature inputs")
    if thermal_boundary_requested:
        assert bottom_temperature is not None
        assert top_temperature is not None
        if wavenumber_lower is None or wavenumber_upper is None:
            raise ValueError(
                "thermal boundary temperatures require wavenumber bounds"
            )
        expected_shape = prop.shape[:2]
        if (
            bottom_temperature.shape != expected_shape
            or top_temperature.shape != expected_shape
        ):
            raise ValueError(
                "thermal boundary temperatures must have shape (nwave, ncol)"
            )
        thermal_bottom = planck_band_radiance(
            bottom_temperature, wavenumber_lower, wavenumber_upper
        )
        if surface_albedo is not None:
            thermal_bottom = thermal_bottom * (1.0 - surface_albedo)
        thermal_top = planck_band_radiance(
            top_temperature, wavenumber_lower, wavenumber_upper
        )
        if top_emissivity is not None:
            if top_emissivity.shape != expected_shape:
                raise ValueError(
                    "top_emissivity must have shape (nwave, ncol)"
                )
            thermal_top = thermal_top * top_emissivity
    else:
        thermal_top = None
        thermal_bottom = None
    thermal0, thermal1 = (
        (None, None)
        if thermal_xr0 is None or thermal_xr1 is None
        else build_tp9_thermal_source(
            optics, quadrature, thermal_xr0, thermal_xr1, nstr=nstr
        )
    )
    constants = solve_tp9_boundary_system(
        build_tp9_boundary_system(
            eigenvectors,
            eigenvalues,
            optics,
            fisot,
            beam_source,
            general_source,
            umu0,
            thermal0,
            thermal1,
            thermal_top,
            thermal_bottom,
            surface_albedo,
            quadrature,
            fbeam,
        )
    )
    return extract_tp9_fluxes(
        eigenvectors,
        eigenvalues,
        optics,
        grid,
        quadrature,
        constants,
        beam_source,
        umu0,
        fbeam,
        thermal0,
        thermal1,
        general_source,
    )


@timed(name="tensor_backend.build_tp9_beam_source")
def build_tp9_beam_source(
    optics: TensorLayerOptics,
    quadrature: TensorQuadrature,
    umu0: torch.Tensor,
    fbeam: torch.Tensor,
    *,
    nstr: int,
) -> torch.Tensor:
    """Compute C-DISORT's plane-parallel mazim=0 beam particular solution.

    This is the batched form of ``c_upbeam`` for the restricted tensor flow.
    It returns ``ZZ`` in C-DISORT quadrature-direction order; boundary RHS and
    flux integration consume it in the following beam-integration increment.
    """
    if umu0.shape != optics.dtaucpr.shape[:2]:
        raise ValueError("umu0 must have shape (nwave, ncol)")
    if fbeam.shape != optics.dtaucpr.shape[:2]:
        raise ValueError("fbeam must have shape (nwave, ncol)")
    if (
        umu0.dtype != optics.dtaucpr.dtype
        or fbeam.dtype != optics.dtaucpr.dtype
    ):
        raise ValueError("beam inputs must use the optics dtype")
    if (
        umu0.device != optics.dtaucpr.device
        or fbeam.device != optics.dtaucpr.device
    ):
        raise ValueError("beam inputs must share the optics device")
    if torch.any(umu0 <= 0):
        raise ValueError("umu0 must be positive for the plane-parallel beam")

    nn = nstr // 2
    mu = quadrature.cmu
    ylm = torch.empty((nstr, nstr), dtype=mu.dtype, device=mu.device)
    ylm0 = torch.empty((*umu0.shape, nstr), dtype=mu.dtype, device=mu.device)
    ylm[0] = 1.0
    ylm0[..., 0] = 1.0
    ylm[1] = mu
    # C-DISORT evaluates the incident direction at -UMU0 before forming its
    # beam particular solution.  Keep this separate from the positive
    # quadrature-angle convention used for outgoing directions.
    ylm0[..., 1] = -umu0
    for degree in range(2, nstr):
        ylm[degree] = (
            (2 * degree - 1) * mu * ylm[degree - 1]
            - (degree - 1) * ylm[degree - 2]
        ) / degree
        ylm0[..., degree] = (
            (2 * degree - 1) * (-umu0) * ylm0[..., degree - 1]
            - (degree - 1) * ylm0[..., degree - 2]
        ) / degree
    cc = 0.5 * torch.einsum(
        "...l,li,lj,j->...ij", optics.gl, ylm, ylm, quadrature.cwt
    )
    system = -cc
    diagonal = torch.arange(nstr, device=mu.device)
    system[..., diagonal, diagonal] += 1.0 + mu.view(
        1, 1, 1, nstr
    ) / umu0.unsqueeze(-1).unsqueeze(-1)
    source = (
        fbeam.unsqueeze(-1).unsqueeze(-1)
        * torch.einsum("abcl,li,abl->abci", optics.gl, ylm, ylm0)
        / (4.0 * torch.pi)
    )
    solution = torch.linalg.solve(system, source.unsqueeze(-1)).squeeze(-1)
    return torch.cat(
        (solution[..., nn:].flip(dims=(-1,)), solution[..., :nn]), dim=-1
    )


@timed(name="tensor_backend.build_tp9_general_source")
def build_tp9_general_source(
    optics: TensorLayerOptics,
    quadrature: TensorQuadrature,
    computational: torch.Tensor,
    *,
    nstr: int,
) -> torch.Tensor:
    """Build C-DISORT's Fourier-zero, layer-constant general source.

    ``computational`` preserves ``DisortOptions.general_source``'s shape:
    ``(nwave, ncol, nstr, nlyr, nstr)``. The tensor flux path consumes only
    Fourier order zero and quadrature-angle sources; user-angle sources belong
    to the future radiance path.
    """
    expected = (
        *optics.dtaucpr.shape[:2],
        nstr,
        optics.dtaucpr.shape[-1],
        nstr,
    )
    if computational.shape != expected:
        raise ValueError("computational general source has incompatible shape")
    if (
        computational.dtype != optics.dtaucpr.dtype
        or computational.device != optics.dtaucpr.device
    ):
        raise ValueError(
            "computational general source must share optics dtype and device"
        )
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
    system = -cc
    diagonal = torch.arange(nstr, device=mu.device)
    system[..., diagonal, diagonal] += 1.0
    source = computational[..., 0, :, :]
    rhs = torch.cat(
        (source[..., nn:], source[..., :nn].flip(dims=(-1,))), dim=-1
    )
    solution = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
    return torch.cat(
        (solution[..., nn:].flip(dims=(-1,)), solution[..., :nn]), dim=-1
    )


@timed(name="tensor_backend.build_tp9_thermal_source")
def build_tp9_thermal_source(
    optics: TensorLayerOptics,
    quadrature: TensorQuadrature,
    xr0: torch.Tensor,
    xr1: torch.Tensor,
    *,
    nstr: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build C-DISORT ``ZPLK0`` and ``ZPLK1`` thermal particular solutions.

    ``xr0`` and ``xr1`` are the layer-wise linear Planck-source coefficients
    used by ``c_upisot``. Boundary and output integration are added separately.
    """
    if xr0.shape != optics.dtaucpr.shape or xr1.shape != optics.dtaucpr.shape:
        raise ValueError("xr0 and xr1 must match the optical layer grid")
    if xr0.dtype != optics.dtaucpr.dtype or xr1.dtype != optics.dtaucpr.dtype:
        raise ValueError("thermal coefficients must use the optics dtype")
    if (
        xr0.device != optics.dtaucpr.device
        or xr1.device != optics.dtaucpr.device
    ):
        raise ValueError("thermal coefficients must share the optics device")

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
    system = -cc
    diagonal = torch.arange(nstr, device=mu.device)
    system[..., diagonal, diagonal] += 1.0
    source1 = (
        ((1.0 - optics.oprim) * xr1).unsqueeze(-1).expand(*xr1.shape, nstr)
    )
    z1 = torch.linalg.solve(system, source1.unsqueeze(-1)).squeeze(-1)
    source0 = (
        ((1.0 - optics.oprim) * xr0).unsqueeze(-1).expand(*xr0.shape, nstr)
    )
    source0 = source0 + mu.view(1, 1, 1, nstr) * z1
    z0 = torch.linalg.solve(system, source0.unsqueeze(-1)).squeeze(-1)
    return (
        torch.cat((z0[..., nn:].flip(dims=(-1,)), z0[..., :nn]), dim=-1),
        torch.cat((z1[..., nn:].flip(dims=(-1,)), z1[..., :nn]), dim=-1),
    )


@timed(name="tensor_backend.planck_band_radiance")
def planck_band_radiance(
    temperature: torch.Tensor,
    wavenumber_lower: torch.Tensor,
    wavenumber_upper: torch.Tensor,
) -> torch.Tensor:
    """Return C-DISORT's finite-band Planck radiance on the tensor device.

    This is the vectorized ``c_planck_func2`` series calculation, including
    its spectral-limit convention.  Matching that routine makes stored C and
    Fortran thermal reference values meaningful without a native callback.
    """
    if temperature.dtype != torch.float64:
        raise ValueError("temperature must use float64")
    if (
        wavenumber_lower.dtype != torch.float64
        or wavenumber_upper.dtype != torch.float64
    ):
        raise ValueError("wavenumber bounds must use float64")
    if (
        temperature.device != wavenumber_lower.device
        or temperature.device != wavenumber_upper.device
    ):
        raise ValueError(
            "temperature and wavenumber bounds must share a device"
        )
    if torch.any(temperature < 0) or torch.any(wavenumber_lower < 0):
        raise ValueError("temperature and wavenumbers must be nonnegative")
    if torch.any(wavenumber_upper < wavenumber_lower):
        raise ValueError("upper wavenumber must not be below lower wavenumber")
    temperature_safe = temperature.clamp_min(
        torch.finfo(temperature.dtype).tiny
    )
    lower = wavenumber_lower
    upper = wavenumber_upper
    while lower.ndim < temperature.ndim:
        lower = lower.unsqueeze(-1)
        upper = upper.unsqueeze(-1)
    c2 = 1.438786
    sigma_over_pi = 5.67032e-8 / torch.pi
    concentration = 15.0 / torch.pi**4
    v_lower = c2 * lower / temperature_safe
    v_upper = c2 * upper / temperature_safe

    def cumulative_fraction(v: torch.Tensor) -> torch.Tensor:
        power = (
            concentration
            * v.square()
            * v
            * (
                1.0 / 3.0
                + v
                * (
                    -1.0 / 8.0
                    + v
                    * (
                        1.0 / 60.0
                        + v.square()
                        * (
                            -1.0 / 5040.0
                            + v.square()
                            * (1.0 / 272160.0 - v.square() / 13305600.0)
                        )
                    )
                )
            )
        )
        thresholds = torch.tensor(
            [10.25, 5.7, 3.9, 2.9, 2.3, 1.9],
            dtype=v.dtype,
            device=v.device,
        )
        mmax = 1 + (v.unsqueeze(-1) < thresholds).sum(dim=-1)
        m = torch.arange(1, 8, dtype=v.dtype, device=v.device)
        mv = v.unsqueeze(-1) * m
        exponential = (
            torch.exp(-mv) * (6.0 + mv * (6.0 + mv * (3.0 + mv))) / m.pow(4)
        )
        tail = concentration * torch.where(
            m <= mmax.unsqueeze(-1), exponential, torch.zeros_like(exponential)
        ).sum(dim=-1)
        return torch.where(v < 1.5, power, tail)

    fraction_lower = cumulative_fraction(v_lower)
    fraction_upper = cumulative_fraction(v_upper)
    lower_small = v_lower < 1.5
    upper_small = v_upper < 1.5
    integral_fraction = torch.where(
        lower_small & upper_small,
        fraction_upper - fraction_lower,
        torch.where(
            lower_small,
            1.0 - fraction_lower - fraction_upper,
            fraction_lower - fraction_upper,
        ),
    )
    integral = sigma_over_pi * temperature_safe.pow(4) * integral_fraction
    max_exponent = torch.log(
        torch.full(
            (),
            torch.finfo(temperature.dtype).max,
            dtype=temperature.dtype,
            device=temperature.device,
        )
    )
    close_band = (
        (v_lower > torch.finfo(temperature.dtype).eps)
        & (v_upper < max_exponent)
        & ((upper - lower) / upper < 1.0e-2)
    )
    # Reproduce c_planck_func2's narrow-band Simpson iteration.  The series
    # above is deliberately truncated as in C, so using it for this case would
    # lose several digits through cancellation.
    old_value = torch.zeros_like(v_lower)
    simpson_value = torch.zeros_like(v_lower)
    converged = torch.zeros_like(close_band)
    interval = v_upper - v_lower
    endpoints = v_lower.pow(3) / torch.expm1(v_lower) + v_upper.pow(
        3
    ) / torch.expm1(v_upper)
    for n in range(1, 11):
        node = torch.arange(
            1, 2 * n, dtype=temperature.dtype, device=temperature.device
        )
        point = v_lower.unsqueeze(-1) + interval.unsqueeze(-1) * node / (2 * n)
        coefficient = torch.where(
            node.remainder(2) == 0,
            torch.full_like(node, 2.0),
            torch.full_like(node, 4.0),
        )
        current = (
            (
                endpoints
                + (coefficient * point.pow(3) / torch.expm1(point)).sum(dim=-1)
            )
            * interval
            / (6 * n)
        )
        newly_converged = ~converged & (
            torch.abs((current - old_value) / current) <= 1.0e-6
        )
        simpson_value = torch.where(newly_converged, current, simpson_value)
        converged = converged | newly_converged
        old_value = current
    simpson_value = torch.where(converged, simpson_value, old_value)
    simpson_integral = (
        sigma_over_pi * temperature_safe.pow(4) * concentration * simpson_value
    )
    integral = torch.where(close_band, simpson_integral, integral)
    spectral_limit = (
        1.1911e-8
        * upper.pow(3)
        * torch.exp(-c2 * upper / temperature_safe)
        / (1.0 - torch.exp(-c2 * upper / temperature_safe))
    )
    integral = torch.where(
        wavenumber_upper == wavenumber_lower, spectral_limit, integral
    )
    return torch.where(
        temperature < 1.0e-4, torch.zeros_like(integral), integral
    )


@timed(name="tensor_backend.prepare_thermal_coefficients")
def prepare_thermal_coefficients(
    temperature: torch.Tensor,
    optics: TensorLayerOptics,
    wavenumber_lower: torch.Tensor,
    wavenumber_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert level temperatures into C-DISORT ``XR0`` and ``XR1`` tensors."""
    if temperature.shape != (
        *optics.dtaucpr.shape[:2],
        optics.dtaucpr.shape[-1] + 1,
    ):
        raise ValueError(
            "temperature must contain one value for each layer boundary"
        )
    planck = planck_band_radiance(
        temperature, wavenumber_lower, wavenumber_upper
    )
    difference = planck[..., 1:] - planck[..., :-1]
    xr1 = torch.where(
        optics.dtaucpr > 1.0e-4,
        difference / optics.dtaucpr,
        torch.zeros_like(difference),
    )
    previous_taucpr = torch.cat(
        (torch.zeros_like(optics.taucpr[..., :1]), optics.taucpr[..., :-1]),
        dim=-1,
    )
    xr0 = planck[..., :-1] - xr1 * previous_taucpr
    return xr0, xr1
