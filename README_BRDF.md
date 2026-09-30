# BRDF capability assessment

## Current public support

`pydisort` supports Lambertian reflection through `flags("lamber")` and a
fixed-parameter Hapke BRDF through `DisortOptions.hapke_brdf()` on CPU.
Hapke is explicitly rejected on CUDA.

RPV, CAM (Cox–Munk ocean), and AMB (Ambrals) do not have public typed
options. They therefore cannot be enabled accidentally through the Python
API. The sibling `disort-pyf` bridge reports noncanonical `brdf_type` values
as `SKIP` with a reason; it maps only the canonical DISOTEST Hapke setup.

## C-DISORT capability

The embedded C-DISORT source already implements allocation, setup, and
reflectivity evaluation for all three models:

| Model | C-DISORT parameters | CPU wrapper effort |
| --- | --- | --- |
| RPV | `rho0`, `k`, `theta`, `sigma`, `t1`, `t2`, `scale` | Moderate |
| CAM | wind speed `u10`, pigment/chlorophyll `pcl`, salinity `xsal` | High |
| AMB | isotropic `iso`, volumetric `vol`, geometric `geo` | Moderate |

CPU support needs a typed immutable configuration for each model, parameter
domain checks, per-wave/per-column C-DISORT state population, and direct
C-DISORT or Fortran references. RPV and AMB can share the same typed-BRDF
infrastructure. CAM also needs compile-feature detection (`HAVE_BRDF`),
spectral bounds, and glint-sensitive validation across geometry and wind.

## Recommendation

For the common case, Lambertian is sufficient; add Hapke for particulate
planetary surfaces. The current reconstruction scope retains those two models and explicitly defers RPV, CAM, and AMB. Implement RPV only when land/vegetation reflectance or
legacy configuration compatibility is required. Defer CAM and AMB until a
concrete science case requires them.

CUDA support is a separate, larger project. The present CUDA path is tuned
for Lambertian scalar solves and does not transfer arbitrary C-DISORT BRDF
state. Completing all three models on CPU is feasible as three focused
changes; completing all three with CPU/CUDA parity needs CUDA state transfer,
kernel validation, and H100 agreement for each model.

## Tensor-backend expansion order

The experimental Python/PyTorch tensor backend currently implements only the
black-Lambertian lower boundary used by TP9a/TP9b. Its next surface increment
should be **nonzero Lambertian albedo**, validated with DISORT Test Problem 6c.
That is the standard active flux reference and supplies the surface-coupling
layout required by every later reflecting boundary.

Add **Hapke** after Lambertian albedo. The existing public C-backend option is
CPU-only and uses fixed C-DISORT constants; a tensor implementation must first
represent the Fourier/azimuthal surface-reflection coupling rather than merely
copying those constants. Freeze the upstream Test Problem 6d--6h values into
self-contained JSON fixtures before enabling it.

For the remaining C-DISORT BRDFs, build one immutable, batched tensor surface
configuration interface, then add each model with its own CPU/CUDA fixture:

| Model | Tensor work after shared surface interface |
| --- | --- |
| RPV | Vectorize the analytic reflectivity over incidence, emergence, relative azimuth, and its seven parameters; validate parameter domains and land-reflectance cases. |
| CAM | Add spectral bounds, wind/pigment/salinity tensors, glint-angle stability handling, and compile-feature-independent fixtures. |
| AMB | Vectorize its isotropic, volumetric, and geometric kernels; validate near-horizon behavior. |

None of RPV, CAM, or AMB is a current active Python reference requirement.
They remain deferred until a science case needs them. The C backend stays the
fallback while the tensor implementations are built and verified.
