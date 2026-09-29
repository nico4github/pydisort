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
planetary surfaces. Implement RPV only when land/vegetation reflectance or
legacy configuration compatibility is required. Defer CAM and AMB until a
concrete science case requires them.

CUDA support is a separate, larger project. The present CUDA path is tuned
for Lambertian scalar solves and does not transfer arbitrary C-DISORT BRDF
state. Completing all three models on CPU is feasible as three focused
changes; completing all three with CPU/CUDA parity needs CUDA state transfer,
kernel validation, and H100 agreement for each model.
