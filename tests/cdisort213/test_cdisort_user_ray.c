/******************************************************************************
 * test_cdisort_user_ray.c
 *
 * A standalone driver for “Test Problem 09” from the original test suite,
 * with command-line control over the number of streams (nstr) and layers
 *(nlyr).
 *
 * Usage:
 *   test_cdisort_09 [nstr nlyr nwave]
 *   Defaults: nstr = 32, nlyr = 100, nwave = 1000
 *
 * Loops through nwave wavenumbers to simulate a longer run.
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cdisort.hpp"

#undef DTAUC
#define DTAUC(lc) ds.dtauc[(lc) - 1]
#undef PHI
#define PHI(j) ds.phi[(j) - 1]
#undef PMOM
#define PMOM(k, lc) ds.pmom[(k) + ((lc) - 1) * (ds.nmom_nstr + 1)]
#undef SSALB
#define SSALB(lc) ds.ssalb[(lc) - 1]
#undef TEMPER
#define TEMPER(lc) ds.temper[(lc)]
#undef UMU
#define UMU(iu) ds.umu[(iu) - 1]
#undef UTAU
#define UTAU(lu) ds.utau[(lu) - 1]

#undef GOODUU
#define GOODUU(iu, lu, j) \
  good.uu[(iu) - 1 + (((lu) - 1 + ((j) - 1) * ds.ntau) * ds.numu)]

void run_disort_user_ray(void);

int main(void) {
  run_disort_user_ray();
  return 0;
}

void run_disort_user_ray(void) {
  const int nstr = 4;
  const int nlyr = 2;
  const double ssalb = 0.;
   int icas, lc, k;
  const int ncase = 1;
  double gg;
  disort_state ds;
  disort_output out, good;

  /* Initialize flags */
  ds.accur = 0.;
  ds.flag.prnt[0] = FALSE;
  ds.flag.prnt[1] = FALSE;
  ds.flag.prnt[2] = FALSE;
  ds.flag.prnt[3] = FALSE;
  ds.flag.prnt[4] = FALSE;

  ds.flag.ibcnd = GENERAL_BC;
  ds.flag.usrtau = TRUE;
  ds.flag.usrang = TRUE;
  ds.flag.lamber = TRUE;
  ds.flag.onlyfl = FALSE;
  ds.flag.quiet = TRUE;
  ds.flag.spher = FALSE;
  ds.flag.general_source = FALSE;
  ds.flag.output_uum = FALSE;
  ds.flag.intensity_correction = TRUE;
  ds.flag.old_intensity_correction = TRUE;

  /* Apply user-specified geometry */
  ds.nstr = nstr;
  ds.nlyr = nlyr;
  ds.nphase = ds.nstr;
  ds.nmom = ds.nstr;
  ds.ntau = 4;
  ds.numu = 4;
  ds.nphi = 1;

  ds.bc.fbeam = 0.;
  ds.bc.fisot = 1. / M_PI;
  ds.bc.phi0 = 0.0;
  ds.bc.umu0 = 0.5;
  ds.bc.fluor = 0.;

  ds.flag.brdf_type = BRDF_NONE;

  for (icas = 1; icas <= ncase; ++icas) {
    switch (icas) {
      case 1:
        ds.flag.planck = FALSE;

        /* Allocate memory */
        c_disort_state_alloc(&ds);
        c_disort_out_alloc(&ds, &out);
        c_disort_out_alloc(&ds, &good);

        /* Set optical properties per layer */
        for (lc = 1; lc <= ds.nlyr; ++lc) {
          DTAUC(lc) = lc == 1 ? 0.2 : 0.5;
          SSALB(lc) = lc == 1 ? 0.4 : 0.7;
        }

        /* Tau grid" (fixed 5 points) */
        UTAU(1) = 0.;
        UTAU(2) = 0.2;
        UTAU(3) = 0.45;
        UTAU(4) = 0.7;

        /* Cosine angles */
        UMU(1) = -0.5;
        UMU(2) = -0.3;
        UMU(3) = 0.4;
        UMU(4) = 0.5;

        /* Azimuthal angle */
        PHI(1) = 60.;

        /* Phase function moments */
        for (lc = 1; lc <= ds.nlyr; ++lc) {
          c_getmom(ISOTROPIC, 0., ds.nmom, &PMOM(0, lc));
        }

        ds.bc.albedo = 0.;

        break;
    } /* Execute DISORT with Planck emission function */
    c_disort(&ds, &out, c_planck_func2);

    /* Clean up */
    c_disort_out_free(&ds, &good);
    c_disort_out_free(&ds, &out);
    c_disort_state_free(&ds);
  }
  return;
}
