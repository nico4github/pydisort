/* Opt-in TP4c trace driver.  Mirrors the immutable published Test Problem 4c
 * inputs without modifying its reference source. */
#include <math.h>
#include <stdio.h>
#include "cdisort.hpp"

#undef UUM
#define UUM(k, iu, lu) out.uum[(iu) - 1 + ((lu) - 1 + (k) * ds.ntau) * ds.numu]
#undef UU
#define UU(j, iu, lu) out.uu[(iu) - 1 + ((lu) - 1 + ((j) - 1) * ds.ntau) * ds.numu]

int main(void) {
  disort_state ds = {};
  disort_output out = {};
  ds.accur = 0.;
  ds.flag.ibcnd = GENERAL_BC;
  ds.flag.usrtau = ds.flag.usrang = ds.flag.lamber = TRUE;
  ds.flag.planck = ds.flag.quiet = ds.flag.spher = FALSE;
  ds.flag.general_source = FALSE;
  ds.flag.output_uum = TRUE;
  ds.flag.intensity_correction = TRUE;
  ds.flag.old_intensity_correction = TRUE;
  ds.nstr = ds.nphase = ds.nmom = 32;
  ds.nlyr = 1; ds.ntau = 3; ds.numu = 6; ds.nphi = 3;
  ds.bc.fbeam = M_PI; ds.bc.fisot = ds.bc.albedo = ds.bc.fluor = 0.;
  ds.bc.umu0 = .5; ds.bc.phi0 = 0.; ds.flag.brdf_type = BRDF_NONE;
  c_disort_state_alloc(&ds); c_disort_out_alloc(&ds, &out);
  c_getmom(HAZE_GARCIA_SIEWERT, 0., ds.nmom, ds.pmom);
  ds.dtauc[0] = 1.; ds.ssalb[0] = .9;
  ds.utau[0] = 0.; ds.utau[1] = .5; ds.utau[2] = 1.;
  ds.umu[0] = -1.; ds.umu[1] = -.5; ds.umu[2] = -.1;
  ds.umu[3] = .1; ds.umu[4] = .5; ds.umu[5] = 1.;
  ds.phi[0] = 0.; ds.phi[1] = 90.; ds.phi[2] = 180.;
  c_disort(&ds, &out, c_planck_func2);
  for (int k = 0; k < ds.nstr; ++k) {
    fprintf(stderr, "PYDISORT_TRACE_TP4_UUM order=%d", k);
    for (int lu = 1; lu <= ds.ntau; ++lu)
      for (int iu = 1; iu <= ds.numu; ++iu) fprintf(stderr, " %.17g", UUM(k, iu, lu));
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "PYDISORT_TRACE_TP4_UU");
  for (int j = 1; j <= ds.nphi; ++j)
    for (int lu = 1; lu <= ds.ntau; ++lu)
      for (int iu = 1; iu <= ds.numu; ++iu) fprintf(stderr, " %.17g", UU(j, iu, lu));
  fprintf(stderr, "\n");
  c_disort_out_free(&ds, &out); c_disort_state_free(&ds);
  return 0;
}
