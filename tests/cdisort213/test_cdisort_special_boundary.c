/* Standalone Problem 13 allocation/run/free regression (compiled as C++).
 * Compare the reciprocity shortcut with unit incident-flux beam solutions.
 */
#include <cmath>
#include <cstdio>
#include <initializer_list>

#include "cdisort.hpp"

static void configure(disort_state *ds, int layers, double albedo) {
  ds->nstr = ds->nmom = 16;
  ds->nlyr = layers;
  ds->flag.quiet = TRUE;
  ds->flag.lamber = TRUE;
  ds->flag.old_intensity_correction = TRUE;
  ds->bc.albedo = albedo;
}

static void atmosphere(disort_state *ds) {
  for (int layer = 0; layer < ds->nlyr; ++layer) {
    ds->dtauc[layer] = 1.0 / ds->nlyr;
    ds->ssalb[layer] = layer == 0 ? 0.99 : 0.5;
    c_getmom(HENYEY_GREENSTEIN, 0.8, ds->nmom,
             ds->pmom + layer * (ds->nmom_nstr + 1));
  }
}

static bool close(double actual, double expected) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <= 1.e-8 + 1.e-6 * std::abs(expected);
}

static bool run_case(int layers, int angles, int azimuths, double albedo) {
  disort_state ds = {};
  disort_output out = {};
  configure(&ds, layers, albedo);
  ds.flag.ibcnd = SPECIAL_BC;
  ds.flag.usrang = TRUE;
  ds.numu = angles;
  ds.nphi = azimuths;
  c_disort_state_alloc(&ds);
  c_disort_out_alloc(&ds, &out);
  atmosphere(&ds);
  for (int angle = 0; angle < angles; ++angle) {
    ds.umu[angle] = (angle + 1.0) / (angles + 1.0);
  }
  bool ok = true;
  // Reuse the state to verify that the solver restores its angle dimensions.
  for (int repeat = 0; repeat < 2; ++repeat) {
    int status = c_disort(&ds, &out, c_planck_func2);
    ok = ok && status == 0 && ds.numu == angles;
    for (int angle = 0; angle < angles; ++angle) {
      double mu = (angle + 1.0) / (angles + 1.0);
      disort_state beam = {};
      disort_output reference = {};
      configure(&beam, layers, albedo);
      beam.flag.onlyfl = TRUE;
      beam.bc.umu0 = mu;
      beam.bc.fbeam = 1.0 / mu;
      c_disort_state_alloc(&beam);
      c_disort_out_alloc(&beam, &reference);
      atmosphere(&beam);
      int beam_status = c_disort(&beam, &reference, c_planck_func2);
      double reflected = reference.rad[0].flup;
      double transmitted = reference.rad[layers].rfldir +
                           reference.rad[layers].rfldn;
      if (status != 0 || beam_status != 0 || !close(ds.umu[angle], mu) ||
          !close(out.albmed[angle], reflected) ||
          !close(out.trnmed[angle], transmitted)) {
        std::fprintf(stderr,
                     "FAIL: layers=%d angles=%d phi=%d albedo=%g mu=%g "
                     "shortcut=(%.12g, %.12g) beam=(%.12g, %.12g)\n",
                     layers, angles, azimuths, albedo, mu,
                     out.albmed[angle], out.trnmed[angle],
                     reflected, transmitted);
        ok = false;
      }
      c_disort_out_free(&beam, &reference);
      c_disort_state_free(&beam);
    }
  }
  c_disort_out_free(&ds, &out);
  c_disort_state_free(&ds);
  return ok;
}

int main() {
  int passed = 0;
  int failed = 0;
  for (int layers : {1, 2}) {
    for (int angles : {1, 3}) {
      for (int azimuths : {0, 2}) {
        for (double albedo : {0.0, 0.5}) {
          if (run_case(layers, angles, azimuths, albedo)) {
            ++passed;
          } else {
            ++failed;
          }
        }
      }
    }
  }
  std::printf("PASS: %d; SKIP: 0; FAIL: %d\n", passed, failed);
  return failed ? 1 : 0;
}
