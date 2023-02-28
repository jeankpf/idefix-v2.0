// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef FLUID_RIEMANNSOLVER_CALCFLUX_HPP_
#define FLUID_RIEMANNSOLVER_CALCFLUX_HPP_

#if MHD == YES
#include "hlldMHD.hpp"
#include "hllMHD.hpp"
#include "roeMHD.hpp"
#include "tvdlfMHD.hpp"
#else
#include "hllcHD.hpp"
#include "hllHD.hpp"
#include "tvdlfHD.hpp"
#include "roeHD.hpp"
#endif

// Compute Riemann fluxes from states
template <typename Phys>
template <int dir>
void RiemannSolver<Phys>::CalcFlux(IdefixArray4D<real> &flux) {
  idfx::pushRegion("RiemannSolver::CalcFlux");
  if constexpr(dir == IDIR) {
    // enable shock flattening
    if(haveShockFlattening) shockFlattening.FindShock();
  }

  if constexpr(Phys::mhd) {
    switch (mySolver) {
      case TVDLF_MHD:
        TvdlfMHD<dir>(flux);
        break;
      case HLL_MHD:
        HllMHD<dir>(flux);
        break;
      case HLLD_MHD:
        HlldMHD<dir>(flux);
        break;
      case ROE_MHD:
        RoeMHD<dir>(flux);
        break;
      default:
        break;
    }
  } else {
    switch (mySolver) {
      case TVDLF:
        TvdlfHD<dir>(flux);
        break;
      case HLL:
        HllHD<dir>(flux);
        break;
      case HLLC:
        HllcHD<dir>(flux);
        break;
      case ROE:
        RoeHD<dir>(flux);
        break;
      default: // do nothing
        break;
    }
  }

  idfx::popRegion();
}
#endif // FLUID_RIEMANNSOLVER_CALCFLUX_HPP_