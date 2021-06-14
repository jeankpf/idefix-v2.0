// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) 2020-2021 Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************
#ifndef HYDRO_CALCPARABOLICFLUX_HPP_
#define HYDRO_CALCPARABOLICFLUX_HPP_

#include "hydro.hpp"
#include "dataBlock.hpp"
#include "addNonIdealMHDFlux.hpp"

// Compute parabolic fluxes
template <int dir>
void Hydro::CalcParabolicFlux(const real t) {
  idfx::pushRegion("Hydro::CalcParabolicFlux");

  IdefixArray3D<real> dMax = this->dMax;

  // Reset Max diffusion coefficient
  idefix_for("HydroParabolicResetStage",0,data->np_tot[KDIR],
                                        0,data->np_tot[JDIR],
                                        0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
        dMax(k,j,i) = ZERO_F;
      }
  );

  if( (resistivityStatus.isExplicit  && (! data->rklCycle))
    || (resistivityStatus.isRKL  && ( data->rklCycle))
    || (ambipolarStatus.isExplicit  && (! data->rklCycle))
    || (ambipolarStatus.isRKL  && ( data->rklCycle)) ) {
      this->AddNonIdealMHDFlux<dir>(t);
  }

  if( (viscosityStatus.isExplicit && (!data->rklCycle))
    || (viscosityStatus.isRKL && data->rklCycle))  {
      // Add fargo velocity if using fargo
    if(haveFargo && viscosityStatus.isExplicit) {
      fargo.AddVelocity(t);
    }
    this->viscosity.AddViscousFlux(dir,t);

    // Remove back Fargo velocity
    if(haveFargo && viscosityStatus.isExplicit) {
      fargo.SubstractVelocity(t);
    }
  }

  idfx::popRegion();
}

#endif //HYDRO_CALCPARABOLICFLUX_HPP_