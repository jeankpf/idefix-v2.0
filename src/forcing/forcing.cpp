// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <string>

#include "forcing.hpp"
#include "dataBlock.hpp"
#include "dataBlockHost.hpp"
#include "input.hpp"

KOKKOS_INLINE_FUNCTION real K_aff_11(real x, real x0, real x1) {
  return 2./(x1-x0)*x - (x1+x0)/(x1-x0);
}

KOKKOS_INLINE_FUNCTION real K_aff_02pi(real x, real x0, real x1) {
  return 2.*M_PI*(x - x0)/(x1 - x0);
}

KOKKOS_INLINE_FUNCTION real K_cheby_fst(int n, real x) {
  if (x >= -1. and x <= 1.) {
    return cos(n*acos(x));
  } else {
    return ZERO_F;
  }
}

KOKKOS_INLINE_FUNCTION real K_cheby_fst_both_hom_dir(int n, real x) {
  if (x >= -1. and x <= 1.) {
    return cos(n*acos(x)) - 1.*((n+1)%2) - x*(n%2);
    return cos((n+1)*acos(x)) - 1.*(n%2) - x*((n+1)%2);
  } else {
    return ZERO_F;
  }
}

KOKKOS_INLINE_FUNCTION real K_cheby_fst_both_hom_neu(int n, real x) {
  if (x >= -1. and x <= 1.) {
    return cos(n*acos(x)) - pow(n/(n+2),2.)*cos((n+2)*acos(x));
//    return cos(n*acos(x)) - pow(n/(n+2),2.)*cos((n+2)*acos(x))*((n+1)%2) - pow(n/(n+2),2.)*cos((n+2)*acos(x))*(n%2);
  } else {
    return ZERO_F;
  }
}

KOKKOS_INLINE_FUNCTION real K_fourier(int n, real x) {
  if (x >= 0. and x <= 2.*M_PI) {
    return cos(n*x) + sin(n*x);
  } else {
    return ZERO_F;
  }
}

KOKKOS_INLINE_FUNCTION real K_sin(int n, real x) {
  if (x >= 0. and x <= 2.*M_PI) {
    return sin(n*x);
  } else {
    return ZERO_F;
  }
}

KOKKOS_INLINE_FUNCTION real K_cos(int n, real x) {
  if (x >= 0. and x <= 2.*M_PI) {
    return cos(n*x);
  } else {
    return ZERO_F;
  }
}

Forcing::Forcing(Input &input, DataBlock *datain) {
  idfx::pushRegion("Forcing::Forcing");
  this->data = datain;
  this->seed = input.GetOrSet<int>("Forcing","seed",0,0);

  this->write = input.GetOrSet<int>("Forcing","write",0, 0);
//  std::string folder = input.GetOrSet<std::string>("Forcing","filename",0,"testOU");
  this->folder = input.GetOrSet<std::string>("Output","folder",0,"output");

  this->stillHaveForcing = true;
  this->stopTime = input.GetOrSet<real>("Forcing","stoptime",0,std::numeric_limits<real>::infinity());

  this->nForcingModes = 0;

  this->kmin = -1.;
  this->kmax = -1.;
  this->haveSolenoidalForcing = false;

  this->xbeg = data->mygrid->xbeg[IDIR];
  this->xend = data->mygrid->xend[IDIR];
  this->ybeg = data->mygrid->xbeg[JDIR];
  this->yend = data->mygrid->xend[JDIR];
  this->zbeg = data->mygrid->xbeg[KDIR];
  this->zend = data->mygrid->xend[KDIR];

  this->kx0 = 2.*M_PI/(xend - xbeg);
  this->ky0 = 2.*M_PI/(yend - ybeg);
  this->kz0 = 2.*M_PI/(zend - zbeg);

  if (input.CheckEntry("Forcing", "iso3D")>=0) {
    this->forcingType = iso3D;
    #if GEOMETRY != CARTESIAN
      IDEFIX_ERROR("You cannot have 3D isotropic forcing in another geometry than Cartesian");
    #endif //GEOMETRY != CARTESIAN
    #if COMPONENTS < 3 or DIMENSIONS < 3
      IDEFIX_ERROR("You cannot have 3D isotropic forcing with less than 3 components and dimensions.");
    #endif //COMPONENTS < 3 or DIMENSIONS < 3
// if () IDEFIX_ERROR("You should not have 3D isotropic forcing with other boundary conditions than fully periodic.");

    this->kmin = input.Get<real>("Forcing","iso3D", 0);
    this->kmax = input.Get<real>("Forcing","iso3D", 1);

    std::vector<std::array<real,3>> k3Disovec;

    int nxmax = kmax/kx0 + 1;
    int nymax = kmax/ky0 + 1;
    int nzmax = kmax/kz0 + 1;
    for (int nx=1; nx<nxmax; nx++) {
      for (int ny=1; ny<nymax; ny++) {
        for (int nz=1; nz<nzmax; nz++) {
          real kx = kx0*nx;
          real ky = ky0*ny;
          real kz = kz0*nz;
          real k_2 = kx*kx + ky*ky + kz*kz;
          if (k_2 >= kmin*kmin and k_2 <= kmax*kmax) {
            nForcingModes ++;
            k3Disovec.push_back({kx, ky, kz});
            modeNames.push_back({"I" + std::to_string(nx) + std::to_string(ny) + std::to_string(nz), "J" + std::to_string(nx) + std::to_string(ny) + std::to_string(nz), "K" + std::to_string(nx) + std::to_string(ny) + std::to_string(nz)});
            nForcingModes ++;
            k3Disovec.push_back({kx, -ky, kz});
            modeNames.push_back({"I" + std::to_string(nx) + std::to_string(-ny) + std::to_string(nz), "J" + std::to_string(nx) + std::to_string(-ny) + std::to_string(nz), "K" + std::to_string(nx) + std::to_string(-ny) + std::to_string(nz)});
            nForcingModes ++;
            k3Disovec.push_back({kx, ky, -kz});
            modeNames.push_back({"I" + std::to_string(nx) + std::to_string(ny) + std::to_string(-nz), "J" + std::to_string(nx) + std::to_string(ny) + std::to_string(-nz), "K" + std::to_string(nx) + std::to_string(ny) + std::to_string(-nz)});
            nForcingModes ++;
            k3Disovec.push_back({kx, -ky, -kz});
            modeNames.push_back({"I" + std::to_string(nx) + std::to_string(-ny) + std::to_string(-nz), "J" + std::to_string(nx) + std::to_string(-ny) + std::to_string(-nz), "K" + std::to_string(nx) + std::to_string(-ny) + std::to_string(-nz)});
          }
        }
      }
    }
    k3DisoHost = IdefixHostArray2D<real>("k3DisoHost", nForcingModes, 3);
    k3Diso = IdefixArray2D<real>("k3Diso", nForcingModes, 3);
    for (int l=0; l<nForcingModes; l++) {
      k3DisoHost(l,0) = k3Disovec[l][0];
      k3DisoHost(l,1) = k3Disovec[l][1];
      k3DisoHost(l,2) = k3Disovec[l][2];
    }
    Kokkos::deep_copy(k3Diso, k3DisoHost);
    EXPAND(
    this->forcingModesIdir = IdefixArray4D<Kokkos::complex<real>>("forcingModesIdir", nForcingModes,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);          ,
    this->forcingModesJdir = IdefixArray4D<Kokkos::complex<real>>("forcingModesJdir", nForcingModes,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);          ,
    this->forcingModesKdir = IdefixArray4D<Kokkos::complex<real>>("forcingModesKdir", nForcingModes,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);          )
  }

  else if (input.CheckEntry("Forcing", "iso2D")>=0) {
    this->forcingType = iso2D;
    this->normal2DisoStr = input.Get<std::string>("Forcing","iso2D", 0);
    if (this->normal2DisoStr == "IDIR") this->normal2Diso = IDIR;
    else if (this->normal2DisoStr == "JDIR") this->normal2Diso = JDIR;
    else if (this->normal2DisoStr == "KDIR") this->normal2Diso = KDIR;
    else IDEFIX_ERROR("The normal component for 2D isotropic forcing cannot not be something else than IDIR, JDIR or KDIR");
    #if COMPONENTS < 2 or DIMENSIONS < 2
      IDEFIX_ERROR("You cannot have 2D isotropic forcing with less than 2 components and dimensions.");
    #endif //COMPONENTS < 2 or DIMENSIONS < 2
    #if GEOMETRY == SPHERICAL
      if (this->normal2DisoStr == "JDIR" or this->normal2DisoStr == "KDIR") IDEFIX_ERROR("Comparing radial with angular wavenumbers is somewhat shady.");
    #endif //GEOMETRY == SPHERICAL

    this->kmin = input.Get<real>("Forcing","iso2D", 1);
    this->kmax = input.Get<real>("Forcing","iso2D", 2);

    std::vector<std::array<real,3>> k2Disovec;

    kx0 = (this->normal2Diso==IDIR) ? ZERO_F : kx0;
    ky0 = (this->normal2Diso==JDIR) ? ZERO_F : ky0;
    kz0 = (this->normal2Diso==KDIR) ? ZERO_F : kz0;
    int nxmax = (this->normal2Diso==IDIR) ? 1 : kmax/kx0 + 1;
    int nymax = (this->normal2Diso==JDIR) ? 1 : kmax/ky0 + 1;
    int nzmax = (this->normal2Diso==KDIR) ? 1 : kmax/kz0 + 1;
    int xsign, ysign, zsign;
    if (this->normal2Diso==IDIR) {
      xsign = 0;
      ysign = 1;
      zsign = -1;
    } else if (this->normal2Diso==JDIR) {
      xsign = 1;
      ysign = 0;
      zsign = -1;
    } else {
      xsign = 1;
      ysign = -1;
      zsign = 0;
    }
    for (int nx=0; nx<nxmax; nx++) {
      for (int ny=0; ny<nymax; ny++) {
        for (int nz=0; nz<nzmax; nz++) {
          real kx = kx0*nx;
          real ky = ky0*ny;
          real kz = kz0*nz;
          real k_2 = kx*kx + ky*ky + kz*kz;
          if (k_2 >= kmin*kmin and k_2 <= kmax*kmax) {
            nForcingModes ++;
            k2Disovec.push_back({kx, ky, kz});
            modeNames.push_back({"I" + std::to_string(nx) + std::to_string(ny) + std::to_string(nz), "J" + std::to_string(nx) + std::to_string(ny) + std::to_string(nz), "K" + std::to_string(nx) + std::to_string(ny) + std::to_string(nz)});
            nForcingModes ++;
            k2Disovec.push_back({xsign*kx, ysign*ky, zsign*kz});
            modeNames.push_back({"I" + std::to_string(xsign*nx) + std::to_string(ysign*ny) + std::to_string(zsign*nz), "J" + std::to_string(xsign*nx) + std::to_string(ysign*ny) + std::to_string(zsign*nz), "K" + std::to_string(xsign*nx) + std::to_string(ysign*ny) + std::to_string(zsign*nz)});
          }
        }
      }
    }
    k2DisoHost = IdefixHostArray2D<real>("k2DisoHost", nForcingModes, 3);
    k2Diso = IdefixArray2D<real>("k2Diso", nForcingModes, 3);
    for (int l=0; l<nForcingModes; l++) {
      k2DisoHost(l,0) = k2Disovec[l][0];
      k2DisoHost(l,1) = k2Disovec[l][1];
      k2DisoHost(l,2) = k2Disovec[l][2];
    }
    Kokkos::deep_copy(k2Diso, k2DisoHost);
    EXPAND(
    this->forcingModesIdir = IdefixArray4D<Kokkos::complex<real>>("forcingModesIdir", nForcingModes,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);          ,
    this->forcingModesJdir = IdefixArray4D<Kokkos::complex<real>>("forcingModesJdir", nForcingModes,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);          ,
    this->forcingModesKdir = IdefixArray4D<Kokkos::complex<real>>("forcingModesKdir", nForcingModes,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);          )
  }

  else if (input.CheckEntry("Forcing", "ani3D")>=0) {
    #if COMPONENTS < 3 or DIMENSIONS < 3
      IDEFIX_ERROR("You cannot have 3D anisotropic forcing with less than 3 components and dimensions.");
    #endif //COMPONENTS < 3 or DIMENSIONS < 3
    this->forcingType = ani3D;
    this->normalAni3DStr = input.Get<std::string>("Forcing","ani3D", 0);
    if (this->normalAni3DStr == "IDIR") this->normalAni3D = IDIR;
    else if (this->normalAni3DStr == "JDIR") this->normalAni3D = JDIR;
    else if (this->normalAni3DStr == "KDIR") this->normalAni3D = KDIR;
    else IDEFIX_ERROR("The normal component for 3D anisotropic forcing cannot not be something else than IDIR, JDIR, or KDIR");

    this->normalAni3DBasisStr = input.Get<std::string>("Forcing","ani3D", 1);
    if (this->normalAni3DBasisStr == "chebyshev") this->normalAni3DBasis = chebyshev;
//    else if (this->normalAni3DBasisStr == "legendre") this->normalAni3DBasis = legendre;
    else if (this->normalAni3DBasisStr == "fourier") this->normalAni3DBasis = fourier;
    else IDEFIX_ERROR("The basis for the normal component of the forcing can only be chebyshev or fourier for now (legendre coming soon).");

    std::vector<NormalBoundType> vecNormalAni3DBound;
    std::string dirbegStr = "X" + std::to_string(this->normalAni3D + 1) + "-beg";
    std::string direndStr = "X" + std::to_string(this->normalAni3D + 1) + "-end";
    if (input.Get<std::string>("Boundary",dirbegStr, 0) != input.Get<std::string>("Boundary",direndStr, 0)) {
      IDEFIX_ERROR("Anisotropic basis with different BCs at the two sides are not yet implemented");
    } else if (input.Get<std::string>("Boundary",dirbegStr, 0) == "reflective") {
      (this->normalAni3D == IDIR) ? vecNormalAni3DBound.push_back(bothHomDir) : vecNormalAni3DBound.push_back(bothHomNeu);
      (this->normalAni3D == JDIR) ? vecNormalAni3DBound.push_back(bothHomDir) : vecNormalAni3DBound.push_back(bothHomNeu);
      (this->normalAni3D == KDIR) ? vecNormalAni3DBound.push_back(bothHomDir) : vecNormalAni3DBound.push_back(bothHomNeu);
    } else if (input.Get<std::string>("Boundary",dirbegStr, 0) == "outflow") {
      vecNormalAni3DBound.push_back(bothHomNeu);
      vecNormalAni3DBound.push_back(bothHomNeu);
      vecNormalAni3DBound.push_back(bothHomNeu);
    } else if (input.Get<std::string>("Boundary",dirbegStr, 0) == "userdef") {
      for (int comp=IDIR; comp<COMPONENTS; comp++) {
        std::string userdefDirbegStr = "begBoundTypeVX" + std::to_string(comp + 1);
        std::string userdefDirendStr = "endBoundTypeVX" + std::to_string(comp + 1);
        if (input.Get<std::string>("Boundary",userdefDirbegStr, 0) != input.Get<std::string>("Boundary",userdefDirendStr, 0)) {
          IDEFIX_ERROR("Anisotropic basis with different BCs at the two sides are not yet implemented");
        } else if (input.Get<std::string>("Boundary",userdefDirbegStr, 0) == "dirichletZero") {
          vecNormalAni3DBound.push_back(bothHomDir);
        } else if (input.Get<std::string>("Boundary",userdefDirbegStr, 0) == "neumann") {
          vecNormalAni3DBound.push_back(bothHomNeu);
        } else {
          IDEFIX_ERROR("userdef BCs not recognised, they are needed to choose a suitable forcing basis.");
        }
      }
    } else if (input.Get<std::string>("Boundary",dirbegStr, 0) == "periodic") {
      IDEFIX_ERROR("periodic BCs recognised, you should use 3D isotropic forcing in this direction.");
    } else {
      IDEFIX_ERROR("BCs not recognised, they are needed to choose the suitable forcing basis.");
    }
    normalAni3DBoundHost = IdefixHostArray1D<NormalBoundType>("normalAni3DBoundHost", COMPONENTS);
    normalAni3DBound = IdefixArray1D<NormalBoundType>("normalAni3DBound", COMPONENTS);
    for (int comp=IDIR; comp<COMPONENTS; comp++) {
      normalAni3DBoundHost(comp) = vecNormalAni3DBound[comp];
      normalAni3DBoundHost(comp) = vecNormalAni3DBound[comp];
      normalAni3DBoundHost(comp) = vecNormalAni3DBound[comp];
    }
    Kokkos::deep_copy(normalAni3DBound, normalAni3DBoundHost);

    this->kmin = input.Get<real>("Forcing","ani3D", 2);
    this->kmax = input.Get<real>("Forcing","ani3D", 3);

    if (input.GetOrSet<std::string>("Forcing","ani3D", 4, "ras") == "write") WriteNormalBasis(folder);

    std::vector<std::vector<real>> kAni3Dvec;

    int nxmax = kmax/kx0 + 1;
    int nymax = kmax/ky0 + 1;
    int nzmax = kmax/kz0 + 1;
    int xsign, ysign, zsign;
    if (this->normalAni3D==IDIR) {
      xsign = 1;
      ysign = 1;
      zsign = -1;
    } else if (this->normalAni3D==JDIR) {
      xsign = 1;
      ysign = 1;
      zsign = -1;
    } else if (this->normalAni3D==KDIR) {
      xsign = 1;
      ysign = -1;
      zsign = 1;
    } else {
      IDEFIX_ERROR("Forcing: normal direction not known");
    }
    for (int nx=1; nx<nxmax; nx++) {
      for (int ny=1; ny<nymax; ny++) {
        for (int nz=1; nz<nzmax; nz++) {
          real kx = kx0*nx;
          real ky = ky0*ny;
          real kz = kz0*nz;
          real factor = 1.;
          #if GEOMETRY == SPHERICAL
            factor *= 4./pow(xbeg+xend, 2.);
          #endif //GEOMETRY == SPHERICAL
          //undimensionalise kx so that kmin and kmax corresponds to angular ktheta and kphi
          real k_2 = kx*kx/factor + (ky*ky + kz*kz);
//          real k_2 = kx*kx + factor*(ky*ky + kz*kz);
          if (k_2 >= kmin*kmin and k_2 <= kmax*kmax) {
            nForcingModes ++;
            kAni3Dvec.push_back({kx, ky, kz});
            modeNames.push_back({"I" + std::to_string(nx) + std::to_string(ny) + std::to_string(nz), "J" + std::to_string(nx) + std::to_string(ny) + std::to_string(nz), "K" + std::to_string(nx) + std::to_string(ny) + std::to_string(nz)});
            nForcingModes ++;
            kAni3Dvec.push_back({xsign*kx, ysign*ky, zsign*kz});
            modeNames.push_back({"I" + std::to_string(xsign*nx) + std::to_string(ysign*ny) + std::to_string(zsign*nz), "J" + std::to_string(xsign*nx) + std::to_string(ysign*ny) + std::to_string(zsign*nz), "K" + std::to_string(xsign*nx) + std::to_string(ysign*ny) + std::to_string(zsign*nz)});
          }
        }
      }
    }
    kAni3DHost = IdefixHostArray2D<real>("kAni3DHost", nForcingModes, 3);
    kAni3D = IdefixArray2D<real>("kAni3D", nForcingModes, 3);
    for (int l=0; l<nForcingModes; l++) {
      kAni3DHost(l,0) = kAni3Dvec[l][0];
      kAni3DHost(l,1) = kAni3Dvec[l][1];
      kAni3DHost(l,2) = kAni3Dvec[l][2];
    }
    Kokkos::deep_copy(kAni3D, kAni3DHost);
    EXPAND(
    this->forcingModesIdir = IdefixArray4D<Kokkos::complex<real>>("forcingModesIdir", nForcingModes,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);          ,
    this->forcingModesJdir = IdefixArray4D<Kokkos::complex<real>>("forcingModesJdir", nForcingModes,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);          ,
    this->forcingModesKdir = IdefixArray4D<Kokkos::complex<real>>("forcingModesKdir", nForcingModes,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);          )
    this->forcingModesNormalAni3D = IdefixArray3D<real>("forcingModesNormalAni3D", COMPONENTS, nForcingModes, data->np_tot[normalAni3D]);
  }

  if (input.CheckEntry("Forcing", "solenoidal")>=0) {
    #if GEOMETRY == POLAR or GEOMETRY == CYLINDRICAL
      IDEFIX_ERROR("Cannot have solenoidal forcing in polar in cylindrical geometry.");
    #endif //GEOMETRY == POLAR or GEOMETRY == CYLINDRICAL
    this->haveSolenoidalForcing = true;
  }

  // Allocate required arrays
  this->forcingTerm = IdefixArray4D<real>("forcingTerm", COMPONENTS,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);
  this->pristineForcingTerm = IdefixArray4D<real>("pristineForcingTerm", COMPONENTS,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);
  this->solenoidalForcingTerm = IdefixArray4D<real>("solenoidalForcingTerm", COMPONENTS,
                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);
//  this->compressiveForcingTerm = IdefixArray4D<real>("compressiveForcingTerm", COMPONENTS,
//                              data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);

  this->means = IdefixArray2D<real>("means", nForcingModes, COMPONENTS);
  this->tcorrs = IdefixArray2D<real>("tcorrs", nForcingModes, COMPONENTS);
  this->epsilons = IdefixArray2D<real>("epsilons", nForcingModes, COMPONENTS);

// WARNING DO SOMETHING WITH MEANS TCORRS AND EPSILONS
  this->hostMeans = IdefixHostArray2D<real>("hostMeans", nForcingModes, COMPONENTS);
  this->hostTcorrs = IdefixHostArray2D<real>("hostTcorrs", nForcingModes, COMPONENTS);
  this->hostEpsilons = IdefixHostArray2D<real>("hostEpsilons", nForcingModes, COMPONENTS);
  this->targetVel = input.GetOrSet<real>("Forcing", "targetVel", 0, -1.);
  if (this->targetVel >= 0) { // in this case they will be reset later
    tcorr = 1.;
    epsilon = -1.;
  } else{
    tcorr = input.Get<real>("Forcing", "t_corr", 0);
    epsilon = input.Get<real>("Forcing", "epsilon", 0);
  }

  this->oUprocesses.InitProcesses(this->folder, this->seed, this->nForcingModes, this->modeNames);
  this->InitForcingModes();
  this->InitForcingParameters();

  if (!input.restartRequested) {
    this->oUprocesses.ResetProcessesValues(); // so that we get the excited modes even if we don't write
    this->oUprocesses.ResetTimestep();
    if(this->write) {
      this->oUprocesses.ResetNormalValues();
    }
  }
  if (input.restartRequested) this->oUprocesses.AdvanceProcessesValues();
//  if (input.restartRequested) this->oUprocesses.AdvanceProcessesValues(data.tabDt);

  idfx::popRegion();
}

void Forcing::ShowConfig() {
  idfx::cout << "Forcing: ENABLED with seed " << seed << "." << std::endl;
  if (stopTime < std::numeric_limits<double>::infinity()) {
    idfx::cout << "Forcing: will be stopped at t=" << stopTime << " ." << std::endl;
  }
  switch(forcingType) {
    case ForcingType::iso3D:
      idfx::cout << "Forcing: 3D isotropic." << std::endl;
      idfx::cout << "Forcing: kmin=" << kmin << " and kmax=" << kmax << " ." << std::endl;
      idfx::cout << "Forcing: There are " << nForcingModes << " different forcing modes." << std::endl;
      if (haveSolenoidalForcing) idfx::cout << "Forcing: solenoidal." << std::endl;
      break;
    case ForcingType::iso2D:
      idfx::cout << "Forcing: 2D isotropic with normal " << normal2DisoStr << "." << std::endl;
      idfx::cout << "Forcing: kmin=" << kmin << " and kmax=" << kmax << " ." << std::endl;
      idfx::cout << "Forcing: There are " << nForcingModes << " different forcing modes." << std::endl;
      if (haveSolenoidalForcing) idfx::cout << "Forcing: solenoidal." << std::endl;
      break;
    case ForcingType::ani3D:
      idfx::cout << "Forcing: 3D anisotropic with normal " << normalAni3DStr << " and " << normalAni3DBasisStr << " basis." << std::endl;
      idfx::cout << "Forcing: kmin=" << kmin << " and kmax=" << kmax << " ." << std::endl;
      idfx::cout << "Forcing: There are " << nForcingModes << " different forcing modes." << std::endl;
      if (haveSolenoidalForcing) idfx::cout << "Forcing: solenoidal." << std::endl;
      break;
  }

  if (targetVel >= ZERO_F) {
    idfx::cout << "Forcing: Targeted velocity=" << targetVel << " ." << std::endl;
  } else {
    idfx::cout << "Forcing: epsilon=" << epsilon << " and tcorr=" << tcorr << " ." << std::endl;
  }


}

void Forcing::InitForcingParameters() {
  idfx::pushRegion("Forcing::InitForcingParameters");

  if (this->targetVel >= 0) {
    real kf = HALF_F*(kmin+kmax);
    this->epsilon = pow(this->targetVel,3.)*kf/(2.*M_PI);
    this->tcorr = 2.*M_PI/(this->targetVel*kf);
  }
  for (int l=0; l<nForcingModes; l++) {
    for (int dir=IDIR; dir<COMPONENTS; dir++) {
      this->hostMeans(l,dir) = ZERO_F;
      this->hostTcorrs(l,dir) = tcorr;
      this->hostEpsilons(l,dir) = epsilon;
    }
  }
  Kokkos::deep_copy(this->means, this->hostMeans);
  Kokkos::deep_copy(this->tcorrs, this->hostTcorrs);
  Kokkos::deep_copy(this->epsilons, this->hostEpsilons);
  this->oUprocesses.SetProcesses(this->means, this->tcorrs, this->epsilons);

  idfx::popRegion();
}

void Forcing::InitForcingModes() {
  idfx::pushRegion("Forcing::InitForcingModes");

  real kx0 = this->kx0;
  real ky0 = this->ky0;
  real kz0 = this->kz0;
  real xbeg = this->xbeg;
  real ybeg = this->ybeg;
  real zbeg = this->zbeg;
  real xend = this->xend;
  real yend = this->yend;
  real zend = this->zend;
  EXPAND(
  IdefixArray4D<Kokkos::complex<real>> forcingModesIdir = this->forcingModesIdir; ,
  IdefixArray4D<Kokkos::complex<real>> forcingModesJdir = this->forcingModesJdir; ,
  IdefixArray4D<Kokkos::complex<real>> forcingModesKdir = this->forcingModesKdir; )
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  IdefixArray1D<real> x3 = data->x[KDIR];
  IdefixArray2D<real> k3Diso = this->k3Diso;
  IdefixArray2D<real> k2Diso = this->k2Diso;
  Kokkos::complex unit_j(0.,1.);
  int nForcingModes = this->nForcingModes;
  ForcingType forcingType = this->forcingType;

  // todo(GL): not sure this is needed
  Kokkos::deep_copy(forcingModesIdir,0);
  Kokkos::deep_copy(forcingModesJdir,0);
  Kokkos::deep_copy(forcingModesKdir,0);


  switch(forcingType) {
    case ForcingType::iso3D:
      idefix_for("iso3D", 0, nForcingModes, 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
                  KOKKOS_LAMBDA (int l, int k, int j, int i) {
                    real kx = k3Diso(l, IDIR);
                    real ky = k3Diso(l, JDIR);
                    real kz = k3Diso(l, KDIR);
                    real kdotx = kx*x1(i) + ky*x2(j) + kz*x3(k);
                    EXPAND(
                    forcingModesIdir(l,k,j,i) = exp(unit_j * kdotx); ,
                    forcingModesJdir(l,k,j,i) = exp(unit_j * kdotx); ,
                    forcingModesKdir(l,k,j,i) = exp(unit_j * kdotx); )
      });
      break;
    case ForcingType::iso2D:
      idefix_for("iso2D", 0, nForcingModes, 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
                  KOKKOS_LAMBDA (int l, int k, int j, int i) {
                    real kx = k2Diso(l, IDIR);
                    real ky = k2Diso(l, JDIR);
                    real kz = k2Diso(l, KDIR);
                    real kdotx = kx*x1(i) + ky*x2(j) + kz*x3(k);
                    EXPAND(
                    forcingModesIdir(l,k,j,i) = exp(unit_j * kdotx); ,
                    forcingModesJdir(l,k,j,i) = exp(unit_j * kdotx); ,
                    forcingModesKdir(l,k,j,i) = exp(unit_j * kdotx); )
      });
      break;
    case ForcingType::ani3D:
      idefix_for("normalAni3D", 0, nForcingModes, 0, data->np_tot[normalAni3D],
                  KOKKOS_LAMBDA (int l, int idx) {
                    real k = kAni3D(l, normalAni3D);
                    int oppx = (normalAni3D==IDIR) ? 1 : 0;
                    int oppy = (normalAni3D==JDIR) ? 1 : 0;
                    int oppz = (normalAni3D==KDIR) ? 1 : 0;
                    int order = 0;
                    order = (normalAni3D==IDIR) ? k/kx0 : order;
                    order = (normalAni3D==JDIR) ? k/ky0 : k/kz0;
                    real rightx = oppx*x1(idx) + oppy*x2(idx) + oppz*x3(idx);
                    real rightx0 = oppx*xbeg + oppy*ybeg + oppz*zbeg;
                    real rightx1 = oppx*xend + oppy*yend + oppz*zend;
                    switch(normalAni3DBasis) {
                      case NormalBasis::chebyshev:
                        for (int comp=IDIR; comp<COMPONENTS; comp++) {
                          int bound = normalAni3DBound(comp);
                          switch(bound) {
                            case NormalBoundType::bothHomDir:
                              forcingModesNormalAni3D(comp,l,idx) = K_cheby_fst_both_hom_dir(order, K_aff_11(rightx, rightx0, rightx1));
                            break;
                            case NormalBoundType::bothHomNeu:
                              forcingModesNormalAni3D(comp,l,idx) = K_cheby_fst_both_hom_neu(order, K_aff_11(rightx, rightx0, rightx1));
                            break;
                          }
                        }
                      break;
                      case NormalBasis::fourier:
                        for (int comp=IDIR; comp<COMPONENTS; comp++) {
                          int bound = normalAni3DBound(comp);
                          switch(bound) {
                            case NormalBoundType::bothHomDir:
                              forcingModesNormalAni3D(comp,l,idx) = K_sin(order, K_aff_02pi(rightx, rightx0, rightx1));
                            break;
                            case NormalBoundType::bothHomNeu:
                              forcingModesNormalAni3D(comp,l,idx) = K_cos(order, K_aff_02pi(rightx, rightx0, rightx1));
                            break;
                          }
                        }
                    }
      });
      idefix_for("ani3D", 0, nForcingModes, 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
                  KOKKOS_LAMBDA (int l, int k, int j, int i) {
                    real kx = kAni3D(l, IDIR);
                    real ky = kAni3D(l, JDIR);
                    real kz = kAni3D(l, KDIR);
                    int offx = (normalAni3D==IDIR) ? 0 : 1;
                    int offy = (normalAni3D==JDIR) ? 0 : 1;
                    int offz = (normalAni3D==KDIR) ? 0 : 1;
                    real kdotx = offx*kx*x1(i) + offy*ky*x2(j) + offz*kz*x3(k);
                    int oppx = (normalAni3D==IDIR) ? 1 : 0;
                    int oppy = (normalAni3D==JDIR) ? 1 : 0;
                    int oppz = (normalAni3D==KDIR) ? 1 : 0;
                    int idx = oppx*i + oppy*j + oppz*k;
                    forcingModesIdir(l,k,j,i) = forcingModesNormalAni3D(IDIR,l,idx) * exp(unit_j*kdotx);
                    forcingModesJdir(l,k,j,i) = forcingModesNormalAni3D(JDIR,l,idx) * exp(unit_j*kdotx);
                    forcingModesKdir(l,k,j,i) = forcingModesNormalAni3D(KDIR,l,idx) * exp(unit_j*kdotx);
      });
      break;

//    case ForcingType::userDef:
//      break;
  }
  idfx::popRegion();
}

// This function compute the required forcing field
void Forcing::ComputeForcing(real dt) {
  idfx::pushRegion("Forcing::ComputeForcing");

  ResetForcingTerms();
  if (haveSolenoidalForcing) {
    ComputePristineForcing(dt);
    ComputeSolenoidalForcing(dt);
    forcingTerm = solenoidalForcingTerm;
  } else {
    ComputePristineForcing(dt);
    forcingTerm = pristineForcingTerm;
  }
  oUprocesses.UpdateProcessesValues(dt);
  idfx::popRegion();
}

// This function compute the pristine forcing field
void Forcing::ComputePristineForcing(real dt) {
  idfx::pushRegion("Forcing::ComputePristineForcing");

  EXPAND(
  IdefixArray4D<Kokkos::complex<real>> forcingModesIdir = this->forcingModesIdir; ,
  IdefixArray4D<Kokkos::complex<real>> forcingModesJdir = this->forcingModesJdir; ,
  IdefixArray4D<Kokkos::complex<real>> forcingModesKdir = this->forcingModesKdir; )
  IdefixArray4D<real> pristineForcingTerm = this->pristineForcingTerm;
  IdefixArray2D<Kokkos::complex<real>> ouValues = this->oUprocesses.ouValues;
  int nForcingModes = this->nForcingModes;
  idefix_for("ComputePristineForcing", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                  Kokkos::complex<real> forcingX1=0.;
                  Kokkos::complex<real> forcingX2=0.;
                  Kokkos::complex<real> forcingX3=0.;
                  for (int l=0; l<nForcingModes; l++) {
                    EXPAND( forcingX1 += ouValues(l,IDIR)*forcingModesIdir(l,k,j,i); ,
                            forcingX2 += ouValues(l,JDIR)*forcingModesJdir(l,k,j,i); ,
                            forcingX3 += ouValues(l,KDIR)*forcingModesKdir(l,k,j,i); )
                  }
                  pristineForcingTerm(IDIR,k,j,i) += forcingX1.real();
                  pristineForcingTerm(JDIR,k,j,i) += forcingX2.real();
                  pristineForcingTerm(KDIR,k,j,i) += forcingX3.real();
  });

  idfx::popRegion();
}

// This function compute the solenoidal part of the forcing field
void Forcing::ComputeSolenoidalForcing(real dt) {
  idfx::pushRegion("Forcing::ComputeSolenoidalForcing");

  IdefixArray2D<real> kiso = (this->forcingType == iso3D) ? this->k3Diso : this->k2Diso;
  EXPAND( IdefixArray4D<Kokkos::complex<real>> forcingModesIdir = this->forcingModesIdir; ,
          IdefixArray4D<Kokkos::complex<real>> forcingModesJdir = this->forcingModesJdir; ,
          IdefixArray4D<Kokkos::complex<real>> forcingModesKdir = this->forcingModesKdir; )

  IdefixArray4D<real> pristineForcingTerm = this->pristineForcingTerm;
  IdefixArray4D<real> solenoidalForcingTerm = this->solenoidalForcingTerm;

  IdefixArray1D<real> dx1 = data->dx[IDIR];
  IdefixArray1D<real> dx2 = data->dx[JDIR];
  IdefixArray1D<real> dx3 = data->dx[KDIR];

  IdefixArray1D<real> x1m = data->xl[IDIR];
  IdefixArray1D<real> x2m = data->xl[JDIR];
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];

  IdefixArray1D<real> sinx2 = data->sinx2;

  real kmean = HALF_F*(kmax + kmin);
  idefix_for("ComputeSolenoidalForcing", 1, data->np_tot[KDIR] - 1, 1, data->np_tot[JDIR] - 1, 1, data->np_tot[IDIR] - 1,
              KOKKOS_LAMBDA (int k, int j, int i) {
                #if GEOMETRY == CARTESIAN
                      solenoidalForcingTerm(IDIR,k,j,i) = D_EXPAND( ZERO_F                                               ,
                                                 + 1/dx2(j) * (pristineForcingTerm(KDIR,k,j+1,i) - pristineForcingTerm(KDIR,k,j-1,i) )  ,
                                                 - 1/dx3(k) * (pristineForcingTerm(JDIR,k+1,j,i) - pristineForcingTerm(JDIR,k-1,j,i) )  );
                  #if DIMENSIONS >= 2
                      solenoidalForcingTerm(JDIR,k,j,i) = D_EXPAND( - 1/dx1(i) * (pristineForcingTerm(KDIR,k,j,i+1) - pristineForcingTerm(KDIR,k,j,i-1) )  ,
                                                                                                      ,
                                                 + 1/dx3(k) * (pristineForcingTerm(IDIR,k+1,j,i) - pristineForcingTerm(IDIR,k-1,j,i) )  );
                  #endif
                  #if DIMENSIONS == 3
                      solenoidalForcingTerm(KDIR,k,j,i) = 1/dx1(i) * (pristineForcingTerm(JDIR,k,j,i+1) - pristineForcingTerm(JDIR,k,j,i-1) )
                                       - 1/dx2(j) * (pristineForcingTerm(IDIR,k,j+1,i) - pristineForcingTerm(IDIR,k,j-1,i) );
                  #endif
                #endif

                solenoidalForcingTerm(IDIR,k,j,i) /= 2.*kmean;
                solenoidalForcingTerm(JDIR,k,j,i) /= 2.*kmean;
                solenoidalForcingTerm(KDIR,k,j,i) /= 2.*kmean;
  });

  idfx::popRegion();
}

// This function writes the normal basis in a txt file to later plot them with matplotlib
// Working in mono-domain only
void Forcing::WriteNormalBasis(std::string folder) {
//  int order_max = kmax/kx0 + 1;
//  int order_min = kmin/kx0 - 1;
//  order_min = std::max(1, order_min);
//  real xbeg = this->xbeg;
//  real xend = this->xend;
//  IdefixArray1D<NormalBoundType> normalAni3DBound = this->normalAni3DBound;
//  int normalAni3DBasis = this->normalAni3DBasis;
//  IdefixHostArray3D<real> normalBasisHost("normalBasisHost", order_max, COMPONENTS, data->np_tot[IDIR]);
//  IdefixArray3D<real> normalBasis("normalBasis", order_max, COMPONENTS, data->np_tot[IDIR]);
//  idefix_for("write", 0, order_max, 0, data->np_tot[IDIR],
//              KOKKOS_LAMBDA (int order, int i) {
//                switch(normalAni3DBasis) {
//                  case NormalBasis::chebyshev:
//                    for (int dir = IDIR; dir < COMPONENTS; dir++) {
//                      int bound = normalAni3DBound(dir);
//                      switch(bound) {
//                        case NormalBoundType::bothHomDir:
//                          normalBasis(order,dir,i) = K_cheby_fst_both_hom_dir(order, K_aff_11(x1(i), xbeg, xend));
//                        break;
//                        case NormalBoundType::bothHomNeu:
//                          normalBasis(order,dir,i) = K_cheby_fst_both_hom_neu(order, K_aff_11(x1(i), xbeg, xend));
//                        break;
//                      }
//                    }
//                    break;
//                  case NormalBasis::fourier:
//                    for (int dir = IDIR; dir < COMPONENTS; dir++) {
//                      int bound = normalAni3DBound(dir);
//                      switch(bound) {
//                        case NormalBoundType::bothHomDir:
//                          normalBasis(order,dir,i) = K_sin(order, K_aff_02pi(x1(i), xbeg, xend));
//                        break;
//                        case NormalBoundType::bothHomNeu:
//                          normalBasis(order,dir,i) = K_cos(order, K_aff_02pi(x1(i), xbeg, xend));
//                        break;
//                      }
//                    }
//                    break;
//                  }
//  });
  IdefixArray1D<real> x = data->x[normalAni3D];
  IdefixHostArray3D<real> forcingModesNormalAni3DHost("forcingModesNormalAni3DHost", COMPONENTS, nForcingModes, data->np_tot[normalAni3D]);
  IdefixHostArray1D<real> xhost("xhost", data->np_tot[normalAni3D]);
  Kokkos::deep_copy(forcingModesNormalAni3DHost, forcingModesNormalAni3D);
  Kokkos::deep_copy(xhost, x);
  if(idfx::prank==0) {
    int precision = 10;
    int col_width = precision + 10;
    std::string filename = folder + "/normalBasis.dat";
    std::ofstream file;
    file.open(filename, std::ios::trunc);
    file << std::setw(col_width) << "x";
    for (int comp=IDIR; comp<COMPONENTS; comp++) {
      for (int l=0; l<nForcingModes; l++) {
        std::string current_name;
        if (comp == IDIR) current_name = "fI"+std::to_string(l);
        else if (comp == JDIR) current_name = "fJ"+std::to_string(l);
        else if (comp == KDIR) current_name = "fK"+std::to_string(l);
        else IDEFIX_ERROR("Weird error");
        file << std::setw(col_width) << current_name;
      }
    }
    file << std::endl;
    file.precision(precision);
    for (int i=0; i<data->np_tot[normalAni3D]; i++) {
      file << std::setw(col_width) << xhost(i);
      for (int comp=IDIR; comp<COMPONENTS; comp++) {
        for (int l=0; l<nForcingModes; l++) {
          file << std::scientific << std::setw(col_width) << forcingModesNormalAni3DHost(comp,l,i);
        }
      }
      file << std::endl;
    }
    file << std::endl;
    file.close();
  }
}

// Fill the forcing term with zeros
void Forcing::ResetForcingTerms() {
  idfx::pushRegion("Forcing::ResetForcingTerms");

  Kokkos::deep_copy(this->forcingTerm,0);
  Kokkos::deep_copy(this->pristineForcingTerm,0);
  Kokkos::deep_copy(this->solenoidalForcingTerm,0);

  idfx::popRegion();
}
