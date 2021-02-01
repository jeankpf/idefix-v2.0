#include "idefix.hpp"
#include "setup.hpp"


// custom parameters meant to be parsed from the inifile
// see Setup constructor

// default values are absurd by design
real aspect_ratio_glob{-1};
real jump_radius_glob{-1};
real jump_width_glob{-1};


real randm(void) {
  // Customized random number generator
  // Allow one to have consistent random numbers
  // generators on different architectures.
  const int a = 16807;
  const int m = 2147483647;
  static int in0 = 13763 + 2417*idfx::prank;
  int q;

  /* find random number  */
  q = (int) fmod((double) a * in0, m);
  in0=q;

  return((real) ((double) q/(double)m));
}


// hydro functions to enroll
// note that everywhere we make the assumption GM = 1, so that
// Phi = - GM/R = - 1/R => Omega_K = R^(-1/2)
// cs = H * Omega_K = h * R * sqrt(GM/R^3) = h / sqrt(R)
// where R is the polar radius

void GravitationalPotential(DataBlock& data, const real t, IdefixArray1D<real>& x1, IdefixArray1D<real>& x2, IdefixArray1D<real>& x3, IdefixArray3D<real>& phi) {

    IdefixArray1D<real> R = x1;
    idefix_for("Potential",0,data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR],
        KOKKOS_LAMBDA (int k, int j, int i) {
        phi(k,j,i) = -1.0/R(i);
    });
}

void LISOTHSoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
  // locally isothermal soundspeed
  // cs = H * Omega_K
  // this is adapted from test/HD/VSI
  IdefixArray1D<real> r=data.x[IDIR];
  real aspect_ratio{aspect_ratio_glob};
  idefix_for("LISOTHSoundSpeed",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                real R = r(i);
                cs(k,j,i) = aspect_ratio/sqrt(R);
              });
}


// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
    // Set the function for userdefboundary
    data.hydro.EnrollGravPotential(&GravitationalPotential);
    data.hydro.EnrollIsoSoundSpeed(&LISOTHSoundSpeed);
    aspect_ratio_glob = input.GetReal("Setup","aspect_ratio",0);
    jump_radius_glob = input.GetReal("Setup", "jump_radius",0);
    jump_width_glob = input.GetReal("Setup", "jump_width",0);
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);

    real x,y,z;

    real vphi,f,r,th;

    // arbitrary values for now, I'll set those as parameters later
    real jump_width{jump_width_glob};
    real jump_radius{jump_radius_glob};
    real aspect_ratio{aspect_ratio_glob};

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
      for(int j = 0; j < d.np_tot[JDIR] ; j++) {
        for(int i = 0; i < d.np_tot[IDIR] ; i++) {
          r = d.x[IDIR](i);
          th = d.x[JDIR](j);

          // note I didn't explicitly give a "rho0" scaling
          d.Vc(RHO,k,j,i) = 1 / r * 0.5 * (1.0 + tanh((r - jump_radius)/jump_width));

          // add some random noise to the radial velocity component break the
          // axial symmetry and let the instability grow
          d.Vc(VX1,k,j,i) = d.Vc(VX2,k,j,i) * aspect_ratio * 1e-1*(0.5-randm());

          // pure Keplerian rotation
          d.Vc(VX2,k,j,i) = pow(r,-0.5);

          // correction to rotational equilibrium, taking pressure gradient into
          // account. This simplified expression was obtained with sympy;
          d.Vc(VX2,k,j,i) *= sqrt(
                               1.0 \
                               - pow(aspect_ratio, 2) / jump_width *
                                  (
                                    r * tanh((r - jump_radius) / jump_width) \
                                    - r \
                                    + 2 * jump_width
                                  )
                             );
        }
      }
    }

    // Send it all, if needed
    d.SyncToDevice();
}

// Analyse data to produce an output
void MakeAnalysis(DataBlock & data) {

}




// Do a specifically designed user step in the middle of the integration
void ComputeUserStep(DataBlock &data, real t, real dt) {

}
