#include "outputVtk.hpp"
#include "gitversion.h"

#define VTK_RECTILINEAR_GRID    14
#define VTK_STRUCTURED_GRID     35

#ifndef VTK_FORMAT
  #if GEOMETRY == CARTESIAN || GEOMETRY == CYLINDRICAL
    #define VTK_FORMAT  VTK_RECTILINEAR_GRID
  #else
    #define VTK_FORMAT  VTK_STRUCTURED_GRID
  #endif
#endif

/* ---------------------------------------------------------
    The following macros are specific to this file only 
    and are used to ease up serial/parallel implementation
    for writing strings and real arrays 
   --------------------------------------------------------- */   
    
#ifdef PARALLEL
 #define VTK_HEADER_WRITE_STRING(header) \
         TOBEDEFINED(header, strlen(header), MPI_CHAR, SZ_Float_Vect);
 #define VTK_HEADER_WRITE_FLTARR(arr, nelem) \
         TOBEDEFINED (arr, nelem, MPI_FLOAT, SZ_Float_Vect);
 #define VTK_HEADER_WRITE_DBLARR(arr, nelem) \
         TOBEDEFINED (arr, nelem, MPI_DOUBLE, SZ_Float_Vect);
#else
 #define VTK_HEADER_WRITE_STRING(header) \
         fprintf (fvtk,header);
 #define VTK_HEADER_WRITE_FLTARR(arr,nelem) \
         fwrite(arr, sizeof(float), nelem, fvtk);
 #define VTK_HEADER_WRITE_DBLARR(arr,nelem) \
         fwrite(arr, sizeof(double), nelem, fvtk);
#endif


/* Main constructor */
OutputVTK::OutputVTK(Input &input, Grid &gridin, real t)
{
    // Init the output period
    this->tperiod=input.GetReal("Output","vtk",0);
    this->tnext = t;

    // Initialize the output structure
    // Create a local gridhost as an image of gridin
    this->grid = GridHost(gridin);
    grid.SyncFromDevice();


    // Create the coordinate array required in VTK files
    this->nx1 = grid.np_tot[IDIR] - 2 * grid.nghost[IDIR];
    this->nx2 = grid.np_tot[JDIR] - 2 * grid.nghost[JDIR];
    this->nx3 = grid.np_tot[KDIR] - 2 * grid.nghost[KDIR];

    // Vector array where we store the pencil before write
    this->Vwrite = new float[nx1+IOFFSET];

    // Temporary storage on host for 3D arrays
    this->vect3D = IdefixHostArray3D<float>("vect3D",grid.np_tot[KDIR],grid.np_tot[JDIR],grid.np_tot[IDIR]);


    // Essentially does nothing
    this->vtkFileNumber = 0;

    // Test endianness
    int tmp1 = 1;
    this->shouldSwapEndian = 0;
    unsigned char *tmp2 = (unsigned char *) &tmp1;
    if (*tmp2 != 0)
        this->shouldSwapEndian = 1;
    
    
#if VTK_FORMAT == VTK_RECTILINEAR_GRID
    this->xnode = new float[nx1+IOFFSET];
    this->ynode = new float[nx2+JOFFSET];
    this->znode = new float[nx3+KOFFSET];

    for (long int i = 0; i < nx1 + IOFFSET; i++) {
        xnode[i] = BigEndian(grid.xl[IDIR](i + grid.nghost[IDIR]));
    }
    for (long int j = 0; j < nx2 + JOFFSET; j++)    {
        ynode[j] = BigEndian(grid.xl[JDIR](j + grid.nghost[JDIR]));
    }
    for (long int k = 0; k < nx3 + KOFFSET; k++)
    {
        if(DIMENSIONS==2) znode[k] = BigEndian(0.0);
        else znode[k] = BigEndian(grid.xl[KDIR](k + grid.nghost[KDIR]));
    }
#else   // VTK_FORMAT
        /* -- Allocate memory for node_coord which is later used -- */
    node_coord = new float[(nx1+IOFFSET)*3]; 
    
#endif

}

int OutputVTK::Write(DataBlock &datain, real t)
{
    FILE *fileHdl;
    char filename[256];

    // Do we need an output?
    if(t<this->tnext) return(0);

    this->tnext+= this->tperiod;
    Kokkos::Profiling::pushRegion("OutputVTK::Write");

    std::cout << "OutputVTK::Write file n " << vtkFileNumber << "..." << std::flush;

    // Create a copy of the dataBlock on Host, and sync it.
    DataBlockHost data(datain);
    data.SyncFromDevice();

    sprintf (filename, "data.%04d.vtk", vtkFileNumber);
    fileHdl = fopen(filename,"wb");
    WriteHeader(fileHdl);
    for(int nv = 0 ; nv < NVAR ; nv++) {
        for(int k = 0; k < grid.np_tot[KDIR] ; k++ ) {
            for(int j = 0; j < grid.np_tot[JDIR] ; j++ ) {
                for(int i = 0; i < grid.np_tot[IDIR] ; i++ ) {
                    vect3D(k,j,i) = float(data.Vc(nv,k,j,i));
                }
            }
        }
        std::string varname="Vc" + std::to_string(nv);
        WriteScalar(fileHdl, vect3D, varname);
    }

    fclose(fileHdl);

    vtkFileNumber++;
    // Make file number

    std::cout << "done." << std::endl;
    Kokkos::Profiling::popRegion();
    // One day, we will have a return code.
    return(0);
}



/* ********************************************************************* */
void OutputVTK::WriteHeader(FILE *fvtk)
/*!
 * Write VTK header in parallel or serial mode.
 *
 * \param [in]  fvtk  pointer to file
 * \param [in]  grid  pointer to an array of Grid structures
 *
 * \todo  Write the grid using several processors.
 *********************************************************************** */
{
    char header[1024];
    float x1, x2, x3;


    /* -------------------------------------------
   1. File version and identifier
   ------------------------------------------- */

    sprintf(header, "# vtk DataFile Version 2.0\n");

    /* -------------------------------------------
   2. Header
   ------------------------------------------- */

    sprintf(header + strlen(header), "Idefix %s VTK Data\n", GITVERSION);

    /* ------------------------------------------
   3. File format
   ------------------------------------------ */

    sprintf(header + strlen(header), "BINARY\n");

    /* ------------------------------------------
   4. Dataset structure
   ------------------------------------------ */

#if VTK_FORMAT == VTK_RECTILINEAR_GRID
    sprintf(header + strlen(header), "DATASET %s\n", "RECTILINEAR_GRID");
#elif VTK_FORMAT == VTK_STRUCTURED_GRID
    sprintf(header + strlen(header), "DATASET %s\n", "STRUCTURED_GRID");
#endif

    
    VTK_HEADER_WRITE_STRING(header);

    /* -- Generate time info (VisIt reader only) -- */

    /*
#if VTK_TIME_INFO == YES
  sprintf (header,"FIELD FieldData 1\n");
  sprintf (header+strlen(header),"TIME 1 1 double\n");
  double tt=g_time;
  if (IsLittleEndian()) SWAP_VAR(tt);
  VTK_HEADER_WRITE_STRING(header);
  VTK_HEADER_WRITE_DBLARR(&tt, 1);
  VTK_HEADER_WRITE_STRING("\n");
#endif /* VTK_TIME_INFO */

    sprintf(header, "DIMENSIONS %d %d %d\n",
            nx1 + IOFFSET, nx2 + JOFFSET, nx3 + KOFFSET);
    VTK_HEADER_WRITE_STRING(header);

#if VTK_FORMAT == VTK_RECTILINEAR_GRID

    /* -- Write rectilinear grid information -- */

    sprintf(header, "X_COORDINATES %d float\n", nx1 + IOFFSET);
    VTK_HEADER_WRITE_STRING(header);
    VTK_HEADER_WRITE_FLTARR(xnode, nx1 + IOFFSET);

    sprintf(header, "\nY_COORDINATES %d float\n", nx2 + JOFFSET);
    VTK_HEADER_WRITE_STRING(header);
    VTK_HEADER_WRITE_FLTARR(ynode, nx2 + JOFFSET);

    sprintf(header, "\nZ_COORDINATES %d float\n", nx3 + KOFFSET);
    VTK_HEADER_WRITE_STRING(header);
    VTK_HEADER_WRITE_FLTARR(znode, nx3 + KOFFSET);

#elif VTK_FORMAT == VTK_STRUCTURED_GRID

    /* -- define node_coord -- */

    sprintf(header, "POINTS %d float\n", (nx1 + IOFFSET) * (nx2 + JOFFSET) * (nx3 + KOFFSET));
    VTK_HEADER_WRITE_STRING(header);

    /* -- Write structured grid information -- */

    x1 = x2 = x3 = 0.0;
    for (long int k = 0; k < nx3 + KOFFSET; k++)
    {
        for (long int j = 0; j < nx2 + JOFFSET; j++)
        {
            for (long int i = 0; i < nx1 + IOFFSET; i++)
            {
                D_EXPAND(x1 = grid.xl[IDIR](i + grid.nghost[IDIR]);,
                         x2 = grid.xl[JDIR](j + grid.nghost[JDIR]);,
                         x3 = grid.xl[KDIR](k + grid.nghost[KDIR]);)

#if (GEOMETRY == CARTESIAN) || (GEOMETRY == CYLINDRICAL)
                node_coord[3*i+IDIR] = BigEndian(x1);
                node_coord[3*i+JDIR] = BigEndian(x2);
                node_coord[3*i+KDIR] = BigEndian(x3);
#elif GEOMETRY == POLAR
                node_coord[3*i+IDIR] = BigEndian(x1 * cos(x2));
                node_coord[3*i+JDIR] = BigEndian(x1 * sin(x2));
                node_coord[3*i+KDIR] = BigEndian(x3);
#elif GEOMETRY == SPHERICAL
#if DIMENSIONS == 2
                node_coord[3*i+IDIR] = BigEndian(x1 * sin(x2));
                node_coord[3*i+JDIR] = BigEndian(x1 * cos(x2));
                node_coord[3*i+KDIR] = BigEndian(0.0);
#elif DIMENSIONS == 3
                node_coord[3*i+IDIR] = BigEndian(x1 * sin(x2) * cos(x3));
                node_coord[3*i+JDIR] = BigEndian(x1 * sin(x2) * sin(x3));
                node_coord[3*i+KDIR] = BigEndian(x1 * cos(x2));
#endif
#endif
            }
            VTK_HEADER_WRITE_FLTARR(node_coord, 3 * (nx1 + IOFFSET));
        }
    }

#endif

    /* -----------------------------------------------------
   5. Dataset attributes [will continue by later calls
      to WriteVTK_Vector() or WriteVTK_Scalar()...]
   ----------------------------------------------------- */

    sprintf(header, "\nCELL_DATA %d\n", nx1 * nx2 * nx3);
    VTK_HEADER_WRITE_STRING(header);
}
#undef VTK_STRUCTERED_GRID
#undef VTK_RECTILINEAR_GRID



/* ********************************************************************* */
void OutputVTK::WriteScalar(FILE *fvtk, IdefixHostArray3D<float> &Vin,  std::string &var_name)
/*!
 * Write VTK scalar field.
 *
 * \param [in]   fvtk       pointer to file (handle)
 * \param [in]   V          pointer to 3D data array
 * \param [in] unit     the corresponding cgs unit (if specified, 1 otherwise)
 * \param [in]   var_name   the variable name appearing in the VTK file
 * \param [in]   grid       pointer to an array of Grid structures
 *********************************************************************** */
{
    int i, j, k;
    char header[512];


    sprintf(header, "\nSCALARS %s float\n", var_name.c_str());
    sprintf(header + strlen(header), "LOOKUP_TABLE default\n");


    fprintf(fvtk, "%s", header);


    for(long int k = 0 ; k < nx3 ; k++ ) {
        for(long int j = 0 ; j < nx2 ; j++ ) {
            for(long int i = 0 ; i < nx1 ; i++ ) {
                Vwrite[i] = BigEndian(Vin(k + grid.nghost[KDIR],j + grid.nghost[JDIR],i + grid.nghost[IDIR]));
            }
            fwrite(Vwrite, sizeof(float), nx1, fvtk);
        }
    }
}

/* ****************************************************************************/
/** Determines if the machine is little-endian.  If so, 
    it will force the data to be big-endian. 
	@param in_number floating point number to be converted in big endian */
/* *************************************************************************** */

float OutputVTK::BigEndian(float in_number)
{
    if (shouldSwapEndian)
    {
		unsigned char *bytes = (unsigned char*) &in_number;
        unsigned char tmp = bytes[0];
        bytes[0] = bytes[3];
        bytes[3] = tmp;
        tmp = bytes[1];
        bytes[1] = bytes[2];
        bytes[2] = tmp;
    }
	return(in_number);
}