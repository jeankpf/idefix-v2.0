import argparse
import os
import sys
import subprocess
import re
sys.path.append(os.getenv("IDEFIX_DIR"))
from pytools.dump_io import readDump
import numpy as np
import shutil
import matplotlib.pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class idfxTest:
  def __init__ (self):
    parser = argparse.ArgumentParser()

    idefix_dir_env = os.getenv("IDEFIX_DIR")

    parser.add_argument("-noplot",
                        default=False,
                        help="disable plotting in standard tests",
                        action="store_true")

    parser.add_argument("-ploterr",
                        default=False,
                        help="Enable plotting on error in regression tests",
                        action="store_true")

    parser.add_argument("-cmake",
                        default="",
                        help="CMake options",
                        nargs='+')

    parser.add_argument("-definitions",
                        default="",
                        help="definitions.hpp file")

    parser.add_argument("-dec",
                        default="",
                        help="MPI domain decomposition",
                        nargs='+')

    parser.add_argument("-check",
                        default=False,
                        help="Only perform regression tests without compilation",
                        action="store_true")

    parser.add_argument("-cuda",
                        default=False,
                        help="Test on Nvidia GPU using CUDA",
                        action="store_true")

    parser.add_argument("-hip",
                        default=False,
                        help="Test on AMD GPU using HIP",
                        action="store_true")

    parser.add_argument("-single",
                        default=False,
                        help="Enable single precision",
                        action="store_true")

    parser.add_argument("-vectPot",
                        default=False,
                        help="Enable vector potential formulation",
                        action="store_true")

    parser.add_argument("-reconstruction",
                        type=int,
                        default=2,
                        help="set reconstruction scheme (2=PLM, 3=LimO3, 4=PPM)")

    parser.add_argument("-idefixDir",
                        default=idefix_dir_env,
                        help="Set directory for idefix source files (default $IDEFIX_DIR)")

    parser.add_argument("-mpi",
                        default=False,
                        help="Enable MPI",
                        action="store_true")

    parser.add_argument("-all",
                    default=False,
                    help="Do all test suite (otherwise, just do the test with the current configuration)",
                    action="store_true")

    parser.add_argument("-init",
                    default=False,
                    help="Reinit reference files for non-regression tests (dangerous!)",
                    action="store_true")

    parser.add_argument("-Werror",
                    default=False,
                    help="Consider warnings as errors",
                    action="store_true")


    args, unknown=parser.parse_known_args()
    self.noplot = args.noplot
    self.ploterr = args.ploterr
    self.cmakeOpts = args.cmake
    self.single = args.single
    self.vectPot = args.vectPot
    self.reconstruction = args.reconstruction
    self.definitions = args.definitions
    self.idefixDir = args.idefixDir
    self.mpi = args.mpi
    self.dec = args.dec
    self.init = args.init
    self.cuda = args.cuda
    self.check = args.check
    self.hip = args.hip
    self.all = args.all
    self.Werror = args.Werror
    self.referenceDirectory = "reference"

  def configure(self,definitionFile=""):
    comm=["cmake"]
    # add source directory
    comm.append(self.idefixDir)
    # add specific options
    if(self.cmakeOpts):
      for opt in self.cmakeOpts:
        comm.append("-D"+opt)

    if self.cuda:
      comm.append("-DKokkos_ENABLE_CUDA=ON")
      # disable fmad operations on Cuda to make it compatible with CPU arithmetics
      comm.append("-DIdefix_CXX_FLAGS=--fmad=false")

    if self.hip:
      comm.append("-DKokkos_ENABLE_HIP=ON")
      # disable fmad operations on HIP to make it compatible with CPU arithmetics
      comm.append("-DIdefix_CXX_FLAGS=-ffp-contract=off")

    #if we use single precision
    if(self.single):
      comm.append("-DIdefix_PRECISION=Single")
    else:
      comm.append("-DIdefix_PRECISION=Double")


    if(self.vectPot):
      comm.append("-DIdefix_EVOLVE_VECTOR_POTENTIAL=ON")
    else:
      comm.append("-DIdefix_EVOLVE_VECTOR_POTENTIAL=OFF")

    if(self.Werror):
      comm.append("-DIdefix_WERROR=ON")

    # add a definition file if provided
    if(definitionFile):
      self.definitions=definitionFile
    else:
      self.definitions="definitions.hpp"

    comm.append("-DIdefix_DEFS="+self.definitions)

    if(self.mpi):
      comm.append("-DIdefix_MPI=ON")
    else:
      comm.append("-DIdefix_MPI=OFF")

    if(self.reconstruction == 2):
      comm.append("-DIdefix_RECONSTRUCTION=Linear")
    elif(self.reconstruction==3):
      comm.append("-DIdefix_RECONSTRUCTION=LimO3")
    elif(self.reconstruction==4):
      comm.append("-DIdefix_RECONSTRUCTION=Parabolic")


    try:
        cmake=subprocess.run(comm)
        cmake.check_returncode()
    except subprocess.CalledProcessError as e:
        print(bcolors.FAIL+"***************************************************")
        print("Cmake failed")
        print("***************************************************"+bcolors.ENDC)
        raise e

  def compile(self,jobs=8):
    try:
        make=subprocess.run(["make","-j"+str(jobs)])
        make.check_returncode()
    except subprocess.CalledProcessError as e:
        print(bcolors.FAIL+"***************************************************")
        print("Compilation failed")
        print("***************************************************"+bcolors.ENDC)
        raise e

  def run(self, inputFile="", np=2, nowrite=False, restart=-1):
      comm=["./idefix"]
      if inputFile:
          comm.append("-i")
          comm.append(inputFile)
      if self.mpi:
          if self.dec:
            np=1
            for n in range(len(self.dec)):
              np=np*int(self.dec[n])

          comm.insert(0,"mpirun")
          comm.insert(1,"-np")
          comm.insert(2,str(np))
          if self.dec:
              comm.append("-dec")
              for n in range(len(self.dec)):
                comm.append(str(self.dec[n]))

      if nowrite:
          comm.append("-nowrite")

      if restart>=0:
        comm.append("-restart")
        comm.append(str(restart))

      try:
          make=subprocess.run(comm)
          make.check_returncode()
      except subprocess.CalledProcessError as e:
          print(bcolors.FAIL+"***************************************************")
          print("Execution failed")
          print("***************************************************"+bcolors.ENDC)
          raise e

      self.__readLog()

  def __readLog(self):
    if not os.path.exists('./idefix.0.log'):
      # When no idefix file is produced, we leave
      return

    with open('./idefix.0.log','r') as file:
      log = file.read()

    if "SINGLE PRECISION" in log:
      self.single = True
    else:
      self.single = False

    if "Kokkos CUDA target ENABLED" in log:
      self.cuda = True
    else:
      self.cuda = False

    if "Kokkos HIP target ENABLED" in log:
      self.hip = True
    else:
      self.hip = False


    self.reconstruction = 2
    if "3rd order (LimO3)" in log:
      self.reconstruction = 3

    if "4th order (PPM)" in log:
      self.reconstruction = 4

    self.mpi=False
    if "MPI ENABLED" in log:
      self.mpi=True


    # Get input file from log
    line = re.search('(?<=Input Parameters using input file )(.*)', log)
    self.inifile=line.group(0)[:-1]

    # Get performances from log
    line = re.search('Main: Perfs are (.*) cell', log)
    self.perf=float(line.group(1))

  def checkOnly(self, filename, tolerance=0):
    # Assumes the code has been run manually using some configuration, so we simply
    # do the test suite witout configure/compile/run
    self.__readLog()
    self.__showConfig()
    if self.cuda or self.hip:
      print(bcolors.WARNING+"***********************************************")
      print("WARNING: Idefix guarantees floating point arithmetic accuracy")
      print("ONLY when fmad instruction are explicitely disabled at compilation.")
      print("Otheriwse, this test will likely fail.")
      print("***********************************************"+bcolors.ENDC)
    self.standardTest()
    self.nonRegressionTest(filename, tolerance)

  def standardTest(self):
    if os.path.exists('python/testidefix.py'):
      os.chdir("python")
      comm = ["python3"]
      comm.append("testidefix.py")
      if self.noplot:
        comm.append("-noplot")

      print(bcolors.OKCYAN+"Running standard test...")
      try:
          make=subprocess.run(comm)
          make.check_returncode()
      except subprocess.CalledProcessError as e:
          print(bcolors.FAIL+"***************************************************")
          print("Standard test execution failed")
          print("***************************************************"+bcolors.ENDC)
          raise e
      print(bcolors.OKCYAN+"Standard test succeeded"+bcolors.ENDC)
      os.chdir("..")
    else:
      print(bcolors.WARNING+"No standard testidefix.py for this test"+bcolors.ENDC)
    sys.stdout.flush()

  def nonRegressionTest(self, filename,tolerance=0):
    fileref=self.referenceDirectory+"/"+self.__getReferenceFilename()
    if not(os.path.exists(fileref)):
      raise Exception("Reference file "+fileref+ " doesn't exist")

    filetest=filename
    if not(os.path.exists(filetest)):
      raise Exception("Test file "+fileref+ " doesn't exist")

    Vref=readDump(fileref)
    Vtest=readDump(filetest)
    error=self.__computeError(Vref,Vtest)
    if error > tolerance:
      print(bcolors.FAIL+"None-Regression test failed!")
      self.__showConfig()
      print(bcolors.ENDC)
      if self.ploterr:
        self.__plotDiff(Vref,Vtest)
      assert error <= tolerance, bcolors.FAIL+"Error (%e) above tolerance (%e)"%(error,tolerance)+bcolors.ENDC
    print(bcolors.OKGREEN+"Non-regression test succeeded with error=%e"%error+bcolors.ENDC)
    sys.stdout.flush()

  def makeReference(self,filename):
    self.__readLog()
    if not os.path.exists(self.referenceDirectory):
      print("Creating reference directory")
      os.mkdir(self.referenceDirectory)
    fileout = self.referenceDirectory+'/'+ self.__getReferenceFilename()
    if(os.path.exists(fileout)):
      ans=input(bcolors.WARNING+"This will overwrite already existing reference file:\n"+fileout+"\nDo you confirm? (type yes to continue): "+bcolors.ENDC)
      if(ans != "yes"):
        print(bcolors.WARNING+"Reference creation aborpted"+bcolors.ENDC)
        return

    shutil.copy(filename,fileout)
    print(bcolors.OKGREEN+"Reference file "+fileout+" created"+bcolors.ENDC)
    sys.stdout.flush()

  def __showConfig(self):
    print("**************************************************************")
    if self.cuda:
      print("Nvidia Cuda enabled.")
    if self.hip:
      print("AMD HIP enabled.")
    print("CMake Opts: " +" ".join(self.cmakeOpts))
    print("Definitions file:"+self.definitions)
    print("Input File: "+self.inifile)
    if(self.single):
      print("Precision: Single")
    else:
      print("Precision: Double")
    if(self.reconstruction==2):
      print("Reconstruction: PLM")
    elif(self.reconstruction==3):
      print("Reconstruction: LimO3")
    elif(self.reconstruction==4):
      print("Reconstruction: PPM")
    if(self.vectPot):
      print("Vector Potential: ON")
    else:
      print("Vector Potential: OFF")
    if self.mpi:
      print("MPI: ON")
    else:
      print("MPI: OFF")

    print("**************************************************************")

  def __getReferenceFilename(self):
    strReconstruction="plm"
    if self.reconstruction == 3:
      strReconstruction = "limo3"
    if self.reconstruction == 4:
      strReconstruction= "ppm"

    strPrecision="double"
    if self.single:
      strPrecision="single"

    fileref='dump.ref.'+strPrecision+"."+strReconstruction+"."+self.inifile
    if self.vectPot:
      fileref=fileref+".vectPot"

    fileref=fileref+'.dmp'
    return(fileref)

  def __computeError(self,Vref,Vtest):
    ntested=0
    error=0
    for fld in Vtest.data.keys():
      if(Vtest.data[fld].ndim==3):
        if fld in Vref.data.keys():
          error = error+np.sqrt(np.mean((Vref.data[fld]-Vtest.data[fld])**2))
          ntested=ntested+1

    if ntested==0:
      raise Exception(bcolors.FAIL+"There is no common field between the reference and current file"+bcolors.ENDC)

    error=error/ntested
    return(error)

  def __plotDiff(self,Vref,Vtest):

    for fld in Vtest.data.keys():
      if(Vtest.data[fld].ndim==3):
        if fld in Vref.data.keys():
          plt.figure()
          plt.title(fld)
          plt.pcolor(Vref.x1, Vref.x2, Vref.data[fld][:,:,0]-Vtest.data[fld][:,:,0],cmap='seismic')
          plt.xlabel("x1")
          plt.ylabel("x2")
          plt.colorbar()
    plt.show()