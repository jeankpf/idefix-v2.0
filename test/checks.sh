#!/bin/bash

rep_HD_list="sod-iso sod MachReflection ViscousFlowPastCylinder ViscousDisk FargoPlanet"
rep_MHD_list="sod-iso sod AxisFluxTube AmbipolarCshock HallWhistler FargoMHDSpherical OrszagTang OrszagTang3D"

# refer to the parent dir of this file, wherever this is called from
# a python equivalent is e.g.
#
# import pathlib
# TEST_DIR = pathlib.Path(__file__).parent
TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


function resolve_path {
    # resolve relative paths
    # work around the fact that `realpath` is not bundled with every UNIX distro
    echo "`cd "$1";pwd`"
}

target_dir=$(resolve_path $TEST_DIR/..)

if [ -z ${var+IDEFIX_DIR} ] & [ -d "$IDEFIX_DIR" ] ; then
    global_dir=$(resolve_path $IDEFIX_DIR)
    if [ $target_dir != $global_dir ] ; then
        echo \
        "Warning: IDEFIX_DIR is set globally to $global_dir" \
        ", but is redefined to $target_dir"
    fi
fi

export IDEFIX_DIR=$target_dir
echo $IDEFIX_DIR

set -e
options=$@

# HD tests
for rep in $rep_HD_list; do
    cd $TEST_DIR/HD/$rep
    echo "***********************************************"
    echo "Configuring  $rep"
    echo "***********************************************"
    def_files=$(ls definitions*.hpp)
    for def in $def_files; do
        python3 $IDEFIX_DIR/configure.py $options -defs=$def || { echo "!!!! HD $rep failed during configuration"; exit 1; }
        echo "***********************************************"
        echo "Making  $rep with $def"
        echo "***********************************************"
        make clean; make -j 4 || { echo "!!!! HD $rep failed during compilation with $def"; exit 1; }

        ini_files=$(ls *.ini)
        for ini in $ini_files; do
            echo "***********************************************"
            echo "Running  $rep with $ini"
            echo "***********************************************"
            ./idefix -i $ini || { echo "!!!! HD $rep failed running with $def and $ini"; exit 1; }

            cd python
            echo "***********************************************"
            echo "Testing  $rep with $ini and $def"
            echo "***********************************************"
            python3 testidefix.py -noplot || { echo "!!!! HD $rep failed validation with $def and $ini"; exit 1; }
            cd ..
        done
        make clean
        rm -f *.vtk *.dbl
    done
done

# MHD tests
for rep in $rep_MHD_list; do
    cd $TEST_DIR/MHD/$rep
    echo "***********************************************"
    echo "Configuring  $rep"
    echo "***********************************************"
    def_files=$(ls definitions*.hpp)
    for def in $def_files; do
        python3 $IDEFIX_DIR/configure.py -mhd $options -defs=$def || { echo "!!!! MHD $rep failed during configuration"; exit 1; }
        echo "***********************************************"
        echo "Making  $rep with $def"
        echo "***********************************************"
        make clean; make -j 4 || { echo "!!!! MHD $rep failed during compilation with $def"; exit 1; }

        ini_files=$(ls *.ini)
        for ini in $ini_files; do
            echo "***********************************************"
            echo "Running  $rep with $ini"
            echo "***********************************************"
            ./idefix -i $ini || { echo "!!!! MHD $rep failed running with $def and $ini"; exit 1; }

            cd python
            echo "***********************************************"
            echo "Testing  $rep with $ini and $def"
            echo "***********************************************"
            python3 testidefix.py -noplot || { echo "!!!! MHD $rep failed validation with $def and $ini"; exit 1; }
            cd ..
        done
    done
    cd $TEST_DIR
done
