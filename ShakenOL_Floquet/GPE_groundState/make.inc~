####################################################################
#  OPENMMF make include file.                                      #
#  OPENMMF, Version 0.5                                            #
#  September 2020                                                  #
####################################################################

SHELL = /bin/sh

#  CC is the C compiler, normally invoked with options CFLAGS.
#
CC     = gcc
CPP    = g++
CPPLAGS = -O3 -lgfortran -lstdc++

#  Modify the GF and GFFLAGS definitions to the desired compiler
#  and desired compiler options for your machine. 
GF         = gfortran
GFFLAGS    = -O3 -llapack -lblas -g -lstdc++

#  Define LDFLAGS to the desired linker options for your machine.
#
LDFLAGS = -g -fPIC 

#  The archiver and the flag(s) to use when building an archive
#  (library).  If your system has no ranlib, set RANLIB = echo.
#
AR         = ar
ARFLAGS    = urv
RANLIB     = ranlib
SHAREFLAGS = -shared -fPIC 
DYLIB_NAME = libopenmmf.so

#  Uncomment the next four command lines to include routines using the 
#  MKL-intel library. Edited as needed for your system.
###
BUILD_MKL = yes
#=========== Set the MKL-intel LIBRARY PATH ======================
MKLLIBS = /mnt/sda5/intel/compilers_and_libraries/linux/mkl/lib/intel64

#=========== Set the MKL-intel INCLUDE PATH ======================
MKLINC = /mnt/sda5/intel/compilers_and_libraries/linux/mkl/include	

#=========== Set the MKL-intel libary flags ======================
MKLFLAGS   = -lstdc++ -lmkl_gf_lp64 -lmkl_sequential -lmkl_core

