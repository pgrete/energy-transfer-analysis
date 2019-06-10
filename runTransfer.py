from mpi4py import MPI


comm.Barrier()
TimeStart = MPI.Wtime()

import numpy as np
from EnergyTransfer import EnergyTransfer
from IOhelperFuncs import readAllFieldsWithYT, readAllFieldsWithHDF 
import time
import pickle
import sys
import os



rhoField = None
pressField = None
velFields = None
magFields = None
accFields = None
needAccFields = False

SplitTerms = Terms.split("-")
thisTerms = []
if "All" in SplitTerms:
    if "mhd" in FluidType:
	    thisTerms += ["UU", "BUT" ,"BUP" ,"UBT" ,"UBPb" ,"BB", "BUPbb", "UBPbb"]
    if "hydro" in FluidType:
    	thisTerms += ["UU"]
if "Int" in SplitTerms:
    thisTerms += ["SS","US","SU"]
if "Pres" in SplitTerms:
    thisTerms += ["PU"]
if "Forc" in SplitTerms:
    thisTerms += ["FU"]
    needAccFields = True
    
if len(thisTerms) == 0:
	print("Unknown Terms... FAIL")
	sys.exit(1)

order = "unset"

if SimType == "Enzo":
    gamma = 1.001 # quasi isothermal
    rhoField = "Density"
    pressField = None
    velFields = ["x-velocity","y-velocity","z-velocity"]
    magFields = ["Bx","By","Bz"]
    accFields = ['x-acceleration','y-acceleration','z-acceleration']
    loadPath = "DD" + ID + "/data" + ID
elif SimType == "EnzoHDF":
    rhoField = "Density"
    velFields = ["x-velocity","y-velocity","z-velocity"]
    magFields = ["Bx","By","Bz"]
    accFields = ['x-acceleration','y-acceleration','z-acceleration']
    loadPath = "DD" + ID 
    order = "F"
elif SimType == "Athena":
    rhoField = "density"
    pressField = None
    velFields = ["velocity_x","velocity_y","velocity_z"]
    magFields = ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    accFields = ['acceleration_x','acceleration_y','acceleration_z']
    loadPath = "id0/Turb." + ID + ".vtk"
elif SimType == "AthenaHDF":
    rhoField = "density"
    velFields = ["velocity_x","velocity_y","velocity_z"]
    magFields = ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    accFields = ['acceleration_x','acceleration_y','acceleration_z']
    if 'adiabatic' in FluidType:
        pressField = 'pressure'
    loadPath = ID
    order = "F"
elif SimType == "AthenaHDFC":
    rhoField = "density"
    velFields = ["velocity_x","velocity_y","velocity_z"]
    magFields = ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    accFields = ['acceleration_x','acceleration_y','acceleration_z']
    loadPath = ID
    if 'adiabatic' in FluidType:
        pressField = 'pressure'
    order = "C"
elif SimType == "Nyx":
    sys.path.append("/home/h/hzfbhsws/Notebooks")
    import parameters as nyx
    import fields
    gamma = nyx.gamma
    rhoField = "density"
    pressField = "pressure"
    velFields = [ "xvel","yvel","zvel" ]
    magFields = None
    accFields = [ "xaccel", "yaccel", "zaccel" ]
    loadPath = "plt" + ID
else:
    print("Unknown SimType - use 'Enzo' or 'Athena' or 'Nyx'... FAIL")
    sys.exit(1)

if FluidType == "hydro":
	magFields = None
elif 'mhd' not in FluidType:
	print("Unknown FluidType - use 'mhd' or 'hydro'... FAIL")
	sys.exit(1)
    
if needAccFields and accFields is None:
	print("Need acceleration fields but got none...")
	sys.exit(1)
if not needAccFields:
    accFields = None

if 'adiabatic' in FluidType and pressField is None:
    print('Adiabatic EOS not tested/implemented yet for this FluidType.')
    sys.exit(1)



TimeDoneStart = MPI.Wtime() - TimeStart
TimeDoneStart = comm.gather(TimeDoneStart)

if rank == 0:
    print("Imports and startup done in %.3g +/- %.3g" % (np.mean(TimeDoneStart),np.std(TimeDoneStart)))
    sys.stdout.flush()

TimeDoneStart = MPI.Wtime() 
if "HDF" in SimType:
    rho, U , B, Acc, P = readAllFieldsWithHDF(loadPath,Res,
        rhoField,velFields,magFields,accFields,pressField,order,useMMAP=False)
elif "Nyx" in SimType:
    rho, U , B, Acc, P = readAllFieldsWithYT(loadPath,Res,
        rhoField,velFields,magFields,accFields,pressField)
else:
    rho, U , B, Acc, P = readAllFieldsWithYT(loadPath,Res,
        rhoField,velFields,magFields,accFields)

TimeDoneReading = MPI.Wtime() - TimeDoneStart
TimeDoneReading = comm.gather(TimeDoneReading)

if rank == 0:
    print("Reading done in %.3g +/- %.3g" % (np.mean(TimeDoneReading),np.std(TimeDoneReading)))
    sys.stdout.flush()


if "PU" not in thisTerms and "SS" not in thisTerms:
    P = None



