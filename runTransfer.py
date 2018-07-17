from mpi4py import MPI

comm  = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
TimeStart = MPI.Wtime()

import numpy as np
from EnergyTransfer import EnergyTransfer
from IOhelperFuncs import readAllFieldsWithYT, readAllFieldsWithHDF 
import time
import pickle
import sys
import os

ID = sys.argv[1]
Terms = sys.argv[2]
Res = int(sys.argv[3])
SimType = sys.argv[4] # Enzo or Athena
FluidType = sys.argv[5] # hydro or mhd
BinType = sys.argv[6] # lin or log



gamma = 1.0 # isothermal

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
    thisTerms += ["SS"]
if "Pres" in SplitTerms:
    thisTerms += ["PU"]
if "Forc" in SplitTerms:
    thisTerms += ["FU"]
    needAccFields = True
    
if len(thisTerms) == 0:
	print("Unknown Terms... FAIL")
	sys.exit(1)

order = "unset"
pField = None

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
        pField = 'pressure'
    loadPath = ID
    order = "F"
elif SimType == "AthenaHDFC":
    rhoField = "density"
    velFields = ["velocity_x","velocity_y","velocity_z"]
    magFields = ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    accFields = ['acceleration_x','acceleration_y','acceleration_z']
    loadPath = ID
    if 'adiabatic' in FluidType:
        pField = 'pressure'
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

if 'adiabatic' in FluidType and pField is None:
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
<<<<<<< HEAD
        rhoField,velFields,magFields,accFields,pField,order,useMMAP=False)
=======
        rhoField,velFields,magFields,accFields,order)
elif "Nyx" in SimType:
    rho, U , B, Acc, P = readAllFieldsWithYT(loadPath,Res,
        rhoField,velFields,magFields,accFields,pressField)
>>>>>>> origin/master
else:
    rho, U , B, Acc, P = readAllFieldsWithYT(loadPath,Res,
        rhoField,velFields,magFields,accFields)

TimeDoneReading = MPI.Wtime() - TimeDoneStart
TimeDoneReading = comm.gather(TimeDoneReading)

if rank == 0:
    print("Reading done in %.3g +/- %.3g" % (np.mean(TimeDoneReading),np.std(TimeDoneReading)))
    sys.stdout.flush()

if BinType == "Lin":
	Bins = np.concatenate((np.linspace(0.5,Res/2-0.5,Res/2,endpoint=True),[float(Res)/2.*np.sqrt(3)]))
elif BinType == "Log":
        ResExp = np.log(Res/8)/np.log(2) * 4 + 1
        Bins = np.concatenate((np.array([0.]),
                        4.* 2** ((np.arange(0,ResExp + 1) - 1.) /4.)))

elif BinType == "testing":
    Bins = [0.5,1.5,2.5,16.0,26.5,28.5,32.0]

else:
	print("Unknown BinType - use 'Lin' or 'Log'... FAIL")
	sys.exit(1)

KBins = Bins
QBins = Bins

if "PU" not in thisTerms and "SS" not in thisTerms:
    P = None

ET = EnergyTransfer(MPI,Res,rho,U,B,Acc,P,gamma)


""" Result dictionary of shape
Result[formalism][term][method][target wavenumber][source wavenumber]
| "WW"
|--| "UUA"
|--|---| "AnyToAny"
|--|---|-----| K
|--|---|-----|-| Q = value
|--|---| "AllToOne"
|--|---|-----| K = value
|--| "UUC
|--| etc.
"""

if rank == 0:
    DumpFile = str(ID).zfill(4) + "-" + Terms + "-" + BinType + "-" + str(Res) + ".pkl"

    if os.path.isfile(DumpFile):
        print("Reading previous transfer file")
        if sys.version_info[0] < 3:
            Result = pickle.load(open(DumpFile,"rb"))
        else:
            Result = pickle.load(open(DumpFile,"rb"),encoding='latin1')

    else:
        Result = {}
else:
    Result = None

ET.getTransferWWAnyToAny(Result,KBins,QBins, Terms = thisTerms)
if comm.Get_rank() == 0:
    pickle.dump(Result,open(DumpFile,"wb"))    

