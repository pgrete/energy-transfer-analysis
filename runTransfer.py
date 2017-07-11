import numpy as np
from mpi4py import MPI
from EnergyTransfer import EnergyTransfer
from IOhelperFuncs import readAllFieldsWithYT
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


comm  = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rhoField = None
velFields = None
magFields = None
accFields = None
needAccFields = False

SplitTerms = Terms.split("-")
thisTerms = []
if "All" in SplitTerms:
    if FluidType == "mhd":
	    thisTerms += ["UU", "BUT" ,"BUP" ,"UBT" ,"UBPb" ,"BB", "BUPbb", "UBPbb"]
    if FluidType == "hydro":
    	thisTerms += ["UU"]
if "Pres" in SplitTerms:
    thisTerms += ["PU"]
if "Forc" in SplitTerms:
    thisTerms += ["FU"]
    needAccFields = True
    
if len(thisTerms) == 0:
	print("Unknown Terms... FAIL")
	sys.exit(1)
        
if SimType == "Enzo":
    rhoField = "Density"
    velFields = ["x-velocity","y-velocity","z-velocity"]
    magFields = ["Bx","By","Bz"]
    accFields = ['x-acceleration','y-acceleration','z-acceleration']
    loadPath = "DD" + ID + "/data" + ID
elif SimType == "Athena":
    rhoField = "density"
    velFields = ["velocity_x","velocity_y","velocity_z"]
    magFields = ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    loadPath = "id0/Turb." + ID + ".vtk"
else:
    print("Unknown SimType - use 'Enzo' or 'Athena'... FAIL")
    sys.exit(1)

if FluidType == "hydro":
	magFields = None
elif FluidType != "mhd":
	print("Unknown FluidType - use 'mhd' or 'hydro'... FAIL")
	sys.exit(1)
    
if needAccFields and accFields is None:
	print("Need acceleration fields but got none...")
	sys.exit(1)
if not needAccFields:
    accFields = None

rho, U , B, Acc, P = readAllFieldsWithYT(loadPath,Res,
    rhoField,velFields,magFields,accFields)


comm.Barrier()

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

if "PU" not in thisTerms:
    P = None

ET = EnergyTransfer(MPI,Res,rho,U,B,Acc,P)


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

if comm.Get_rank() == 0:
    DumpFile = str(ID).zfill(4) + "-" + Terms + "-" + BinType + "-" + str(Res) + ".pkl"

    if os.path.isfile(DumpFile):
        print("Reading previous transfer file")
        Result = pickle.load(open(DumpFile,"rb"))

    else:
        Result = {}
else:
    Result = None


ET.getTransferWWAnyToAny(Result,KBins,QBins, Terms = thisTerms)
if comm.Get_rank() == 0:
    pickle.dump(Result,open(DumpFile,"wb"))    

