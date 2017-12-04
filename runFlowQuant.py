from mpi4py import MPI

comm  = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
TimeStart = MPI.Wtime()

import numpy as np
from mpiFFT4py.slab import R2C
import time
import pickle
import sys
import h5py
import os
from scipy.stats import binned_statistic
from math import ceil
from configobj import ConfigObj
from IOhelperFuncs import readAllFieldsWithYT, readAllFieldsWithHDF 

#Res = int(sys.argv[3])
ID = sys.argv[1]
Res = int(sys.argv[2])
SimType = sys.argv[3] # Enzo or Athena
FluidType = sys.argv[4] # hydro or mhd



def normedSpec(k,quantity):
    kMaxInt = ceil(Res*0.5*np.sqrt(3.))
    Bins = np.linspace(0.,kMaxInt,kMaxInt+1)
    histSum = binned_statistic(k,quantity,bins=Bins,statistic='sum')[0] 
    kSum = binned_statistic(k,k,bins=Bins,statistic='sum')[0]
    histCount = np.histogram(k,bins=Bins)[0]
    
    totalHistSum = comm.reduce(histSum.astype(np.float64))
    totalKSum = comm.reduce(kSum.astype(np.float64))
    totalHistCount = comm.reduce(histCount.astype(np.float64))

    if rank == 0:

    
        # get rid of bins with 0 cells -> no division by 0 in normalization
        mask = totalHistCount != 0

        # calculate corresponding k to to bin
        # this is more accurate (wrt energy conservation) than using the bin center
        centeredK = totalKSum[mask] / totalHistCount[mask] 

        #  "integrate" over k-shells
        vals = 4. * np.pi * centeredK**2. * (totalHistSum[mask] / totalHistCount[mask])
    
        # normalize by cell count/resolution to work with mpiFFT4py 
        vals /= float(Res**6.)
        return [Bins,vals,centeredK]
    else:
        return [None,None,None]



        
order='unset'

if SimType == "AthenaHDF":
    rhoField = "density"
    velFields = ["velocity_x","velocity_y","velocity_z"]
    magFields = ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    #accFields = ['acceleration_x','acceleration_y','acceleration_z']
    accFields = None
    loadPath = ID
    order = "F"
elif SimType == "AthenaHDFC":
    rhoField = "density"
    velFields = ["velocity_x","velocity_y","velocity_z"]
    magFields = ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    #accFields = ['acceleration_x','acceleration_y','acceleration_z']
    accFields = None
    loadPath = ID
    order = "C"
else:
    print("Unknown SimType - use 'Enzo' or 'Athena'... FAIL")
    sys.exit(1)

if FluidType == "hydro":
	magFields = None
elif FluidType != "mhd":
	print("Unknown FluidType - use 'mhd' or 'hydro'... FAIL")
	sys.exit(1)
    

TimeDoneStart = MPI.Wtime() - TimeStart
TimeDoneStart = comm.gather(TimeDoneStart)

if rank == 0:
    print("Imports and startup done in %.3g +/- %.3g" % (np.mean(TimeDoneStart),np.std(TimeDoneStart)))
    sys.stdout.flush()

TimeDoneStart = MPI.Wtime() 
rho, U , B, Acc, P = readAllFieldsWithHDF(loadPath,Res,
    rhoField,velFields,magFields,accFields,order)

TimeDoneReading = MPI.Wtime() - TimeDoneStart
TimeDoneReading = comm.gather(TimeDoneReading)

if rank == 0:
    print("Reading done in %.3g +/- %.3g" % (np.mean(TimeDoneReading),np.std(TimeDoneReading)))
    sys.stdout.flush()

TimeDoneReading = MPI.Wtime()

N = np.array([Res,Res,Res], dtype=int)
# using L = 2pi as we work (e.g. when binning) with integer wavenumbers
L = np.array([2*np.pi, 2*np.pi, 2*np.pi], dtype=float)
FFT = R2C(N, L, comm, "double", communication='Alltoallw')

localK = FFT.get_local_wavenumbermesh(scaled=True)
localKmag = np.linalg.norm(localK,axis=0)

#if rank == 0:
#    print localKmag


W = np.zeros((3,) + FFT.real_shape(),dtype =  FFT.float)

sqrtRho = np.sqrt(rho)
for i in range(3):
    W[i] = sqrtRho * U[i]
del sqrtRho

FT_W = np.zeros((3,) + FFT.complex_shape(),dtype =  FFT.complex)
for i in range(3):
    FT_W[i] = FFT.fftn(W[i], FT_W[i])
        
FTKinEn = 0.5 * np.sum(np.abs(FT_W)**2.,axis=0)
PSKinEn = normedSpec(localKmag.reshape(-1),FTKinEn.reshape(-1))

if rank == 0:
    np.save(str(ID).zfill(4) + "-PSKinEn-" + str(Res) + ".npy",PSKinEn)

del FT_W, FTKinEn, PSKinEn

TimeDoneKinSpec = MPI.Wtime() - TimeDoneReading
TimeDoneKinSpec = comm.gather(TimeDoneKinSpec)

if rank == 0:
    print("KinSpec done in %.3g +/- %.3g" % (np.mean(TimeDoneKinSpec),np.std(TimeDoneKinSpec)))
    sys.stdout.flush()


if rank == 0:
    Quantities = ConfigObj() 
    Quantities.filename = str(ID).zfill(4) + "-flowQuantities-" + str(Res) + ".txt"

    for quantName in ["Mean","RMS"]:
        if quantName not in Quantities.keys():
            Quantities[quantName] = {}


KinEn = 0.5 * np.sum(W**2.,axis=0)
totalKinEn = comm.reduce(np.sum(KinEn))
del KinEn

V2 = np.sum(U**2.,axis=0)
totalV2 = comm.reduce(np.sum(V2))
V = np.sqrt(V2)
totalV = comm.reduce(np.sum(V))

if rank == 0:
    Quantities["Mean"]["KineticEnergy"] = totalKinEn / float(Res**3.)
    
    # assuming isothermal EOS with c_s = 1
    Quantities["RMS"]["SonicMach"] = np.sqrt(totalV2 / float(Res**3.))
    Quantities["Mean"]["SonicMach"] = totalV / float(Res**3.)


if FluidType != "mhd":
    if rank == 0:
        Quantities.write()
    sys.exit(0)

FT_B = np.zeros((3,) + FFT.complex_shape(),dtype =  FFT.complex)
for i in range(3):
    FT_B[i] = FFT.fftn(B[i], FT_B[i])
        
FTMagEn = 0.5 * np.sum(np.abs(FT_B)**2.,axis=0)
PSMagEn = normedSpec(localKmag.reshape(-1),FTMagEn.reshape(-1))

if rank == 0:
    np.save(str(ID).zfill(4) + "-PSMagEn-" + str(Res) + ".npy",PSMagEn)

del FT_B, FTMagEn, PSMagEn


B2 = np.sum(B**2.,axis=0)
totalB2 = comm.reduce(np.sum(B2))

AlfMach2 = V2*rho/B2
totalAlfMach2 = comm.reduce(np.sum(AlfMach2))

AlfMach = np.sqrt(AlfMach2)
totalAlfMach = comm.reduce(np.sum(AlfMach))

plasmaBeta = 2.*rho/B2
totalPlasmaBeta = comm.reduce(np.sum(plasmaBeta))


if rank == 0:
    Quantities["Mean"]["MagneticEnergy"] = 0.5 * totalB2 / float(Res**3.)
    
    # assuming isothermal EOS with c_s = 1
    Quantities["RMS"]["AlfvenicMach"] = np.sqrt(totalAlfMach2 / float(Res**3.))
    Quantities["Mean"]["AlfvenicMach"] = totalAlfMach / float(Res**3.)
    Quantities["Mean"]["plasmabeta"] = totalPlasmaBeta / float(Res**3.)
    
    Quantities.write()



   

# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 ai
