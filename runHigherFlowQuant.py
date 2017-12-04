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
from scipy.stats import binned_statistic, pearsonr
from math import ceil
from configobj import ConfigObj
from IOhelperFuncs import readAllFieldsWithYT, readAllFieldsWithHDF 
from MPIderivHelperFuncs import MPIderiv2, MPIXdotGradY, MPIdivX, MPIdivXY, MPIgradX, MPIrotX

#Res = int(sys.argv[3])
ID = sys.argv[1]
Res = int(sys.argv[2])
SimType = sys.argv[3] # Enzo or Athena
FluidType = sys.argv[4] # hydro or mhd
TurbType = sys.argv[5] # forced or decay

globalMinMax = {
    'rho' : [0.16159,3.7477],
    'lnrho' : [-1.8227,1.3211],
    'log10rho' : [-0.79159,0.57376],
    'u' : [2.6225e-05,2.3007],
    'a' : [0.00033966,1436.7],
    'AbsDivU' : [0,522.2],
    'AbsRotU' : [0.001079,611.2],
    'B' : [6.9895e-05,2.0628],
    'AlfvenicMach' : [0.00015687,10037],
    'plasmabeta' : [0.14199,3.9747e+08],
    'DM_x' : [0.01,100],
    'DM_y' : [0.01,100],
    'DM_z' : [0.01,100],
    'lnDM_x' : [-3.,3.],
    'lnDM_y' : [-3.,3.],
    'lnDM_z' : [-3.,3.],
    'RM_x' : [-2.,2.],
    'RM_y' : [-2.,2.],
    'RM_z' : [-2.,2.],
}
        
order='unset'

if SimType == "AthenaHDF":
    rhoField = "density"
    velFields = ["velocity_x","velocity_y","velocity_z"]
    magFields = ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    accFields = ['acceleration_x','acceleration_y','acceleration_z']
    #accFields = None
    loadPath = ID
    order = "F"
elif SimType == "AthenaHDFC":
    rhoField = "density"
    velFields = ["velocity_x","velocity_y","velocity_z"]
    magFields = ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    accFields = ['acceleration_x','acceleration_y','acceleration_z']
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
    
if TurbType == 'decay':
    accFields = None

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


if rank == 0:
    Outfile = h5py.File(str(ID).zfill(4) + "-stats-" + str(Res) + ".hdf5", "w")


def getAndWriteStatisticsToFile(field,name,bounds=None):
    """
        field - 3d scalar field to get statistics from
        name - human readable name of the field
        bounds - tuple, lower and upper bound for histogram, if None then min/max
    """
    
    N = float(comm.allreduce(field.size))
    total = comm.allreduce(np.sum(field))
    mean = total / N

    totalSqrd = comm.allreduce(np.sum(field**2.))
    rms = np.sqrt(totalSqrd / N)

    var = comm.allreduce(np.sum((field - mean)**2.)) / (N - 1.)
    
    stddev = np.sqrt(var)

    skew = comm.allreduce(np.sum((field - mean)**3. / stddev**3.)) / N
    
    kurt = comm.allreduce(np.sum((field - mean)**4. / stddev**4.)) / N - 3.

    globMin = comm.allreduce(np.min(field),op=MPI.MIN)
    globMax = comm.allreduce(np.max(field),op=MPI.MAX)

    globAbsMin = comm.allreduce(np.min(np.abs(field)),op=MPI.MIN)
    globAbsMax = comm.allreduce(np.max(np.abs(field)),op=MPI.MAX)

    if rank == 0:
        Outfile.require_dataset(name + '/moments/mean', (1,), dtype='f')[0] = mean
        Outfile.require_dataset(name + '/moments/rms', (1,), dtype='f')[0] = rms
        Outfile.require_dataset(name + '/moments/var', (1,), dtype='f')[0] = var
        Outfile.require_dataset(name + '/moments/stddev', (1,), dtype='f')[0] = stddev
        Outfile.require_dataset(name + '/moments/skew', (1,), dtype='f')[0] = skew
        Outfile.require_dataset(name + '/moments/kurt', (1,), dtype='f')[0] = kurt
        Outfile.require_dataset(name + '/moments/min', (1,), dtype='f')[0] = globMin
        Outfile.require_dataset(name + '/moments/max', (1,), dtype='f')[0] = globMax
        Outfile.require_dataset(name + '/moments/absmin', (1,), dtype='f')[0] = globAbsMin
        Outfile.require_dataset(name + '/moments/absmax', (1,), dtype='f')[0] = globAbsMax

    if bounds is None:
        bounds = [globMin,globMax]
        HistBins = "Snap"
    else:
        HistBins = "Sim"

    Bins = np.linspace(bounds[0],bounds[1],129)
    hist = np.histogram(field.reshape(-1),bins=Bins)[0]
    totalHist = comm.allreduce(hist)

    if rank == 0:
        tmp  = Outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax', (2,129), dtype='f')
        tmp[0] = Bins
        tmp[1,:-1] = totalHist.astype(float)

    if name in globalMinMax.keys():
        Bins = np.linspace(globalMinMax[name][0],globalMinMax[name][1],129)
        hist = np.histogram(field.reshape(-1),bins=Bins)[0]
        totalHist = comm.allreduce(hist)

        HistBins = 'globalMinMax'

        if rank == 0:
            tmp  = Outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax', (2,129), dtype='f')
            tmp[0] = Bins
            tmp[1,:-1] = totalHist.astype(float)



getAndWriteStatisticsToFile(rho,"rho")
getAndWriteStatisticsToFile(np.log(rho),"lnrho")
V2 = np.sum(U**2.,axis=0)
getAndWriteStatisticsToFile(np.sqrt(V2),"u")
if Acc is not None:
    getAndWriteStatisticsToFile(np.sqrt(np.sum(Acc**2.,axis=0)),"a")
getAndWriteStatisticsToFile(np.abs(MPIdivX(comm,U)),"AbsDivU")
getAndWriteStatisticsToFile(np.sqrt(np.sum(MPIrotX(comm,U)**2.,axis=0)),"AbsRotU")


if FluidType != "mhd":
    if rank == 0:
        Outfile.close()
    sys.exit(0)


B2 = np.sum(B**2.,axis=0)

getAndWriteStatisticsToFile(np.sqrt(B2),"B")

AlfMach2 = V2*rho/B2
AlfMach = np.sqrt(AlfMach2)

getAndWriteStatisticsToFile(AlfMach,"AlfvenicMach")

plasmaBeta = 2.*rho/B2
getAndWriteStatisticsToFile(plasmaBeta,"plasmabeta")


# this is cheap... and only works for slab decomp on x-axis
# np.sum is required for slabs with width > 1
DM = comm.allreduce(np.sum(rho,axis=0))/float(Res)
chunkSize = int(Res/size)
endIdx = int((rank + 1) * chunkSize)
if endIdx == size:
    endIdx = None
getAndWriteStatisticsToFile(DM[rank*chunkSize:endIdx,:],"DM_x")
getAndWriteStatisticsToFile(np.log(DM[rank*chunkSize:endIdx,:]),"lnDM_x")

DM = np.mean(rho,axis=1)
getAndWriteStatisticsToFile(DM,"DM_y")
getAndWriteStatisticsToFile(np.log(DM),"lnDM_y")

DM = np.mean(rho,axis=2)
getAndWriteStatisticsToFile(DM,"DM_z")
getAndWriteStatisticsToFile(np.log(DM),"lnDM_z")

# this is cheap... and only works for slab decomp on x-axis
# np.sum is required for slabs with width > 1
RM = comm.allreduce(np.sum(B[0]*rho,axis=0))/float(Res)
chunkSize = int(Res/size)
endIdx = int((rank + 1) * chunkSize)
if endIdx == size:
    endIdx = None
getAndWriteStatisticsToFile(RM[rank*chunkSize:endIdx,:],"RM_x")

RM = np.mean(B[1]*rho,axis=1)
getAndWriteStatisticsToFile(RM,"RM_y")

RM = np.mean(B[2]*rho,axis=2)
getAndWriteStatisticsToFile(RM,"RM_z")


def getCorrCoeff(X,Y):
    N = float(comm.allreduce(X.size))
    meanX = comm.allreduce(np.sum(X)) / N
    meanY = comm.allreduce(np.sum(Y)) / N

    stdX = np.sqrt(comm.allreduce(np.sum((X - meanX)**2.)))
    stdY = np.sqrt(comm.allreduce(np.sum((Y - meanY)**2.)))

    cov = comm.allreduce(np.sum( (X - meanX)*(Y - meanY)))

    return cov / (stdX*stdY)
    
corrRhoB = getCorrCoeff(rho,np.sqrt(B2))

if rank == 0:
    Outfile.require_dataset('rho-B/corr', (1,), dtype='f')[0] = corrRhoB

def get2dHist(name,X,Y,bounds = None):
    if bounds is None:
        globXMin = comm.allreduce(np.min(X),op=MPI.MIN)
        globXMax = comm.allreduce(np.max(X),op=MPI.MAX)
        
        globYMin = comm.allreduce(np.min(Y),op=MPI.MIN)
        globYMax = comm.allreduce(np.max(Y),op=MPI.MAX)

        bounds = [[globXMin,globXMax],[globYMin,globYMax]]
        HistBins = "Snap"
    else:
        HistBins = "Sim"
    
    
    XBins = np.linspace(bounds[0][0],bounds[0][1],129)
    YBins = np.linspace(bounds[1][0],bounds[1][1],129)

    hist = np.histogram2d(X.reshape(-1),Y.reshape(-1),bins=[XBins,YBins])[0]
    totalHist = comm.allreduce(hist)

    if rank == 0:
        tmp = Outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/edges', (2,129), dtype='f')
        tmp[0] = XBins
        tmp[1] = YBins
        tmp = Outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/counts', (128,128), dtype='f')
        tmp[:,:] = totalHist.astype(float)
    
    Xname, Yname = name.split('-')
    if Xname in globalMinMax.keys() and Yname in globalMinMax.keys():
        XBins = np.linspace(globalMinMax[Xname][0],globalMinMax[Xname][1],129)
        YBins = np.linspace(globalMinMax[Yname][0],globalMinMax[Yname][1],129)

        hist = np.histogram2d(X.reshape(-1),Y.reshape(-1),bins=[XBins,YBins])[0]
        totalHist = comm.allreduce(hist)
        
        HistBins = 'globalMinMax'

        if rank == 0:
            tmp = Outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/edges', (2,129), dtype='f')
            tmp[0] = XBins
            tmp[1] = YBins
            tmp = Outfile.require_dataset(name + '/hist/' + HistBins + 'MinMax/counts', (128,128), dtype='f')
            tmp[:,:] = totalHist.astype(float)


get2dHist('rho-B',rho,np.sqrt(B2))
get2dHist('log10rho-B',np.log10(rho),np.sqrt(B2))

if rank == 0:
    Outfile.close()
    


# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 ai
