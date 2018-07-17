from mpi4py import MPI

comm  = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
TimeStart = MPI.Wtime()

import numpy as np
from mpi4py_fft.mpifft import PFFT, Function
import time
import pickle
import sys
import h5py
import os
from scipy.stats import binned_statistic
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
globalMinMaxFile = sys.argv[6]

try:
    globalMinMax = pickle.load(open(globalMinMaxFile,'rb'))
    if rank == 0:
        print("Successfully loaded globalMinMax dict: %s" % globalMinMaxFile)
except:
    globalMinMax = {}

if 'adiabatic' in FluidType:
    Gamma = float(sys.argv[7])
    if rank == 0:
        print("Using gamma = %.3f for adiabatic EOS" % Gamma)

        
order='unset'
pField = None

if SimType == "AthenaHDF":
    rhoField = "density"
    velFields = ["velocity_x","velocity_y","velocity_z"]
    magFields = ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    accFields = ['acceleration_x','acceleration_y','acceleration_z']
    #accFields = None
    if 'adiabatic' in FluidType:
        pField = 'pressure'
    loadPath = ID
    order = "F"
elif SimType == "AthenaHDFC":
    rhoField = "density"
    velFields = ["velocity_x","velocity_y","velocity_z"]
    magFields = ["cell_centered_B_x","cell_centered_B_y","cell_centered_B_z"]
    accFields = ['acceleration_x','acceleration_y','acceleration_z']
    if 'adiabatic' in FluidType:
        pField = 'pressure'
    loadPath = ID
    order = "C"
else:
    print("Unknown SimType - use 'Enzo' or 'Athena'... FAIL")
    sys.exit(1)

if FluidType == "hydro":
	magFields = None
elif 'mhd' not in FluidType:
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
    rhoField,velFields,magFields,accFields,pField,order)

TimeDoneReading = MPI.Wtime() - TimeDoneStart
TimeDoneReading = comm.gather(TimeDoneReading)

if rank == 0:
    print("Reading done in %.3g +/- %.3g" % (np.mean(TimeDoneReading),np.std(TimeDoneReading)))
    sys.stdout.flush()

TimeDoneReading = MPI.Wtime()


if rank == 0:
    Outfile = h5py.File(str(ID).zfill(4) + "-stats-" + str(Res) + ".hdf5", "w")

kMaxInt = ceil(Res*0.5*np.sqrt(3.))
Bins = np.linspace(0.5,kMaxInt + 0.5,kMaxInt+1)
# if max k does not fall in last bin, remove last bin
if Res*0.5*np.sqrt(3.) < Bins[-2]:
    Bins = Bins[:-1]

N = np.array([Res,Res,Res], dtype=int)
# using L = 2pi as we work (e.g. when binning) with integer wavenumbers
L = np.array([2*np.pi, 2*np.pi, 2*np.pi], dtype=float)
#FFT = PFFT(comm,N, axes=(0,1,2),collapse=False, slab=True,dtype=np.float64)
FFT = PFFT(comm,N, axes=(0,1,2),collapse=False, slab=True,dtype=np.complex128)

localK = FFT.get_local_wavenumbermesh(L)
localKmag = np.linalg.norm(localK,axis=0)

localKunit = np.copy(localK)
# fixing division by 0 for harmonic part
if rank == 0:
    if localKmag[0,0,0] == 0.:
        localKmag[0,0,0] = 1.
    else:
        print("[0] kmag not zero where it's supposed to be")
        sys.exit(1)
else:
    if (localKmag == 0.).any():
        print("[%d] kmag not zero where it's supposed to be" % rank)
        sys.exit(1)

localKunit /= localKmag
if rank == 0:
    localKmag[0,0,0] = 0.


def normedSpec(k,quantity,Bins):
    histSum = binned_statistic(k,quantity,bins=Bins,statistic='sum')[0]
    kSum = binned_statistic(k,k,bins=Bins,statistic='sum')[0]
    histCount = np.histogram(k,bins=Bins)[0]

    totalHistSum = comm.reduce(histSum.astype(np.float64))
    totalKSum = comm.reduce(kSum.astype(np.float64))
    totalHistCount = comm.reduce(histCount.astype(np.float64))

    if rank == 0:

        if (totalHistCount == 0.).any():
            print("totalHistCount is 0. Check desired binning!")
            print(Bins)
            print(totalHistCount)
            sys.exit(1)

        # calculate corresponding k to to bin
        # this help to overcome statistics for low k bins
        centeredK = totalKSum / totalHistCount

        ###  "integrate" over k-shells
        # normalized by mean shell surface
        valsShell = 4. * np.pi * centeredK**2. * (totalHistSum / totalHistCount)
        # normalized by mean shell volume
        valsVol = 4. * np.pi / 3.* (Bins[1:]**3. - Bins[:-1]**3.) * (totalHistSum / totalHistCount)
        # unnormalized
        valsNoNorm = totalHistSum

        return [centeredK,valsShell,valsVol,valsNoNorm]
    else:
        return None


def getComponents(vec):
    """ decomposed input vector into harmonic, rotational and compressive part
    """
    
    N = float(comm.allreduce(vec.size))
    total = comm.allreduce(np.sum(vec,axis=(1,2,3)))
    Harm = total / N
    
    FT_vec = Function(FFT,tensor=3)
    for i in range(3):
        FT_vec[i] = FFT.forward(vec[i], FT_vec[i])
    
    # project components
    localVecDotKunit = np.sum(FT_vec*localKunit,axis = 0)

    FT_Dil = localVecDotKunit * localKunit
    
    FT_Sol = FT_vec - FT_Dil
    if rank == 0:
        # remove harmonic part from solenoidal component
        FT_Sol[:,0,0,0] = 0.

    Dil = Function(FFT,False,tensor=3)
    Sol = Function(FFT,False,tensor=3)

    for i in range(3):
        Dil[i] = FFT.backward(FT_Dil[i],Dil[i])
        Sol[i] = FFT.backward(FT_Sol[i],Sol[i])

    return Harm, Sol.real, Dil.real


def getScaPowSpec(name,field):

    FT_field = Function(FFT)
    FT_field = FFT.forward(field, FT_field)

    FT_fieldAbs2 = np.abs(FT_field)**2.
    PS_Full = normedSpec(localKmag.reshape(-1),FT_fieldAbs2.reshape(-1),Bins)

    if rank == 0:
        Outfile.require_dataset(name + '/PowSpec/Bins', (1,len(Bins)), dtype='f')[0] = Bins
        Outfile.require_dataset(name + '/PowSpec/Full', (4,len(Bins)-1), dtype='f')[:,:] = PS_Full

def getVecPowSpecs(name,vec):

    FT_vec = Function(FFT,tensor=3)
    for i in range(3):
        FT_vec[i] = FFT.forward(vec[i], FT_vec[i])

    FT_vecAbs2 = np.linalg.norm(FT_vec,axis=0)**2.
    PS_Full = normedSpec(localKmag.reshape(-1),FT_vecAbs2.reshape(-1),Bins)

    # project components
    localVecDotKunit = np.sum(FT_vec*localKunit,axis = 0)

    FT_Dil = localVecDotKunit * localKunit
    FT_DilAbs2 = np.linalg.norm(FT_Dil,axis=0)**2.
    PS_Dil = normedSpec(localKmag.reshape(-1),FT_DilAbs2.reshape(-1),Bins)
    
    FT_Sol = FT_vec - FT_Dil
    if rank == 0:
        # remove harmonic part from solenoidal component
        FT_Sol[:,0,0,0] = 0.
    FT_SolAbs2 = np.linalg.norm(FT_Sol,axis=0)**2.
    PS_Sol = normedSpec(localKmag.reshape(-1),FT_SolAbs2.reshape(-1),Bins)



    totPowFull = comm.allreduce(np.sum(FT_vecAbs2))
    totPowDil = comm.allreduce(np.sum(FT_DilAbs2))
    totPowSol = comm.allreduce(np.sum(FT_SolAbs2))
    totPowHarm = np.linalg.norm(FT_vec[:,0,0,0],axis=0)**2.

    if rank == 0:
        Outfile.require_dataset(name + '/PowSpec/Bins', (1,len(Bins)), dtype='f')[0] = Bins
        Outfile.require_dataset(name + '/PowSpec/Full', (4,len(Bins)-1), dtype='f')[:,:] = PS_Full
        Outfile.require_dataset(name + '/PowSpec/Dil', (4,len(Bins)-1), dtype='f')[:,:] = PS_Dil
        Outfile.require_dataset(name + '/PowSpec/Sol', (4,len(Bins)-1), dtype='f')[:,:] = PS_Sol
        Outfile.require_dataset(name + '/PowSpec/TotSol', (1,), dtype='f')[0] = totPowSol
        Outfile.require_dataset(name + '/PowSpec/TotDil', (1,), dtype='f')[0] = totPowDil
        Outfile.require_dataset(name + '/PowSpec/TotFull', (1,), dtype='f')[0] = totPowFull
        Outfile.require_dataset(name + '/PowSpec/TotHarm', (1,), dtype='f')[0] = totPowHarm

getVecPowSpecs('u',U)
getVecPowSpecs('rhoU',np.sqrt(rho)*U)
getVecPowSpecs('rhoThirdU',rho**(1./3.)*U)


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


def getCorrCoeff(X,Y):
    N = float(comm.allreduce(X.size))
    meanX = comm.allreduce(np.sum(X)) / N
    meanY = comm.allreduce(np.sum(Y)) / N

    stdX = np.sqrt(comm.allreduce(np.sum((X - meanX)**2.)))
    stdY = np.sqrt(comm.allreduce(np.sum((Y - meanY)**2.)))

    cov = comm.allreduce(np.sum( (X - meanX)*(Y - meanY)))

    return cov / (stdX*stdY)

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

getAndWriteStatisticsToFile(rho,"rho")
getAndWriteStatisticsToFile(np.log(rho),"lnrho")
getScaPowSpec('rho',rho)
getScaPowSpec('lnrho',np.log(rho))

V2 = np.sum(U**2.,axis=0)
getAndWriteStatisticsToFile(np.sqrt(V2),"u")

getAndWriteStatisticsToFile(0.5 * rho * V2,"KinEnDensity")
getAndWriteStatisticsToFile(0.5 * V2,"KinEnSpecific")

if Acc is not None:
    Amag = np.sqrt(np.sum(Acc**2.,axis=0))
    getAndWriteStatisticsToFile(Amag,"a")
    getVecPowSpecs('a',Acc)

    corrUA = getCorrCoeff(np.sqrt(V2),Amag)
    if rank == 0:
        Outfile.require_dataset('U-A/corr', (1,), dtype='f')[0] = corrUA
    
    UHarm, USol, UDil = getComponents(U)
    getAndWriteStatisticsToFile(
        np.sum(Acc*U,axis=0)/(
            np.linalg.norm(Acc,axis=0)*np.linalg.norm(U,axis=0)),"Angle_u_a")
    getAndWriteStatisticsToFile(
        np.sum(Acc*USol,axis=0)/(
            np.linalg.norm(Acc,axis=0)*np.linalg.norm(USol,axis=0)),"Angle_uSol_a")
    getAndWriteStatisticsToFile(
        np.sum(Acc*UDil,axis=0)/(
            np.linalg.norm(Acc,axis=0)*np.linalg.norm(UDil,axis=0)),"Angle_uDil_a")

getAndWriteStatisticsToFile(np.abs(MPIdivX(comm,U)),"AbsDivU")
getAndWriteStatisticsToFile(np.sqrt(np.sum(MPIrotX(comm,U)**2.,axis=0)),"AbsRotU")

if 'adiabatic' in FluidType:
    Ms2 = V2 / (Gamma * P / rho)
    getAndWriteStatisticsToFile(np.sqrt(Ms2),"Ms")
    get2dHist('rho-P',rho,P)
    getAndWriteStatisticsToFile(P,"P")
    getAndWriteStatisticsToFile(np.log10(P),"log10P")
    T = P / (Gamma - 1.0) /rho
    getAndWriteStatisticsToFile(T,"T")
    getAndWriteStatisticsToFile(np.log10(T),"log10T")
    get2dHist('rho-T',rho,T)


if "mhd" not in FluidType:
    if rank == 0:
        Outfile.close()
    sys.exit(0)


B2 = np.sum(B**2.,axis=0)

getAndWriteStatisticsToFile(np.sqrt(B2),"B")
getAndWriteStatisticsToFile(0.5 * B2,"MagEnDensity")

if 'adiabatic' in FluidType:
    TotPres = P + B2/2.
    corrPBcomp = getCorrCoeff(P,np.sqrt(B2)/rho**(2./3.))
else:
    TotPres = rho + B2/2.
    corrPBcomp = getCorrCoeff(rho,np.sqrt(B2)/rho**(2./3.))

getAndWriteStatisticsToFile(TotPres,"TotPres")

if rank == 0:
    Outfile.require_dataset('P-Bcomp/corr', (1,), dtype='f')[0] = corrPBcomp

AlfMach2 = V2*rho/B2
AlfMach = np.sqrt(AlfMach2)

getAndWriteStatisticsToFile(AlfMach,"AlfvenicMach")

if 'adiabatic' in FluidType:
    plasmaBeta = 2.* P / B2
else:
    plasmaBeta = 2.*rho/B2
getAndWriteStatisticsToFile(plasmaBeta,"plasmabeta")
getAndWriteStatisticsToFile(np.log10(plasmaBeta),"log10plasmabeta")

getVecPowSpecs('B',B)

# this is cheap... and only works for slab decomp on x-axis
# np.sum is required for slabs with width > 1
DM = comm.allreduce(np.sum(rho,axis=0))/float(Res)
RM = comm.allreduce(np.sum(B[0]*rho,axis=0))/float(Res)
chunkSize = int(Res/size)
endIdx = int((rank + 1) * chunkSize)
if endIdx == size:
    endIdx = None
getAndWriteStatisticsToFile(DM[rank*chunkSize:endIdx,:],"DM_x")
getAndWriteStatisticsToFile(np.log(DM[rank*chunkSize:endIdx,:]),"lnDM_x")
getAndWriteStatisticsToFile(RM[rank*chunkSize:endIdx,:],"RM_x")
getAndWriteStatisticsToFile(RM[rank*chunkSize:endIdx,:]/DM[rank*chunkSize:endIdx,:],"LOSB_x")

DM = np.mean(rho,axis=1)
RM = np.mean(B[1]*rho,axis=1)
getAndWriteStatisticsToFile(DM,"DM_y")
getAndWriteStatisticsToFile(np.log(DM),"lnDM_y")
getAndWriteStatisticsToFile(RM,"RM_y")
getAndWriteStatisticsToFile(RM/DM,"LOSB_y")

DM = np.mean(rho,axis=2)
RM = np.mean(B[2]*rho,axis=2)
getAndWriteStatisticsToFile(DM,"DM_z")
getAndWriteStatisticsToFile(np.log(DM),"lnDM_z")
getAndWriteStatisticsToFile(RM,"RM_z")
getAndWriteStatisticsToFile(RM/DM,"LOSB_z")


    
corrRhoB = getCorrCoeff(rho,np.sqrt(B2))
if rank == 0:
    Outfile.require_dataset('rho-B/corr', (1,), dtype='f')[0] = corrRhoB


get2dHist('rho-B',rho,np.sqrt(B2))
get2dHist('log10rho-B',np.log10(rho),np.sqrt(B2))

if rank == 0:
    Outfile.close()
    


# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 ai
