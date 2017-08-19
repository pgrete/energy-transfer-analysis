import yt
import numpy as np
from mpi4py import MPI
import sys
import h5py

comm  = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def readAllFieldsWithYT(loadPath,Res,
    rhoField,velFields,magFields,accFields):
    """
    Reads all fields using the yt frontend. Data is read in parallel.

    """
	
    if Res % size != 0:
        print("Data cannot be split evenly among processes. Abort (for now) - fix me!")
        sys.exit(1)
        
    FinalShape = (Res//size,Res,Res)   
    
    ds = yt.load(loadPath)
    dims = ds.domain_dimensions
    dims[0] /= size

    startPos = rank * 1./np.float(size)
    if rank == 0:
        print("WARNING: remember assuming domain of L = 1")
    

    ad = ds.h.covering_grid(level=0, left_edge=[startPos,0.0,0.0],dims=dims)
    
    if rhoField is not None:
        rho = ad[rhoField].d
    else:
        rho = np.ones(FinalShape,dtype=np.float64) 
    
    if velFields is not None:
        U = np.zeros((3,) + FinalShape,dtype=np.float64)
        U[0] = ad[velFields[0]].d
        U[1] = ad[velFields[1]].d
        U[2] = ad[velFields[2]].d
    else:
        U = None
        
    if magFields is not None:
        B = np.zeros((3,) + FinalShape,dtype=np.float64)  
        B[0] = ad[magFields[0]].d
        B[1] = ad[magFields[1]].d
        B[2] = ad[magFields[2]].d
    else:
        B = None
    
    if accFields is not None:
        Acc = np.zeros((3,) + FinalShape,dtype=np.float64)  
        Acc[0] = ad[accFields[0]].d
        Acc[1] = ad[accFields[1]].d
        Acc[2] = ad[accFields[2]].d
    else:
        Acc = None
        
    # CAREFUL assuming isothermal EOS here with c_s = 1 -> P = rho in code units
    if rank == 0:
        print("WARNING: rememer assuming isothermal EOS with c_s = 1, i.e. P = rho hardcoded")
    return rho, U, B, Acc, rho

def readOneFieldWithHDF(loadPath,FieldName,Res,order):
    Filename = loadPath + '/' + FieldName + '-' + str(Res) + '.hdf5'

    h5Data = h5py.File(Filename, 'r')[FieldName]

    chunkSize = Res/size
    startIdx = int(rank * chunkSize)
    endIdx = int((rank + 1) * chunkSize)
    if endIdx == Res:                
        endIdx = None

    if order == 'C':
        data = np.float64(h5Data[0,startIdx:endIdx,:,:])     
    elif order == 'F':                    
        data = np.float64(h5Data[0,:,:,startIdx:endIdx].T)

    return np.ascontiguousarray(data)
    

def readAllFieldsWithHDF(loadPath,Res,
    rhoField,velFields,magFields,accFields,order):
    """
    Reads all fields using the HDF5. Data is read in parallel.

    """
	
    if Res % size != 0:
        print("Data cannot be split evenly among processes. Abort (for now) - fix me!")
        sys.exit(1)

    if order is not "C" and order is not "F":
        print("For safety reasons you have to specify the order (row or column major) for your data.")
        sys.exit(1)
        
    FinalShape = (Res//size,Res,Res)   
    
    if rhoField is not None:
        rho = readOneFieldWithHDF(loadPath,rhoField,Res,order)
    else:
        rho = np.ones(FinalShape,dtype=np.float64) 
    
    if velFields is not None:
        U = np.zeros((3,) + FinalShape,dtype=np.float64)
        U[0] = readOneFieldWithHDF(loadPath,velFields[0],Res,order)
        U[1] = readOneFieldWithHDF(loadPath,velFields[1],Res,order)
        U[2] = readOneFieldWithHDF(loadPath,velFields[2],Res,order)
    else:
        U = None
        
    if magFields is not None:
        B = np.zeros((3,) + FinalShape,dtype=np.float64)  
        B[0] = readOneFieldWithHDF(loadPath,magFields[0],Res,order)
        B[1] = readOneFieldWithHDF(loadPath,magFields[1],Res,order)
        B[2] = readOneFieldWithHDF(loadPath,magFields[2],Res,order)
    else:
        B = None
    
    if accFields is not None:
        Acc = np.zeros((3,) + FinalShape,dtype=np.float64)  
        Acc[0] = readOneFieldWithHDF(loadPath,accFields[0],Res,order)
        Acc[1] = readOneFieldWithHDF(loadPath,accFields[1],Res,order)
        Acc[2] = readOneFieldWithHDF(loadPath,accFields[2],Res,order)
    else:
        Acc = None
        
    # CAREFUL assuming isothermal EOS here with c_s = 1 -> P = rho in code units
    if rank == 0:
        print("WARNING: rememer assuming isothermal EOS with c_s = 1, i.e. P = rho hardcoded")
    return rho, U, B, Acc, rho


