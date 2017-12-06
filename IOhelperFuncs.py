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

# mmet by https://gist.github.com/rossant/7b4704e8caeb8f173084
def _mmap_h5(path, h5path):
    with h5py.File(path) as f:
        ds = f[h5path]
        # We get the dataset address in the HDF5 fiel.
        offset = ds.id.get_offset()
        # We ensure we have a non-compressed contiguous array.
        assert ds.chunks is None
        assert ds.compression is None
        assert offset > 0
        dtype = ds.dtype
        shape = ds.shape
    arr = np.memmap(path, mode='r', shape=shape, offset=offset, dtype=dtype)
    return arr



def readOneFieldWithHDFmmap(loadPath,FieldName,Res,order):
    Filename = loadPath + '/' + FieldName + '-' + str(Res) + '.hdf5'

    chunkSize = Res/size
    startIdx = int(rank * chunkSize)
    endIdx = int((rank + 1) * chunkSize)
    if endIdx == Res:                
        endIdx = None

    if order == 'F':                    
        data = np.float64(_mmap_h5(Filename, '/' + FieldName)[0,:,:,startIdx:endIdx].T)
    else:
        print('readOneFieldWithHDFmmap: Order %s not tested yet. Abort...' % order)
        sys.exit(1)


    return np.ascontiguousarray(data)

def readOneFieldWithHDF(loadPath,FieldName,Res,order):

    if order == 'F':
        Filename = loadPath + '/' + FieldName + '-' + str(Res) + '.hdf5'

        if rank == 0:

            h5Data = h5py.File(Filename, 'r')[FieldName]
        
            tmp = np.float64(h5Data[0,:,:,:]).T.reshape((size,int(Res/size),Res,Res))

            data = comm.scatter(tmp)

        else:
            data = comm.scatter(None)

    elif order == 'C':
        Filename = loadPath + '/' + FieldName + '-' + str(Res) + '-C.hdf5'
        
        chunkSize = Res/size
        startIdx = int(rank * chunkSize)
        endIdx = int((rank + 1) * chunkSize)
        if endIdx == Res:                
            endIdx = None
        
        h5Data = h5py.File(Filename, 'r')[FieldName]
        data = np.float64(h5Data[0,startIdx:endIdx,:,:])

    if rank == 0:
        print("[%03d] done reading %s" % (rank,FieldName))

    return np.ascontiguousarray(data)

def readAllFieldsWithHDF(loadPath,Res,
    rhoField,velFields,magFields,accFields,pField,order,useMMAP=False):
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

    if useMMAP:
        readOneFieldWithX = readOneFieldWithHDFmmap
    else:
        readOneFieldWithX = readOneFieldWithHDF
    
    if rhoField is not None:
        rho = readOneFieldWithX(loadPath,rhoField,Res,order)
    else:
        rho = np.ones(FinalShape,dtype=np.float64) 
    
    if velFields is not None:
        U = np.zeros((3,) + FinalShape,dtype=np.float64)
        U[0] = readOneFieldWithX(loadPath,velFields[0],Res,order)
        U[1] = readOneFieldWithX(loadPath,velFields[1],Res,order)
        U[2] = readOneFieldWithX(loadPath,velFields[2],Res,order)
    else:
        U = None
        
    if magFields is not None:
        B = np.zeros((3,) + FinalShape,dtype=np.float64)  
        B[0] = readOneFieldWithX(loadPath,magFields[0],Res,order)
        B[1] = readOneFieldWithX(loadPath,magFields[1],Res,order)
        B[2] = readOneFieldWithX(loadPath,magFields[2],Res,order)
    else:
        B = None
    
    if accFields is not None:
        Acc = np.zeros((3,) + FinalShape,dtype=np.float64)  
        Acc[0] = readOneFieldWithX(loadPath,accFields[0],Res,order)
        Acc[1] = readOneFieldWithX(loadPath,accFields[1],Res,order)
        Acc[2] = readOneFieldWithX(loadPath,accFields[2],Res,order)
    else:
        Acc = None
    
    if pField is not None:
        P = readOneFieldWithX(loadPath,pField,Res,order)
    else:
        # CAREFUL assuming isothermal EOS here with c_s = 1 -> P = rho in code units
        if rank == 0:
            print("WARNING: remember assuming isothermal EOS with c_s = 1, i.e. P = rho")
        P = rho
        
    return rho, U, B, Acc, P


