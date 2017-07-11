import yt
import numpy as np
from mpi4py import MPI

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


