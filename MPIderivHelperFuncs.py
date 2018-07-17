import numpy as np

def MPIderiv2(comm,var,dim):
    """Returns first derivative (2-point central finite difference)
    
    of a 3-dimensional, real space, uniform grid (with L = 1) variable.
    Assumes that the field is split on axis 0 between processes.
    
    Args:
        comm -- MPI world communicator
        var -- input field 
        dim -- axis along the derivative should be taken   
    """
    
    rank = comm.Get_rank()
    size = comm.Get_size()
         
    sl_m1 = slice(None,-2,None)
    sl_p1 = slice(2,None,None)
    sl_c = slice(1,-1,None)
    ds = 2.0/float(var.shape[1]) # assumes axis 1 covers entire grid with L = 1       
               
    # send right slice of local proc as left slab to follow proc
    leftSlice = None
    leftSlice = comm.sendrecv(sendobj=var[-1:,:,:],dest=(rank+1)%size,source=(rank-1)%size)
    
    # send left slice of local proc as right slab to follow proc
    rightSlice = None
    rightSlice = comm.sendrecv(sendobj=var[:1,:,:],dest=(rank-1)%size,source=(rank+1)%size)
    
    tmp = np.concatenate((leftSlice,var,rightSlice),axis=0)
    tmp = np.concatenate((tmp[:,-1:,:],tmp,tmp[:,:1,:]),axis=1)
    tmp = np.concatenate((tmp[:,:,-1:],tmp,tmp[:,:,:1]),axis=2)    

    #TODO further optimization could be done here, i.e. communicate only once for all three derivs
    if (dim == 0):
        p1 = tmp[sl_p1,sl_c,sl_c]
        m1 = tmp[sl_m1,sl_c,sl_c]
    elif (dim == 1):
        p1 = tmp[sl_c,sl_p1,sl_c]
        m1 = tmp[sl_c,sl_m1,sl_c]
    elif (dim == 2):
        p1 = tmp[sl_c,sl_c,sl_p1]
        m1 = tmp[sl_c,sl_c,sl_m1]
    else:
        print("watch out for dimension!")
                
    del tmp 
            
    return np.array((p1 - m1)/ds)

def MPIXdotGradYScalar(comm,X,Y):
    """ returns  (X . grad) Y
    """
    
    return X[0] * MPIderiv2(comm,Y,0) + X[1] * MPIderiv2(comm,Y,1) + X[2] * MPIderiv2(comm,Y,2)

def MPIXdotGradY(comm,X,Y):
    """ returns  (X . grad) Y
    """
    
    res = np.zeros_like(X)
    for i in range(3):
        res[i] = (X[0] * MPIderiv2(comm,Y[i],0) + X[1] * MPIderiv2(comm,Y[i],1) + X[2] * MPIderiv2(comm,Y[i],2))
        
    return res

def MPIdivX(comm,X):
    """ returns  div X = x_dx + y_dy + z_dz
    """
    
    return MPIderiv2(comm,X[0],0) + MPIderiv2(comm,X[1],1) + MPIderiv2(comm,X[2],2)

def MPIdivXY(comm,X,Y):
    """ returns  pd_j X_j Y_i
    """
    res = np.zeros_like(Y)
    
    for i in range(3):
        res[i] = MPIderiv2(comm,X[0]*Y[i],0) + MPIderiv2(comm,X[1]*Y[i],1) + MPIderiv2(comm,X[2]*Y[i],2)
    
    return res

def MPIgradX(comm,X):
    """ returns  grad X = [ x_dx, y_dy, z_dz ]
    """
    
    return np.array([MPIderiv2(comm,X,0),
                     MPIderiv2(comm,X,1),
                     MPIderiv2(comm,X,2),
                     ])
def MPIrotX(comm,X):
    """ returns  curl X = [ z_dy - y_dz, x_dz - z_dx, y_dx - x_dy ]
    """
    
    return np.array([MPIderiv2(comm,X[2],1) - MPIderiv2(comm,X[1],2),
                     MPIderiv2(comm,X[0],2) - MPIderiv2(comm,X[2],0),
                     MPIderiv2(comm,X[1],0) - MPIderiv2(comm,X[0],1),
                     ])
