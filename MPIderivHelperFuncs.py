import numpy as np
import FFTHelperFuncs

def MPIderiv2(comm,var,dim,deriv=1):
    """Returns first derivative (2-point central finite difference)
    
    of a 3-dimensional, real space, uniform grid (with L = 1) variable.
    Assumes that the field is split on axis 0 between processes.
    
    Args:
        comm -- MPI world communicator
        var -- input field 
        dim -- axis along the derivative should be taken   
        deriv (int) -- first (default) or second derivative
    """
    
    rank = comm.Get_rank()
    size = comm.Get_size()
         
    sl_m1 = slice(None,-2,None)
    sl_p1 = slice(2,None,None)
    sl_c = slice(1,-1,None)
    N = np.array(FFTHelperFuncs.global_shape, dtype=int)
    loc_slc = FFTHelperFuncs.local_shape
    n_proc = N // loc_slc
               
    if (dim == 0):
        next_proc = (rank + n_proc[1]) % (n_proc[0] * n_proc[1])
        prev_proc = (rank - n_proc[1]) % (n_proc[0] * n_proc[1])
        # send right slice of local proc as left slab to follow proc
        leftSlice = None
        leftSlice = comm.sendrecv(sendobj=var[-1:,:,:],dest=next_proc,source=prev_proc)
        # send left slice of local proc as right slab to follow proc
        rightSlice = None
        rightSlice = comm.sendrecv(sendobj=var[:1,:,:],dest=prev_proc,source=next_proc)

        tmp = np.concatenate((leftSlice,var,rightSlice),axis=0)
        p1 = tmp[sl_p1,:,:]
        c0 = tmp[sl_c,:,:]
        m1 = tmp[sl_m1,:,:]
    elif (dim == 1):
        next_proc = (rank + 1) % n_proc[1] + (rank // n_proc[1]) * n_proc[1]
        prev_proc = (rank - 1) % n_proc[1] + (rank // n_proc[1]) * n_proc[1]
        # send right slice of local proc as left slab to follow proc
        leftSlice = None
        leftSlice = comm.sendrecv(sendobj=var[:,-1:,:],dest=next_proc,source=prev_proc)
        # send left slice of local proc as right slab to follow proc
        rightSlice = None
        rightSlice = comm.sendrecv(sendobj=var[:,:1,:],dest=prev_proc,source=next_proc)

        tmp = np.concatenate((leftSlice,var,rightSlice),axis=1)
        p1 = tmp[:,sl_p1,:]
        c0 = tmp[:,sl_c,:]
        m1 = tmp[:,sl_m1,:]
    elif (dim == 2):
        # nothing special required here as we do pencil decomp in x-y
        tmp = np.concatenate((var[:,:,-1:],var,var[:,:,:1]),axis=2)
        p1 = tmp[:,:,sl_p1]
        c0 = tmp[:,:,sl_c]
        m1 = tmp[:,:,sl_m1]
    else:
        print("watch out for dimension!")
                
    del tmp 

    dx = 1.0 / float(var.shape[2])  # assumes axis 2 covers entire grid with L = 1
    if (deriv == 1):
        return np.array((p1 - m1)/(2.*dx))
    elif (deriv == 2):
        return np.array((m1 - 2.0 * c0 + p1)/dx**2.)
    else:
        raise SystemExit('Unknown derivative')

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
def MPIVecLaplacian(comm, X):
    """ return the vector Laplacian of the vector X
    """

    return np.array([(MPIderiv2(comm,X[0],0,deriv=2) +
                      MPIderiv2(comm,X[0],1,deriv=2) +
                      MPIderiv2(comm,X[0],2,deriv=2)),
                     (MPIderiv2(comm,X[1],0,deriv=2) +
                      MPIderiv2(comm,X[1],1,deriv=2) +
                      MPIderiv2(comm,X[1],2,deriv=2)),
                     (MPIderiv2(comm,X[2],0,deriv=2) +
                      MPIderiv2(comm,X[2],1,deriv=2) +
                      MPIderiv2(comm,X[2],2,deriv=2))])
