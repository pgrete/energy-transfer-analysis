"""
BSD 2-Clause License
Author: Lisandro Dalcin and Mikael Mortensen
Contact:    dalcinl@gmail.com or mikaem@math.uio.no

Copyright (c) 2017, Lisandro Dalcin and Mikael Mortensen. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import sys
import heffte
from mpi4py import MPI
#from mpi4py_fft import PFFT, newDistArray
#from fluidfft.fft3d.mpi_with_p3dfft import FFT3DMPIWithP3DFFT as PFFT
#from fluidfft.fft3d.mpi_with_mpi4pyfft import FFT3DMPIWithMPI4PYFFT as PFFT
#from fluidfft.fft3d.mpi_with_fftw1d import FFT3DMPIWithFFTW1D as PFFT

comm  = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()
FFT = None
local_wavenumbermesh = None
local_shape = None
global_shape = None

def setup_fft(res, dtype=np.complex128):
    """ Setup shared FFT object and properties
        res - linear resolution
    """

    global FFT
    global local_wavenumbermesh
    global local_shape
    global global_shape

    if comm.Get_rank() == 0:
        print("""!!! WARNING - CURRENT PITFALLS !!!
        - data units are ignored
        - data is assumed to live on a 3d uniform grid with L = 1
        - for the FFT L = 2 pi is implicitly assumed to work with integer wavenumbers
        """)

    time_start = MPI.Wtime()

    #N = np.array([res, res, res], dtype=int)
    global_shape = np.array([res, res, res], dtype=int)

    r2c = False
    N = res
    # boxes are the indices of the arrays
    # we assume pencil decomposition here
    real_box = heffte.box3d([0, 0, 0], [N - 1, N - 1, N - 1])
    if r2c:
        complex_box = heffte.box3d([0, 0, 0], [N - 1, N - 1, (N//2 + 1) - 1])
        r2c_direction = 2
    else:
        complex_box = real_box

    # decompose into stencils with first dim being smallest
    proc_x = 1
    proc_y = 1
    while (proc_x * proc_y != comm_size):
        proc_x *= 2
        if proc_x * proc_y == comm_size:
            break
        proc_y *= 2

    best_grid = [proc_x, proc_y, 1]
    # get local indices
    real_box = heffte.split_world(real_box, best_grid)[rank]
    complex_box = heffte.split_world(complex_box, best_grid)[rank]
    print(rank, real_box.low, real_box.high, real_box.size, complex_box.low, complex_box.high, complex_box.size)

    if r2c:
        FFT = heffte.fft3d_r2c(heffte.backend.fftw,
                       real_box, complex_box, r2c_direction, comm)
    else:
        FFT = heffte.fft3d(heffte.backend.fftw,
                       real_box, complex_box, comm)

    #local_wavenumbermesh = get_local_wavenumbermesh(real_box.size, complex_box.size, global_shape, r2c)
#    #local_wavenumbermesh = get_local_wavenumbermesh(FFT, L)
    #localK = FFT.get_k_adim_loc()
    local_k_x = np.arange(complex_box.low[0], complex_box.high[0] + 1)
    local_k_y = np.arange(complex_box.low[1], complex_box.high[1] + 1)
    local_k_z = np.arange(complex_box.low[2], complex_box.high[2] + 1)
    local_k_x[local_k_x >= res//2] -= res
    local_k_y[local_k_y >= res//2] -= res
    local_k_z[local_k_z >= res//2] -= res
    localK = [local_k_x, local_k_y, local_k_z]
    #localKdims = FFT.get_shapeK_loc()
    localKdims = complex_box.size

    #k-x
    ifreq = np.fromfunction(lambda i,j,k : localK[0][i], 
        (localKdims[0],localKdims[1],localKdims[2]), dtype=int)
#    #k-y
    jfreq = np.fromfunction(lambda i,j,k : localK[1][j], 
        (localKdims[0],localKdims[1],localKdims[2]), dtype=int)
    #k-z
    kfreq = np.fromfunction(lambda i,j,k : localK[2][k], 
        (localKdims[0],localKdims[1],localKdims[2]), dtype=int)
    
    #freq= np.fft.fftfreq(res, 1./res)

    ##k-x
    #ifreq = np.fromfunction(lambda i,j,k : freq[i], (res,res,res), dtype=int)
    ##k-y
    #jfreq = np.fromfunction(lambda i,j,k : freq[j], (res,res,res), dtype=int)
    ##k-z
    #kfreq = np.fromfunction(lambda i,j,k : freq[k], (res,res,res), dtype=int)

    local_wavenumbermesh = np.array([ifreq,jfreq,kfreq],dtype=np.float64)

    local_shape = tuple(real_box.size)
    #local_wavenumbermesh = get_local_wavenumbermesh(tuple(real_box.size), tuple(complex_box.size), global_shape, r2c)

    time_elapsed = MPI.Wtime() - time_start
    time_elapsed = comm.gather(time_elapsed)

    if comm.Get_rank() == 0:
        print("Setup up FFT and wavenumbers done in %.3g +/- %.3g" %
            (np.mean(time_elapsed), np.std(time_elapsed)))
        sys.stdout.flush()


# from
# https://bitbucket.org/mpi4py/mpi4py-fft/raw/67dfed980115108c76abb7e865860b5da98674f9/examples/spectral_dns_solver.py
# with modification for complex numbers
def get_local_wavenumbermesh(real_shape, complex_shape, global_shape, r2c = False):
    """Returns local wavenumber mesh."""
    s = real_shape
    N = global_shape
    # Set wavenumbers in grid
    if not r2c:
        k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N]
    else:
        k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1./N[-1]).astype(int))
    K = [ki[si] for ki, si in zip(k, s)]
    Ks = np.meshgrid(*K, indexing='ij', sparse=True)
    #Lp = 2*np.pi/L
    Lp = np.array([1,1,1])
    for i in range(3):
        Ks[i] = (Ks[i]*Lp[i]).astype(float)
    return [np.broadcast_to(k, complex_shape) for k in Ks]
