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
from mpi4py import MPI
#from mpi4py_fft import PFFT, newDistArray
from fluidfft.fft3d.mpi_with_p3dfft import FFT3DMPIWithP3DFFT as PFFT

comm  = MPI.COMM_WORLD

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
    # using L = 2pi as we work (e.g. when binning) with integer wavenumbers
    L = np.array([2*np.pi, 2*np.pi, 2*np.pi], dtype=float)
    #FFT = PFFT(comm, N, axes=(0,1,2), collapse=False, dtype=dtype)
    FFT = PFFT(res, res, res)

    #local_wavenumbermesh = get_local_wavenumbermesh(FFT, L)
    localK = FFT.get_k_adim_loc()
    localKdims = FFT.get_shapeK_loc()

    #k-x
    ifreq = np.fromfunction(lambda i,j,k : localK[0][i], 
        (localKdims[0],localKdims[1],localKdims[2]), dtype=int)
    #k-y
    jfreq = np.fromfunction(lambda i,j,k : localK[1][j], 
        (localKdims[0],localKdims[1],localKdims[2]), dtype=int)
    #k-z
    kfreq = np.fromfunction(lambda i,j,k : localK[2][k], 
        (localKdims[0],localKdims[1],localKdims[2]), dtype=int)
    
    local_wavenumbermesh = np.array([ifreq,jfreq,kfreq])

    #local_shape = newDistArray(FFT,False).shape
    local_shape = FFT.get_shapeX_loc()

    time_elapsed = MPI.Wtime() - time_start
    time_elapsed = comm.gather(time_elapsed)

    if comm.Get_rank() == 0:
        print("Setup up FFT and wavenumbers done in %.3g +/- %.3g" %
            (np.mean(time_elapsed), np.std(time_elapsed)))
        sys.stdout.flush()


# from
# https://bitbucket.org/mpi4py/mpi4py-fft/raw/67dfed980115108c76abb7e865860b5da98674f9/examples/spectral_dns_solver.py
# with modification for complex numbers
def get_local_wavenumbermesh(FFT, L):
    """Returns local wavenumber mesh."""
    s = FFT.local_slice()
    N = FFT.global_shape()
    # Set wavenumbers in grid
    if FFT.dtype() == np.complex128:
        k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N]
    else:
        k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1./N[-1]).astype(int))
    K = [ki[si] for ki, si in zip(k, s)]
    Ks = np.meshgrid(*K, indexing='ij', sparse=True)
    Lp = 2*np.pi/L
    for i in range(3):
        Ks[i] = (Ks[i]*Lp[i]).astype(float)
    return [np.broadcast_to(k, FFT.shape(True)) for k in Ks]
