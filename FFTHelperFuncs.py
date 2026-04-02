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
from mpi4py_fft import PFFT, newDistArray
comm  = MPI.COMM_WORLD

FFT = None
local_wavenumbermesh = None
local_shape = None

def setup_fft(res, dtype=np.complex64):
    """ Setup shared FFT object and properties
        res - linear resolution
    """

    global FFT
    global local_wavenumbermesh
    global local_shape

    if comm.Get_rank() == 0:
        print("""!!! WARNING - CURRENT PITFALLS !!!
        - data units are ignored
        - data is assumed to live on a 3d uniform grid with L = 1
        - for the FFT L = 2 pi is implicitly assumed to work with integer wavenumbers
        """)

    time_start = MPI.Wtime()

    if comm.Get_rank() == 0:
        print("Setting up FFT and wavenumbers...")

    N = np.array([res, res, res], dtype=int)
    # using L = 2pi as we work (e.g. when binning) with integer wavenumbers
    L = np.array([2*np.pi, 2*np.pi, 2*np.pi], dtype=float)
    FFT = PFFT(comm, N, axes=(0,1,2), collapse=False, dtype=dtype) # here, we can configure how the data is distributed

    local_wavenumbermesh = get_local_wavenumbermesh(FFT, L)
    local_shape = newDistArray(FFT,False).shape

    time_elapsed = MPI.Wtime() - time_start
    time_elapsed = comm.gather(time_elapsed)

    if comm.Get_rank() == 0:
        print("Setup up FFT and wavenumbers done in %.3g +/- %.3g" %
            (np.mean(time_elapsed), np.std(time_elapsed)))
        sys.stdout.flush()

def get_local_wavenumbermesh(FFT, L):
    """
    Returns the local wavenumber mesh for a 3D FFT.

    Parameters
    ----------
    FFT : mpi4py_fft.FFT or compatible object
        The parallel FFT object with methods .local_slice() and .global_shape().
    L : float or sequence of float
        Box size. If float, same for all dimensions; if sequence, must be length 3.

    Returns
    -------
    Ks : list of np.ndarray
        Local 3D wavenumber arrays [Kx, Ky, Kz] shaped to FFT.local_shape().
    """
    # Ensure L is a sequence of length 3
    if np.isscalar(L):
        L = [L]*3

    # Local slice and global shape
    s = FFT.local_slice()
    N = FFT.global_shape()

    # Compute wavenumber arrays for each dimension
    if FFT.dtype() == np.complex128:
        k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N]
    else:  # real-to-complex last axis
        k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1./N[-1]).astype(int))

    # Select local slices
    K_local = [ki[si] for ki, si in zip(k, s)]

    # Make 3D sparse meshgrid and convert to list for mutability
    Ks = list(np.meshgrid(*K_local, indexing='ij', sparse=True))

    # Scale by 2*pi/L
    for i in range(3):
        Ks[i] = (Ks[i] * (2*np.pi / L[i])).astype(float)

    # Broadcast to full local shape
    Ks_broadcast = [np.broadcast_to(k, FFT.shape(True)) for k in Ks]

    return Ks_broadcast

