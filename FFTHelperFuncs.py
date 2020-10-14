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
box_spec_local = None
box_real_local = None

def setup_fft(res, dtype=np.complex128):
    """ Setup shared FFT object and properties
        res - linear resolution
    """

    global FFT
    global local_wavenumbermesh
    global local_shape
    global global_shape
    global box_spec_local
    global box_real_local

    if comm.Get_rank() == 0:
        print("""!!! WARNING - CURRENT PITFALLS !!!
        - data units are ignored
        - data is assumed to live on a 3d uniform grid with L = 1
        - for the FFT L = 2 pi is implicitly assumed to work with integer wavenumbers
        """)

    time_start = MPI.Wtime()

    #N = np.array([res, res, res], dtype=int)
    N = tuple(np.array([res, res, res], dtype=int))
    global_shape = N

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
    real_box.order = np.array([2, 1, 0], np.int32)
    complex_box.order = np.array([2, 1, 0], np.int32)

    if r2c:
        FFT = heffte.fft3d_r2c(heffte.backend.fftw,
                       real_box, complex_box, r2c_direction, comm)
    else:
        FFT = heffte.fft3d(heffte.backend.fftw,
                       real_box, complex_box, comm)

    box_spec_local = complex_box
    box_real_local = real_box
    local_wavenumbermesh = get_local_wavenumbermesh(res, complex_box)

    local_shape = tuple(real_box.size)

    time_elapsed = MPI.Wtime() - time_start
    time_elapsed = comm.gather(time_elapsed)

    if comm.Get_rank() == 0:
        print("Setup up FFT and wavenumbers done in %.3g +/- %.3g" %
            (np.mean(time_elapsed), np.std(time_elapsed)))
        sys.stdout.flush()

def get_local_wavenumbermesh(N, box_spec_local):
    freq = np.fft.fftfreq(N, 1./N)
    #k-x
    ifreq = np.fromfunction(lambda i,j,k : freq[i + box_spec_local.low[0]], box_spec_local.size, dtype=int)
    #k-y
    jfreq = np.fromfunction(lambda i,j,k : freq[j + box_spec_local.low[1]], box_spec_local.size, dtype=int)
    #k-z
    kfreq = np.fromfunction(lambda i,j,k : freq[k + box_spec_local.low[2]], box_spec_local.size, dtype=int)

    return np.ascontiguousarray([ifreq,jfreq,kfreq], np.float64)
